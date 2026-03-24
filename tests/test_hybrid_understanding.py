import asyncio
import random

import pytest

from agents.intent_router import (
    _build_cache_key,
    build_turn_understanding_hybrid,
    build_turn_understanding_rules,
    clear_turn_understanding_cache,
)
from llm.prompts import (
    _SYSTEM_PROMPTS,
    _pick_system_prompt,
    _system_prompt_mandates_casual_fillers,
    build_agent_prompt,
)
from population.synthesis import generate_population


class _FakeClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, *_args, **_kwargs):
        self.calls += 1
        return self.payload


def _llm_payload(
    *,
    interaction_mode: str,
    question_type: str = "categorical",
    scale_type: str = "categorical",
    topic: str = "general",
    domain: str = "general",
    question_model_key_candidate: str = "generic_likert",
    persona_anchor_allowed: bool = False,
    action_type_candidate: str = "choose",
    target_candidate: str = "behavior",
    intensity_scale_candidate: str = "ordinal",
    confidence: float = 0.95,
    reason: str = "test_stub",
) -> str:
    return f"""
{{
  "interaction_mode": "{interaction_mode}",
  "question_type": "{question_type}",
  "scale_type": "{scale_type}",
  "topic": "{topic}",
  "domain": "{domain}",
  "location_related": false,
  "question_model_key_candidate": "{question_model_key_candidate}",
  "persona_anchor_allowed": {str(persona_anchor_allowed).lower()},
  "action_type_candidate": "{action_type_candidate}",
  "target_candidate": "{target_candidate}",
  "intensity_scale_candidate": "{intensity_scale_candidate}",
  "normalization_candidates": [],
  "confidence": {confidence},
  "reason": "{reason}"
}}
""".strip()


def test_clear_turn_understanding_cache():
    from agents.intent_router import _TURN_UNDERSTANDING_CACHE

    _TURN_UNDERSTANDING_CACHE["_test_marker"] = None  # type: ignore[assignment]
    clear_turn_understanding_cache()
    assert len(_TURN_UNDERSTANDING_CACHE) == 0


def test_cache_key_differs_by_question_id_and_options():
    q = "How are grocery prices affecting your household?"
    k1 = _build_cache_key(q, None, None, survey_run_id="run1", question_id="id-a")
    k2 = _build_cache_key(q, None, None, survey_run_id="run1", question_id="id-b")
    k3 = _build_cache_key(q, None, ["opt1", "opt2"], survey_run_id="run1", question_id="id-a")
    assert k1 != k2
    assert k1 != k3


def test_same_question_different_question_id_triggers_two_llm_calls(monkeypatch):
    monkeypatch.setattr("agents.intent_router._TURN_UNDERSTANDING_CACHE", {})
    fake = _FakeClient(_llm_payload(interaction_mode="survey", topic="cost_of_living", domain="housing"))
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: fake)

    q = "Same survey wording for two studies?"
    asyncio.run(build_turn_understanding_hybrid(q, question_id="study-1"))
    asyncio.run(build_turn_understanding_hybrid(q, question_id="study-2"))

    assert fake.calls == 2


def test_greeting_keeps_conversation_when_llm_disagrees(monkeypatch):
    monkeypatch.setattr("agents.intent_router._TURN_UNDERSTANDING_CACHE", {})
    fake = _FakeClient(_llm_payload(interaction_mode="survey", confidence=0.98))
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: fake)

    result = asyncio.run(build_turn_understanding_hybrid("hi"))

    assert fake.calls == 1
    assert result.interaction_mode == "conversation"
    assert result.question_type == "open_text"
    assert result.question_model_key_candidate == "generic_open_text"


def test_explicit_survey_keeps_survey_when_llm_disagrees(monkeypatch):
    monkeypatch.setattr("agents.intent_router._TURN_UNDERSTANDING_CACHE", {})
    fake = _FakeClient(_llm_payload(interaction_mode="conversation", confidence=0.99))
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: fake)

    result = asyncio.run(
        build_turn_understanding_hybrid("How likely are you to try a new fintech wallet next month?")
    )

    assert fake.calls == 1
    assert result.interaction_mode == "survey"
    assert result.question_type == "likelihood"
    assert result.scale_type == "likelihood"


def test_prompt_uses_turn_understanding_for_anchor_gating():
    persona = generate_population(1, method="bayesian", seed=133)[0]
    prompt = build_agent_prompt(
        persona=persona,
        question="How likely are you to try a new fintech wallet next month?",
        sampled_option="Likely",
        distribution={
            "Very unlikely": 0.05,
            "Unlikely": 0.10,
            "Neutral": 0.15,
            "Likely": 0.45,
            "Very likely": 0.25,
        },
        memories=[],
        turn_understanding={
            "scale_type": "likelihood",
            "persona_anchor_allowed": False,
        },
    )
    assert "Cuisine preference:" not in prompt
    assert "Diet:" not in prompt
    assert "Hobby:" not in prompt


def test_housing_rules_omit_lifestyle_anchors_in_prompt():
    persona = generate_population(1, method="bayesian", seed=201)[0]
    q = "How satisfied are you with your current housing situation?"
    tu = build_turn_understanding_rules(q)
    prompt = build_agent_prompt(
        persona=persona,
        question=q,
        sampled_option="Neutral",
        distribution={"Dissatisfied": 0.2, "Neutral": 0.6, "Satisfied": 0.2},
        memories=[],
        turn_understanding=dict(tu),
    )
    assert "Cuisine preference:" not in prompt
    assert "Hobby:" not in prompt


def test_build_agent_prompt_always_includes_voice_hedge_limit():
    persona = generate_population(1, method="bayesian", seed=502)[0]
    prompt = build_agent_prompt(
        persona=persona,
        question="How often do you use food delivery apps?",
        sampled_option="1-2 per week",
        distribution={"rarely": 0.1, "1-2 per week": 0.5, "3-4 per week": 0.4},
        memories=[],
    )
    assert "VOICE LIMIT" in prompt


def test_pick_system_prompt_strict_skips_filler_mandates():
    if not any(_system_prompt_mandates_casual_fillers(p) for p in _SYSTEM_PROMPTS):
        pytest.skip("no filler-mandate system prompts in domain")
    rng = random.Random(0)
    for _ in range(50):
        sp = _pick_system_prompt(
            rng, voice_register="conversational", strict_demographic_voice=True,
        )
        assert not _system_prompt_mandates_casual_fillers(sp)


def test_build_agent_prompt_includes_demographic_voice_for_senior_low_income():
    persona = generate_population(1, method="bayesian", seed=404)[0]
    persona = persona.model_copy(update={"age": "60-65", "income": "<10k"})
    prompt = build_agent_prompt(
        persona=persona,
        question="How are rising food prices affecting your shopping?",
        sampled_option="Cutting back on extras",
        distribution={"Cutting back on extras": 0.5, "No change": 0.5},
        memories=[],
    )
    assert "VOICE (demographics)" in prompt
    assert "budget" in prompt.lower() or "Strictly avoid" in prompt or "filler" in prompt.lower()


def test_demographic_voice_strict_filler_when_rhetorical_habit_direct():
    persona = generate_population(1, method="bayesian", seed=407)[0]
    pa = persona.personal_anchors
    ns = pa.narrative_style.model_copy(update={"rhetorical_habit": "direct"})
    persona = persona.model_copy(
        update={"age": "55-60", "personal_anchors": pa.model_copy(update={"narrative_style": ns})},
    )
    prompt = build_agent_prompt(
        persona=persona,
        question="How satisfied are you with public transport?",
        sampled_option="Neutral",
        distribution={"Dissatisfied": 0.2, "Neutral": 0.6, "Satisfied": 0.2},
        memories=[],
    )
    assert "VOICE (demographics)" in prompt
    assert "Strictly avoid" in prompt


def test_demographic_voice_on_open_text_prompt():
    persona = generate_population(1, method="bayesian", seed=406)[0]
    persona = persona.model_copy(update={"age": "60-65", "income": "25k-40k"})
    prompt = build_agent_prompt(
        persona=persona,
        question="What is stressing you most about daily life lately?",
        sampled_option="",
        distribution={},
        memories=[],
        response_contract={
            "expression_mode": "open_expression",
            "confidence_band": "low",
            "latent_stance": "mixed",
        },
        turn_understanding={"scale_type": "open_text", "persona_anchor_allowed": True},
    )
    assert "VOICE (demographics)" in prompt
    assert "Strictly avoid" in prompt or "measured" in prompt.lower()


def test_build_agent_prompt_skips_demographic_filler_ban_for_narrative_habit():
    persona = generate_population(1, method="bayesian", seed=405)[0]
    pa = persona.personal_anchors
    ns = pa.narrative_style.model_copy(update={"rhetorical_habit": "narrative"})
    persona = persona.model_copy(
        update={
            "age": "60-65",
            "income": "<10k",
            "personal_anchors": pa.model_copy(update={"narrative_style": ns}),
        },
    )
    prompt = build_agent_prompt(
        persona=persona,
        question="How are rising food prices affecting your shopping?",
        sampled_option="Cutting back on extras",
        distribution={"Cutting back on extras": 0.6, "No change": 0.4},
        memories=[],
    )
    assert "VOICE (demographics)" in prompt
    assert "Do NOT start sentences with" not in prompt
    assert "budget" in prompt.lower() or "tradeoff" in prompt.lower()


def test_prompt_uses_turn_understanding_scale_type_for_open_text():
    persona = generate_population(1, method="bayesian", seed=144)[0]
    prompt = build_agent_prompt(
        persona=persona,
        question="Tell me how you feel about your commute lately.",
        sampled_option="",
        distribution={},
        memories=[],
        turn_understanding={
            "scale_type": "open_text",
            "persona_anchor_allowed": True,
        },
        response_contract={
            "expression_mode": "open_expression",
            "confidence_band": "low",
            "latent_stance": "mixed",
        },
    )
    assert "Do NOT mention ratings, scales, or numeric scores." in prompt
