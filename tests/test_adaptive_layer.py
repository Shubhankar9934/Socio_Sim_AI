import asyncio

import numpy as np

from agents.adaptive_layer import adaptive_generate_and_register
from agents.decision import compute_distribution
from agents.factor_graph import DecisionContext
from agents.perception import perceive_with_llm
from agents.personality import personality_from_persona
from agents.response_contract import build_response_contract
from agents.state import AgentState
from config.generated_registry import load_generated_registry
from config.question_models import QUESTION_MODELS
from llm.prompts import build_agent_prompt
from population.synthesis import generate_population


class _FakeClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, *_args, **_kwargs):
        self.calls += 1
        return self.payload


def _valid_generated_payload() -> str:
    return """
{
  "model_key": "night_safety_satisfaction",
  "scale": ["1", "2", "3", "4", "5"],
  "reference_distribution": {"1": 0.22, "2": 0.24, "3": 0.23, "4": 0.18, "5": 0.13},
  "factors": ["income", "location", "social", "belief"],
  "dominant_factors": ["location", "belief"],
  "semantic_profile": "safety_perception",
  "narrative_guidance": {"1": "sound unsafe and worried", "5": "sound secure and confident"},
  "constraints": [
    {"if": {"income": "<10k", "household_size": "5+"}, "disallow": ["5"]}
  ]
}
""".strip()


def _exercise_generated_payload() -> str:
    return """
{
  "model_key": "exercise_intent",
  "scale": ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"],
  "reference_distribution": {"Very unlikely": 0.12, "Unlikely": 0.18, "Neutral": 0.22, "Likely": 0.28, "Very likely": 0.20},
  "factors": ["behavioral", "personality", "belief"],
  "dominant_factors": ["behavioral", "personality"],
  "semantic_profile": "health_behavior",
  "narrative_guidance": {"Very likely": "sound motivated and active", "Very unlikely": "sound resistant and unmotivated"},
  "constraints": []
}
""".strip()


def _conversation_only_payload() -> str:
    return """
{
  "classification": "conversation_only"
}
""".strip()


def _intent_payload(intent: str, confidence: float = 0.95, reason: str = "test_stub") -> str:
    return f"""
{{
  "intent": "{intent}",
  "confidence": {confidence},
  "reason": "{reason}"
}}
""".strip()


def test_adaptive_generation_cached_and_valid(monkeypatch, tmp_path):
    registry_path = tmp_path / "generated_models.json"
    monkeypatch.setattr("config.generated_registry._REGISTRY_PATH", registry_path)
    fake = _FakeClient(_valid_generated_payload())
    monkeypatch.setattr("agents.adaptive_layer.get_llm_client", lambda: fake)

    question = "How safe do you feel walking alone at night in your area?"
    key1 = asyncio.run(adaptive_generate_and_register(question))
    key2 = asyncio.run(adaptive_generate_and_register(question))

    assert key1 == "night_safety_satisfaction"
    assert key2 == key1
    assert fake.calls == 1

    registry = load_generated_registry()
    ref = registry["references"][key1]
    assert abs(sum(ref.values()) - 1.0) < 1e-9
    uniform = 1.0 / len(ref)
    assert max(abs(v - uniform) for v in ref.values()) >= 0.03


def test_adaptive_generated_constraints_are_applied(monkeypatch, tmp_path):
    registry_path = tmp_path / "generated_models.json"
    monkeypatch.setattr("config.generated_registry._REGISTRY_PATH", registry_path)
    fake = _FakeClient(_valid_generated_payload())
    monkeypatch.setattr("agents.adaptive_layer.get_llm_client", lambda: fake)

    key = asyncio.run(adaptive_generate_and_register("How safe do you feel walking at night?"))
    assert key in QUESTION_MODELS

    base = generate_population(1, method="bayesian", seed=920)[0]
    persona = base.model_copy(update={"income": "<10k", "household_size": "5+"})
    state = AgentState.from_persona(persona)
    traits = personality_from_persona(persona)
    ctx = DecisionContext(
        persona=persona,
        traits=traits,
        perception=type("P", (), {"raw_question": "How safe do you feel walking at night?"})(),
        friends_using=0.1,
        location_quality=0.5,
        memories=[],
        environment={"activation": 0.0},
    )
    dist = compute_distribution(QUESTION_MODELS[key], ctx, agent_state=state, persona=persona, traits=traits, rng=np.random.default_rng(5))
    assert dist["5"] == 0.0


def test_perceive_with_llm_bootstraps_known_question(monkeypatch):
    calls = {"n": 0}
    intent_fake = _FakeClient(_intent_payload("survey"))

    async def _fake_bootstrap(question: str, **_kwargs):
        calls["n"] += 1
        return "cost_of_living_satisfaction"

    monkeypatch.setattr("agents.adaptive_layer.adaptive_generate_and_register", _fake_bootstrap)
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: intent_fake)
    perception = asyncio.run(perceive_with_llm("How satisfied are you with the cost of living in your city?"))
    assert calls["n"] == 1
    assert intent_fake.calls == 1
    assert perception.question_model_key == "cost_of_living_satisfaction"
    assert perception.resolution_source == "adaptive"


def test_adaptive_generation_can_abstain_on_conversation(monkeypatch, tmp_path):
    registry_path = tmp_path / "generated_models.json"
    monkeypatch.setattr("config.generated_registry._REGISTRY_PATH", registry_path)
    fake = _FakeClient(_conversation_only_payload())
    monkeypatch.setattr("agents.adaptive_layer.get_llm_client", lambda: fake)

    key = asyncio.run(adaptive_generate_and_register("hello there"))
    assert key is None
    assert not registry_path.exists()


def test_perceive_with_llm_skips_adaptive_for_conversation(monkeypatch):
    calls = {"n": 0}
    intent_fake = _FakeClient(_intent_payload("survey", confidence=0.91, reason="intentional_conflict"))

    async def _fake_bootstrap(question: str, **_kwargs):
        calls["n"] += 1
        return "cost_of_living_satisfaction"

    monkeypatch.setattr("agents.adaptive_layer.adaptive_generate_and_register", _fake_bootstrap)
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: intent_fake)
    perception = asyncio.run(perceive_with_llm("hi"))
    assert calls["n"] == 0
    assert intent_fake.calls == 1
    assert perception.interaction_mode == "conversation"
    assert perception.question_model_key == "generic_open_text"


def test_perceive_with_llm_uses_hybrid_router_for_qualitative(monkeypatch):
    intent_fake = _FakeClient(_intent_payload("qualitative_interview"))
    monkeypatch.setattr("agents.intent_router.get_llm_client", lambda: intent_fake)

    perception = asyncio.run(perceive_with_llm("Tell me about your daily routine and what you enjoy most."))

    assert intent_fake.calls == 1
    assert perception.interaction_mode == "qualitative_interview"
    assert perception.adaptive_allowed is False
    assert perception.question_model_key == "generic_open_text"


def test_health_behavior_dominance_separates_active_and_inactive_personas(monkeypatch, tmp_path):
    registry_path = tmp_path / "generated_models.json"
    monkeypatch.setattr("config.generated_registry._REGISTRY_PATH", registry_path)
    fake = _FakeClient(_exercise_generated_payload())
    monkeypatch.setattr("agents.adaptive_layer.get_llm_client", lambda: fake)
    key = asyncio.run(adaptive_generate_and_register("How likely are you to exercise three times a week?"))
    assert key == "exercise_intent"

    base = generate_population(1, method="bayesian", seed=1200)[0]
    active = base.model_copy(
        update={
            "personal_anchors": base.personal_anchors.model_copy(
                update={"health_focus": "fitness-focused", "hobby": "gym"}
            )
        }
    )
    inactive = base.model_copy(
        update={
            "personal_anchors": base.personal_anchors.model_copy(
                update={"health_focus": "don't think about it", "hobby": "watching TV"}
            )
        }
    )

    def _score(persona):
        state = AgentState.from_persona(persona)
        traits = personality_from_persona(persona)
        ctx = DecisionContext(
            persona=persona,
            traits=traits,
            perception=type("P", (), {"raw_question": "How likely are you to exercise three times a week?"})(),
            friends_using=0.1,
            location_quality=0.5,
            memories=[],
            environment={"activation": 0.0},
        )
        dist = compute_distribution(QUESTION_MODELS[key], ctx, agent_state=state, persona=persona, traits=traits, rng=np.random.default_rng(9))
        scale = QUESTION_MODELS[key].scale
        index = {label: i for i, label in enumerate(scale)}
        return sum(index[label] * prob for label, prob in dist.items()), max(dist, key=dist.get)

    active_score, active_top = _score(active)
    inactive_score, inactive_top = _score(inactive)
    assert active_score > inactive_score
    assert active_top in {"Likely", "Very likely"}
    assert inactive_top in {"Very unlikely", "Unlikely", "Neutral"}


def test_prompt_includes_dominant_factor_guidance():
    persona = generate_population(1, method="bayesian", seed=888)[0]
    dist = {"1": 0.08, "2": 0.17, "3": 0.19, "4": 0.31, "5": 0.25}
    contract = build_response_contract(
        scale_type="likert",
        sampled_option="4",
        distribution=dist,
        ordered_options=list(dist.keys()),
        latent_state=None,
        decision_trace={
            "dominance": {"factor": "income", "raw_scalar": 0.82},
            "question_spec": {"narrative_guidance": {"4": "sound cautiously positive, not mixed"}},
        },
    ).to_dict()
    prompt = build_agent_prompt(
        persona,
        "How satisfied are you with the cost of living in your city?",
        "4",
        dist,
        memories=[],
        response_contract=contract,
    )
    assert "mainly driven by this factor: income" in prompt
    assert "sound cautiously positive, not mixed" in prompt

