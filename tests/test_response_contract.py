from types import SimpleNamespace

from agents.response_contract import (
    build_response_contract,
    compute_confidence_band,
    compute_expected_score,
    enforce_survey_response,
)
from llm.prompts import build_agent_prompt
from population.synthesis import generate_population
from simulation.archetype_runner import _build_response_dict


def test_confidence_band_and_expected_score_for_structured_distribution():
    dist = {
        "never": 0.05,
        "rarely": 0.10,
        "sometimes": 0.15,
        "often": 0.20,
        "very often": 0.50,
    }
    expected = compute_expected_score(dist, ordered_options=list(dist.keys()))
    assert expected is not None
    assert expected > 0.70
    assert compute_confidence_band(max(dist.values())) == "medium"

    contract = build_response_contract(
        scale_type="frequency",
        sampled_option="very often",
        distribution=dist,
        ordered_options=list(dist.keys()),
    )
    payload = contract.to_dict()
    assert payload["expression_mode"] == "structured_expression"
    assert payload["confidence_band"] == "medium"
    assert payload["tone_selected"] in {"casual", "reflective", "emotional_practical"}


def test_open_expression_contract_uses_latent_state_when_distribution_empty():
    latent = SimpleNamespace(to_vector=lambda: [0.82, 0.76, 0.80])
    contract = build_response_contract(
        scale_type="open_text",
        sampled_option="",
        distribution={},
        latent_state=latent,
    )
    payload = contract.to_dict()
    assert payload["expression_mode"] == "open_expression"
    assert payload["expected_score"] is not None
    assert payload["expected_score"] > 0.70
    assert payload["latent_stance"] in {"positive", "positive_strong"}


def test_open_prompt_includes_no_scale_leakage_guardrail_from_contract():
    persona = generate_population(1, method="bayesian", seed=77)[0]
    prompt = build_agent_prompt(
        persona=persona,
        question="How do you feel about your commute these days?",
        sampled_option="",
        distribution={},
        memories=[],
        response_contract={
            "expression_mode": "open_expression",
            "confidence_band": "low",
            "latent_stance": "mixed",
        },
    )
    assert "Do NOT mention ratings, scales, or numeric scores." in prompt
    assert "Use mild uncertainty language" in prompt


def test_archetype_response_dict_hides_diagnostics_by_default():
    persona = generate_population(1, method="bayesian", seed=91)[0]
    agent = {"persona": persona}

    no_diag = _build_response_dict(agent, "answer text", "often", {"often": 1.0})
    assert "response_diagnostics" not in no_diag

    with_diag = _build_response_dict(
        agent,
        "answer text",
        "often",
        {"often": 1.0},
        response_diagnostics={"expression_mode": "structured_expression"},
    )
    assert with_diag["response_diagnostics"]["expression_mode"] == "structured_expression"


def test_enforce_survey_response_prefixes_option_when_missing():
    out = enforce_survey_response(
        "I think it is fine for my budget.",
        scale_type="likert",
        sampled_option="Agree",
        interaction_mode="survey",
    )
    assert out.lower().startswith("agree")
    assert out.endswith(".")


def test_enforce_skips_option_echo_for_open_text():
    out = enforce_survey_response(
        "just feeling tired lately",
        scale_type="open_text",
        sampled_option="",
        interaction_mode="survey",
    )
    assert "just feeling tired lately" in out.lower()
    assert out.endswith(".")


def test_enforce_long_categorical_uses_compact_anchor():
    long_opt = (
        "Exploring alternatives — I will start bulk-buying staples at discount stores "
        "and reduce restaurant meals significantly for the next quarter."
    )
    text = "Honestly prices pushed me to cook more at home anyway."
    out = enforce_survey_response(
        text,
        scale_type="categorical",
        sampled_option=long_opt,
        interaction_mode="survey",
    )
    assert not out.startswith(long_opt)
    assert "exploring alternatives" in out.lower()
    assert "I'd go with" in out
    assert "…" not in out.split(".")[0]


def test_short_option_label_hyphen_takes_lead_clause():
    long_opt = (
        "Exploring alternatives - I will start bulk-buying, hunting for weekly deals, "
        "and reduce restaurant meals."
    )
    out = enforce_survey_response(
        "Prices nudged me toward cooking more anyway.",
        scale_type="categorical",
        sampled_option=long_opt,
        interaction_mode="survey",
    )
    assert "Exploring alternatives" in out
    assert "bulk-buying, hunting" not in out.split(".")[0]
    assert "…" not in out.split(".")[0]


def test_enforce_long_option_token_overlap_skips_verbatim_prefix():
    long_opt = (
        "Exploring alternatives — I will start bulk-buying staples at discount stores."
    )
    text = "I'm exploring alternatives and started bulk-buying staples to save money."
    out = enforce_survey_response(
        text,
        scale_type="categorical",
        sampled_option=long_opt,
        interaction_mode="survey",
    )
    assert "going with this choice" not in out.lower()
    assert not out.startswith(long_opt)


def test_opening_pool_excludes_trainer_for_low_income():
    import random

    from agents.narrative import NarrativeStyleProfile, pick_opening_deduplicated

    p = NarrativeStyleProfile(
        verbosity="medium",
        preferred_tone="casual",
        preferred_style="routine",
        slang_level=0.4,
        grammar_quality=0.6,
        voice_register="conversational",
        rhetorical_habit="direct",
        avoid_phrases=(),
    )
    rng = random.Random(42)
    used: set = set()
    for _ in range(50):
        o = pick_opening_deduplicated({}, used_openings=used, rng=rng, profile=p, income_band="<10k")
        assert "trainer" not in o.lower()
