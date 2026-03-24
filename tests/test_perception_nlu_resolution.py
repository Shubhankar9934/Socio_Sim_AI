"""NLU effective options resolution and turn_understanding telemetry (intent-agnostic)."""

from agents.perception import (
    Perception,
    _enrich_turn_understanding_telemetry,
    _resolve_effective_nlu_options,
)


class _S:
    nlu_fallback_options_from_model_scale = True


class _SNoFallback:
    nlu_fallback_options_from_model_scale = False


def _survey_food_frequency():
    return Perception(
        topic="food_delivery",
        domain="transport",
        location_related=False,
        keywords=[],
        raw_question="How often do you order food delivery?",
        scale_type="frequency",
        question_type="frequency",
        question_model_key="food_delivery_frequency",
        structured_response_expected=True,
        interaction_mode="survey",
    )


def test_resolve_prefers_caller_options():
    p = _survey_food_frequency()
    opts, src = _resolve_effective_nlu_options(
        p, ["rarely", "often"], settings=_S(),
    )
    assert src == "caller"
    assert opts == ["rarely", "often"]


def test_resolve_model_scale_fallback_for_structured_without_caller_options():
    p = _survey_food_frequency()
    opts, src = _resolve_effective_nlu_options(p, None, settings=_S())
    assert src == "model_scale_fallback"
    assert len(opts) >= 3
    assert "rarely" in [x.lower() for x in opts]


def test_resolve_skips_fallback_when_setting_off():
    p = _survey_food_frequency()
    opts, src = _resolve_effective_nlu_options(p, None, settings=_SNoFallback())
    assert src == "none"
    assert opts == []


def test_resolve_no_fallback_for_conversation_intent():
    p = Perception(
        topic="general",
        domain="general",
        location_related=False,
        keywords=[],
        raw_question="hi there",
        structured_response_expected=False,
        interaction_mode="conversation",
    )
    opts, src = _resolve_effective_nlu_options(p, None, settings=_S())
    assert opts == [] and src == "none"


def test_enrich_turn_understanding_telemetry_in_place():
    tu = {"rule_payload": {"provided_options": []}}
    _enrich_turn_understanding_telemetry(
        tu,
        interaction_mode="survey",
        effective_options=["A", "B"],
        options_source="caller",
    )
    assert tu["nlu_interaction_mode"] == "survey"
    assert tu["nlu_provided_options_count"] == 2
    assert len(tu["nlu_provided_options_preview"]) == 2
    assert tu["nlu_options_source"] == "caller"


def test_enrich_open_mode_zero_count():
    tu = {}
    _enrich_turn_understanding_telemetry(
        tu,
        interaction_mode="conversation",
        effective_options=[],
        options_source="none",
    )
    assert tu["nlu_provided_options_count"] == 0
    assert tu["nlu_provided_options_preview"] == []
