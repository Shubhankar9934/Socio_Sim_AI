from agents.narrative import validate_narrative_consistency
from llm.prompts import _deterministic_alignment_repair
from simulation.orchestrator import _derive_fallback_flags


def test_rarely_text_conflicts_with_likely_scale():
    scale = ["very unlikely", "unlikely", "neutral", "likely", "very likely"]
    text = "I rarely use ride-hailing apps and almost never book one."
    consistent, _ = validate_narrative_consistency(text, "likely", scale)
    assert consistent is False


def test_deterministic_repair_is_consistent_for_frequency():
    scale = ["never", "rarely", "sometimes", "often", "very often"]
    repaired = _deterministic_alignment_repair("often", scale)
    consistent, _ = validate_narrative_consistency(repaired, "often", scale)
    assert consistent is True


def test_fallback_flags_are_derived_from_trace_markers():
    trace = {
        "fallback_used": "uniform_after_invariant_failure",
        "invariant_failure": "distribution_not_normalized",
        "post_sampling_guard": {"hard_constraint_violation_avoided": True},
        "demographic_plausibility_resample": {"original_choice": "often", "resampled_choice": "rarely"},
    }
    flags = _derive_fallback_flags(trace)
    assert "decision:uniform_after_invariant_failure" in flags
    assert "decision:invariant_failure" in flags
    assert "sampling:hard_constraint_guard" in flags
    assert "sampling:plausibility_resample" in flags
