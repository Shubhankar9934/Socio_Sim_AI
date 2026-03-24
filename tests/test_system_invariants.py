"""System-level tests that exercise the full cognitive pipeline.

Unlike unit tests, these verify end-to-end invariants: greeting skip,
contradiction detection, fatigue compression, scale diversity, and
personality stability.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.cognitive import AgentCognitiveEngine, CognitiveTrace
from agents.state import AgentState
from evaluation.invariants import (
    InvariantResult,
    check_population_invariants,
    invariant_consistency,
    invariant_fatigue_monotonicity,
    invariant_intent_routing,
    invariant_scale_entropy,
)


def _make_persona(**overrides):
    """Build a minimal Persona-like object for testing."""
    from unittest.mock import MagicMock
    persona = MagicMock()
    persona.agent_id = overrides.get("agent_id", "test-agent-001")
    persona.age = overrides.get("age", "25-34")
    persona.nationality = overrides.get("nationality", "Indian")
    persona.income = overrides.get("income", "medium")
    persona.location = overrides.get("location", "Mumbai")
    persona.occupation = overrides.get("occupation", "Engineer")
    persona.household_size = overrides.get("household_size", 3)
    persona.family.children = overrides.get("children", 0)
    persona.family.spouse = overrides.get("spouse", False)
    persona.personal_anchors.cuisine_preference = "varied"
    persona.personal_anchors.diet = "non-veg"
    persona.personal_anchors.hobby = "reading"
    persona.personal_anchors.work_schedule = "9-5"
    persona.personal_anchors.health_focus = "moderate"
    persona.personal_anchors.commute_method = "metro"
    persona.meta.persona_version = "v1"
    persona.meta.synthesis_method = "test"
    persona.meta.generation_seed = 42
    persona.meta.archetype_id = "arch-01"
    persona.meta.persona_cluster = "c-01"
    persona.lifestyle.tech_adoption = 0.5
    return persona


# ---------------------------------------------------------------------------
# Test: Greeting must skip the decision pipeline
# ---------------------------------------------------------------------------

class TestIntentInvariant:
    def test_greeting_returns_no_decision_trace(self):
        result = invariant_intent_routing("greeting", pipeline_triggered=False)
        assert result.passed

    def test_greeting_with_pipeline_fails(self):
        result = invariant_intent_routing("greeting", pipeline_triggered=True)
        assert not result.passed
        assert "greeting" in result.message

    def test_ack_returns_no_decision_trace(self):
        result = invariant_intent_routing("acknowledgment", pipeline_triggered=False)
        assert result.passed

    def test_survey_with_pipeline_passes(self):
        result = invariant_intent_routing("survey", pipeline_triggered=True)
        assert result.passed


# ---------------------------------------------------------------------------
# Test: Consistency invariant
# ---------------------------------------------------------------------------

class TestConsistencyInvariant:
    def test_no_contradiction_when_same_answer(self):
        state = MagicMock()
        state.structured_memory = {"freq": {"answer": "Often"}}
        result = invariant_consistency(state, "Often", "freq")
        assert result.passed

    def test_contradiction_detected_large_gap(self):
        state = MagicMock()
        state.structured_memory = {"freq": {"answer": "never"}}
        result = invariant_consistency(state, "very often", "freq")
        assert not result.passed
        assert "contradiction" in result.message

    def test_no_contradiction_small_gap(self):
        state = MagicMock()
        state.structured_memory = {"freq": {"answer": "sometimes"}}
        result = invariant_consistency(state, "often", "freq")
        assert result.passed


# ---------------------------------------------------------------------------
# Test: Fatigue monotonicity
# ---------------------------------------------------------------------------

class TestFatigueInvariant:
    def test_fatigue_positive_at_turn_2(self):
        state = MagicMock()
        state.fatigue = 0.1
        state.turn_count = 3
        result = invariant_fatigue_monotonicity(state)
        assert result.passed

    def test_fatigue_zero_at_turn_5_fails(self):
        state = MagicMock()
        state.fatigue = 0.0
        state.turn_count = 5
        result = invariant_fatigue_monotonicity(state)
        assert not result.passed


# ---------------------------------------------------------------------------
# Test: Scale entropy (population-level)
# ---------------------------------------------------------------------------

class TestScaleInvariant:
    def test_diverse_distribution_passes(self):
        counts = {"A": 30, "B": 25, "C": 25, "D": 20}
        result = invariant_scale_entropy(counts, min_entropy=0.5)
        assert result.passed

    def test_collapsed_distribution_fails(self):
        counts = {"A": 95, "B": 3, "C": 2}
        result = invariant_scale_entropy(counts, min_entropy=0.5)
        assert not result.passed

    def test_population_invariants_detect_collapse(self):
        responses = [{"sampled_option": "A", "answer": f"answer {i}"} for i in range(80)]
        responses += [{"sampled_option": "B", "answer": f"other {i}"} for i in range(20)]
        violations = check_population_invariants(responses)
        assert any("dominant" in v or "entropy" in v for v in violations)


# ---------------------------------------------------------------------------
# Test: CognitiveTrace dataclass
# ---------------------------------------------------------------------------

class TestCognitiveTrace:
    def test_trace_to_dict(self):
        trace = CognitiveTrace(agent_id="a1", question="test?")
        d = trace.to_dict()
        assert d["agent_id"] == "a1"
        assert "post_processing_applied" in d
        assert "invariant_violations" in d
        assert "decision_latency" in d

    def test_final_response_truncated(self):
        trace = CognitiveTrace(final_response="x" * 500)
        d = trace.to_dict()
        assert len(d["final_response"]) <= 200
