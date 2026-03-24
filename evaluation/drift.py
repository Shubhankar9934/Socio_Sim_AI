"""
Behavioral drift: flag agents whose responses deviate from persona baseline.
Returns drift magnitudes per agent and can auto-reset drifted agents.
"""

from typing import Any, Dict, List, Optional

from config.option_space import canonicalize_option
from population.personas import Persona

_ANSWER_TO_BEHAVIOR: Dict[str, float] = {
    "multiple": 0.95,
    "daily": 0.85,
    "3-4": 0.65,
    "3x": 0.65,
    "1-2": 0.45,
    "rarely": 0.2,
    "never": 0.1,
}


def drift_score(
    initial_behavior: float,
    current_behavior: float,
) -> float:
    """Absolute difference; 0 = no drift, 1 = max drift."""
    return abs(initial_behavior - current_behavior)


def infer_current_behavior(response_history: List[Dict[str, Any]]) -> float:
    """Map the latest survey answers to a 0-1 behavioral estimate."""
    if not response_history:
        return 0.5
    current = 0.5
    value_map = {
        "rarely": 0.2,
        "1-2 per week": 0.45,
        "3-4 per week": 0.65,
        "daily": 0.85,
        "multiple per day": 0.95,
    }
    for r in response_history:
        sampled = r.get("sampled_option_canonical") or r.get("sampled_option") or r.get("answer") or ""
        canonical = canonicalize_option("food_delivery_frequency", str(sampled))
        if canonical in value_map:
            current = value_map[canonical]
    return current


def detect_drift(
    agent_id: str,
    persona: Persona,
    response_history: List[Dict[str, Any]],
    threshold: float = 0.3,
) -> tuple[bool, float]:
    """Returns (is_drifted, drift_magnitude)."""
    baseline = persona.lifestyle.primary_service_preference
    if not response_history:
        return False, 0.0
    current = infer_current_behavior(response_history)
    magnitude = drift_score(baseline, current)
    return magnitude > threshold, magnitude


def drift_report(
    personas: List[Persona],
    response_histories: Dict[str, List[Dict[str, Any]]],
    threshold: float = 0.3,
    agent_states: Optional[Dict[str, Any]] = None,
    auto_reset: bool = False,
) -> Dict[str, Any]:
    """Returns drift analysis with per-agent magnitudes.

    If auto_reset=True and agent_states are provided, resets drifted agents.
    """
    drifted = []
    per_agent: Dict[str, float] = {}
    reset_count = 0

    for p in personas:
        hist = response_histories.get(p.agent_id, [])
        is_drifted, magnitude = detect_drift(p.agent_id, p, hist, threshold)
        per_agent[p.agent_id] = magnitude
        if is_drifted:
            drifted.append(p.agent_id)
            if auto_reset and agent_states and p.agent_id in agent_states:
                state = agent_states[p.agent_id]
                if hasattr(state, "reset_behavior"):
                    state.reset_behavior(blend=0.7)
                    reset_count += 1

    n = len(personas)
    return {
        "drifted_agent_ids": drifted,
        "count": len(drifted),
        "rate": len(drifted) / n if n else 0.0,
        "threshold": threshold,
        "per_agent_magnitude": per_agent,
        "auto_reset_count": reset_count,
    }
