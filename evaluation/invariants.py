"""Formal system invariants that must hold after every agent response.

Each invariant returns an ``InvariantResult`` with pass/fail, message, and
severity.  ``run_agent_invariants`` is the entry-point called from
``AgentCognitiveEngine.think()`` after each response.  Population-level
invariants are exposed separately for use by the survey engine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import math

if TYPE_CHECKING:
    from agents.cognitive import CognitiveTrace
    from agents.state import AgentState


@dataclass
class InvariantResult:
    passed: bool
    message: str
    severity: str = "warning"  # "info" | "warning" | "error"


# ---------------------------------------------------------------------------
# Agent-level invariants
# ---------------------------------------------------------------------------

def invariant_consistency(agent_state: AgentState, new_answer: str, semantic_key: str) -> InvariantResult:
    """No answer may contradict a stored answer for same/related key by > 2 levels."""
    if not semantic_key or not agent_state.structured_memory:
        return InvariantResult(passed=True, message="no memory to check")
    prev = agent_state.structured_memory.get(semantic_key)
    if prev is None:
        return InvariantResult(passed=True, message="first answer for key")
    prev_answer = str(prev.get("answer", ""))
    if prev_answer.lower().strip() == new_answer.lower().strip():
        return InvariantResult(passed=True, message="identical answer")

    _ANSWER_LEVEL_MAP = {
        "never": 0, "rarely": 1, "sometimes": 2, "often": 3,
        "very often": 4, "always": 5,
        "strongly disagree": 0, "disagree": 1, "neutral": 2, "agree": 3, "strongly agree": 4,
        "very dissatisfied": 0, "dissatisfied": 1, "satisfied": 3, "very satisfied": 4,
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    }
    prev_level = _ANSWER_LEVEL_MAP.get(prev_answer.lower().strip())
    new_level = _ANSWER_LEVEL_MAP.get(new_answer.lower().strip())
    if prev_level is not None and new_level is not None:
        gap = abs(prev_level - new_level)
        if gap > 2:
            return InvariantResult(
                passed=False,
                message=f"contradiction: '{prev_answer}' -> '{new_answer}' (gap={gap}) for key={semantic_key}",
                severity="warning",
            )
    return InvariantResult(passed=True, message="consistent")


def invariant_intent_routing(intent_class: str, pipeline_triggered: bool) -> InvariantResult:
    """GREETING/ACKNOWLEDGMENT intents must NOT trigger LPFG decision pipeline."""
    non_pipeline_intents = {"greeting", "acknowledgment"}
    if intent_class in non_pipeline_intents and pipeline_triggered:
        return InvariantResult(
            passed=False,
            message=f"intent '{intent_class}' triggered full decision pipeline",
            severity="error",
        )
    return InvariantResult(passed=True, message="intent routing correct")


def invariant_personality_stability(trace: CognitiveTrace) -> InvariantResult:
    """Confidence band should roughly correlate with personality (informational only)."""
    return InvariantResult(passed=True, message="personality check: informational only")


def invariant_fatigue_monotonicity(agent_state: AgentState) -> InvariantResult:
    """Fatigue must be non-decreasing within a session (when feature is enabled)."""
    from config.settings import get_settings
    if not getattr(get_settings(), "enable_fatigue", True):
        return InvariantResult(passed=True, message="fatigue disabled")
    fatigue = getattr(agent_state, "fatigue", 0.0)
    turn = getattr(agent_state, "turn_count", 0)
    if turn > 1 and fatigue <= 0.0:
        return InvariantResult(
            passed=False,
            message=f"fatigue={fatigue} at turn={turn} (should be > 0)",
            severity="warning",
        )
    return InvariantResult(passed=True, message=f"fatigue={fatigue:.2f} at turn={turn}")


def run_agent_invariants(
    agent_state: AgentState,
    intent_class: str,
    pipeline_triggered: bool,
    trace: CognitiveTrace,
) -> List[str]:
    """Run all agent-level invariants, return list of violation messages."""
    violations: List[str] = []
    checks = [
        invariant_intent_routing(intent_class, pipeline_triggered),
        invariant_fatigue_monotonicity(agent_state),
        invariant_personality_stability(trace),
    ]
    for result in checks:
        if not result.passed:
            violations.append(f"[{result.severity}] {result.message}")
    return violations


# ---------------------------------------------------------------------------
# Population-level invariants
# ---------------------------------------------------------------------------

def invariant_scale_entropy(answer_counts: Dict[str, int], min_entropy: float = 0.5) -> InvariantResult:
    """Population entropy must exceed threshold (anti-collapse)."""
    total = sum(answer_counts.values())
    if total == 0:
        return InvariantResult(passed=True, message="no answers")
    probs = [c / total for c in answer_counts.values() if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    if normalized < min_entropy:
        return InvariantResult(
            passed=False,
            message=f"population entropy={normalized:.3f} < threshold={min_entropy}",
            severity="warning",
        )
    return InvariantResult(passed=True, message=f"entropy={normalized:.3f}")


def check_population_invariants(round_responses: List[Dict[str, Any]]) -> List[str]:
    """After each round, check population-level health."""
    violations: List[str] = []
    if not round_responses:
        return violations

    answer_counts: Dict[str, int] = Counter()
    for resp in round_responses:
        opt = resp.get("sampled_option", "")
        if opt:
            answer_counts[opt] += 1

    if answer_counts:
        entropy_check = invariant_scale_entropy(answer_counts)
        if not entropy_check.passed:
            violations.append(f"[{entropy_check.severity}] {entropy_check.message}")

        total = sum(answer_counts.values())
        if total > 0:
            max_frac = max(answer_counts.values()) / total
            if max_frac > 0.60:
                violations.append(
                    f"[warning] dominant option has {max_frac:.1%} of responses (collapse risk)"
                )

    dup_answers = Counter(r.get("answer", "") for r in round_responses if r.get("answer"))
    total_answers = sum(dup_answers.values())
    if total_answers > 10:
        exact_dup_rate = sum(1 for c in dup_answers.values() if c > 1) / max(1, len(dup_answers))
        if exact_dup_rate > 0.3:
            violations.append(f"[warning] high duplicate narrative rate: {exact_dup_rate:.1%}")

    return violations
