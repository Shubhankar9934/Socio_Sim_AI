"""
Factor Graph Decision Engine: composable behavioral factors for agent decisions.

Each factor is a callable (DecisionContext) -> float in [0, 1].
The FactorGraph computes a weighted average across all registered factors,
producing a single behavioral score that feeds into the softmax distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.perception import Perception
from agents.personality import PersonalityTraits
from population.personas import Persona

FactorFn = Callable[["DecisionContext"], float]


@dataclass
class DecisionContext:
    """All inputs a factor can read when computing its score."""

    persona: Persona
    traits: PersonalityTraits
    perception: Perception
    friends_using: float = 0.0
    location_quality: float = 0.5
    memories: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)


class FactorGraph:
    """Weighted collection of behavioral factors.

    Each factor returns a score in [0, 1].  The graph computes
    a normalised weighted average across all factors.
    """

    __slots__ = ("_factors",)

    def __init__(self) -> None:
        self._factors: List[Tuple[FactorFn, float]] = []

    def add_factor(self, func: FactorFn, weight: float = 1.0) -> None:
        self._factors.append((func, weight))

    def compute(self, context: DecisionContext) -> float:
        if not self._factors:
            return 0.5

        score = 0.0
        total_weight = 0.0
        for fn, w in self._factors:
            raw = fn(context)
            clamped = max(0.0, min(1.0, raw))
            score += w * clamped
            total_weight += abs(w)

        if total_weight == 0:
            return 0.5
        return max(0.0, min(1.0, score / total_weight))


# ---------------------------------------------------------------------------
# Precompiled graph cache — one FactorGraph instance per QuestionModel name
# ---------------------------------------------------------------------------

_GRAPH_CACHE: Dict[str, FactorGraph] = {}


def get_or_build_graph(
    model_name: str,
    builder: Callable[[], FactorGraph],
) -> FactorGraph:
    """Return a cached FactorGraph for *model_name*, building it on first access."""
    if model_name not in _GRAPH_CACHE:
        _GRAPH_CACHE[model_name] = builder()
    return _GRAPH_CACHE[model_name]


def clear_graph_cache() -> None:
    """Reset the precompiled graph cache (useful in tests)."""
    _GRAPH_CACHE.clear()
