"""Unified stochastic controller for post-processing transforms.

Replaces scattered independent probability constants with a coherent
``BehaviorBudget`` that adapts to agent state and caps the total
number of transforms that can fire on a single response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


@dataclass
class BehaviorBudget:
    """Probability bundle for one response's post-processing."""
    hedge_prob: float = 0.20
    thinking_marker_prob: float = 0.35
    redundancy_prob: float = 0.18
    micro_contradiction_prob: float = 0.12
    fragment_prob: float = 0.32
    polish_prob: float = 0.45
    vague_prob: float = 0.30
    max_transforms: int = 3


class BehaviorController:
    """Computes a coherent BehaviorBudget from agent state."""

    @staticmethod
    def compute_budget(
        fatigue: float = 0.0,
        confidence_band: str = "medium",
        grammar_quality: float = 0.5,
        settings: "Settings | None" = None,
    ) -> BehaviorBudget:
        budget = BehaviorBudget()

        # Fatigue increases vague/fragment, decreases hedging and redundancy
        if fatigue > 0.5:
            budget.vague_prob = min(0.6, 0.30 + fatigue * 0.3)
            budget.hedge_prob = max(0.05, 0.20 - fatigue * 0.15)
            budget.redundancy_prob = max(0.05, 0.18 - fatigue * 0.1)
            budget.fragment_prob = min(0.55, 0.32 + fatigue * 0.2)

        # High confidence decreases hedging and contradiction
        if confidence_band == "high":
            budget.hedge_prob = max(0.03, budget.hedge_prob * 0.4)
            budget.micro_contradiction_prob = 0.0
            budget.max_transforms = 2
        elif confidence_band == "low":
            budget.hedge_prob = min(0.40, budget.hedge_prob * 1.5)
            budget.thinking_marker_prob = min(0.50, budget.thinking_marker_prob * 1.3)
            budget.max_transforms = 4

        # Low grammar quality increases markers and fragments
        if grammar_quality < 0.4:
            budget.thinking_marker_prob = min(0.55, budget.thinking_marker_prob * 1.4)
            budget.fragment_prob = min(0.55, budget.fragment_prob * 1.3)

        # Chaotic profile: raise cap
        if settings is not None and getattr(settings, "simulation_profile", "realistic") == "chaotic":
            budget.max_transforms = 5
        elif settings is not None and getattr(settings, "simulation_profile", "realistic") == "strict":
            budget.max_transforms = 0

        return budget
