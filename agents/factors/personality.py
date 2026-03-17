"""
Personality factor: scores the agent based on trait–dimension alignment.

Reads dimension_weights from the QuestionModel (stored in
context.environment["dimension_weights"]) and computes a weighted
sum over the agent's PersonalityTraits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext


def personality_factor(ctx: "DecisionContext") -> float:
    """Weighted sum of traits using the question model's dimension_weights."""
    weights = ctx.environment.get("dimension_weights", {})
    if not weights:
        return 0.5

    score = 0.0
    total = 0.0
    for trait_name, w in weights.items():
        value = getattr(ctx.traits, trait_name, 0.5)
        score += w * value
        total += abs(w)

    if total == 0:
        return 0.5
    return max(0.0, min(1.0, score / total))
