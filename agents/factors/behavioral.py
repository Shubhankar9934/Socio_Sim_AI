"""
Behavioral factor: reads the agent's latent behavioral state dimensions
relevant to the current question and produces a single [0,1] factor score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext


def behavioral_factor(ctx: "DecisionContext") -> float:
    """Score derived from the agent's BehavioralLatentState.

    Uses ``dimension_weights`` from the active QuestionModel (stored in
    ``ctx.environment["behavioral_dimension_weights"]``) to produce a
    weighted combination of the agent's latent dimensions.
    Falls back to 0.5 if the latent state or weights are unavailable.
    """
    latent = ctx.environment.get("latent_state")
    if latent is None:
        return 0.5

    dim_weights = ctx.environment.get("behavioral_dimension_weights", {})
    if not dim_weights:
        return 0.5

    return latent.behavioral_score(dim_weights)
