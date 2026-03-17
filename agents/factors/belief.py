"""
Belief factor: reads the agent's BeliefNetwork dimensions relevant to the
current question and produces a single [0,1] factor score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext


def belief_factor(ctx: "DecisionContext") -> float:
    """Score derived from the agent's BeliefNetwork.

    Uses ``belief_dimension_weights`` from the environment (populated by
    config/belief_mappings.py) to produce a weighted combination of the
    agent's belief dimensions.  Falls back to 0.5 if unavailable.
    """
    beliefs = ctx.environment.get("beliefs")
    if beliefs is None:
        return 0.5

    dim_weights = ctx.environment.get("belief_dimension_weights", {})
    if not dim_weights:
        return 0.5

    return beliefs.belief_score(dim_weights)
