"""
Cognitive Dissonance Layer: adjusts answer probability distributions toward
consistency with the agent's stored beliefs and behavioral state.

Without this layer, an agent could answer "I rarely order delivery" in one
survey and "I order delivery daily" in the next.  With dissonance modeling,
the second answer shifts only slightly (e.g. "1-2 per week") unless strong
external signals push the agent far enough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from agents.state import AgentState
    from config.question_models import QuestionModel


def compute_dynamic_strength(agent_state: "AgentState | None" = None) -> float:
    """Derive dissonance strength from calcification and identity.

    strength = 0.3 + 0.4 * calcification + 0.3 * identity_strength

    More calcified / identity-strong agents resist change harder.
    Falls back to 0.5 if no state is available.
    """
    if agent_state is None:
        return 0.5
    calc = getattr(agent_state, "calcification", 0.0)
    identity = getattr(agent_state, "identity", None)
    id_strength = identity.identity_strength if identity is not None else 0.0
    return 0.3 + 0.4 * float(calc) + 0.3 * float(id_strength)


def dissonance_penalty(old_score: float, new_score: float, strength: float = 0.5) -> float:
    """Exponential penalty for deviating from a prior belief/behavior score.

    Returns 1.0 when old == new, decaying toward 0 as the gap grows.
    ``strength`` controls how aggressively large deviations are penalized.
    """
    return float(np.exp(-strength * abs(old_score - new_score)))


def apply_cognitive_dissonance(
    dist: Dict[str, float],
    consistency_score: float,
    scale: List[str],
    strength: float = 0.5,
    agent_state: "AgentState | None" = None,
) -> Dict[str, float]:
    """Re-weight a probability distribution toward belief-consistent options.

    Each answer option is mapped to a numeric position on [0, 1] based on its
    ordinal rank in ``scale``.  Options closer to ``consistency_score`` receive
    a multiplicative bonus; distant options are penalized.

    The result is re-normalized so probabilities sum to 1.
    """
    n = len(scale)
    if n <= 1:
        return dict(dist)

    # Use dynamic strength derived from agent identity + calcification when available.
    if agent_state is not None:
        strength = compute_dynamic_strength(agent_state)

    adjusted: Dict[str, float] = {}
    for i, option in enumerate(scale):
        option_score = i / max(1, n - 1)
        penalty = dissonance_penalty(consistency_score, option_score, strength)
        adjusted[option] = dist.get(option, 0.0) * penalty

    total = sum(adjusted.values())
    if total <= 0:
        return dict(dist)
    return {k: v / total for k, v in adjusted.items()}


def compute_consistency_score(
    agent_state: "AgentState",
    question_model: "QuestionModel",
) -> float:
    """Blend behavioral and belief scores into a single consistency anchor.

    The consistency score is a 50/50 mix of:
    - BehavioralLatentState score (behavioral inertia)
    - BeliefNetwork score (attitudinal consistency)

    Falls back to 0.5 if either component is unavailable.
    """
    from config.belief_mappings import get_belief_dimensions
    from config.question_models import get_behavioral_dimensions

    behavior_score = 0.5
    belief_score = 0.5

    dim_weights = get_behavioral_dimensions(question_model.name)
    if dim_weights and hasattr(agent_state, "latent_state"):
        behavior_score = agent_state.latent_state.behavioral_score(dim_weights)

    belief_weights = get_belief_dimensions(question_model.name)
    if belief_weights and hasattr(agent_state, "beliefs"):
        belief_score = agent_state.beliefs.belief_score(belief_weights)

    return 0.5 * behavior_score + 0.5 * belief_score
