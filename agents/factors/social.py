"""
Social influence factor: combines scalar friends_using with latent-state
neighbor alignment for multi-dimensional social context.

When neighbor latent means are available (passed via environment during
survey mode), the factor considers how similar the agent's behavioral
profile is to their social circle.  Otherwise falls back to the scalar
friends_using metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext

_SATURATION_THRESHOLD = 0.6


def social_factor(ctx: "DecisionContext") -> float:
    """Multi-dimensional social influence score.

    If ``neighbor_latent_mean`` is in the environment (injected by the
    survey orchestrator's warmup pass), the factor blends scalar
    friends_using with the cosine similarity between the agent's
    latent state and their neighbors' mean state.

    This captures richer social context than a simple adoption fraction:
    an agent surrounded by high-convenience neighbors will score differently
    than one surrounded by budget-conscious neighbors, even if the same
    fraction of friends "use delivery".
    """
    env = ctx.environment or {}
    neighbor_mean = env.get("neighbor_latent_mean")
    agent_latent = env.get("latent_state")

    base = min(1.0, ctx.friends_using / _SATURATION_THRESHOLD) if _SATURATION_THRESHOLD > 0 else 0.5

    if neighbor_mean is not None and agent_latent is not None:
        try:
            agent_vec = agent_latent.to_vector()
            if isinstance(neighbor_mean, np.ndarray) and len(neighbor_mean) == len(agent_vec):
                dot = float(np.dot(agent_vec, neighbor_mean))
                norm = float(np.linalg.norm(agent_vec) * np.linalg.norm(neighbor_mean))
                similarity = dot / max(norm, 1e-8)
                return min(1.0, 0.55 * base + 0.45 * max(0.0, similarity))
        except (AttributeError, TypeError):
            pass

    return base
