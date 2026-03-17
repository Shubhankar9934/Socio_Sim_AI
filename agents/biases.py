"""
Bounded-Rational Cognitive Bias Engine.

Implements five psychological biases as distribution transforms, stabilized by
residual mixing (reality anchor) and entropy injection (uncertainty floor).

Master equation:
    D_final = Normalize( (1 - eps) * [gamma * F(D0) + (1 - gamma) * D0] + eps * U )

where F = B_availability . B_bandwagon . B_anchoring . B_loss . B_confirmation,
gamma is dynamic bias susceptibility, eps is the entropy factor, and U is
uniform (pure uncertainty).

Stability guarantees:
  - gamma clipped to [0.05, 0.95]   -> agents never fully irrational or rigid
  - epsilon clipped to [0.01, 0.3]  -> irreducible uncertainty floor
  - residual mixing applied once at the end (not per-bias)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def normalize(dist: np.ndarray) -> np.ndarray:
    """Clip to positive and normalize to sum=1."""
    d = np.clip(dist, 1e-6, None)
    return d / d.sum()


def _dict_to_array(dist: Dict[str, float], scale: List[str]) -> np.ndarray:
    return np.array([dist.get(opt, 0.0) for opt in scale], dtype=np.float64)


def _array_to_dict(arr: np.ndarray, scale: List[str]) -> Dict[str, float]:
    normed = normalize(arr)
    return {opt: float(normed[i]) for i, opt in enumerate(scale)}


# ---------------------------------------------------------------------------
# Dynamic gamma (bias susceptibility)
# ---------------------------------------------------------------------------

def compute_gamma(
    agent_state: Any,
    context: Dict[str, Any],
    *,
    gamma_floor: float = 0.05,
    gamma_ceiling: float = 0.95,
) -> float:
    """Bias susceptibility: how much the bias chain overrides baseline reasoning.

    gamma = effective_malleability * topic_term * (1 - calcification)
    """
    base_malleability = getattr(agent_state, "base_malleability", 0.5)
    effective = max(0.1, base_malleability)

    topic_importance = context.get("topic_importance", 0.5)
    topic_term = (1.0 - topic_importance) ** 1.5

    calcification = getattr(agent_state, "calcification", 0.0)

    gamma = effective * topic_term * (1.0 - calcification)
    return float(np.clip(gamma, gamma_floor, gamma_ceiling))


# ---------------------------------------------------------------------------
# Dynamic epsilon (entropy / uncertainty)
# ---------------------------------------------------------------------------

def compute_media_conflict(media_distributions: List[np.ndarray]) -> float:
    """Mean pairwise Euclidean distance between media frame distributions."""
    if len(media_distributions) < 2:
        return 0.0
    distances: List[float] = []
    for i in range(len(media_distributions)):
        for j in range(i + 1, len(media_distributions)):
            distances.append(float(np.linalg.norm(
                media_distributions[i] - media_distributions[j]
            )))
    return float(np.clip(np.mean(distances), 0.0, 1.0))


def compute_epsilon(
    agent_state: Any,
    context: Dict[str, Any],
    *,
    epsilon_base: float = 0.05,
    w_knowledge: float = 0.1,
    w_conflict: float = 0.15,
    epsilon_floor: float = 0.01,
    epsilon_ceiling: float = 0.3,
) -> float:
    """Entropy factor: irreducible uncertainty from ignorance and conflict."""
    knowledge_levels = getattr(agent_state, "knowledge_levels", {})
    topic = context.get("topic", "")
    knowledge_level = knowledge_levels.get(topic, 0.5) if knowledge_levels else 0.5

    media_conflict = context.get("media_conflict", 0.0)

    epsilon = epsilon_base + w_knowledge * (1.0 - knowledge_level) + w_conflict * media_conflict
    return float(np.clip(epsilon, epsilon_floor, epsilon_ceiling))


# ---------------------------------------------------------------------------
# Individual bias functions
# ---------------------------------------------------------------------------

def apply_confirmation_bias(
    dist: np.ndarray,
    agent_beliefs: np.ndarray,
    option_belief_vectors: Optional[np.ndarray] = None,
    strength: float = 0.3,
) -> np.ndarray:
    """Boost options aligned with existing beliefs via cosine similarity.

    Parameters
    ----------
    dist : (K,) current distribution
    agent_beliefs : (B,) agent belief vector (7-dim)
    option_belief_vectors : (K, B) implied belief vector per option, or None
    strength : multiplicative strength of the bias
    """
    k = len(dist)
    if option_belief_vectors is None:
        positions = np.linspace(0.0, 1.0, k)
        belief_center = float(np.mean(agent_beliefs))
        alignment = 1.0 - np.abs(positions - belief_center)
        boost = 1.0 + strength * alignment
    else:
        norms_a = np.linalg.norm(agent_beliefs)
        if norms_a < 1e-8:
            return dist.copy()
        sims = option_belief_vectors @ agent_beliefs / (
            np.linalg.norm(option_belief_vectors, axis=1) * norms_a + 1e-8
        )
        boost = 1.0 + strength * np.clip(sims, -1, 1)

    return dist * boost


def apply_loss_aversion(
    dist: np.ndarray,
    current_score: float,
    scale_len: int,
    aversion_factor: float = 2.0,
) -> np.ndarray:
    """Weight options representing 'loss' with aversion_factor penalty.

    Options below the agent's current behavioral score are penalized more
    heavily than equivalent gains above it.
    """
    k = scale_len
    positions = np.linspace(0.0, 1.0, k)
    delta = positions - current_score

    weights = np.where(
        delta < 0,
        1.0 / (1.0 + aversion_factor * np.abs(delta)),
        1.0 + 0.5 * delta,
    )
    return dist * weights


def apply_anchoring(
    dist: np.ndarray,
    prior_dist: Optional[np.ndarray] = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """Weighted blend of prior distribution and new evidence.

    new = alpha * prior + (1 - alpha) * current
    """
    if prior_dist is None or len(prior_dist) != len(dist):
        return dist.copy()
    prior_norm = normalize(prior_dist)
    return alpha * prior_norm + (1.0 - alpha) * dist


def apply_bandwagon_effect(
    dist: np.ndarray,
    neighbor_dist: np.ndarray,
    susceptibility: float = 0.5,
    strength: float = 0.3,
) -> np.ndarray:
    """Shift toward modal choice in social neighborhood.

    Scaled by agent's social_influence_susceptibility.
    """
    if neighbor_dist is None or len(neighbor_dist) != len(dist):
        return dist.copy()
    neighbor_norm = normalize(neighbor_dist)
    blend_weight = strength * susceptibility
    blend_weight = min(blend_weight, 0.5)
    return (1.0 - blend_weight) * dist + blend_weight * neighbor_norm


def apply_availability_heuristic(
    dist: np.ndarray,
    recent_event_scores: Optional[np.ndarray] = None,
    recency_weight: float = 0.25,
) -> np.ndarray:
    """Boost options related to recent/vivid events.

    recent_event_scores: (K,) per-option relevance scores from recent events.
    Uses exponential weighting so recent events dominate.
    """
    if recent_event_scores is None:
        return dist.copy()
    boost = 1.0 + recency_weight * np.clip(recent_event_scores, 0.0, 1.0)
    return dist * boost


# ---------------------------------------------------------------------------
# Master bias pipeline
# ---------------------------------------------------------------------------

def apply_all_biases(
    dist_dict: Dict[str, float],
    scale: List[str],
    agent_state: Any,
    context: Dict[str, Any],
    *,
    neighbor_dist_dict: Optional[Dict[str, float]] = None,
    prior_dist_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Apply the full stabilized bias pipeline.

    D_final = Normalize( (1-eps) * [gamma * F(D0) + (1-gamma) * D0] + eps * U )

    Parameters
    ----------
    dist_dict : initial distribution from factor graph {option: prob}
    scale : ordered list of options
    agent_state : AgentState with beliefs, latent_state, malleability, etc.
    context : dict with topic_importance, media_conflict, topic, etc.
    neighbor_dist_dict : social neighborhood distribution (optional)
    prior_dist_dict : agent's prior distribution from previous round (optional)
    """
    if len(scale) < 2:
        return dict(dist_dict)

    D0 = _dict_to_array(dist_dict, scale)
    D = D0.copy()

    beliefs = getattr(agent_state, "beliefs", None)
    agent_beliefs = beliefs.to_vector() if beliefs is not None else np.full(7, 0.5)

    latent = getattr(agent_state, "latent_state", None)

    # --- 1. Confirmation bias ---
    D = apply_confirmation_bias(D, agent_beliefs)

    # --- 2. Loss aversion ---
    current_score = 0.5
    if latent is not None:
        dim_weights = context.get("behavioral_dimension_weights", {})
        if dim_weights:
            current_score = latent.behavioral_score(dim_weights)
    D = apply_loss_aversion(D, current_score, len(scale))

    # --- 3. Anchoring ---
    prior = _dict_to_array(prior_dist_dict, scale) if prior_dist_dict else None
    D = apply_anchoring(D, prior)

    # --- 4. Bandwagon ---
    if neighbor_dist_dict is not None:
        neighbor_arr = _dict_to_array(neighbor_dist_dict, scale)
        susceptibility = latent.social_influence_susceptibility if latent else 0.5
        D = apply_bandwagon_effect(D, neighbor_arr, susceptibility)

    # --- 5. Availability heuristic ---
    event_scores = context.get("recent_event_scores")
    if event_scores is not None:
        D = apply_availability_heuristic(D, np.asarray(event_scores))

    D_biased = normalize(D)

    # --- 6. Residual mixing (reality anchor) ---
    gamma = compute_gamma(agent_state, context)
    D_mixed = gamma * D_biased + (1.0 - gamma) * normalize(D0)

    # --- 7. Entropy injection (uncertainty floor) ---
    k = len(scale)
    U = np.ones(k) / k
    epsilon = compute_epsilon(agent_state, context)
    D_final = (1.0 - epsilon) * D_mixed + epsilon * U

    return _array_to_dict(normalize(D_final), scale)
