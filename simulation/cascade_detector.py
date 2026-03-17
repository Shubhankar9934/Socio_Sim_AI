"""
Emergent Event Generation via Cascade Detection.

Activation dynamics -> sparse cluster detection -> score-based event
classification -> issue fatigue.

Core activation equation:
    A_{t+1} = clip( decay * A_t + sat * (media_term + social_term), 0, 1 )

where sat = (1 - A_t) provides diminishing returns and prevents runaway.
Outrage is weighted 2x heavier than validation (w_out=0.6 vs w_val=0.3).

Cluster detection uses boolean masking + scipy sparse connected_components
for O(V_active + E_active) performance at 100k+ agents.

Event classification is score-based (not brittle if-else) with continuous
transitions between protest, viral campaign, rumor, and movement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy.sparse import issparse
    from scipy.sparse.csgraph import connected_components
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Activation dynamics (vectorized)
# ---------------------------------------------------------------------------

def update_activation(
    A_t: np.ndarray,
    exposure: np.ndarray,
    emotional_intensity: np.ndarray,
    topic_importance: np.ndarray,
    alignment: np.ndarray,
    neighbor_activation: np.ndarray,
    susceptibility: np.ndarray,
    *,
    decay: float = 0.85,
    w_val: float = 0.3,
    w_out: float = 0.6,
    lambda_social: float = 0.2,
) -> np.ndarray:
    """Vectorized activation update for N agents.

    Parameters
    ----------
    A_t : (N,) current activation
    exposure : (N,) exposure intensity
    emotional_intensity : (N,) emotional charge of consumed media
    topic_importance : (N,) per-agent topic importance
    alignment : (N,) belief-media alignment in [-1, 1]
    neighbor_activation : (N,) mean activation of graph neighbors
    susceptibility : (N,) social influence susceptibility
    """
    validation = np.maximum(0.0, alignment)
    outrage = np.maximum(0.0, -alignment)

    media_term = (
        exposure * emotional_intensity * topic_importance
        * (w_val * validation + w_out * outrage)
    )
    social_term = lambda_social * susceptibility * neighbor_activation

    sat = 1.0 - A_t
    delta_A = sat * (media_term + social_term)

    A_next = decay * A_t + delta_A
    return np.clip(A_next, 0.0, 1.0)


def compute_neighbor_activation(
    A: np.ndarray,
    adjacency_norm: Any,
) -> np.ndarray:
    """Compute mean neighbor activation via sparse matrix multiply.

    adjacency_norm should be a row-normalized sparse adjacency matrix
    (or dense). Returns (N,) array.
    """
    if adjacency_norm is None:
        return np.zeros_like(A)
    result = adjacency_norm @ A
    if hasattr(result, 'A1'):
        return result.A1
    return np.asarray(result).ravel()


# ---------------------------------------------------------------------------
# Issue fatigue (cascade stability)
# ---------------------------------------------------------------------------

def apply_fatigue(
    activation: np.ndarray,
    cluster: Dict[str, Any],
    cooldown_topics: Optional[Dict[str, Dict[str, int]]] = None,
    *,
    fatigue_factor: float = 0.3,
    cooldown_days: int = 5,
    topic: str = "",
) -> None:
    """Post-event fatigue: drop activation for cluster participants.

    Modifies activation array in-place. Optionally updates per-agent
    cooldown_topics dict for suppressed re-activation.
    """
    indices = cluster["indices"]
    activation[indices] *= fatigue_factor

    if cooldown_topics is not None and topic:
        for idx in indices:
            agent_key = str(idx)
            if agent_key not in cooldown_topics:
                cooldown_topics[agent_key] = {}
            cooldown_topics[agent_key][topic] = cooldown_days


def apply_cooldown_suppression(
    delta_A: np.ndarray,
    agent_indices: np.ndarray,
    cooldown_topics: Dict[str, Dict[str, int]],
    topic: str,
    cooldown_decay: float = 0.1,
) -> np.ndarray:
    """Suppress activation growth for agents in cooldown for a topic."""
    for idx in agent_indices:
        key = str(idx)
        remaining = cooldown_topics.get(key, {}).get(topic, 0)
        if remaining > 0:
            delta_A[idx] *= cooldown_decay
    return delta_A


def tick_cooldowns(cooldown_topics: Dict[str, Dict[str, int]]) -> None:
    """Decrement all cooldown counters by 1 day; remove expired."""
    expired_agents = []
    for agent_key, topics in cooldown_topics.items():
        expired = [t for t, d in topics.items() if d <= 1]
        for t in expired:
            del topics[t]
        if not topics:
            expired_agents.append(agent_key)
        else:
            for t in topics:
                topics[t] -= 1
    for k in expired_agents:
        del cooldown_topics[k]


# ---------------------------------------------------------------------------
# Cluster detection (scalable)
# ---------------------------------------------------------------------------

def detect_activation_clusters(
    A: np.ndarray,
    adjacency_matrix: Any,
    *,
    activation_threshold: float = 0.8,
    min_size_absolute: int = 200,
    min_size_fraction: float = 0.005,
    min_density: float = 0.01,
) -> List[Dict[str, Any]]:
    """Detect dense clusters of highly activated agents.

    Uses boolean masking + scipy connected_components for O(V_active + E_active).
    Hybrid threshold: max(absolute_floor, population * fraction).
    Density filter rejects loose chains.
    """
    if not HAS_SCIPY or adjacency_matrix is None:
        return []

    if not issparse(adjacency_matrix):
        return []

    active_mask = A > activation_threshold
    n_active = active_mask.sum()
    if n_active == 0:
        return []

    active_indices = np.where(active_mask)[0]
    sub_adj = adjacency_matrix[active_mask][:, active_mask]

    n_components, labels = connected_components(sub_adj, directed=False)

    total_pop = len(A)
    dynamic_threshold = max(min_size_absolute, int(total_pop * min_size_fraction))

    clusters: List[Dict[str, Any]] = []
    for c in range(n_components):
        local_idx = np.where(labels == c)[0]
        size = len(local_idx)

        if size < dynamic_threshold:
            continue

        subgraph = sub_adj[local_idx][:, local_idx]
        edges = subgraph.nnz / 2
        max_edges = size * (size - 1) / 2
        density = edges / (max_edges + 1e-6)

        if density < min_density:
            continue

        global_idx = active_indices[local_idx]
        clusters.append({
            "indices": global_idx,
            "size": size,
            "density": density,
            "mean_activation": float(A[global_idx].mean()),
        })

    return clusters


# ---------------------------------------------------------------------------
# Score-based event classification
# ---------------------------------------------------------------------------

def generate_emergent_event(
    cluster: Dict[str, Any],
    agent_states: Dict[str, np.ndarray],
    total_population: int,
    *,
    activation_gate: float = 0.75,
) -> Optional[Dict[str, Any]]:
    """Classify a cluster into an event type using continuous scoring.

    agent_states is a dict of vectorized arrays keyed by field name:
      activation, activation_prev, novelty_seeking, social_influence_susceptibility,
      government_trust, beliefs (2D array), topic_importance
    """
    indices = cluster["indices"]
    size = cluster["size"]
    density = cluster["density"]

    A = agent_states.get("activation")
    if A is None:
        return None

    mean_activation = float(A[indices].mean())
    if mean_activation < activation_gate:
        return None

    size_norm = size / max(total_population, 1)

    A_prev = agent_states.get("activation_prev")
    if A_prev is not None:
        activation_growth = float((A[indices] - A_prev[indices]).mean())
    else:
        activation_growth = 0.0
    activation_growth = max(0.0, activation_growth)

    novelty = agent_states.get("novelty_seeking")
    novelty_mean = float(novelty[indices].mean()) if novelty is not None else 0.5

    suscept = agent_states.get("social_influence_susceptibility")
    suscept_mean = float(suscept[indices].mean()) if suscept is not None else 0.5

    trust = agent_states.get("government_trust")
    trust_mean = float(trust[indices].mean()) if trust is not None else 0.5

    beliefs = agent_states.get("beliefs")
    if beliefs is not None and beliefs.ndim == 2:
        belief_polarization = float(np.var(beliefs[indices], axis=0).mean())
    else:
        belief_polarization = 0.0

    topic_imp = agent_states.get("topic_importance")
    topic_importance = float(topic_imp[indices].mean()) if topic_imp is not None else 0.5

    stability = max(0.0, 1.0 - activation_growth)

    # --- Scores ---
    score_protest = (
        0.30 * mean_activation
        + 0.25 * density
        + 0.20 * size_norm
        + 0.15 * (1.0 - trust_mean)
        + 0.10 * topic_importance
    )
    score_viral = (
        0.35 * activation_growth
        + 0.25 * novelty_mean
        + 0.20 * suscept_mean
        + 0.10 * size_norm
        + 0.10 * density
    )
    score_rumor = (
        0.30 * (1.0 - trust_mean)
        + 0.25 * activation_growth
        + 0.20 * belief_polarization
        + 0.15 * size_norm
        + 0.10 * (1.0 - density)
    )
    score_movement = (
        0.30 * size_norm
        + 0.25 * density
        + 0.20 * mean_activation
        + 0.15 * stability
        + 0.10 * topic_importance
    )

    scores = {
        "protest": score_protest,
        "viral_campaign": score_viral,
        "rumor": score_rumor,
        "movement": score_movement,
    }
    event_type = max(scores, key=scores.get)

    return {
        "type": event_type,
        "size": size,
        "density": density,
        "mean_activation": mean_activation,
        "confidence": scores[event_type],
        "scores": scores,
        "agent_indices": indices.tolist() if hasattr(indices, "tolist") else list(indices),
    }


# ---------------------------------------------------------------------------
# Regime / Phase Transition Detection
# ---------------------------------------------------------------------------

def compute_regime_metrics(
    activation: np.ndarray,
    belief_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Compute global metrics that characterize the system-level regime.

    Returns a dict with:
      - polarization: mean inter-agent belief variance (higher = more divided)
      - activation_mean / activation_std: population-level arousal
      - belief_entropy: mean per-agent Shannon entropy over beliefs
      - consensus_index: 1 - polarization (higher = more aligned)
      - instability: activation_std * polarization (high = volatile & divided)
      - regime: classified label ('consensus', 'polarized', 'unstable', 'dormant')
    """
    N = len(activation)
    if N == 0:
        return {
            "polarization": 0.0, "activation_mean": 0.0,
            "activation_std": 0.0, "belief_entropy": 0.0,
            "consensus_index": 1.0, "instability": 0.0,
            "regime": "dormant",
        }

    act_mean = float(activation.mean())
    act_std = float(activation.std())

    var_per_dim = np.var(belief_matrix, axis=0)
    polarization = float(var_per_dim.mean())
    consensus = 1.0 - polarization

    bmat_safe = np.clip(belief_matrix, 1e-9, 1.0 - 1e-9)
    entropy = float(-np.sum(bmat_safe * np.log(bmat_safe), axis=1).mean())

    instability = act_std * polarization

    if polarization > 0.08 and act_mean > 0.4:
        regime = "unstable"
    elif polarization > 0.06:
        regime = "polarized"
    elif act_mean < 0.15:
        regime = "dormant"
    else:
        regime = "consensus"

    return {
        "polarization": polarization,
        "activation_mean": act_mean,
        "activation_std": act_std,
        "belief_entropy": entropy,
        "consensus_index": consensus,
        "instability": instability,
        "regime": regime,
    }
