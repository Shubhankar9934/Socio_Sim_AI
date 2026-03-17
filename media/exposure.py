"""
Selective Exposure, Alignment Computation, and Echo Chamber Metrics.

Computes exposure matrices, prior-belief-dominant updates,
gated peak alignment, and media conflict for the epsilon equation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.belief_network import BELIEF_DIMENSIONS, _N_BELIEFS
from media.framing import MediaFrame
from media.sources import MediaSource, get_source


# ---------------------------------------------------------------------------
# Exposure matrices
# ---------------------------------------------------------------------------

def compute_exposure_matrices(
    agents: List[Dict[str, Any]],
    frames: List[MediaFrame],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (N, K) matrices for exposure, alignment, and emotion.

    Parameters
    ----------
    agents : list of agent dicts (with "persona" and "state" keys)
    frames : list of MediaFrame objects

    Returns
    -------
    raw_exposure : (N, K) binary/weighted exposure (1.0 if subscribed, else 0)
    alignment_matrix : (N, K) cosine alignment between agent beliefs and frame bias
    emotion_matrix : (N, K) emotional intensity per frame (broadcast to agents)
    """
    n_agents = len(agents)
    n_frames = len(frames)

    if n_agents == 0 or n_frames == 0:
        return (
            np.zeros((n_agents, max(n_frames, 1))),
            np.zeros((n_agents, max(n_frames, 1))),
            np.zeros((n_agents, max(n_frames, 1))),
        )

    raw_exposure = np.zeros((n_agents, n_frames), dtype=np.float64)
    alignment_matrix = np.zeros((n_agents, n_frames), dtype=np.float64)
    emotion_matrix = np.zeros((n_agents, n_frames), dtype=np.float64)

    frame_biases = np.array([f.framing_bias for f in frames])  # (K, 7)
    frame_emotions = np.array([f.emotional_intensity for f in frames])  # (K,)

    for i, agent in enumerate(agents):
        persona = agent.get("persona")
        state = agent.get("state")
        if persona is None or state is None:
            continue

        subs = getattr(persona, "media_subscriptions", None) or []
        beliefs = state.beliefs.to_vector() if hasattr(state, "beliefs") else np.full(_N_BELIEFS, 0.5)

        belief_norm = np.linalg.norm(beliefs)
        if belief_norm < 1e-8:
            belief_norm = 1.0

        for j, frame in enumerate(frames):
            if frame.source_type in subs or not subs:
                raw_exposure[i, j] = 1.0

            frame_norm = np.linalg.norm(frame.framing_bias)
            if frame_norm < 1e-8:
                frame_norm = 1.0
            alignment_matrix[i, j] = float(np.dot(beliefs, frame.framing_bias)) / (
                belief_norm * frame_norm
            )

            emotion_matrix[i, j] = frame.emotional_intensity

    return raw_exposure, alignment_matrix, emotion_matrix


# ---------------------------------------------------------------------------
# Gated Peak Alignment
# ---------------------------------------------------------------------------

def compute_alignment(
    alignment_matrix: np.ndarray,
    exposure_matrix: np.ndarray,
    beta: float = 0.15,
) -> np.ndarray:
    """Compute per-agent alignment using gated peak selection + smoothing.

    Returns (N,) alignment in [-1, 1].
    """
    n = alignment_matrix.shape[0]
    if n == 0 or alignment_matrix.shape[1] == 0:
        return np.zeros(n)

    scores = exposure_matrix * np.abs(alignment_matrix)
    peak_idx = np.argmax(scores, axis=1)
    alignment_peak = alignment_matrix[np.arange(n), peak_idx]

    weighted_sum = (alignment_matrix * exposure_matrix).sum(axis=1)
    exposure_sum = exposure_matrix.sum(axis=1) + 1e-6
    alignment_mean = weighted_sum / exposure_sum

    return (1.0 - beta) * alignment_peak + beta * alignment_mean


# ---------------------------------------------------------------------------
# Prior-Belief-Dominant Update
# ---------------------------------------------------------------------------

def update_beliefs_from_media(
    agents: List[Dict[str, Any]],
    adjusted_exposure: np.ndarray,
    frames: List[MediaFrame],
    *,
    w_prior: float = 0.70,
    w_media: float = 0.15,
    w_social: float = 0.15,
) -> None:
    """Update agent beliefs using prior-dominant weighting.

    new_belief = w_prior * prior + w_media * media_influence + w_social * social
    Social influence is handled separately in the diffusion step, so here
    we apply: new = w_prior * prior + w_media * media + (1 - w_prior - w_media) * prior
    effectively: new = (1 - w_media) * prior + w_media * media_signal
    """
    n_agents = len(agents)
    n_frames = len(frames)
    if n_agents == 0 or n_frames == 0:
        return

    for i, agent in enumerate(agents):
        state = agent.get("state")
        if state is None or not hasattr(state, "beliefs"):
            continue

        total_exposure = adjusted_exposure[i].sum()
        if total_exposure < 1e-6:
            continue

        media_signal = np.zeros(_N_BELIEFS)
        for j, frame in enumerate(frames):
            weight = adjusted_exposure[i, j]
            if weight < 1e-8:
                continue
            for dim_idx, dim in enumerate(BELIEF_DIMENSIONS):
                impact = frame.belief_impacts.get(dim, 0.0)
                media_signal[dim_idx] += weight * impact

        media_signal /= total_exposure

        current = state.beliefs.to_vector()
        effective_media_w = w_media
        topic_imp = getattr(state, "topic_importances", {})
        if topic_imp:
            avg_importance = np.mean(list(topic_imp.values())) if topic_imp else 0.5
            effective_media_w *= (1.0 - avg_importance)

        new_beliefs = current + effective_media_w * media_signal
        new_beliefs = np.clip(new_beliefs, 0.0, 1.0)

        from agents.belief_network import BeliefNetwork
        state.beliefs = BeliefNetwork.from_vector(new_beliefs)


# ---------------------------------------------------------------------------
# Echo Chamber and Media Conflict
# ---------------------------------------------------------------------------

def compute_echo_chamber_index(
    belief_vector: np.ndarray,
    subscribed_sources: List[str],
) -> float:
    """Measure homogeneity of an agent's media diet.

    Low variance of source bias vectors = strong echo chamber.
    Returns 0-1 where 1 = maximum echo chamber.
    """
    if len(subscribed_sources) < 2:
        return 1.0

    from media.sources import get_source
    vectors = []
    for name in subscribed_sources:
        src = get_source(name)
        if src is not None:
            vectors.append(src.bias_vector)

    if len(vectors) < 2:
        return 1.0

    mat = np.array(vectors)
    variance = np.var(mat, axis=0).mean()
    return float(1.0 - np.clip(variance * 10.0, 0.0, 1.0))


def compute_media_conflict_from_frames(
    agent_exposure: np.ndarray,
    frames: List[MediaFrame],
) -> float:
    """Mean pairwise distance between consumed frame bias vectors.

    Used to feed epsilon (uncertainty) in the bias engine.
    """
    consumed = [
        frames[j].framing_bias for j in range(len(frames))
        if j < len(agent_exposure) and agent_exposure[j] > 0.01
    ]
    if len(consumed) < 2:
        return 0.0

    distances = []
    for i in range(len(consumed)):
        for j in range(i + 1, len(consumed)):
            distances.append(float(np.linalg.norm(consumed[i] - consumed[j])))
    return float(np.clip(np.mean(distances), 0.0, 1.0))
