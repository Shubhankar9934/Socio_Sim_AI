"""
Adaptive Attention Layer.

Emotion reshapes what information is even seen. When activation is low,
agents have broad cognitive bandwidth. As activation spikes, attention
concentrates on the most emotionally threatening or validating frames
(tunnel vision / doomscrolling).

sharpness = 1 + k * A^p

An entropy floor (min_attention) prevents permanent cognitive collapse.
"""

from __future__ import annotations

import numpy as np


def adaptive_attention(
    activation: np.ndarray,
    exposure: np.ndarray,
    emotional_intensity: np.ndarray,
    *,
    k: float = 5.0,
    p: float = 2.0,
    min_attention: float = 0.05,
) -> np.ndarray:
    """Reweight raw exposure based on emotional state.

    Parameters
    ----------
    activation : (N,) current activation per agent
    exposure : (N, K) raw exposure intensities per agent per frame
    emotional_intensity : (N, K) emotional charge per frame per agent
    k : sharpness scaling factor
    p : nonlinear exponent
    min_attention : entropy floor preventing total cognitive collapse

    Returns
    -------
    adjusted_exposure : (N, K)
    """
    if exposure.size == 0:
        return exposure.copy()

    salience = exposure * emotional_intensity

    sharpness = (1.0 + k * (activation ** p))[:, None]

    logits = sharpness * salience
    logits = logits - np.max(logits, axis=1, keepdims=True)

    weights = np.exp(logits)
    weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-9)

    n_frames = exposure.shape[1]
    if n_frames > 0:
        weights = (1.0 - min_attention) * weights + min_attention / n_frames

    # Hard attention budget: each agent processes only K_i frames, where
    # K_i = 1 + 3*(1 - activation_i).  High-activation agents tunnel-vision
    # on 1-2 frames; calm agents see up to 4.
    budget = np.clip(1 + 3 * (1.0 - activation), 1, n_frames).astype(int)
    for i in range(weights.shape[0]):
        k_i = int(budget[i])
        if k_i < n_frames:
            idx_sorted = np.argsort(weights[i])
            cutoff = idx_sorted[:-k_i]
            weights[i, cutoff] = 0.0
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = weights / row_sums

    return exposure * weights
