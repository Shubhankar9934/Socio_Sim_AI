"""
Media Source Definitions.

Each source has a 7-dim editorial bias vector over the belief dimensions
(from agents.belief_network) and an emotional intensity baseline.
Agents are assigned 2-3 sources via homophilic cosine selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from agents.belief_network import BELIEF_DIMENSIONS, _N_BELIEFS
from core.rng import ensure_np_rng


@dataclass
class MediaSource:
    """A media outlet with an editorial bias vector."""

    name: str
    bias_vector: np.ndarray  # (_N_BELIEFS,) editorial lean over belief dims
    emotional_intensity: float = 0.5  # baseline emotional charge 0-1
    narrative_style: str = "objective"  # objective | sensational | analytical

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "bias_vector": self.bias_vector.tolist(),
            "emotional_intensity": self.emotional_intensity,
            "narrative_style": self.narrative_style,
        }


# Belief dimension order:
# technology_optimism, brand_loyalty, environmental_concern,
# health_priority, government_trust, price_consciousness, innovation_curiosity

DEFAULT_SOURCES: List[MediaSource] = [
    MediaSource(
        name="TechForward Daily",
        bias_vector=np.array([0.85, 0.40, 0.30, 0.35, 0.50, 0.30, 0.85]),
        emotional_intensity=0.55,
        narrative_style="analytical",
    ),
    MediaSource(
        name="Economic Watch",
        bias_vector=np.array([0.40, 0.25, 0.30, 0.30, 0.45, 0.85, 0.35]),
        emotional_intensity=0.50,
        narrative_style="analytical",
    ),
    MediaSource(
        name="Green Dubai",
        bias_vector=np.array([0.45, 0.30, 0.90, 0.80, 0.50, 0.40, 0.50]),
        emotional_intensity=0.60,
        narrative_style="objective",
    ),
    MediaSource(
        name="Community Voice",
        bias_vector=np.array([0.35, 0.30, 0.50, 0.45, 0.20, 0.75, 0.30]),
        emotional_intensity=0.70,
        narrative_style="sensational",
    ),
    MediaSource(
        name="Lifestyle Plus",
        bias_vector=np.array([0.60, 0.80, 0.35, 0.55, 0.55, 0.30, 0.75]),
        emotional_intensity=0.45,
        narrative_style="objective",
    ),
]

_SOURCE_MAP: Dict[str, MediaSource] = {s.name: s for s in DEFAULT_SOURCES}


def get_source(name: str) -> Optional[MediaSource]:
    return _SOURCE_MAP.get(name)


def get_all_sources() -> List[MediaSource]:
    return list(DEFAULT_SOURCES)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-8:
        return 0.0
    return dot / norm


def assign_media_diet(
    belief_vector: np.ndarray,
    sources: Optional[List[MediaSource]] = None,
    n_sources: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Assign media sources to an agent via homophilic cosine selection.

    Agents are more likely to subscribe to sources whose editorial bias
    vector aligns with their own beliefs, but with stochastic variation.
    """
    gen = ensure_np_rng(rng, key="media_assign_diet")
    pool = sources or DEFAULT_SOURCES

    scores = np.array([_cosine_sim(belief_vector, s.bias_vector) for s in pool])
    scores = np.clip(scores, 0.01, None)
    probs = scores / scores.sum()

    n = min(n_sources, len(pool))
    chosen_idx = gen.choice(len(pool), size=n, replace=False, p=probs)
    return [pool[i].name for i in chosen_idx]
