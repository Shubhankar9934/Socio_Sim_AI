"""
Narrative Frame Generation.

Transforms simulation events into multiple media frames, one per source type.
Each MediaFrame carries both cognitive (belief_impacts) and emotional
(emotional_intensity) payloads for the single-payload architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agents.belief_network import BELIEF_DIMENSIONS, _N_BELIEFS
from media.sources import MediaSource, get_all_sources


@dataclass
class MediaFrame:
    """A single narrative frame from one media source about one event."""

    source_type: str
    headline: str
    framing_bias: np.ndarray  # (7,) which beliefs are emphasised
    belief_impacts: Dict[str, float]  # specific belief dimension shifts
    sentiment: float  # -1 to 1
    emotional_intensity: float  # 0 to 1
    dimension_impacts: Dict[str, float]  # behavioral dimension shifts
    topic: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "headline": self.headline,
            "framing_bias": self.framing_bias.tolist(),
            "belief_impacts": dict(self.belief_impacts),
            "sentiment": self.sentiment,
            "emotional_intensity": self.emotional_intensity,
            "dimension_impacts": dict(self.dimension_impacts),
            "topic": self.topic,
        }


def _derive_sentiment(source: MediaSource, event_payload: Dict) -> float:
    """Heuristic sentiment from source bias and event type."""
    etype = event_payload.get("type", "")
    if etype in ("price_change",):
        return -0.3 if source.bias_vector[5] > 0.6 else 0.1
    if etype in ("infrastructure", "new_metro_station"):
        return 0.4 if source.bias_vector[0] > 0.5 else -0.1
    return 0.0


def _derive_belief_impacts(
    source: MediaSource, event_payload: Dict,
) -> Dict[str, float]:
    """Small belief nudges toward the source's editorial lean."""
    impacts: Dict[str, float] = {}
    base_shift = 0.02
    for i, dim in enumerate(BELIEF_DIMENSIONS):
        lean = source.bias_vector[i] - 0.5
        if abs(lean) > 0.1:
            impacts[dim] = base_shift * lean
    return impacts


def _derive_dimension_impacts(
    source: MediaSource, event_payload: Dict,
) -> Dict[str, float]:
    """Behavioral dimension shifts from the event, coloured by source bias."""
    raw = event_payload.get("dimension_impacts", {})
    if not raw:
        return {}
    out: Dict[str, float] = {}
    for dim, shift in raw.items():
        amplification = 1.0
        if dim == "technology_openness" and source.bias_vector[0] > 0.6:
            amplification = 1.3
        if dim == "price_sensitivity" and source.bias_vector[5] > 0.6:
            amplification = 1.3
        out[dim] = shift * amplification
    return out


def generate_frames(
    events: List[Dict[str, Any]],
    sources: Optional[List[MediaSource]] = None,
) -> List[MediaFrame]:
    """Generate narrative frames for a batch of events.

    One frame per source per event. For 5 sources and 2 events = 10 frames.
    """
    pool = sources or get_all_sources()
    frames: List[MediaFrame] = []

    for event in events:
        etype = event.get("type", "event")
        payload = event.get("payload", event)
        topic = payload.get("name", etype)

        for src in pool:
            sentiment = _derive_sentiment(src, payload)
            emotion = min(1.0, src.emotional_intensity + abs(sentiment) * 0.3)

            headline = f"{src.name}: {topic}"
            belief_impacts = _derive_belief_impacts(src, payload)
            dim_impacts = _derive_dimension_impacts(src, payload)

            frames.append(MediaFrame(
                source_type=src.name,
                headline=headline,
                framing_bias=src.bias_vector.copy(),
                belief_impacts=belief_impacts,
                sentiment=sentiment,
                emotional_intensity=emotion,
                dimension_impacts=dim_impacts,
                topic=topic,
            ))

    return frames
