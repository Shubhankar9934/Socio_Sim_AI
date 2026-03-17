"""
Behavioral Latent State: 12 universal dimensions that drive agent decisions
across any survey domain.  Evolves over time via EMA updates from decisions,
social influence, and macro feedback.

The dimensions are grounded in behavioral psychology and work for consumer,
policy, health, technology, and financial survey domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from agents.personality import PersonalityTraits
    from population.personas import Persona

DIMENSION_NAMES: List[str] = [
    "convenience_seeking",
    "price_sensitivity",
    "technology_openness",
    "risk_aversion",
    "health_orientation",
    "routine_stability",
    "novelty_seeking",
    "social_influence_susceptibility",
    "time_pressure",
    "financial_confidence",
    "environmental_consciousness",
    "institutional_trust",
]

_N_DIMS = len(DIMENSION_NAMES)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass
class BehavioralLatentState:
    """12 core universal behavioral dimensions (all 0.0-1.0) plus extensible extras."""

    convenience_seeking: float = 0.5
    price_sensitivity: float = 0.5
    technology_openness: float = 0.5
    risk_aversion: float = 0.5
    health_orientation: float = 0.5
    routine_stability: float = 0.5
    novelty_seeking: float = 0.5
    social_influence_susceptibility: float = 0.5
    time_pressure: float = 0.5
    financial_confidence: float = 0.5
    environmental_consciousness: float = 0.5
    institutional_trust: float = 0.5
    extra: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        core = [getattr(self, d) for d in DIMENSION_NAMES]
        return np.array(core, dtype=np.float64)

    def to_extended_vector(self, extra_names: Optional[List[str]] = None) -> np.ndarray:
        """Core dims + extra dims in the order given."""
        core = [getattr(self, d) for d in DIMENSION_NAMES]
        if extra_names:
            core.extend(self.extra.get(n, 0.5) for n in extra_names)
        return np.array(core, dtype=np.float64)

    @classmethod
    def from_vector(cls, arr: np.ndarray) -> "BehavioralLatentState":
        vals = {d: _clamp(float(arr[i])) for i, d in enumerate(DIMENSION_NAMES)}
        return cls(**vals)

    def to_dict(self) -> Dict[str, float]:
        d = {dim: getattr(self, dim) for dim in DIMENSION_NAMES}
        d.update(self.extra)
        return d

    def get(self, dim: str, default: float = 0.5) -> float:
        if hasattr(self, dim) and dim != "extra":
            return getattr(self, dim, default)
        return self.extra.get(dim, default)

    def set(self, dim: str, value: float) -> None:
        value = _clamp(value)
        if dim in DIMENSION_NAMES:
            setattr(self, dim, value)
        else:
            self.extra[dim] = value

    def update_dimensions(
        self,
        dimension_weights: Dict[str, float],
        answer_score: float,
        learning_rate: float = 0.05,
    ) -> None:
        """EMA update: nudge relevant dimensions toward the answer_score."""
        for dim, weight in dimension_weights.items():
            current = self.get(dim, None)
            if current is None:
                continue
            target = answer_score if weight > 0 else (1.0 - answer_score)
            delta = learning_rate * abs(weight) * (target - current)
            self.set(dim, current + delta)

    def apply_social_influence(
        self,
        neighbor_mean: "np.ndarray | float",
        influence_rate: float = 0.02,
    ) -> None:
        """Nudge dimensions toward the mean latent state of neighbors.

        Accepts either an (N,) numpy vector (full latent diffusion) or a
        scalar float (legacy friends_using_fraction for backward compat).
        """
        sus = self.social_influence_susceptibility
        if isinstance(neighbor_mean, (int, float)):
            nudge = influence_rate * sus * (float(neighbor_mean) - 0.5)
            self.convenience_seeking = _clamp(self.convenience_seeking + nudge)
            self.novelty_seeking = _clamp(self.novelty_seeking + nudge * 0.5)
            return
        v = self.to_vector()
        n = min(len(v), len(neighbor_mean))
        delta = influence_rate * sus * (neighbor_mean[:n] - v[:n])
        updated = np.clip(v[:n] + delta, 0.0, 1.0)
        for i, d in enumerate(DIMENSION_NAMES[:n]):
            setattr(self, d, float(updated[i]))

    def apply_macro_influence(
        self,
        macro_signals: Dict[str, float],
        macro_rate: float = 0.01,
    ) -> None:
        """Population-level trends nudge individual dimensions."""
        sus = self.social_influence_susceptibility
        for dim, trend_value in macro_signals.items():
            current = self.get(dim, None)
            if current is None:
                continue
            delta = macro_rate * sus * (trend_value - current)
            self.set(dim, current + delta)

    def apply_event_impact(self, impacts: Dict[str, float]) -> None:
        """Direct dimension shifts from world events."""
        for dim, shift in impacts.items():
            current = self.get(dim, None)
            if current is not None:
                self.set(dim, current + shift)

    def behavioral_score(self, dimension_weights: Dict[str, float]) -> float:
        """Weighted combination of relevant dimensions -> single 0-1 score."""
        score = 0.0
        total = 0.0
        for dim, w in dimension_weights.items():
            val = self.get(dim, 0.5)
            score += val * w
            total += abs(w)
        if total == 0:
            return 0.5
        return _clamp(score / total)


def init_from_persona(
    persona: "Persona",
    traits: "PersonalityTraits",
) -> BehavioralLatentState:
    """Derive initial behavioral state from persona and personality traits.

    If the persona has a population_segment assigned, the latent dimensions
    are sampled from the segment's Gaussian priors and blended 70/30 with
    the trait-derived values.  This creates tight within-cluster distributions
    while still respecting individual persona variation.
    """
    ls = persona.lifestyle
    pa = persona.personal_anchors

    income_high = persona.income in ("25-50k", "50k+")

    novelty = _clamp(1.0 - traits.brand_loyalty * 0.6 + ls.tech_adoption * 0.3)
    social_sus = _clamp(traits.social_activity * 0.6 + 0.2)
    financial_conf = _clamp(0.3 + (0.3 if income_high else 0.0) + ls.luxury_preference * 0.2)

    health_map = {"high": 0.85, "very health-conscious": 0.90, "fitness-focused": 0.92,
                  "active": 0.70, "moderate": 0.50, "relaxed": 0.30}
    health_val = health_map.get(pa.health_focus.lower(), 0.50)

    env_consciousness = _clamp(health_val * 0.4 + (1.0 - ls.convenience_preference) * 0.3 + 0.15)
    inst_trust = _clamp(0.5 + (0.1 if income_high else -0.05) + traits.risk_aversion * 0.2)

    trait_derived = {
        "convenience_seeking": ls.convenience_preference,
        "price_sensitivity": ls.price_sensitivity,
        "technology_openness": ls.tech_adoption,
        "risk_aversion": traits.risk_aversion,
        "health_orientation": health_val,
        "routine_stability": _clamp(traits.brand_loyalty * 0.7 + 0.15),
        "novelty_seeking": novelty,
        "social_influence_susceptibility": social_sus,
        "time_pressure": traits.time_pressure,
        "financial_confidence": financial_conf,
        "environmental_consciousness": env_consciousness,
        "institutional_trust": inst_trust,
    }

    segment_name = getattr(persona.meta, "population_segment", None)
    if segment_name:
        from population.segments import sample_latent_from_segment
        rng = np.random.default_rng(hash(persona.agent_id) & 0xFFFFFFFF)
        segment_vals = sample_latent_from_segment(segment_name, rng)
        _SEGMENT_WEIGHT = 0.70
        final = {}
        for dim in DIMENSION_NAMES:
            sv = segment_vals.get(dim, 0.5)
            tv = trait_derived.get(dim, 0.5)
            final[dim] = _clamp(_SEGMENT_WEIGHT * sv + (1.0 - _SEGMENT_WEIGHT) * tv)
        return BehavioralLatentState(**final)

    return BehavioralLatentState(**trait_derived)
