"""
Belief Network: high-level belief dimensions that persist across surveys
and influence agent decisions, spread socially, and evolve through events.

Beliefs are distinct from behavioral dimensions -- they represent attitudes
and opinions (e.g. "technology improves life") rather than behavioral tendencies
(e.g. "convenience_seeking").  Both layers feed into the factor graph and
cognitive dissonance adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from agents.personality import PersonalityTraits
    from population.personas import Persona

BELIEF_DIMENSIONS: List[str] = [
    "technology_optimism",
    "brand_loyalty",
    "environmental_concern",
    "health_priority",
    "government_trust",
    "price_consciousness",
    "innovation_curiosity",
]

_N_BELIEFS = len(BELIEF_DIMENSIONS)

# Cross-belief coupling matrix W (7x7).
# W[i,j] > 0 means high belief-j *raises* belief-i.
# W[i,j] < 0 means high belief-j *lowers* belief-i.
# Diagonal is zero (self-coupling handled by update rules).
# Rows: tech_opt, brand_loy, env_concern, health_pri, gov_trust, price_con, innov_cur
BELIEF_COUPLING_MATRIX: np.ndarray = np.array([
    # tech   brand  env    health gov    price  innov
    [ 0.00,  0.05,  0.00,  0.00,  0.08, -0.05,  0.15],  # tech_optimism
    [ 0.05,  0.00, -0.03,  0.00,  0.05, -0.08,  0.05],  # brand_loyalty
    [ 0.00, -0.03,  0.00,  0.12,  0.05,  0.05, -0.03],  # environmental_concern
    [ 0.00,  0.00,  0.12,  0.00,  0.03,  0.00, -0.02],  # health_priority
    [ 0.08,  0.05,  0.05,  0.03,  0.00, -0.05,  0.05],  # government_trust
    [-0.05, -0.08,  0.05,  0.00, -0.05,  0.00, -0.05],  # price_consciousness
    [ 0.15,  0.05, -0.03, -0.02,  0.05, -0.05,  0.00],  # innovation_curiosity
], dtype=np.float64)

_COUPLING_RATE = 0.03


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass
class BeliefNetwork:
    """7 core belief dimensions (all 0.0-1.0) plus extensible extras."""

    technology_optimism: float = 0.5
    brand_loyalty: float = 0.5
    environmental_concern: float = 0.5
    health_priority: float = 0.5
    government_trust: float = 0.5
    price_consciousness: float = 0.5
    innovation_curiosity: float = 0.5
    extra: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, d) for d in BELIEF_DIMENSIONS], dtype=np.float64)

    def to_extended_vector(self, extra_names: Optional[List[str]] = None) -> np.ndarray:
        core = [getattr(self, d) for d in BELIEF_DIMENSIONS]
        if extra_names:
            core.extend(self.extra.get(n, 0.5) for n in extra_names)
        return np.array(core, dtype=np.float64)

    @classmethod
    def from_vector(cls, arr: np.ndarray) -> BeliefNetwork:
        vals = {d: _clamp(float(arr[i])) for i, d in enumerate(BELIEF_DIMENSIONS)}
        return cls(**vals)

    def to_dict(self) -> Dict[str, float]:
        d = {dim: getattr(self, dim) for dim in BELIEF_DIMENSIONS}
        d.update(self.extra)
        return d

    def get(self, dim: str, default: float = 0.5) -> float:
        if hasattr(self, dim) and dim != "extra":
            return getattr(self, dim, default)
        return self.extra.get(dim, default)

    def set(self, dim: str, value: float) -> None:
        value = _clamp(value)
        if dim in BELIEF_DIMENSIONS:
            setattr(self, dim, value)
        else:
            self.extra[dim] = value

    def update_from_answer(
        self,
        dimension_weights: Dict[str, float],
        answer_score: float,
        learning_rate: float = 0.05,
    ) -> None:
        """Slow EMA update: nudge relevant belief dimensions toward answer_score."""
        for dim, weight in dimension_weights.items():
            current = self.get(dim, None)
            if current is None:
                continue
            target = answer_score if weight > 0 else (1.0 - answer_score)
            delta = learning_rate * abs(weight) * (target - current)
            self.set(dim, current + delta)

    def apply_social_diffusion(
        self,
        neighbor_mean: np.ndarray | float,
        rate: float = 0.03,
    ) -> None:
        """Nudge beliefs toward the mean belief state of social neighbors."""
        if isinstance(neighbor_mean, (int, float)):
            nudge = rate * (float(neighbor_mean) - 0.5)
            self.technology_optimism = _clamp(self.technology_optimism + nudge)
            self.innovation_curiosity = _clamp(self.innovation_curiosity + nudge * 0.5)
            return
        v = self.to_vector()
        n = min(len(v), len(neighbor_mean))
        delta = rate * (neighbor_mean[:n] - v[:n])
        updated = np.clip(v[:n] + delta, 0.0, 1.0)
        for i, d in enumerate(BELIEF_DIMENSIONS[:n]):
            setattr(self, d, float(updated[i]))

    def apply_event_impact(self, impacts: Dict[str, float]) -> None:
        """Direct belief shifts from world events."""
        for dim, shift in impacts.items():
            current = self.get(dim, None)
            if current is not None:
                self.set(dim, current + shift)

    def propagate_coupling(self, rate: float = _COUPLING_RATE) -> None:
        """Apply cross-belief coupling (core dims only; extras are independent)."""
        v = self.to_vector()
        delta = rate * (BELIEF_COUPLING_MATRIX @ v)
        updated = np.clip(v + delta, 0.0, 1.0)
        for i, d in enumerate(BELIEF_DIMENSIONS):
            setattr(self, d, float(updated[i]))

    def belief_score(self, dimension_weights: Dict[str, float]) -> float:
        """Weighted combination of relevant belief dimensions -> single 0-1 score.
        When belief_nonlinearity > 0, applies sigmoid for non-linear, thresholded effect.
        """
        score = 0.0
        total = 0.0
        for dim, w in dimension_weights.items():
            val = self.get(dim, 0.5)
            score += val * w
            total += abs(w)
        if total == 0:
            return 0.5
        raw = _clamp(score / total)
        try:
            from config.settings import get_settings
            k = getattr(get_settings(), "belief_nonlinearity", 0.0)
        except Exception:
            k = 0.0
        if k <= 0:
            return raw
        sigmoid = 1.0 / (1.0 + np.exp(-k * (raw - 0.5)))
        return float(_clamp(sigmoid))


_BELIEF_LANGUAGE = {
    "technology_optimism": ("technology makes life better", "technology complicates things"),
    "brand_loyalty": ("trusted brands are worth paying more for", "brands don't matter much"),
    "environmental_concern": ("environmental issues matter a lot", "environmental issues are overblown"),
    "health_priority": ("health comes first in decisions", "health isn't a top priority"),
    "government_trust": ("the government generally does the right thing", "the government can't be trusted"),
    "price_consciousness": ("price matters more than anything", "price isn't the main concern"),
    "innovation_curiosity": ("new things are exciting to try", "sticking with what works is better"),
}


def surface_top_beliefs(beliefs: "BeliefNetwork", top_n: int = 3) -> List[str]:
    """Return natural language statements for the agent's strongest beliefs.

    Beliefs far from 0.5 (strong conviction either way) are surfaced.
    """
    scores = []
    for dim in BELIEF_DIMENSIONS:
        val = getattr(beliefs, dim, 0.5)
        distance = abs(val - 0.5)
        scores.append((distance, dim, val))
    scores.sort(key=lambda x: x[0], reverse=True)

    statements = []
    for _, dim, val in scores[:top_n]:
        positive_text, negative_text = _BELIEF_LANGUAGE.get(dim, (dim, f"not {dim}"))
        if val >= 0.65:
            strength = "strongly" if val >= 0.80 else ""
            statements.append(f"You {strength} believe that {positive_text}".replace("  ", " ").strip())
        elif val <= 0.35:
            strength = "strongly" if val <= 0.20 else ""
            statements.append(f"You {strength} believe that {negative_text}".replace("  ", " ").strip())
    return statements


def init_beliefs_from_persona(
    persona: "Persona",
    traits: "PersonalityTraits",
) -> BeliefNetwork:
    """Derive initial beliefs from persona demographics, lifestyle, and traits."""
    ls = persona.lifestyle
    pa = persona.personal_anchors

    income_high = persona.income in ("25-50k", "50k+")

    health_map = {
        "high": 0.85, "very health-conscious": 0.90, "fitness-focused": 0.92,
        "active": 0.70, "moderate": 0.50, "relaxed": 0.30, "low": 0.25,
    }
    health_val = health_map.get(pa.health_focus.lower(), 0.50)

    technology_optimism = _clamp(ls.tech_adoption * 0.7 + 0.15 + (0.1 if income_high else 0.0))
    brand_loyalty_val = _clamp(
        ls.luxury_preference * 0.5 + (1.0 - ls.price_sensitivity) * 0.3 + ls.tech_adoption * 0.2
    )
    environmental_concern = _clamp(
        health_val * 0.4 + (1.0 - ls.convenience_preference) * 0.3 + 0.15
    )
    health_priority = health_val
    government_trust = _clamp(
        0.5 + (0.1 if income_high else -0.05) + traits.risk_aversion * 0.2
    )
    price_consciousness = _clamp(ls.price_sensitivity * 0.7 + (0.2 if not income_high else 0.0))
    innovation_curiosity = _clamp(
        ls.tech_adoption * 0.5 + (1.0 - traits.risk_aversion) * 0.3 + 0.1
    )

    return BeliefNetwork(
        technology_optimism=technology_optimism,
        brand_loyalty=brand_loyalty_val,
        environmental_concern=environmental_concern,
        health_priority=health_priority,
        government_trust=government_trust,
        price_consciousness=price_consciousness,
        innovation_curiosity=innovation_curiosity,
    )
