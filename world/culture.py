"""
Cultural Norm Fields: per-district cultural vectors that nudge agent
behavioral dimensions toward local norms.

Two identical agents in different districts will drift toward different
behavioral profiles, reflecting real-world neighborhood culture effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.behavior import BehavioralLatentState

from agents.behavior import DIMENSION_NAMES, _N_DIMS


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass
class CulturalField:
    """Per-district cultural norms over the 12 behavioral dimensions.

    Only dimensions with explicit norms exert cultural pressure.
    Unspecified dimensions default to 0.5 (neutral) and have no pull.
    """

    norms: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# District Cultural Profiles (grounded in Dubai geography)
# ---------------------------------------------------------------------------

DISTRICT_CULTURAL_FIELDS: Dict[str, CulturalField] = {
    "Dubai Marina": CulturalField(norms={
        "convenience_seeking": 0.80,
        "technology_openness": 0.75,
        "novelty_seeking": 0.70,
        "social_influence_susceptibility": 0.65,
        "financial_confidence": 0.65,
    }),
    "Jumeirah": CulturalField(norms={
        "financial_confidence": 0.80,
        "health_orientation": 0.70,
        "routine_stability": 0.65,
        "environmental_consciousness": 0.60,
    }),
    "Deira": CulturalField(norms={
        "price_sensitivity": 0.75,
        "routine_stability": 0.70,
        "social_influence_susceptibility": 0.60,
    }),
    "Business Bay": CulturalField(norms={
        "technology_openness": 0.75,
        "convenience_seeking": 0.75,
        "financial_confidence": 0.70,
        "time_pressure": 0.70,
    }),
    "Al Barsha": CulturalField(norms={
        "routine_stability": 0.65,
        "price_sensitivity": 0.60,
        "health_orientation": 0.55,
    }),
    "JLT": CulturalField(norms={
        "convenience_seeking": 0.70,
        "technology_openness": 0.70,
        "novelty_seeking": 0.65,
        "social_influence_susceptibility": 0.65,
    }),
    "Downtown": CulturalField(norms={
        "financial_confidence": 0.75,
        "convenience_seeking": 0.70,
        "novelty_seeking": 0.65,
        "technology_openness": 0.70,
    }),
    "Al Karama": CulturalField(norms={
        "price_sensitivity": 0.70,
        "routine_stability": 0.70,
        "social_influence_susceptibility": 0.55,
    }),
    "JVC": CulturalField(norms={
        "routine_stability": 0.70,
        "health_orientation": 0.60,
        "price_sensitivity": 0.60,
    }),
    "Others": CulturalField(norms={
        "routine_stability": 0.60,
        "price_sensitivity": 0.55,
    }),
}


_NEUTRAL_FIELD = CulturalField()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_cultural_field(district_name: str) -> CulturalField:
    """Look up a district's cultural field; returns neutral if unknown."""
    return DISTRICT_CULTURAL_FIELDS.get(district_name, _NEUTRAL_FIELD)


# ---------------------------------------------------------------------------
# Vector conversion
# ---------------------------------------------------------------------------

def cultural_field_to_vector(cf: CulturalField) -> np.ndarray:
    """Convert a CulturalField to a 12-dim vector (0.5 for unspecified dims)."""
    return np.array(
        [cf.norms.get(d, 0.5) for d in DIMENSION_NAMES],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Scalar application
# ---------------------------------------------------------------------------

def apply_cultural_influence(
    latent_state: "BehavioralLatentState",
    field: CulturalField,
    rate: float = 0.02,
) -> None:
    """Nudge an agent's latent state toward local cultural norms.

    Only dimensions with explicit norms in the field exert pull.
    """
    for dim, norm_val in field.norms.items():
        if not hasattr(latent_state, dim):
            continue
        current = getattr(latent_state, dim)
        delta = rate * (norm_val - current)
        setattr(latent_state, dim, _clamp(current + delta))


# ---------------------------------------------------------------------------
# Vectorized operations
# ---------------------------------------------------------------------------

def build_cultural_matrix(agents: List[Dict[str, Any]]) -> np.ndarray:
    """Build an (N, 12) matrix of cultural norm vectors based on agent locations."""
    rows = []
    for a in agents:
        persona = a.get("persona")
        if persona is not None:
            cf = get_cultural_field(persona.location)
        else:
            cf = _NEUTRAL_FIELD
        rows.append(cultural_field_to_vector(cf))
    if not rows:
        return np.empty((0, _N_DIMS), dtype=np.float64)
    return np.vstack(rows)


def vectorized_cultural_influence(
    trait_matrix: np.ndarray,
    cultural_matrix: np.ndarray,
    rate: float = 0.02,
) -> np.ndarray:
    """Batch cultural nudge: move trait_matrix toward cultural_matrix.

    Only dimensions where cultural_matrix != 0.5 exert pull.
    """
    mask = cultural_matrix != 0.5
    delta = rate * (cultural_matrix - trait_matrix) * mask
    return np.clip(trait_matrix + delta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mutable cultural fields (for events that shift district culture)
# ---------------------------------------------------------------------------

_RUNTIME_OVERRIDES: Dict[str, CulturalField] = {}


def apply_cultural_shift(district: str, shifts: Dict[str, float]) -> None:
    """Shift a district's cultural norms at runtime (from world events)."""
    base = DISTRICT_CULTURAL_FIELDS.get(district, CulturalField())
    current_norms = dict(base.norms)

    override = _RUNTIME_OVERRIDES.get(district)
    if override is not None:
        current_norms.update(override.norms)

    for dim, delta in shifts.items():
        old = current_norms.get(dim, 0.5)
        current_norms[dim] = _clamp(old + delta)

    _RUNTIME_OVERRIDES[district] = CulturalField(norms=current_norms)


def get_effective_cultural_field(district_name: str) -> CulturalField:
    """Return cultural field with runtime overrides applied."""
    override = _RUNTIME_OVERRIDES.get(district_name)
    if override is not None:
        return override
    return get_cultural_field(district_name)


def reset_cultural_overrides() -> None:
    """Clear all runtime cultural overrides (useful in tests)."""
    _RUNTIME_OVERRIDES.clear()


# ---------------------------------------------------------------------------
# Emergent cultural norms: agent behavior feeds back into district culture
# ---------------------------------------------------------------------------

def update_emergent_norms(
    agents: List[Dict[str, Any]],
    rate: float = 0.01,
) -> None:
    """Adjust district cultural norms toward the observed mean behavior of
    agents in each district.

    This closes the culture loop: district -> agent AND agent -> district.
    Only dimensions already present in the district's cultural field are
    updated so that culturally irrelevant dimensions stay neutral.
    """
    from collections import defaultdict

    district_agents: Dict[str, List[np.ndarray]] = defaultdict(list)
    for a in agents:
        persona = a.get("persona")
        state = a.get("state")
        if persona is None or state is None:
            continue
        if not hasattr(state, "latent_state"):
            continue
        vec = np.array(
            [getattr(state.latent_state, d, 0.5) for d in DIMENSION_NAMES],
            dtype=np.float64,
        )
        district_agents[persona.location].append(vec)

    for district, vecs in district_agents.items():
        if not vecs:
            continue
        observed_mean = np.mean(vecs, axis=0)

        base = DISTRICT_CULTURAL_FIELDS.get(district, CulturalField())
        override = _RUNTIME_OVERRIDES.get(district)
        current_norms = dict(base.norms)
        if override is not None:
            current_norms.update(override.norms)

        updated = dict(current_norms)
        for i, dim in enumerate(DIMENSION_NAMES):
            if dim not in current_norms:
                continue
            old = current_norms[dim]
            updated[dim] = _clamp(old + rate * (float(observed_mean[i]) - old))

        _RUNTIME_OVERRIDES[district] = CulturalField(norms=updated)
