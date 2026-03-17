"""
Population Segments: behavioral archetypes with statistical priors.

Each segment defines a distribution over the 12 latent behavioral dimensions
so that agents within a segment cluster tightly while segments are well-separated.
This produces the multimodal population distributions that statisticians expect
in real survey data.

Segments are assigned during population synthesis based on demographic fit and
used to prime the initial BehavioralLatentState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.behavior import DIMENSION_NAMES


@dataclass(frozen=True)
class SegmentPrior:
    """Gaussian prior for one latent dimension within a segment."""
    mean: float
    std: float


@dataclass(frozen=True)
class PopulationSegment:
    """A behavioral population segment with demographic affinities and latent priors."""
    name: str
    weight: float
    location_affinity: Dict[str, float]
    age_affinity: Dict[str, float]
    income_affinity: Dict[str, float]
    latent_priors: Dict[str, SegmentPrior]
    delivery_mode: Dict[str, float]


POPULATION_SEGMENTS: Dict[str, PopulationSegment] = {
    "young_professional": PopulationSegment(
        name="young_professional",
        weight=0.22,
        location_affinity={
            "Business Bay": 0.28, "Dubai Marina": 0.22, "JLT": 0.18,
            "Downtown": 0.12, "Al Barsha": 0.08, "Jumeirah": 0.05,
            "Deira": 0.03, "Al Karama": 0.02, "JVC": 0.01, "Others": 0.01,
        },
        age_affinity={"25-34": 0.55, "35-44": 0.25, "18-24": 0.15, "45-54": 0.05},
        income_affinity={"25-50k": 0.45, "50k+": 0.30, "10-25k": 0.20, "<10k": 0.05},
        latent_priors={
            "convenience_seeking":           SegmentPrior(0.72, 0.08),
            "price_sensitivity":             SegmentPrior(0.32, 0.10),
            "technology_openness":           SegmentPrior(0.78, 0.07),
            "risk_aversion":                 SegmentPrior(0.35, 0.10),
            "health_orientation":            SegmentPrior(0.50, 0.12),
            "routine_stability":             SegmentPrior(0.40, 0.10),
            "novelty_seeking":               SegmentPrior(0.68, 0.09),
            "social_influence_susceptibility": SegmentPrior(0.55, 0.10),
            "time_pressure":                 SegmentPrior(0.74, 0.08),
            "financial_confidence":          SegmentPrior(0.65, 0.10),
            "environmental_consciousness":   SegmentPrior(0.42, 0.12),
            "institutional_trust":           SegmentPrior(0.52, 0.10),
        },
        delivery_mode={
            "rarely": 0.08, "1-2 per week": 0.20,
            "3-4 per week": 0.35, "daily": 0.27, "multiple per day": 0.10,
        },
    ),
    "family_homemaker": PopulationSegment(
        name="family_homemaker",
        weight=0.23,
        location_affinity={
            "JVC": 0.25, "Al Barsha": 0.20, "Deira": 0.12,
            "Al Karama": 0.12, "Others": 0.10, "Jumeirah": 0.08,
            "JLT": 0.05, "Downtown": 0.04, "Dubai Marina": 0.02, "Business Bay": 0.02,
        },
        age_affinity={"35-44": 0.40, "25-34": 0.25, "45-54": 0.25, "55+": 0.10},
        income_affinity={"10-25k": 0.35, "25-50k": 0.30, "<10k": 0.20, "50k+": 0.15},
        latent_priors={
            "convenience_seeking":           SegmentPrior(0.38, 0.10),
            "price_sensitivity":             SegmentPrior(0.65, 0.09),
            "technology_openness":           SegmentPrior(0.42, 0.12),
            "risk_aversion":                 SegmentPrior(0.60, 0.10),
            "health_orientation":            SegmentPrior(0.62, 0.10),
            "routine_stability":             SegmentPrior(0.72, 0.08),
            "novelty_seeking":               SegmentPrior(0.30, 0.10),
            "social_influence_susceptibility": SegmentPrior(0.45, 0.10),
            "time_pressure":                 SegmentPrior(0.55, 0.12),
            "financial_confidence":          SegmentPrior(0.42, 0.12),
            "environmental_consciousness":   SegmentPrior(0.55, 0.10),
            "institutional_trust":           SegmentPrior(0.58, 0.10),
        },
        delivery_mode={
            "rarely": 0.32, "1-2 per week": 0.30,
            "3-4 per week": 0.20, "daily": 0.12, "multiple per day": 0.06,
        },
    ),
    "budget_worker": PopulationSegment(
        name="budget_worker",
        weight=0.20,
        location_affinity={
            "Deira": 0.30, "Al Karama": 0.25, "Others": 0.15,
            "Al Barsha": 0.10, "JVC": 0.08, "JLT": 0.05,
            "Business Bay": 0.03, "Dubai Marina": 0.02, "Jumeirah": 0.01, "Downtown": 0.01,
        },
        age_affinity={"25-34": 0.35, "35-44": 0.30, "45-54": 0.20, "18-24": 0.10, "55+": 0.05},
        income_affinity={"<10k": 0.45, "10-25k": 0.40, "25-50k": 0.12, "50k+": 0.03},
        latent_priors={
            "convenience_seeking":           SegmentPrior(0.35, 0.10),
            "price_sensitivity":             SegmentPrior(0.82, 0.07),
            "technology_openness":           SegmentPrior(0.38, 0.12),
            "risk_aversion":                 SegmentPrior(0.65, 0.10),
            "health_orientation":            SegmentPrior(0.35, 0.12),
            "routine_stability":             SegmentPrior(0.60, 0.10),
            "novelty_seeking":               SegmentPrior(0.28, 0.10),
            "social_influence_susceptibility": SegmentPrior(0.50, 0.10),
            "time_pressure":                 SegmentPrior(0.60, 0.12),
            "financial_confidence":          SegmentPrior(0.22, 0.08),
            "environmental_consciousness":   SegmentPrior(0.30, 0.10),
            "institutional_trust":           SegmentPrior(0.45, 0.12),
        },
        delivery_mode={
            "rarely": 0.40, "1-2 per week": 0.30,
            "3-4 per week": 0.18, "daily": 0.08, "multiple per day": 0.04,
        },
    ),
    "student_explorer": PopulationSegment(
        name="student_explorer",
        weight=0.15,
        location_affinity={
            "Al Karama": 0.18, "Deira": 0.15, "JVC": 0.12,
            "Al Barsha": 0.12, "Others": 0.12, "JLT": 0.10,
            "Dubai Marina": 0.08, "Business Bay": 0.05, "Downtown": 0.05, "Jumeirah": 0.03,
        },
        age_affinity={"18-24": 0.60, "25-34": 0.30, "35-44": 0.10},
        income_affinity={"<10k": 0.50, "10-25k": 0.35, "25-50k": 0.12, "50k+": 0.03},
        latent_priors={
            "convenience_seeking":           SegmentPrior(0.58, 0.12),
            "price_sensitivity":             SegmentPrior(0.72, 0.09),
            "technology_openness":           SegmentPrior(0.80, 0.07),
            "risk_aversion":                 SegmentPrior(0.28, 0.10),
            "health_orientation":            SegmentPrior(0.35, 0.14),
            "routine_stability":             SegmentPrior(0.28, 0.10),
            "novelty_seeking":               SegmentPrior(0.78, 0.08),
            "social_influence_susceptibility": SegmentPrior(0.72, 0.08),
            "time_pressure":                 SegmentPrior(0.48, 0.14),
            "financial_confidence":          SegmentPrior(0.25, 0.10),
            "environmental_consciousness":   SegmentPrior(0.38, 0.12),
            "institutional_trust":           SegmentPrior(0.40, 0.12),
        },
        delivery_mode={
            "rarely": 0.15, "1-2 per week": 0.25,
            "3-4 per week": 0.30, "daily": 0.20, "multiple per day": 0.10,
        },
    ),
    "health_premium": PopulationSegment(
        name="health_premium",
        weight=0.12,
        location_affinity={
            "Dubai Marina": 0.22, "Jumeirah": 0.20, "Downtown": 0.15,
            "JLT": 0.12, "Business Bay": 0.10, "Al Barsha": 0.08,
            "JVC": 0.05, "Al Karama": 0.03, "Deira": 0.03, "Others": 0.02,
        },
        age_affinity={"25-34": 0.35, "35-44": 0.30, "45-54": 0.20, "18-24": 0.10, "55+": 0.05},
        income_affinity={"50k+": 0.35, "25-50k": 0.35, "10-25k": 0.20, "<10k": 0.10},
        latent_priors={
            "convenience_seeking":           SegmentPrior(0.48, 0.10),
            "price_sensitivity":             SegmentPrior(0.30, 0.10),
            "technology_openness":           SegmentPrior(0.65, 0.10),
            "risk_aversion":                 SegmentPrior(0.50, 0.12),
            "health_orientation":            SegmentPrior(0.85, 0.06),
            "routine_stability":             SegmentPrior(0.55, 0.10),
            "novelty_seeking":               SegmentPrior(0.52, 0.10),
            "social_influence_susceptibility": SegmentPrior(0.40, 0.10),
            "time_pressure":                 SegmentPrior(0.50, 0.12),
            "financial_confidence":          SegmentPrior(0.68, 0.10),
            "environmental_consciousness":   SegmentPrior(0.72, 0.08),
            "institutional_trust":           SegmentPrior(0.55, 0.10),
        },
        delivery_mode={
            "rarely": 0.25, "1-2 per week": 0.30,
            "3-4 per week": 0.22, "daily": 0.15, "multiple per day": 0.08,
        },
    ),
    "convenience_maximizer": PopulationSegment(
        name="convenience_maximizer",
        weight=0.08,
        location_affinity={
            "Dubai Marina": 0.22, "Downtown": 0.18, "Business Bay": 0.18,
            "JLT": 0.12, "Jumeirah": 0.10, "Al Barsha": 0.08,
            "JVC": 0.05, "Al Karama": 0.03, "Deira": 0.02, "Others": 0.02,
        },
        age_affinity={"25-34": 0.40, "35-44": 0.30, "18-24": 0.15, "45-54": 0.10, "55+": 0.05},
        income_affinity={"50k+": 0.40, "25-50k": 0.35, "10-25k": 0.20, "<10k": 0.05},
        latent_priors={
            "convenience_seeking":           SegmentPrior(0.88, 0.05),
            "price_sensitivity":             SegmentPrior(0.22, 0.08),
            "technology_openness":           SegmentPrior(0.82, 0.06),
            "risk_aversion":                 SegmentPrior(0.30, 0.10),
            "health_orientation":            SegmentPrior(0.38, 0.12),
            "routine_stability":             SegmentPrior(0.45, 0.10),
            "novelty_seeking":               SegmentPrior(0.65, 0.10),
            "social_influence_susceptibility": SegmentPrior(0.55, 0.10),
            "time_pressure":                 SegmentPrior(0.80, 0.07),
            "financial_confidence":          SegmentPrior(0.72, 0.08),
            "environmental_consciousness":   SegmentPrior(0.28, 0.10),
            "institutional_trust":           SegmentPrior(0.48, 0.10),
        },
        delivery_mode={
            "rarely": 0.03, "1-2 per week": 0.10,
            "3-4 per week": 0.25, "daily": 0.38, "multiple per day": 0.24,
        },
    ),
}


def _demographic_fit_score(
    persona_age: str,
    persona_income: str,
    persona_location: str,
    segment: PopulationSegment,
) -> float:
    """Score how well a persona's demographics match a segment's affinities."""
    age_score = segment.age_affinity.get(persona_age, 0.05)
    income_score = segment.income_affinity.get(persona_income, 0.05)
    loc_score = segment.location_affinity.get(persona_location, 0.02)
    return age_score * income_score * loc_score


def assign_segment(
    age: str,
    income: str,
    location: str,
    rng: np.random.Generator,
) -> str:
    """Assign a population segment based on demographic fit + segment weights.

    Uses a product of demographic affinity and segment population weight,
    then samples proportionally.  This produces correlated segment-geography
    patterns (Business Bay => young_professional, Deira => budget_worker).
    """
    segments = list(POPULATION_SEGMENTS.values())
    scores = np.array([
        seg.weight * _demographic_fit_score(age, income, location, seg)
        for seg in segments
    ])
    total = scores.sum()
    if total <= 0:
        return rng.choice([s.name for s in segments])
    probs = scores / total
    idx = int(rng.choice(len(segments), p=probs))
    return segments[idx].name


def sample_latent_from_segment(
    segment_name: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Sample initial latent dimension values from a segment's Gaussian priors.

    Returns a dict of dimension_name -> value (clamped to [0, 1]).
    Dimensions not specified in the segment use default N(0.5, 0.1).
    """
    segment = POPULATION_SEGMENTS.get(segment_name)
    if segment is None:
        return {d: float(np.clip(rng.normal(0.5, 0.10), 0.0, 1.0)) for d in DIMENSION_NAMES}

    values: Dict[str, float] = {}
    for dim in DIMENSION_NAMES:
        prior = segment.latent_priors.get(dim)
        if prior is not None:
            val = rng.normal(prior.mean, prior.std)
        else:
            val = rng.normal(0.5, 0.10)
        values[dim] = float(np.clip(val, 0.0, 1.0))
    return values


def get_segment_location_bias(segment_name: str) -> Dict[str, float]:
    """Return location affinity weights for biasing location sampling."""
    segment = POPULATION_SEGMENTS.get(segment_name)
    if segment is None:
        return {}
    return dict(segment.location_affinity)
