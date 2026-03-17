"""
Question Model Registry: declarative config mapping survey domains to
scale definitions, personality-dimension weights, and factor weights.

Adding a new survey type = adding one QuestionModel entry here.
No engine code changes required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class QuestionModel:
    """Describes how to score a particular kind of survey question.

    Attributes
    ----------
    name : str
        Unique key (matches registry key).
    scale : list[str]
        Ordered answer options (low → high).
    dimension_weights : dict[str, float]
        Personality-trait name → weight used by the personality factor.
        Negative weights invert the trait (e.g. price_sensitivity = -0.2).
    factor_weights : dict[str, float]
        Factor name → weight for the FactorGraph.
    temperature : float
        Softmax temperature (higher = more uniform distribution).
    """

    name: str
    scale: List[str]
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

QUESTION_MODELS: Dict[str, QuestionModel] = {

    # --- Food & Delivery --------------------------------------------------
    "food_delivery_frequency": QuestionModel(
        name="food_delivery_frequency",
        scale=["rarely", "1-2 per week", "3-4 per week", "daily", "multiple per day"],
        dimension_weights={
            "convenience_preference": 0.4,
            "primary_service_preference": 0.3,
            "price_sensitivity": -0.2,
            "dining_out": 0.1,
        },
        factor_weights={
            "personality": 0.30,
            "income": 0.16,
            "social": 0.16,
            "location": 0.08,
            "memory": 0.07,
            "behavioral": 0.11,
            "belief": 0.12,
        },
    ),

    # --- Parking -----------------------------------------------------------
    "parking_satisfaction": QuestionModel(
        name="parking_satisfaction",
        scale=["1", "2", "3", "4", "5"],
        dimension_weights={
            "mobility_dependence": 0.5,
            "price_sensitivity": -0.2,
            "convenience_preference": 0.3,
        },
        factor_weights={
            "personality": 0.26,
            "income": 0.08,
            "social": 0.04,
            "location": 0.32,
            "memory": 0.07,
            "behavioral": 0.11,
            "belief": 0.12,
        },
    ),

    # --- Transport / Metro -------------------------------------------------
    "transport_satisfaction": QuestionModel(
        name="transport_satisfaction",
        scale=["1", "2", "3", "4", "5"],
        dimension_weights={
            "mobility_dependence": 0.4,
            "convenience_preference": 0.3,
            "price_sensitivity": -0.15,
            "tech_adoption": 0.15,
        },
        factor_weights={
            "personality": 0.23,
            "income": 0.08,
            "social": 0.08,
            "location": 0.32,
            "memory": 0.07,
            "behavioral": 0.10,
            "belief": 0.12,
        },
    ),

    # --- Shopping / Retail -------------------------------------------------
    "shopping_frequency": QuestionModel(
        name="shopping_frequency",
        scale=["rarely", "1-2 per month", "weekly", "2-3 per week", "daily"],
        dimension_weights={
            "luxury_preference": 0.3,
            "convenience_preference": 0.2,
            "price_sensitivity": -0.2,
            "social_activity": 0.2,
            "brand_loyalty": 0.1,
        },
        factor_weights={
            "personality": 0.33,
            "income": 0.19,
            "social": 0.11,
            "location": 0.08,
            "memory": 0.05,
            "behavioral": 0.12,
            "belief": 0.12,
        },
    ),

    # --- Housing -----------------------------------------------------------
    "housing_satisfaction": QuestionModel(
        name="housing_satisfaction",
        scale=["1", "2", "3", "4", "5"],
        dimension_weights={
            "price_sensitivity": -0.3,
            "convenience_preference": 0.3,
            "luxury_preference": 0.2,
            "social_activity": 0.2,
        },
        factor_weights={
            "personality": 0.23,
            "income": 0.19,
            "social": 0.04,
            "location": 0.23,
            "memory": 0.08,
            "behavioral": 0.11,
            "belief": 0.12,
        },
    ),

    # --- NPS / Recommendation (0-10) ---------------------------------------
    "nps_recommendation": QuestionModel(
        name="nps_recommendation",
        scale=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        dimension_weights={
            "tech_adoption": 0.35,
            "convenience_preference": 0.25,
            "brand_loyalty": 0.20,
            "price_sensitivity": -0.15,
        },
        factor_weights={
            "personality": 0.35,
            "income": 0.10,
            "social": 0.15,
            "location": 0.03,
            "memory": 0.07,
            "behavioral": 0.15,
            "belief": 0.15,
        },
    ),

    # --- Tech / Fintech Adoption -------------------------------------------
    "tech_adoption_likelihood": QuestionModel(
        name="tech_adoption_likelihood",
        scale=["very unlikely", "unlikely", "neutral", "likely", "very likely"],
        dimension_weights={
            "tech_adoption": 0.45,
            "convenience_preference": 0.2,
            "risk_aversion": -0.2,
            "brand_loyalty": 0.15,
        },
        factor_weights={
            "personality": 0.36,
            "income": 0.15,
            "social": 0.15,
            "location": 0.03,
            "memory": 0.04,
            "behavioral": 0.13,
            "belief": 0.14,
        },
    ),

    # --- Policy / Opinion --------------------------------------------------
    "policy_support": QuestionModel(
        name="policy_support",
        scale=["strongly oppose", "oppose", "neutral", "support", "strongly support"],
        dimension_weights={
            "price_sensitivity": -0.25,
            "convenience_preference": 0.25,
            "social_activity": 0.2,
            "health_consciousness": 0.15,
            "mobility_dependence": 0.15,
        },
        factor_weights={
            "personality": 0.29,
            "income": 0.11,
            "social": 0.19,
            "location": 0.08,
            "memory": 0.07,
            "behavioral": 0.12,
            "belief": 0.14,
        },
    ),
}


# ---------------------------------------------------------------------------
# Generic fallback models — used when classification cannot find a match
# ---------------------------------------------------------------------------

GENERIC_LIKERT = QuestionModel(
    name="generic_likert",
    scale=["1", "2", "3", "4", "5"],
    dimension_weights={
        "convenience_preference": 0.20,
        "price_sensitivity": -0.20,
        "social_activity": 0.20,
        "health_consciousness": 0.10,
        "tech_adoption": 0.15,
        "time_pressure": 0.15,
    },
    factor_weights={
        "personality": 0.43,
        "income": 0.08,
        "social": 0.08,
        "location": 0.08,
        "memory": 0.08,
        "behavioral": 0.13,
        "belief": 0.12,
    },
)

GENERIC_FREQUENCY = QuestionModel(
    name="generic_frequency",
    scale=["never", "rarely", "sometimes", "often", "very often"],
    dimension_weights={
        "convenience_preference": 0.25,
        "price_sensitivity": -0.20,
        "social_activity": 0.20,
        "time_pressure": 0.20,
        "tech_adoption": 0.15,
    },
    factor_weights={
        "personality": 0.36,
        "income": 0.11,
        "social": 0.11,
        "location": 0.07,
        "memory": 0.08,
        "behavioral": 0.15,
        "belief": 0.12,
    },
)

# Open-ended / free-form questions — no scale, no distribution
GENERIC_OPEN_TEXT = QuestionModel(
    name="generic_open_text",
    scale=[],
    dimension_weights={
        "convenience_preference": 0.10,
        "price_sensitivity": -0.10,
        "tech_adoption": 0.10,
    },
    factor_weights={
        "personality": 0.40,
        "income": 0.10,
        "behavioral": 0.20,
        "belief": 0.15,
    },
)

# Duration / tenure questions — no discrete scale; open-ended time span
GENERIC_DURATION = QuestionModel(
    name="generic_duration",
    scale=[],
    dimension_weights={
        "convenience_preference": 0.10,
        "tech_adoption": 0.10,
        "routine_stability": 0.15,
    },
    factor_weights={
        "personality": 0.40,
        "income": 0.10,
        "behavioral": 0.20,
        "belief": 0.15,
    },
)

# Default fallback for completely unknown questions
GENERIC_FALLBACK = GENERIC_LIKERT

# Register generic_duration for topic map lookups
QUESTION_MODELS["generic_duration"] = GENERIC_DURATION


def get_question_model(key: str, domain_id: str = None) -> QuestionModel:
    """Look up a question model by key; returns the generic fallback if not found."""
    return QUESTION_MODELS.get(key, GENERIC_FALLBACK)


# ---------------------------------------------------------------------------
# Question → Behavioral-Latent-Dimension mapping
# ---------------------------------------------------------------------------

QUESTION_DIMENSION_MAP: Dict[str, Dict[str, float]] = {
    "food_delivery_frequency": {
        "convenience_seeking": 0.35,
        "price_sensitivity": 0.25,
        "time_pressure": 0.15,
        "health_orientation": -0.10,
        "routine_stability": 0.15,
    },
    "parking_satisfaction": {
        "convenience_seeking": 0.30,
        "financial_confidence": 0.20,
        "institutional_trust": 0.20,
        "environmental_consciousness": 0.15,
        "routine_stability": 0.15,
    },
    "transport_satisfaction": {
        "convenience_seeking": 0.25,
        "technology_openness": 0.20,
        "environmental_consciousness": 0.20,
        "price_sensitivity": 0.15,
        "institutional_trust": 0.20,
    },
    "shopping_frequency": {
        "convenience_seeking": 0.20,
        "novelty_seeking": 0.20,
        "price_sensitivity": 0.25,
        "social_influence_susceptibility": 0.20,
        "financial_confidence": 0.15,
    },
    "housing_satisfaction": {
        "financial_confidence": 0.25,
        "convenience_seeking": 0.20,
        "routine_stability": 0.20,
        "institutional_trust": 0.20,
        "environmental_consciousness": 0.15,
    },
    "nps_recommendation": {
        "technology_openness": 0.30,
        "novelty_seeking": 0.15,
        "risk_aversion": -0.15,
        "financial_confidence": 0.20,
        "social_influence_susceptibility": 0.20,
    },
    "tech_adoption_likelihood": {
        "technology_openness": 0.35,
        "novelty_seeking": 0.20,
        "risk_aversion": -0.20,
        "financial_confidence": 0.15,
        "social_influence_susceptibility": 0.10,
    },
    "policy_support": {
        "institutional_trust": 0.25,
        "environmental_consciousness": 0.20,
        "social_influence_susceptibility": 0.20,
        "price_sensitivity": -0.15,
        "risk_aversion": 0.20,
    },
    "generic_duration": {
        "routine_stability": 0.20,
        "institutional_trust": 0.15,
        "financial_confidence": 0.15,
    },
}

GENERIC_DIMENSION_WEIGHTS: Dict[str, float] = {
    "convenience_seeking": 0.15,
    "price_sensitivity": 0.15,
    "technology_openness": 0.10,
    "social_influence_susceptibility": 0.15,
    "routine_stability": 0.10,
    "novelty_seeking": 0.10,
    "financial_confidence": 0.10,
    "institutional_trust": 0.15,
}


def get_behavioral_dimensions(key: str) -> Dict[str, float]:
    """Return the behavioral-latent-dimension weights for a question model key."""
    return QUESTION_DIMENSION_MAP.get(key, GENERIC_DIMENSION_WEIGHTS)
