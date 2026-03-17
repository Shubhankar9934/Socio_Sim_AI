"""
Question-to-Belief-Dimension mappings: which beliefs are relevant for each
survey question model, and how answers update beliefs.

Analogous to QUESTION_DIMENSION_MAP in config/question_models.py but for
the BeliefNetwork layer rather than BehavioralLatentState.
"""

from __future__ import annotations

from typing import Dict


# Maps question model name -> belief dimension -> relevance weight.
# Positive weight: high answer score reinforces that belief.
# Negative weight: high answer score reduces that belief.
QUESTION_BELIEF_MAP: Dict[str, Dict[str, float]] = {
    "food_delivery_frequency": {
        "price_consciousness": -0.25,
        "technology_optimism": 0.20,
        "innovation_curiosity": 0.15,
        "health_priority": -0.10,
    },
    "parking_satisfaction": {
        "government_trust": 0.35,
        "environmental_concern": 0.15,
        "price_consciousness": -0.15,
    },
    "transport_satisfaction": {
        "government_trust": 0.30,
        "environmental_concern": 0.25,
        "technology_optimism": 0.20,
        "price_consciousness": -0.10,
    },
    "shopping_frequency": {
        "brand_loyalty": 0.30,
        "price_consciousness": -0.25,
        "innovation_curiosity": 0.15,
    },
    "housing_satisfaction": {
        "government_trust": 0.25,
        "price_consciousness": -0.20,
        "environmental_concern": 0.15,
    },
    "tech_adoption_likelihood": {
        "technology_optimism": 0.40,
        "innovation_curiosity": 0.30,
        "brand_loyalty": 0.10,
    },
    "policy_support": {
        "government_trust": 0.35,
        "environmental_concern": 0.25,
        "price_consciousness": -0.15,
        "health_priority": 0.15,
    },
}

GENERIC_BELIEF_WEIGHTS: Dict[str, float] = {
    "technology_optimism": 0.15,
    "brand_loyalty": 0.10,
    "environmental_concern": 0.10,
    "health_priority": 0.10,
    "government_trust": 0.15,
    "price_consciousness": 0.15,
    "innovation_curiosity": 0.10,
}


def get_belief_dimensions(key: str, domain_id: str = None) -> Dict[str, float]:
    """Return belief-dimension weights for a question model key.

    Checks domain-specific overrides first, then falls back to the static registry.
    """
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config(domain_id)
        domain_map = cfg.question_belief_map.get(key)
        if domain_map:
            return domain_map
    except Exception:
        pass
    return QUESTION_BELIEF_MAP.get(key, GENERIC_BELIEF_WEIGHTS)
