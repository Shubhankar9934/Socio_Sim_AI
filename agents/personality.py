"""
Personality model: traits and behavioral coefficients that feed into decision engine.
Derived from Persona lifestyle coefficients with heuristic derivations for
generic dimensions (social_activity, health_consciousness, mobility_dependence,
time_pressure, brand_loyalty).
"""

from dataclasses import dataclass
from typing import Optional

from population.personas import Persona


@dataclass
class PersonalityTraits:
    """Numeric traits 0.0-1.0 for decision model."""

    risk_aversion: float
    convenience_preference: float
    price_sensitivity: float
    tech_adoption: float
    luxury_preference: float
    primary_service_preference: float
    dining_out: float

    social_activity: float = 0.5
    health_consciousness: float = 0.5
    mobility_dependence: float = 0.5
    time_pressure: float = 0.5
    brand_loyalty: float = 0.5

    openness_to_experience: float = 0.5
    conscientiousness: float = 0.5
    agreeableness: float = 0.5
    emotional_stability: float = 0.5
    extraversion: float = 0.5
    impulsivity: float = 0.5
    optimism: float = 0.5


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


_HEALTH_FOCUS_MAP = {
    "high": 0.85,
    "moderate": 0.50,
    "low": 0.25,
}

_WORK_SCHEDULE_TIME_PRESSURE = {
    "9-to-5": 0.45,
    "shift": 0.60,
    "flexible": 0.30,
    "remote": 0.25,
    "freelance": 0.35,
}


def personality_from_persona(persona: Persona) -> PersonalityTraits:
    """Build personality traits from persona lifestyle coefficients and anchors."""
    ls = persona.lifestyle
    pa = persona.personal_anchors
    mob = persona.mobility

    # --- social_activity: high dining_out + social hobbies push upward ---
    hobby_social_boost = 0.15 if pa.hobby.lower() in (
        "sports", "gym", "socializing", "travel", "dancing", "team sports",
    ) else 0.0
    social_activity = _clamp(0.3 + ls.dining_out * 0.4 + hobby_social_boost + ls.convenience_preference * 0.15)

    # --- health_consciousness: from personal anchors health_focus ---
    health_consciousness = _HEALTH_FOCUS_MAP.get(pa.health_focus.lower(), 0.50)

    # --- mobility_dependence: car owners + frequent metro users score higher ---
    metro_score = {"frequent": 0.7, "occasional": 0.4, "rare": 0.15}.get(mob.metro_usage, 0.15)
    mobility_dependence = _clamp(0.3 * float(mob.car) + 0.4 * metro_score + 0.3 * ls.convenience_preference)

    # --- time_pressure: derived from work_schedule ---
    time_pressure = _WORK_SCHEDULE_TIME_PRESSURE.get(pa.work_schedule.lower(), 0.45)

    # --- brand_loyalty: luxury seekers with low price sensitivity tend to be brand-loyal ---
    brand_loyalty = _clamp(ls.luxury_preference * 0.5 + (1.0 - ls.price_sensitivity) * 0.3 + ls.tech_adoption * 0.2)

    pv = getattr(persona, "personality", None)
    risk_aversion = pv.risk_aversion if pv else 0.5
    openness = pv.openness_to_experience if pv else 0.5
    conscientiousness = pv.conscientiousness if pv else 0.5
    agreeableness = pv.agreeableness if pv else 0.5
    emotional_stability = pv.emotional_stability if pv else 0.5
    extraversion_val = pv.extraversion if pv else 0.5
    impulsivity = pv.impulsivity if pv else 0.5
    optimism = pv.optimism if pv else 0.5

    return PersonalityTraits(
        risk_aversion=risk_aversion,
        convenience_preference=ls.convenience_preference,
        price_sensitivity=ls.price_sensitivity,
        tech_adoption=ls.tech_adoption,
        luxury_preference=ls.luxury_preference,
        primary_service_preference=ls.primary_service_preference,
        dining_out=ls.dining_out,
        social_activity=social_activity,
        health_consciousness=health_consciousness,
        mobility_dependence=mobility_dependence,
        time_pressure=time_pressure,
        brand_loyalty=brand_loyalty,
        openness_to_experience=openness,
        conscientiousness=conscientiousness,
        agreeableness=agreeableness,
        emotional_stability=emotional_stability,
        extraversion=extraversion_val,
        impulsivity=impulsivity,
        optimism=optimism,
    )
