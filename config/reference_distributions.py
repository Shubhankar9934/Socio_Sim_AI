"""
Reference distributions for survey validation, keyed by question_model name.

Each entry represents approximate real-world distributions from Dubai
market research or general UAE survey benchmarks.  Used by the
distribution_validation module to compute JS divergence and pass/fail.

Includes a dynamic fallback estimator for unknown question models.
"""

from typing import Dict, List, Optional

REFERENCE_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "food_delivery_frequency": {
        "rarely": 0.30,
        "1-2 per week": 0.35,
        "3-4 per week": 0.20,
        "daily": 0.10,
        "multiple per day": 0.05,
    },
    "parking_satisfaction": {
        "1": 0.15,
        "2": 0.25,
        "3": 0.30,
        "4": 0.20,
        "5": 0.10,
    },
    "transport_satisfaction": {
        "1": 0.08,
        "2": 0.17,
        "3": 0.30,
        "4": 0.30,
        "5": 0.15,
    },
    "housing_satisfaction": {
        "1": 0.05,
        "2": 0.15,
        "3": 0.30,
        "4": 0.35,
        "5": 0.15,
    },
    "cost_of_living_satisfaction": {
        "1": 0.18,
        "2": 0.24,
        "3": 0.26,
        "4": 0.20,
        "5": 0.12,
    },
    "shopping_frequency": {
        "rarely": 0.15,
        "1-2 per month": 0.30,
        "weekly": 0.30,
        "2-3 per week": 0.15,
        "daily": 0.10,
    },
    "nps_recommendation": {
        "0": 0.02,
        "1": 0.02,
        "2": 0.03,
        "3": 0.04,
        "4": 0.05,
        "5": 0.10,
        "6": 0.15,
        "7": 0.20,
        "8": 0.22,
        "9": 0.12,
        "10": 0.05,
    },
    "tech_adoption_likelihood": {
        "very unlikely": 0.05,
        "unlikely": 0.15,
        "neutral": 0.25,
        "likely": 0.35,
        "very likely": 0.20,
    },
    "policy_support": {
        "strongly oppose": 0.10,
        "oppose": 0.15,
        "neutral": 0.30,
        "support": 0.30,
        "strongly support": 0.15,
    },
    "generic_likert": {
        "1": 0.10,
        "2": 0.20,
        "3": 0.35,
        "4": 0.25,
        "5": 0.10,
    },
    "generic_frequency": {
        "never": 0.10,
        "rarely": 0.20,
        "sometimes": 0.35,
        "often": 0.25,
        "very often": 0.10,
    },
    "generic_duration": {},  # No discrete scale; open-ended tenure
}


def estimate_reference_from_scale(scale: List[str]) -> Dict[str, float]:
    """Generate a bell-curve-like reference for an unknown scale.

    Produces a symmetric distribution that peaks at the center option(s),
    which serves as a plausible default expectation for most surveys.
    """
    n = len(scale)
    if n == 0:
        return {}
    if n == 1:
        return {scale[0]: 1.0}
    import numpy as np
    center = (n - 1) / 2.0
    sigma = n / 4.0
    raw = [float(np.exp(-0.5 * ((i - center) / sigma) ** 2)) for i in range(n)]
    total = sum(raw)
    return {scale[i]: round(raw[i] / total, 4) for i in range(n)}


def get_reference_distribution(
    question_model_key: str,
    scale: Optional[List[str]] = None,
    domain_id: Optional[str] = None,
) -> Dict[str, float]:
    """Look up a reference distribution by question model key.

    Checks domain-specific overrides first, then falls back to the static
    registry, then to dynamic estimation from the scale.
    """
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config(domain_id)
        domain_ref = cfg.reference_distributions.get(question_model_key)
        if domain_ref:
            return domain_ref
    except Exception:
        pass
    try:
        from config.generated_registry import load_generated_registry

        generated = load_generated_registry().get("references", {})
        if question_model_key in generated and isinstance(generated[question_model_key], dict):
            return generated[question_model_key]
    except Exception:
        pass
    ref = REFERENCE_DISTRIBUTIONS.get(question_model_key)
    if ref:
        return ref
    from config.settings import get_settings

    if get_settings().strict_mode:
        raise ValueError(
            f"Missing reference distribution for '{question_model_key}' in strict_mode.",
        )
    if scale:
        return estimate_reference_from_scale(scale)
    return {}
