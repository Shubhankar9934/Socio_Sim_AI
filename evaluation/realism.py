"""
Population realism: JS divergence, KL divergence vs target distributions.
"""

from typing import Dict, List

from population.personas import Persona
from population.validator import population_realism_score, validate_population


def compute_realism_score(personas: List[Persona]) -> float:
    """Aggregate realism score in [0, 1]. Higher = more realistic."""
    return population_realism_score(personas)


def compute_realism_report(
    personas: List[Persona],
    threshold: float = 0.85,
) -> Dict[str, any]:
    """Full realism report: score, passed, per-attribute scores."""
    passed, score, per_attr = validate_population(personas, realism_threshold=threshold)
    return {
        "population_realism_score": score,
        "passed": passed,
        "threshold": threshold,
        "per_attribute": per_attr,
    }
