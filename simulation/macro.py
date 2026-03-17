"""
Macro Feedback Loop: aggregate agent states into population-level metrics,
then feed those metrics back as per-dimension nudges to individual agents.

This creates self-reinforcing trends (adoption bandwagons, opinion shifts)
that emerge from the collective behavior of the population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from agents.behavior import DIMENSION_NAMES


@dataclass
class MacroMetrics:
    """Population-level aggregated behavioral metrics."""

    dimension_means: Dict[str, float] = field(default_factory=dict)
    dimension_stds: Dict[str, float] = field(default_factory=dict)
    adoption_rate: float = 0.0
    avg_satisfaction: float = 0.5
    tech_penetration: float = 0.5
    market_trend: float = 0.0
    population_size: int = 0

    # Supply-side feedback signals
    demand_pressure: float = 0.0
    price_elasticity_signal: float = 0.0
    service_utilization: float = 0.0


def compute_macro_metrics(agents: List[Dict[str, Any]]) -> MacroMetrics:
    """Aggregate latent states across the population into macro signals."""
    vectors = []
    satisfactions = []
    adopters = 0

    for a in agents:
        state = a.get("state")
        if state is None:
            continue
        latent = getattr(state, "latent_state", None)
        if latent is None:
            continue
        vectors.append(latent.to_vector())
        sat = getattr(state, "satisfaction_general", None)
        if sat is not None:
            satisfactions.append(sat)
        if getattr(state, "latent_state", None) and state.latent_state.get("convenience_seeking", 0) >= 0.5:
            adopters += 1

    n = len(vectors)
    if n == 0:
        return MacroMetrics()

    mat = np.array(vectors)
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)

    dim_means = {d: float(means[i]) for i, d in enumerate(DIMENSION_NAMES)}
    dim_stds = {d: float(stds[i]) for i, d in enumerate(DIMENSION_NAMES)}

    convenience = dim_means.get("convenience_seeking", 0.5)
    price_sens = dim_means.get("price_sensitivity", 0.5)

    return MacroMetrics(
        dimension_means=dim_means,
        dimension_stds=dim_stds,
        adoption_rate=adopters / n,
        avg_satisfaction=float(np.mean(satisfactions)) if satisfactions else 0.5,
        tech_penetration=dim_means.get("technology_openness", 0.5),
        market_trend=convenience - 0.5,
        population_size=n,
        demand_pressure=convenience,
        price_elasticity_signal=price_sens,
        service_utilization=min(1.0, (adopters / n) * 1.2),
    )


def macro_influence(
    macro: MacroMetrics,
    macro_rate: float = 0.01,
) -> Dict[str, float]:
    """Convert macro metrics into per-dimension trend signals.

    Returns a dict of dimension_name -> trend_value that can be passed
    to ``BehavioralLatentState.apply_macro_influence()``.

    The signal is the population mean for each dimension, so agents
    are gently nudged toward the population average (conformity pressure).
    Stronger susceptibility in the agent amplifies this effect.
    """
    return dict(macro.dimension_means)
