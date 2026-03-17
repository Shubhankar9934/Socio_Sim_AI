"""
Simulation parameter space for calibration.

Wraps all tunable parameters (personality weights, factor weights,
trait distributions, temperature) into a single vector interface
suitable for scipy optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SimulationParameters:
    """All tunable parameters in vector form for optimiser consumption."""

    personality_weights: Dict[str, float] = field(default_factory=lambda: {
        "convenience_preference": 0.4,
        "price_sensitivity": -0.2,
        "primary_service_preference": 0.3,
        "dining_out": 0.1,
    })
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "personality": 0.35,
        "income": 0.18,
        "social": 0.18,
        "location": 0.09,
        "memory": 0.08,
        "behavioral": 0.12,
    })
    temperature: float = 1.0

    _key_order_personality: List[str] = field(init=False, repr=False, default_factory=list)
    _key_order_factors: List[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._key_order_personality = sorted(self.personality_weights.keys())
        self._key_order_factors = sorted(self.factor_weights.keys())

    @property
    def n_params(self) -> int:
        return len(self._key_order_personality) + len(self._key_order_factors) + 1

    def to_vector(self) -> np.ndarray:
        vals = [self.personality_weights[k] for k in self._key_order_personality]
        vals += [self.factor_weights[k] for k in self._key_order_factors]
        vals.append(self.temperature)
        return np.array(vals, dtype=np.float64)

    def from_vector(self, vec: np.ndarray) -> "SimulationParameters":
        """Return a new SimulationParameters from a flat vector."""
        idx = 0
        pw = {}
        for k in self._key_order_personality:
            pw[k] = float(vec[idx])
            idx += 1
        fw = {}
        for k in self._key_order_factors:
            fw[k] = float(vec[idx])
            idx += 1
        temp = float(vec[idx])
        return SimulationParameters(
            personality_weights=pw,
            factor_weights=fw,
            temperature=max(0.1, temp),
        )

    def bounds(self) -> List[Tuple[float, float]]:
        """Return (lower, upper) bounds for each parameter."""
        b = []
        for k in self._key_order_personality:
            b.append((-1.0, 1.0))
        for k in self._key_order_factors:
            b.append((0.0, 1.0))
        b.append((0.1, 5.0))  # temperature
        return b
