"""
Gradient-free calibration: minimize JS divergence between simulated and
real-world survey distributions using scipy differential evolution.

Usage:
    from calibration.optimizer import calibrate
    result = calibrate(real_distribution, parameter_space)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.spatial.distance import jensenshannon

from calibration.parameter_space import SimulationParameters


@dataclass
class CalibrationResult:
    """Outcome of a calibration run."""

    best_params: SimulationParameters
    best_loss: float
    loss_history: List[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


def calibration_loss(
    simulated: Dict[str, float],
    real: Dict[str, float],
) -> float:
    """Jensen-Shannon divergence between two distributions over the same keys."""
    keys = sorted(set(simulated.keys()) | set(real.keys()))
    p = np.array([simulated.get(k, 0.0) for k in keys], dtype=np.float64)
    q = np.array([real.get(k, 0.0) for k in keys], dtype=np.float64)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum
    return float(jensenshannon(p, q))


def calibration_report(
    result: CalibrationResult,
) -> Dict[str, Any]:
    """Summarise a calibration run for logging/export."""
    return {
        "best_loss": result.best_loss,
        "n_iterations": result.n_iterations,
        "converged": result.converged,
        "best_params": {
            "personality_weights": result.best_params.personality_weights,
            "factor_weights": result.best_params.factor_weights,
            "temperature": result.best_params.temperature,
        },
        "loss_curve": result.loss_history[-20:],
    }


def calibrate(
    real_distribution: Dict[str, float],
    parameter_space: Optional[SimulationParameters] = None,
    simulator: Optional[Callable[[SimulationParameters], Dict[str, float]]] = None,
    n_iterations: int = 100,
    population_size: int = 15,
    seed: Optional[int] = None,
) -> CalibrationResult:
    """Run differential evolution to find parameters that minimise JS divergence.

    Parameters
    ----------
    real_distribution : dict
        Target distribution to match (option -> probability).
    parameter_space : SimulationParameters, optional
        Starting parameter template (defines keys and bounds).
    simulator : callable, optional
        Function that takes SimulationParameters and returns a simulated
        distribution dict.  If None, a dummy pass-through is used (for testing).
    n_iterations : int
        Maximum generations for the optimizer.
    population_size : int
        DE population multiplier.
    seed : int, optional
        RNG seed for reproducibility.
    """
    if parameter_space is None:
        parameter_space = SimulationParameters()

    bounds = parameter_space.bounds()
    loss_history: List[float] = []

    def objective(vec: np.ndarray) -> float:
        params = parameter_space.from_vector(vec)
        if simulator is not None:
            simulated = simulator(params)
        else:
            keys = list(real_distribution.keys())
            n = len(keys)
            uniform = {k: 1.0 / n for k in keys}
            simulated = uniform
        loss = calibration_loss(simulated, real_distribution)
        loss_history.append(loss)
        return loss

    result: OptimizeResult = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=n_iterations,
        popsize=population_size,
        seed=seed,
        tol=1e-6,
        disp=False,
    )

    best_params = parameter_space.from_vector(result.x)
    return CalibrationResult(
        best_params=best_params,
        best_loss=float(result.fun),
        loss_history=loss_history,
        n_iterations=result.nit if hasattr(result, "nit") else n_iterations,
        converged=result.success,
    )
