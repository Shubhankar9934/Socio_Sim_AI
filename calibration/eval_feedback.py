"""
Closed-loop feedback: evaluation metrics -> composite loss -> calibration.

This module bridges the evaluation pipeline (`evaluation.report`) with the
calibration optimizer (`calibration.optimizer`) so that the system can
self-improve by minimising a multi-objective loss derived from realism,
drift, consistency, distribution fit, and narrative quality metrics.

Usage:
    from calibration.eval_feedback import closed_loop_calibrate

    result = await closed_loop_calibrate(
        agents, personas, real_distribution, make_simulator,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from calibration.optimizer import CalibrationResult, calibrate, calibration_loss
from calibration.parameter_space import SimulationParameters

logger = logging.getLogger(__name__)


def evaluation_composite_loss(
    eval_report: Dict[str, Any],
    simulated_dist: Dict[str, float],
    real_dist: Dict[str, float],
    *,
    w_distribution: float = 0.35,
    w_realism: float = 0.20,
    w_drift: float = 0.15,
    w_consistency: float = 0.15,
    w_narrative: float = 0.15,
) -> float:
    """Combine evaluation metrics into a single scalar loss in [0, 1].

    The loss is designed so that *lower is better* and can be directly
    consumed by `scipy.optimize.differential_evolution`.
    """
    js = calibration_loss(simulated_dist, real_dist)

    realism = eval_report.get("population_realism", {}).get("population_realism_score", 0.0)
    realism_loss = 1.0 - float(realism)

    drift_rate = eval_report.get("drift", {}).get("rate", 0.0)
    drift_loss = float(np.clip(drift_rate, 0.0, 1.0))

    consistency = eval_report.get("consistency_score", 1.0)
    consistency_loss = 1.0 - float(consistency)

    dup_rate = eval_report.get("narrative_similarity", {}).get("duplicate_rate", 0.0)
    narrative_loss = float(np.clip(dup_rate, 0.0, 1.0))

    composite = (
        w_distribution * js
        + w_realism * realism_loss
        + w_drift * drift_loss
        + w_consistency * consistency_loss
        + w_narrative * narrative_loss
    )
    return float(np.clip(composite, 0.0, 1.0))


def make_eval_aware_simulator(
    run_fn: Callable[[SimulationParameters], Dict[str, Any]],
    personas: List[Any],
    real_distribution: Dict[str, float],
) -> Callable[[SimulationParameters], Dict[str, float]]:
    """Wrap a simulation runner so the calibration optimizer's objective
    uses the full composite loss instead of raw JS divergence.

    Parameters
    ----------
    run_fn : callable
        (SimulationParameters) -> {"survey_responses": [...], "distribution": {...}}
    personas : list of Persona
    real_distribution : dict
        Target survey distribution.
    """
    import asyncio
    from evaluation.report import run_evaluation

    def simulator(params: SimulationParameters) -> Dict[str, float]:
        result = run_fn(params)
        sim_dist = result.get("distribution", {})
        survey_responses = result.get("survey_responses", [])

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    eval_report = pool.submit(
                        asyncio.run,
                        run_evaluation(personas, survey_responses),
                    ).result()
            else:
                eval_report = loop.run_until_complete(
                    run_evaluation(personas, survey_responses),
                )
        except Exception:
            logger.exception("Evaluation during calibration failed; using JS only")
            return sim_dist

        composite = evaluation_composite_loss(eval_report, sim_dist, real_distribution)
        # Return a synthetic distribution whose JS from real = composite loss.
        # The calibration optimizer uses calibration_loss() on the returned dict,
        # so we need to return the actual sim_dist and override loss externally.
        # Instead we monkey-patch the score via a sentinel key.
        sim_dist["__composite_loss__"] = composite
        return sim_dist

    return simulator


async def closed_loop_calibrate(
    run_fn: Callable[[SimulationParameters], Dict[str, Any]],
    personas: List[Any],
    real_distribution: Dict[str, float],
    parameter_space: Optional[SimulationParameters] = None,
    n_iterations: int = 50,
    seed: Optional[int] = None,
) -> CalibrationResult:
    """End-to-end calibration loop that uses full evaluation metrics."""
    sim = make_eval_aware_simulator(run_fn, personas, real_distribution)
    return calibrate(
        real_distribution,
        parameter_space=parameter_space,
        simulator=sim,
        n_iterations=n_iterations,
        seed=seed,
    )
