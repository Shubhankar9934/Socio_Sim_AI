"""
End-to-end calibration pipeline: dimension discovery -> weight learning ->
simulation -> comparison -> validation.

Connects Phases 1, 4, 5 into a single fit() call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon

from calibration.auto_weights import FactorWeightLearner, AutoWeightsResult
from calibration.data_loader import RealSurveyData

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Full calibration result with train/test metrics."""

    weights_result: Optional[AutoWeightsResult] = None
    train_js_divergence: Dict[str, float] = field(default_factory=dict)
    test_js_divergence: Dict[str, float] = field(default_factory=dict)
    overall_train_loss: float = 0.0
    overall_test_loss: float = 0.0
    n_questions: int = 0
    converged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_questions": self.n_questions,
            "overall_train_loss": self.overall_train_loss,
            "overall_test_loss": self.overall_test_loss,
            "converged": self.converged,
            "train_js_divergence": self.train_js_divergence,
            "test_js_divergence": self.test_js_divergence,
            "learned_weights": {
                r.question: r.learned_weights
                for r in (self.weights_result.results if self.weights_result else [])
            },
        }


class CalibrationPipeline:
    """End-to-end: real data -> holdout split -> weight learning -> validation."""

    def __init__(
        self,
        n_iterations: int = 50,
        holdout_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.n_iterations = n_iterations
        self.holdout_fraction = holdout_fraction
        self.seed = seed

    def fit(
        self,
        real_data: Dict[str, RealSurveyData],
        agents: List[Dict[str, Any]],
    ) -> CalibrationReport:
        """Run the full calibration pipeline.

        Parameters
        ----------
        real_data : dict
            question_text -> RealSurveyData with real responses.
        agents : list
            Agent dicts from the population store.
        """
        train_refs: Dict[str, Dict[str, float]] = {}
        test_refs: Dict[str, Dict[str, float]] = {}
        train_data_map: Dict[str, RealSurveyData] = {}
        test_data_map: Dict[str, RealSurveyData] = {}

        for q, data in real_data.items():
            if data.n_responses < 10:
                logger.warning("Skipping %s: only %d responses", q, data.n_responses)
                continue
            train, test = data.holdout_split(
                train_fraction=1.0 - self.holdout_fraction, seed=self.seed
            )
            train_refs[q] = train.to_reference_distribution()
            test_refs[q] = test.to_reference_distribution()
            train_data_map[q] = train
            test_data_map[q] = test

        if not train_refs:
            return CalibrationReport(converged=False)

        learner = FactorWeightLearner(
            n_iterations=self.n_iterations, seed=self.seed
        )
        weights_result = learner.learn_weights(
            questions=list(train_refs.keys()),
            reference_distributions=train_refs,
            agents=agents,
        )

        train_js: Dict[str, float] = {}
        test_js: Dict[str, float] = {}

        for wr in weights_result.results:
            train_js[wr.question] = wr.best_loss

            sim_fn = learner._build_default_simulator(wr.question, agents)
            try:
                sim_dist = sim_fn(wr.learned_weights)
                ref = test_refs.get(wr.question, {})
                keys = sorted(set(sim_dist.keys()) | set(ref.keys()))
                p = np.array([sim_dist.get(k, 0.0) for k in keys])
                q_arr = np.array([ref.get(k, 0.0) for k in keys])
                if p.sum() > 0:
                    p = p / p.sum()
                if q_arr.sum() > 0:
                    q_arr = q_arr / q_arr.sum()
                test_js[wr.question] = float(jensenshannon(p, q_arr))
            except Exception:
                test_js[wr.question] = 1.0

        overall_train = np.mean(list(train_js.values())) if train_js else 0.0
        overall_test = np.mean(list(test_js.values())) if test_js else 0.0

        return CalibrationReport(
            weights_result=weights_result,
            train_js_divergence=train_js,
            test_js_divergence=test_js,
            overall_train_loss=float(overall_train),
            overall_test_loss=float(overall_test),
            n_questions=len(train_refs),
            converged=all(r.converged for r in weights_result.results),
        )
