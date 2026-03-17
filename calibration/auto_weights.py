"""
Auto-learning factor weights: optimize factor weights per question
against reference distributions using differential evolution.

Leverages the existing calibration infrastructure but targets
factor_weights specifically, producing a per-question-model mapping.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)

FACTOR_NAMES = [
    "personality", "income", "social", "location",
    "memory", "behavioral", "belief",
]


@dataclass
class WeightLearningResult:
    question: str
    learned_weights: Dict[str, float]
    best_loss: float
    n_iterations: int
    converged: bool


@dataclass
class AutoWeightsResult:
    results: List[WeightLearningResult] = field(default_factory=list)
    overall_loss: float = 0.0


def _weights_to_vector(weights: Dict[str, float]) -> np.ndarray:
    return np.array([weights.get(f, 0.1) for f in FACTOR_NAMES], dtype=np.float64)


def _vector_to_weights(vec: np.ndarray) -> Dict[str, float]:
    return {FACTOR_NAMES[i]: round(float(vec[i]), 4) for i in range(len(FACTOR_NAMES))}


def _weight_bounds() -> list:
    return [(-0.5, 1.0)] * len(FACTOR_NAMES)


class FactorWeightLearner:
    """Optimize factor weights for one or more questions against reference data."""

    def __init__(self, n_iterations: int = 50, seed: Optional[int] = 42):
        self.n_iterations = n_iterations
        self.seed = seed

    def learn_weights_for_question(
        self,
        question: str,
        reference_distribution: Dict[str, float],
        agents: List[Dict[str, Any]],
        simulate_fn: Optional[Callable] = None,
    ) -> WeightLearningResult:
        """Optimize factor weights for a single question.

        Parameters
        ----------
        simulate_fn : callable, optional
            (factor_weights_dict) -> simulated_distribution_dict.
            If None, a default simulator is built from the agents.
        """
        if simulate_fn is None:
            simulate_fn = self._build_default_simulator(question, agents)

        def objective(vec: np.ndarray) -> float:
            weights = _vector_to_weights(vec)
            try:
                simulated = simulate_fn(weights)
            except Exception:
                return 1.0
            keys = sorted(set(simulated.keys()) | set(reference_distribution.keys()))
            p = np.array([simulated.get(k, 0.0) for k in keys], dtype=np.float64)
            q = np.array([reference_distribution.get(k, 0.0) for k in keys], dtype=np.float64)
            if p.sum() > 0:
                p = p / p.sum()
            if q.sum() > 0:
                q = q / q.sum()
            return float(jensenshannon(p, q))

        result = differential_evolution(
            objective,
            bounds=_weight_bounds(),
            maxiter=self.n_iterations,
            popsize=10,
            seed=self.seed,
            tol=1e-5,
            disp=False,
        )

        best_weights = _vector_to_weights(result.x)
        return WeightLearningResult(
            question=question,
            learned_weights=best_weights,
            best_loss=float(result.fun),
            n_iterations=result.nit if hasattr(result, "nit") else self.n_iterations,
            converged=result.success,
        )

    def learn_weights(
        self,
        questions: List[str],
        reference_distributions: Dict[str, Dict[str, float]],
        agents: List[Dict[str, Any]],
    ) -> AutoWeightsResult:
        """Optimize factor weights for multiple questions."""
        results: List[WeightLearningResult] = []
        total_loss = 0.0

        for q in questions:
            ref_dist = reference_distributions.get(q)
            if not ref_dist:
                logger.warning("No reference distribution for question: %s", q)
                continue
            wr = self.learn_weights_for_question(q, ref_dist, agents)
            results.append(wr)
            total_loss += wr.best_loss

        avg_loss = total_loss / max(1, len(results))
        return AutoWeightsResult(results=results, overall_loss=avg_loss)

    def _build_default_simulator(
        self, question: str, agents: List[Dict[str, Any]]
    ) -> Callable:
        """Build a quick simulator that runs the decision pipeline with given weights."""
        def simulate(factor_weights: Dict[str, float]) -> Dict[str, float]:
            from agents.perception import perceive, detect_question_model
            from agents.decision import compute_distribution, sample_from_distribution
            from agents.personality import personality_from_persona
            from agents.factor_graph import DecisionContext

            perception = perceive(question)
            question_model = detect_question_model(perception)

            patched_model = copy.copy(question_model)
            patched_model.factor_weights = factor_weights

            counts: Dict[str, int] = {}
            sample_size = min(100, len(agents))
            rng = np.random.default_rng(self.seed)

            for a in agents[:sample_size]:
                persona = a.get("persona")
                state = a.get("state")
                if not persona:
                    continue
                traits = personality_from_persona(persona)
                env = {"dimension_weights": dict(patched_model.dimension_weights)}
                if state and hasattr(state, "beliefs"):
                    env["beliefs"] = state.beliefs
                ctx = DecisionContext(
                    persona=persona, traits=traits, perception=perception,
                    friends_using=a.get("social_trait_fraction", 0.0),
                    location_quality=a.get("location_quality", 0.5),
                    environment=env,
                )
                try:
                    dist = compute_distribution(patched_model, ctx, agent_state=state,
                                                persona=persona, traits=traits, rng=rng)
                    chosen = sample_from_distribution(dist, rng=rng)
                    counts[chosen] = counts.get(chosen, 0) + 1
                except Exception:
                    pass

            total = sum(counts.values())
            if total == 0:
                return {}
            return {k: v / total for k, v in counts.items()}

        return simulate
