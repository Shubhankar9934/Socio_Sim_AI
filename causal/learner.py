"""
Causal structure learning from simulation time-series data.

Uses Granger-causality-inspired analysis on dimension trajectories
to discover and weight causal edges.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from causal.graph import CausalGraph

logger = logging.getLogger(__name__)


class CausalLearner:
    """Learn causal structure from simulation timeline data."""

    def __init__(self, significance_threshold: float = 0.1, lag: int = 1):
        self.significance_threshold = significance_threshold
        self.lag = lag

    def learn_from_timeline(
        self,
        timeline: List[Dict[str, Any]],
        dimension_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """Learn causal graph from a timeline of dimension_means snapshots.

        Each timeline entry should have {"day": int, "dimension_means": dict}.
        Uses lagged cross-correlation as a proxy for Granger causality.
        """
        if not timeline or len(timeline) < 3:
            from causal.graph import build_default_causal_graph
            return build_default_causal_graph()

        if dimension_names is None:
            dimension_names = sorted(timeline[0].get("dimension_means", {}).keys())

        n_dims = len(dimension_names)
        n_steps = len(timeline)

        mat = np.zeros((n_steps, n_dims))
        for t, snap in enumerate(timeline):
            means = snap.get("dimension_means", {})
            for j, dim in enumerate(dimension_names):
                mat[t, j] = means.get(dim, 0.5)

        graph = CausalGraph()
        for d in dimension_names:
            graph.add_node(d)

        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                weight = self._lagged_correlation(mat[:, i], mat[:, j])
                if abs(weight) >= self.significance_threshold:
                    graph.add_edge(
                        dimension_names[i],
                        dimension_names[j],
                        round(weight, 4),
                        mechanism="learned_from_data",
                    )

        return graph

    def _lagged_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute lagged Pearson correlation: does past x predict future y?"""
        if len(x) <= self.lag:
            return 0.0

        x_lagged = x[:-self.lag]
        y_future = y[self.lag:]

        if np.std(x_lagged) < 1e-8 or np.std(y_future) < 1e-8:
            return 0.0

        corr = np.corrcoef(x_lagged, y_future)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    def learn_from_interventions(
        self,
        baseline_means: Dict[str, float],
        intervention_results: List[Dict[str, Any]],
    ) -> CausalGraph:
        """Learn causal edges from intervention experiments (scenarios).

        Each intervention_result should have:
          - "intervention": {"variable": str, "value": float}
          - "outcome_means": {dim: float, ...}
        """
        graph = CausalGraph()
        all_dims = set(baseline_means.keys())
        for d in all_dims:
            graph.add_node(d)

        for result in intervention_results:
            iv = result.get("intervention", {})
            iv_var = iv.get("variable", "")
            outcome_means = result.get("outcome_means", {})

            if not iv_var or iv_var not in all_dims:
                continue

            for dim in all_dims:
                if dim == iv_var:
                    continue
                delta = outcome_means.get(dim, 0.5) - baseline_means.get(dim, 0.5)
                if abs(delta) >= self.significance_threshold:
                    graph.add_edge(iv_var, dim, round(delta * 5.0, 4),
                                   mechanism="intervention_observed")

        return graph

    def merge_graphs(self, *graphs: CausalGraph) -> CausalGraph:
        """Merge multiple causal graphs, averaging edge weights."""
        merged = CausalGraph()
        edge_weights: Dict[Tuple[str, str], List[float]] = {}

        for g in graphs:
            for n in g.nodes:
                merged.add_node(n)
            for (c, e), edge in g.edges.items():
                edge_weights.setdefault((c, e), []).append(edge.weight)

        for (c, e), weights in edge_weights.items():
            avg_weight = float(np.mean(weights))
            if abs(avg_weight) >= self.significance_threshold:
                merged.add_edge(c, e, round(avg_weight, 4), mechanism="merged")

        return merged
