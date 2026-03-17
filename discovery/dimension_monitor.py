"""
Dimension Evolution Monitor: detect when existing dimensions are insufficient
and suggest new dimensions at epoch boundaries.

Runs between simulation epochs (not mid-step) to avoid breaking the
vectorized N x 12 / N x 7 matrix pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdequacyReport:
    """Result of a dimension adequacy check."""

    day: int
    adequacy_score: float  # 0-1, higher = dimensions are sufficient
    residual_variance: float  # unexplained variance in agent behavior
    suggested_new_dimensions: List[str] = field(default_factory=list)
    event_signals: List[str] = field(default_factory=list)
    needs_extension: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "adequacy_score": round(self.adequacy_score, 4),
            "residual_variance": round(self.residual_variance, 4),
            "suggested_new_dimensions": self.suggested_new_dimensions,
            "event_signals": self.event_signals,
            "needs_extension": self.needs_extension,
        }


class DimensionEvolutionMonitor:
    """Monitor dimension adequacy across simulation epochs."""

    def __init__(
        self,
        variance_threshold: float = 0.15,
        min_agents: int = 50,
    ):
        self.variance_threshold = variance_threshold
        self.min_agents = min_agents
        self._history: List[AdequacyReport] = []

    def check_adequacy(
        self,
        agents: List[Dict[str, Any]],
        events_log: Optional[List[Dict[str, Any]]] = None,
        day: int = 0,
    ) -> AdequacyReport:
        """Analyze whether current dimensions adequately capture agent behavior.

        Computes residual variance by measuring how much of the behavioral
        spread is *not* captured by the core dimension axes.
        """
        from agents.behavior import DIMENSION_NAMES

        vecs = []
        for a in agents:
            state = a.get("state")
            if state and hasattr(state, "latent_state"):
                vecs.append(state.latent_state.to_vector())

        if len(vecs) < self.min_agents:
            report = AdequacyReport(day=day, adequacy_score=1.0, residual_variance=0.0)
            self._history.append(report)
            return report

        mat = np.array(vecs)

        total_var = np.var(mat)
        per_dim_var = np.var(mat, axis=0)
        explained_var = np.sum(per_dim_var)
        residual_var = max(0.0, total_var * mat.shape[1] - explained_var)

        from sklearn.decomposition import PCA
        try:
            n_components = min(mat.shape[1], mat.shape[0])
            pca = PCA(n_components=n_components)
            pca.fit(mat)
            explained_ratio = np.sum(pca.explained_variance_ratio_)
            residual_fraction = 1.0 - explained_ratio
        except Exception:
            residual_fraction = 0.0

        event_signals = self._analyze_unmapped_events(events_log or [])

        needs_extension = (
            residual_fraction > self.variance_threshold
            or len(event_signals) > 2
        )

        adequacy = 1.0 - residual_fraction

        suggested = []
        if needs_extension:
            suggested = self._suggest_dimensions(mat, event_signals)

        report = AdequacyReport(
            day=day,
            adequacy_score=float(adequacy),
            residual_variance=float(residual_fraction),
            suggested_new_dimensions=suggested,
            event_signals=event_signals,
            needs_extension=needs_extension,
        )
        self._history.append(report)
        return report

    def _analyze_unmapped_events(
        self, events_log: List[Dict[str, Any]]
    ) -> List[str]:
        """Find event types that don't map to any existing dimension."""
        from agents.behavior import DIMENSION_NAMES
        from agents.belief_network import BELIEF_DIMENSIONS

        all_dims = set(DIMENSION_NAMES) | set(BELIEF_DIMENSIONS)
        unmapped: List[str] = []

        for event in events_log:
            impacts = event.get("dimension_impacts", {})
            belief_impacts = event.get("belief_impacts", {})
            event_type = event.get("type", "")
            all_impact_dims = set(impacts.keys()) | set(belief_impacts.keys())

            unmapped_dims = all_impact_dims - all_dims
            if unmapped_dims:
                unmapped.append(f"{event_type}: {', '.join(unmapped_dims)}")

        return unmapped

    def _suggest_dimensions(
        self,
        mat: np.ndarray,
        event_signals: List[str],
    ) -> List[str]:
        """Suggest new dimension names based on PCA residuals and events."""
        suggestions = []

        try:
            from sklearn.decomposition import PCA
            n_extra = min(3, mat.shape[0] - mat.shape[1])
            if n_extra > 0:
                pca = PCA(n_components=mat.shape[1] + n_extra)
                pca.fit(mat)
                for i in range(mat.shape[1], mat.shape[1] + n_extra):
                    if pca.explained_variance_ratio_[i] > 0.02:
                        suggestions.append(f"latent_factor_{i - mat.shape[1]}")
        except Exception:
            pass

        for signal in event_signals[:3]:
            dim_name = signal.split(":")[0].strip().replace(" ", "_").lower()
            suggestions.append(f"event_{dim_name}")

        return suggestions[:5]

    def extend_dimensions(
        self,
        new_dim_names: List[str],
        agents: List[Dict[str, Any]],
        kind: str = "behavioral",
    ) -> None:
        """Add new dimensions to all agents at epoch boundary.

        Only call this between simulation epochs, not mid-step.
        """
        for a in agents:
            state = a.get("state")
            if not state:
                continue

            if kind == "behavioral" and hasattr(state, "latent_state"):
                for dim in new_dim_names:
                    if dim not in state.latent_state.extra:
                        state.latent_state.extra[dim] = 0.5
            elif kind == "belief" and hasattr(state, "beliefs"):
                for dim in new_dim_names:
                    if dim not in state.beliefs.extra:
                        state.beliefs.extra[dim] = 0.5

        logger.info("Extended %s dimensions with %s for %d agents",
                     kind, new_dim_names, len(agents))

    @property
    def history(self) -> List[AdequacyReport]:
        return list(self._history)
