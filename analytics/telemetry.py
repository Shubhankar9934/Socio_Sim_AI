"""Per-step telemetry collector for simulation runs.

Accumulates lightweight metrics each day so post-hoc analysis can trace
population-level belief/activation trajectories, event counts, and drift
without storing full agent snapshots.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DailySnapshot:
    day: int
    activation_mean: float
    activation_std: float
    belief_means: Dict[str, float]
    belief_variances: Dict[str, float]
    latent_means: Dict[str, float]
    polarization: float
    belief_entropy: float
    event_count: int
    cascade_count: int = 0
    population_size: int = 0
    regime: str = "dormant"
    consensus_index: float = 1.0
    instability: float = 0.0


class TelemetryCollector:
    """Accumulates per-step metrics during a simulation run."""

    def __init__(self) -> None:
        self.snapshots: List[DailySnapshot] = []

    def record(
        self,
        agents: List[Dict[str, Any]],
        day: int,
        activation_state: Dict[str, np.ndarray],
        event_count: int = 0,
        cascade_count: int = 0,
    ) -> None:
        N = len(agents)
        if N == 0:
            return

        act = activation_state.get("activation", np.zeros(N))

        from agents.behavior import DIMENSION_NAMES
        from agents.belief_network import BELIEF_DIMENSIONS

        belief_vecs = []
        latent_vecs = []
        for a in agents:
            state = a.get("state")
            if state and hasattr(state, "beliefs"):
                belief_vecs.append(state.beliefs.to_vector())
            if state and hasattr(state, "latent_state"):
                latent_vecs.append(state.latent_state.to_vector())

        belief_means: Dict[str, float] = {}
        belief_vars: Dict[str, float] = {}
        polarization = 0.0
        entropy = 0.0
        if belief_vecs:
            bmat = np.array(belief_vecs)
            var_per_dim = np.var(bmat, axis=0)
            polarization = float(var_per_dim.mean())
            bmat_safe = np.clip(bmat, 1e-9, 1.0 - 1e-9)
            entropy = float(
                -np.sum(bmat_safe * np.log(bmat_safe), axis=1).mean()
            )
            for i, dim in enumerate(BELIEF_DIMENSIONS):
                if i < bmat.shape[1]:
                    belief_means[dim] = float(bmat[:, i].mean())
                    belief_vars[dim] = float(var_per_dim[i])

        latent_means: Dict[str, float] = {}
        if latent_vecs:
            lmat = np.array(latent_vecs)
            for i, dim in enumerate(DIMENSION_NAMES):
                if i < lmat.shape[1]:
                    latent_means[dim] = float(lmat[:, i].mean())

        regime = "dormant"
        consensus_idx = 1.0
        instability_val = 0.0
        if belief_vecs:
            try:
                from simulation.cascade_detector import compute_regime_metrics
                regime_data = compute_regime_metrics(act, np.array(belief_vecs))
                regime = regime_data["regime"]
                consensus_idx = regime_data["consensus_index"]
                instability_val = regime_data["instability"]
            except Exception:
                pass

        snap = DailySnapshot(
            day=day,
            activation_mean=float(act.mean()),
            activation_std=float(act.std()),
            belief_means=belief_means,
            belief_variances=belief_vars,
            latent_means=latent_means,
            polarization=polarization,
            belief_entropy=entropy,
            event_count=event_count,
            cascade_count=cascade_count,
            population_size=N,
            regime=regime,
            consensus_index=consensus_idx,
            instability=instability_val,
        )
        self.snapshots.append(snap)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Export snapshots as a list of plain dicts."""
        from dataclasses import asdict
        return [asdict(s) for s in self.snapshots]
