"""
Identity Layer: stable self-concept that evolves at 1/10 the rate of beliefs.

IdentityState captures *who the agent is* across time -- core values that
resist rapid change.  When beliefs diverge from identity, cognitive
dissonance rises, producing realistic opinion inertia and consistent
long-horizon personas.

Update rule:
    core_values += identity_lr * (beliefs - core_values)
    identity_strength += calcification_delta

Identity strength replaces raw calcification as the semantic measure of
resistance to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from agents.belief_network import BeliefNetwork
    from population.personas import Persona

from agents.belief_network import BELIEF_DIMENSIONS, _N_BELIEFS


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass
class IdentityState:
    """Stable self-concept anchoring an agent's belief trajectory."""

    core_values: np.ndarray = field(
        default_factory=lambda: np.full(_N_BELIEFS, 0.5),
    )
    identity_strength: float = 0.1
    narrative_self: str = ""

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "core_values": {d: float(self.core_values[i]) for i, d in enumerate(BELIEF_DIMENSIONS)},
            "identity_strength": self.identity_strength,
            "narrative_self": self.narrative_self,
        }

    def divergence_from_beliefs(self, beliefs_vec: np.ndarray) -> float:
        """L2 distance between current beliefs and core values (0-1 range)."""
        diff = beliefs_vec - self.core_values
        return float(np.clip(np.linalg.norm(diff) / np.sqrt(_N_BELIEFS), 0.0, 1.0))

    def update(
        self,
        beliefs_vec: np.ndarray,
        *,
        identity_lr: float = 0.005,
        strength_growth: float = 0.001,
    ) -> None:
        """Slow drift of core values toward current beliefs.

        Identity learns at ~1/10 the rate of beliefs, so it acts as a
        stabilising attractor rather than a follower.
        """
        delta = beliefs_vec - self.core_values
        self.core_values = np.clip(
            self.core_values + identity_lr * delta, 0.0, 1.0,
        )
        self.identity_strength = _clamp(self.identity_strength + strength_growth)


def init_identity_from_beliefs(beliefs: "BeliefNetwork") -> IdentityState:
    """Seed identity directly from initial beliefs -- they are the same at t=0."""
    return IdentityState(
        core_values=beliefs.to_vector().copy(),
        identity_strength=0.1,
        narrative_self="",
    )


def build_narrative_self(persona: "Persona", identity: IdentityState) -> str:
    """Generate a compressed self-concept string from top identity dimensions."""
    vals = {d: float(identity.core_values[i]) for i, d in enumerate(BELIEF_DIMENSIONS)}
    top = sorted(vals.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:3]
    parts = []
    for dim, val in top:
        direction = "high" if val > 0.6 else ("low" if val < 0.4 else "moderate")
        parts.append(f"{direction} {dim.replace('_', ' ')}")
    return f"{persona.nationality} {persona.age}, {', '.join(parts)}"
