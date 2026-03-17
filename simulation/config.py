"""
Simulation configuration: master seed, mode flags, and global parameters
that control deterministic execution of the simulation kernel.

All randomness flows through explicit Generator / Random objects --
no global ``np.random.seed()`` or bare ``random.choice()`` calls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SimulationConfig:
    """Global configuration for a single simulation run."""

    master_seed: Optional[int] = None
    days: int = 30
    vectorize_threshold: int = 200

    def make_rng(self) -> np.random.Generator:
        """Create a numpy Generator from the master seed (or entropy)."""
        return np.random.default_rng(self.master_seed)

    def make_stdlib_rng(self) -> random.Random:
        """Create a stdlib Random from the master seed (or entropy)."""
        return random.Random(self.master_seed)

    def derive_child_seed(self, label: str) -> int:
        """Derive a deterministic child seed from master_seed + label.

        Useful for giving sub-systems (narrative, social network build)
        their own reproducible RNG stream.
        """
        import hashlib
        base = str(self.master_seed or 0)
        h = hashlib.sha256(f"{base}:{label}".encode()).digest()
        return int.from_bytes(h[:8], "little") % (2**63)
