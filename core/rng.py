"""Central seeded RNG helpers for deterministic stochastic behavior."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.settings import get_settings


def stable_seed_from_key(key: str, base_seed: Optional[int] = None) -> int:
    """Build a stable 32-bit seed from (base_seed, key)."""
    settings = get_settings()
    seed = settings.master_seed if base_seed is None else int(base_seed)
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


@dataclass(frozen=True)
class RngPack:
    """Companion NumPy + Python RNG pair derived from same child seed."""

    seed: int
    np_rng: np.random.Generator
    py_rng: random.Random


def make_rng_pack(key: str, base_seed: Optional[int] = None) -> RngPack:
    """Create deterministic RNGs for a logical component."""
    child_seed = stable_seed_from_key(key, base_seed=base_seed)
    return RngPack(
        seed=child_seed,
        np_rng=np.random.default_rng(child_seed),
        py_rng=random.Random(child_seed),
    )


def ensure_np_rng(rng: Optional[np.random.Generator], key: str) -> np.random.Generator:
    """Return provided rng or a deterministic fallback rng."""
    if rng is not None:
        return rng
    settings = get_settings()
    if settings.allow_unseeded_rng:
        return np.random.default_rng()
    return make_rng_pack(key).np_rng


def ensure_py_rng(rng: Optional[random.Random], key: str) -> random.Random:
    """Return provided rng or a deterministic fallback rng."""
    if rng is not None:
        return rng
    settings = get_settings()
    if settings.allow_unseeded_rng:
        return random.Random()
    return make_rng_pack(key).py_rng


def agent_seed_from_id(agent_id: str, base_seed: Optional[int] = None) -> int:
    """Derive a unique 32-bit seed from an agent ID using SHA-256.

    Unlike ``stable_seed_from_key`` this hashes the agent_id itself
    (not just a component key), producing statistically independent
    random streams for every agent regardless of creation order.
    """
    settings = get_settings()
    seed = settings.master_seed if base_seed is None else int(base_seed)
    digest = hashlib.sha256(f"agent:{seed}:{agent_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def agent_rng_pack(agent_id: str, base_seed: Optional[int] = None) -> RngPack:
    """Create a per-agent deterministic RNG pair from the agent's ID."""
    child_seed = agent_seed_from_id(agent_id, base_seed=base_seed)
    return RngPack(
        seed=child_seed,
        np_rng=np.random.default_rng(child_seed),
        py_rng=random.Random(child_seed),
    )

