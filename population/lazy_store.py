"""
Lazy population store for extreme-scale simulations.

Instead of generating all N agents upfront, ``LazyPopulationStore``
creates agents on first access via deterministic per-agent seeds.
An LRU cache bounds memory usage so that theoretically infinite
populations can be served with constant memory overhead.

Usage:
    store = LazyPopulationStore(total=1_000_000, seed=42)
    agent = store[42]           # generates agent 42 on-demand
    agent = store["DXB_0042"]   # lookup by ID
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Union

from core.rng import agent_rng_pack, agent_seed_from_id
from population.personas import Persona


class LazyPopulationStore:
    """LRU-cached on-demand agent generator.

    Parameters
    ----------
    total : int
        Logical population size (for index bounds).
    seed : int | None
        Master seed for deterministic generation.
    id_prefix : str
        Agent ID prefix (e.g. "DXB").
    method : str
        Synthesis method forwarded to the single-agent generator.
    cache_size : int
        Maximum number of personas held in memory.
    """

    def __init__(
        self,
        total: int = 100_000,
        seed: int | None = 42,
        id_prefix: str = "DXB",
        method: Literal["monte_carlo", "bayesian", "ipf"] = "bayesian",
        cache_size: int = 10_000,
    ):
        self.total = total
        self.seed = seed
        self.id_prefix = id_prefix
        self.method = method
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Persona] = OrderedDict()
        self._id_to_index: Dict[str, int] = {}

    def _agent_id(self, index: int) -> str:
        return f"{self.id_prefix}_{index:04d}"

    def _generate_one(self, index: int) -> Persona:
        """Generate a single persona using its deterministic hash seed."""
        from population.synthesis import (
            _family_from_household,
            _lifestyle_from_demographics,
            _mobility_from_location,
            _personal_anchors_from_demographics,
            _generate_personality_vector,
            _weighted_choice,
            _sample_income_given_nationality,
            _sample_location_given_income,
            _sample_occupation_given_nationality,
            HOUSEHOLD_GIVEN_AGE,
        )
        from population.personas import PersonaMeta
        from population.constraints import repair
        from population.life_path import generate_life_path
        from config.demographics import get_demographics

        agent_id = self._agent_id(index)
        pack = agent_rng_pack(agent_id, base_seed=self.seed)
        rng = pack.py_rng
        np_rng = pack.np_rng
        demo = get_demographics()

        age = _weighted_choice(demo.age, rng=rng, np_rng=np_rng)
        nationality = _weighted_choice(demo.nationality, rng=rng, np_rng=np_rng)
        income = _sample_income_given_nationality(nationality, rng, np_rng=np_rng)
        location = _sample_location_given_income(income, rng, np_rng=np_rng)
        household_size = _weighted_choice(
            HOUSEHOLD_GIVEN_AGE.get(age, demo.household_size), rng=rng, np_rng=np_rng,
        )
        occupation = _sample_occupation_given_nationality(nationality, rng, np_rng=np_rng)

        family = _family_from_household(household_size, age, nationality, rng)
        mobility = _mobility_from_location(location, rng)
        lifestyle = _lifestyle_from_demographics(income, location, nationality, rng)
        anchors = _personal_anchors_from_demographics(
            nationality, occupation, income, location, mobility, rng,
        )
        agent_hash = agent_seed_from_id(agent_id, base_seed=self.seed)
        personality = _generate_personality_vector(agent_hash, age, income, occupation)

        p = Persona(
            agent_id=agent_id,
            age=age,
            nationality=nationality,
            income=income,
            location=location,
            occupation=occupation,
            household_size=household_size,
            family=family,
            mobility=mobility,
            lifestyle=lifestyle,
            personal_anchors=anchors,
            personality=personality,
            meta=PersonaMeta(synthesis_method=self.method, generation_seed=agent_hash),
        )
        repair(p)
        p.life_path = generate_life_path(p, seed=self.seed)
        return p

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def __getitem__(self, key: Union[int, str]) -> Persona:
        if isinstance(key, str):
            idx = self._id_to_index.get(key)
            if idx is None:
                parts = key.rsplit("_", 1)
                idx = int(parts[-1]) if len(parts) == 2 and parts[-1].isdigit() else None
            if idx is None:
                raise KeyError(f"Unknown agent ID: {key}")
            return self[idx]

        if key < 0 or key >= self.total:
            raise IndexError(f"Agent index {key} out of range [0, {self.total})")

        agent_id = self._agent_id(key)
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id)
            return self._cache[agent_id]

        persona = self._generate_one(key)
        self._cache[agent_id] = persona
        self._id_to_index[agent_id] = key
        self._evict_if_needed()
        return persona

    def __len__(self) -> int:
        return self.total

    def __contains__(self, key: Union[int, str]) -> bool:
        try:
            self[key]
            return True
        except (KeyError, IndexError):
            return False

    def batch(self, start: int, count: int) -> List[Persona]:
        """Generate a contiguous batch of personas."""
        end = min(start + count, self.total)
        return [self[i] for i in range(start, end)]

    @property
    def cached_count(self) -> int:
        return len(self._cache)
