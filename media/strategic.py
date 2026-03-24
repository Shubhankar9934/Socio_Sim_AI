"""
Strategic Media Actors: intentional agents that inject targeted narrative
frames into the media ecosystem based on population state.

Each StrategicActor has a goal (e.g. "increase delivery adoption"), a
target demographic segment, a framing strategy, and a budget that limits
how many frames they can inject per day.

Usage in the simulation loop:
    from media.strategic import get_active_actors, inject_strategic_frames
    actors = get_active_actors()
    strategic_frames = inject_strategic_frames(actors, agents, day)
    frames.extend(strategic_frames)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agents.belief_network import BELIEF_DIMENSIONS, _N_BELIEFS
from core.rng import stable_seed_from_key
from media.framing import MediaFrame


@dataclass
class FramingPolicy:
    """How a strategic actor frames its messages."""

    belief_emphasis: Dict[str, float] = field(default_factory=dict)
    sentiment_target: float = 0.3
    emotional_intensity: float = 0.6
    headline_templates: List[str] = field(default_factory=lambda: [
        "New study shows {topic} benefits for {segment}",
        "Why {segment} residents are choosing {topic}",
        "Experts recommend {topic} for better {benefit}",
    ])


@dataclass
class StrategicActor:
    """An intentional agent that injects targeted media frames."""

    name: str
    goal: str
    target_segments: List[str] = field(default_factory=list)
    strategy: FramingPolicy = field(default_factory=FramingPolicy)
    budget_per_day: int = 2
    active: bool = True
    start_day: int = 1
    end_day: int = 999

    def is_active(self, day: int) -> bool:
        return self.active and self.start_day <= day <= self.end_day

    def matches_segment(self, persona_attrs: Dict[str, str]) -> bool:
        """Check if an agent belongs to any target segment."""
        if not self.target_segments:
            return True
        for seg in self.target_segments:
            key, _, val = seg.partition("=")
            if persona_attrs.get(key.strip()) == val.strip():
                return True
        return False

    def generate_frames(
        self,
        population_state: Dict[str, float],
        day: int,
    ) -> List[MediaFrame]:
        """Generate targeted frames based on population state."""
        frames = []
        rng = np.random.default_rng(stable_seed_from_key(f"strategic:{self.name}:{day}"))

        for _ in range(self.budget_per_day):
            bias_vec = np.full(_N_BELIEFS, 0.5)
            belief_impacts: Dict[str, float] = {}

            for dim, emphasis in self.strategy.belief_emphasis.items():
                if dim in BELIEF_DIMENSIONS:
                    idx = BELIEF_DIMENSIONS.index(dim)
                    bias_vec[idx] = 0.5 + 0.3 * emphasis
                    belief_impacts[dim] = 0.03 * emphasis

            template = rng.choice(self.strategy.headline_templates)
            segment_str = self.target_segments[0] if self.target_segments else "residents"
            headline = template.format(
                topic=self.goal,
                segment=segment_str,
                benefit=self.goal.replace("_", " "),
            )

            frames.append(MediaFrame(
                source_type=f"strategic_{self.name}",
                headline=headline,
                framing_bias=bias_vec,
                belief_impacts=belief_impacts,
                sentiment=self.strategy.sentiment_target,
                emotional_intensity=self.strategy.emotional_intensity,
                dimension_impacts={},
                topic=f"strategic_{self.goal}",
            ))

        return frames


def _load_default_actors() -> List[StrategicActor]:
    """Load default strategic actors from domain config."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        actors = []
        for a_cfg in cfg.strategic_actors:
            strat_data = a_cfg.get("strategy", {})
            strategy = FramingPolicy(
                belief_emphasis=strat_data.get("belief_emphasis", {}),
                sentiment_target=strat_data.get("sentiment_target", 0.3),
                emotional_intensity=strat_data.get("emotional_intensity", 0.6),
                headline_templates=strat_data.get("headline_templates", []),
            )
            actors.append(StrategicActor(
                name=a_cfg["name"],
                goal=a_cfg.get("goal", ""),
                target_segments=a_cfg.get("target_segments", []),
                strategy=strategy,
                budget_per_day=a_cfg.get("budget_per_day", 1),
                start_day=a_cfg.get("start_day", 1),
                end_day=a_cfg.get("end_day", 999),
            ))
        return actors
    except Exception:
        return []


_DEFAULT_ACTORS: List[StrategicActor] = _load_default_actors()

_custom_actors: List[StrategicActor] = []


def register_actor(actor: StrategicActor) -> None:
    """Add a custom strategic actor at runtime."""
    _custom_actors.append(actor)


def get_active_actors(day: int = 1) -> List[StrategicActor]:
    """Return all strategic actors active on the given day."""
    return [a for a in _DEFAULT_ACTORS + _custom_actors if a.is_active(day)]


def inject_strategic_frames(
    actors: List[StrategicActor],
    agents: List[Dict[str, Any]],
    day: int,
) -> List[MediaFrame]:
    """Generate targeted frames from all active strategic actors.

    Population state is aggregated to let actors adapt their messaging.
    """
    pop_state: Dict[str, float] = {}
    if agents:
        n = len(agents)
        for dim in BELIEF_DIMENSIONS:
            total = sum(
                getattr(a.get("state", {}).beliefs if hasattr(a.get("state"), "beliefs") else object(), dim, 0.5)
                for a in agents
                if a.get("state") is not None
            )
            pop_state[dim] = total / max(n, 1)

    all_frames: List[MediaFrame] = []
    for actor in actors:
        all_frames.extend(actor.generate_frames(pop_state, day))
    return all_frames
