"""
Goal-directed utility layer: adds intentional, utility-maximising behavior
to the otherwise reactive factor-graph decision pipeline.

Each agent can hold a small set of active goals (e.g. "save money",
"eat healthier") that bias decisions by blending a utility score into
the factor-graph probability distribution.

    final_score = alpha * factor_graph_score + (1 - alpha) * utility_score

Where alpha is configurable per-agent and typically 0.7-0.8 (factor graph
dominant, utility as a gentle nudge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.state import AgentState
    from population.personas import Persona


@dataclass
class AgentGoal:
    """A single active goal with dimension weights and priority."""

    name: str
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    priority: float = 1.0
    ttl_days: int = 90


@dataclass
class GoalProfile:
    """Collection of active goals for one agent."""

    goals: List[AgentGoal] = field(default_factory=list)

    def add_goal(self, goal: AgentGoal) -> None:
        existing = {g.name for g in self.goals}
        if goal.name not in existing:
            self.goals.append(goal)

    def tick(self) -> None:
        """Decrement TTL and remove expired goals."""
        surviving = []
        for g in self.goals:
            g.ttl_days -= 1
            if g.ttl_days > 0:
                surviving.append(g)
        self.goals = surviving

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {"name": g.name, "priority": g.priority, "ttl_days": g.ttl_days}
            for g in self.goals
        ]


# Pre-defined goal templates that can be assigned based on persona/life events.
GOAL_TEMPLATES: Dict[str, AgentGoal] = {
    "save_money": AgentGoal(
        name="save_money",
        dimension_weights={"price_sensitivity": 0.4, "convenience_seeking": -0.2, "financial_confidence": 0.1},
        priority=1.2,
    ),
    "eat_healthier": AgentGoal(
        name="eat_healthier",
        dimension_weights={"health_orientation": 0.5, "novelty_seeking": 0.1},
        priority=1.0,
    ),
    "try_new_things": AgentGoal(
        name="try_new_things",
        dimension_weights={"novelty_seeking": 0.4, "technology_openness": 0.2, "routine_stability": -0.2},
        priority=0.8,
    ),
    "be_more_social": AgentGoal(
        name="be_more_social",
        dimension_weights={"social_influence_susceptibility": 0.3, "convenience_seeking": 0.1},
        priority=0.9,
    ),
}


def compute_utility_scores(
    options: List[str],
    goal_profile: GoalProfile,
    agent_state: "AgentState",
    n_options: int = 0,
) -> np.ndarray:
    """Compute a utility score per option based on active goals.

    Maps ordinal options to positions on [0, 1] and scores each position
    by alignment with goal-weighted behavioral dimensions.

    Returns an array of shape (n_options,) with values in [0, 1].
    """
    n = n_options or len(options)
    if n == 0 or not goal_profile.goals:
        return np.full(n, 0.5)

    position = np.linspace(0.0, 1.0, n)
    utility = np.zeros(n)
    total_priority = sum(g.priority for g in goal_profile.goals)

    for goal in goal_profile.goals:
        weight = goal.priority / max(total_priority, 1e-9)
        goal_score = 0.0
        for dim, w in goal.dimension_weights.items():
            val = getattr(agent_state.latent_state, dim, 0.5) if hasattr(agent_state, "latent_state") else 0.5
            goal_score += w * val
        utility += weight * goal_score * position

    u_min, u_max = utility.min(), utility.max()
    if u_max - u_min > 1e-9:
        utility = (utility - u_min) / (u_max - u_min)
    else:
        utility = np.full(n, 0.5)

    return utility


def blend_utility_into_distribution(
    dist: Dict[str, float],
    goal_profile: Optional[GoalProfile],
    agent_state: "AgentState",
    alpha: float = 0.8,
) -> Dict[str, float]:
    """Blend utility scores into an existing probability distribution.

    alpha controls the mix: 1.0 = pure factor-graph, 0.0 = pure utility.
    Default 0.8 keeps the factor graph dominant with a gentle goal nudge.
    """
    if goal_profile is None or not goal_profile.goals:
        return dist

    options = list(dist.keys())
    probs = np.array([dist[o] for o in options])
    utility = compute_utility_scores(options, goal_profile, agent_state, len(options))

    u_sum = utility.sum()
    if u_sum > 0:
        utility_norm = utility / u_sum
    else:
        utility_norm = np.full(len(options), 1.0 / len(options))

    blended = alpha * probs + (1 - alpha) * utility_norm
    total = blended.sum()
    if total > 0:
        blended /= total

    return dict(zip(options, blended.tolist()))


def assign_initial_goals(persona: "Persona") -> GoalProfile:
    """Heuristically assign starting goals based on persona attributes."""
    profile = GoalProfile()

    if hasattr(persona, "lifestyle"):
        if persona.lifestyle.price_sensitivity > 0.7:
            profile.add_goal(AgentGoal(**{**GOAL_TEMPLATES["save_money"].__dict__}))
        if getattr(persona, "personal_anchors", None) and "health" in getattr(persona.personal_anchors, "health_focus", "").lower():
            profile.add_goal(AgentGoal(**{**GOAL_TEMPLATES["eat_healthier"].__dict__}))

    return profile
