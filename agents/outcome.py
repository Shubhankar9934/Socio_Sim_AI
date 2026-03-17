"""
Outcome Engine: compute action outcomes and apply reinforcement signals.

After agents take actions (from survey, intents, or events), the outcome
engine computes rewards/costs and feeds them back into beliefs and
latent state, creating a closed learning loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionOutcome:
    """The result of an agent's action."""

    action_type: str
    target: str
    reward: float = 0.0            # -1 to 1: satisfaction signal
    cost: float = 0.0              # 0 to 1: normalized resource cost
    social_approval: float = 0.0   # -1 to 1: peer reaction
    day: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Temporal discount factor for outcome influence
_DEFAULT_DECAY_RATE = 0.05


class OutcomeEngine:
    """Compute outcomes for actions and apply reinforcement to agent state."""

    def __init__(
        self,
        base_reward_noise: float = 0.1,
        decay_rate: float = _DEFAULT_DECAY_RATE,
    ):
        self.base_reward_noise = base_reward_noise
        self.decay_rate = decay_rate

    def compute_outcome(
        self,
        action_type: str,
        target: str,
        intensity: float,
        agent_state: Any,
        environment: Dict[str, Any],
        social_trait_fraction: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> ActionOutcome:
        """Compute the outcome of an action.

        Components:
          1. Economic cost: from budget via price sensitivity
          2. Social approval: alignment with peer behavior
          3. Personal reward: goal alignment + bounded rationality noise
        """
        if rng is None:
            rng = np.random.default_rng()

        cost = self._compute_cost(action_type, intensity, agent_state, environment)
        social_approval = self._compute_social_approval(
            action_type, intensity, social_trait_fraction, agent_state
        )
        reward = self._compute_reward(
            action_type, intensity, agent_state, cost, social_approval, rng
        )

        return ActionOutcome(
            action_type=action_type,
            target=target,
            reward=float(np.clip(reward, -1.0, 1.0)),
            cost=float(np.clip(cost, 0.0, 1.0)),
            social_approval=float(np.clip(social_approval, -1.0, 1.0)),
            metadata={"intensity": intensity},
        )

    def _compute_cost(
        self, action_type: str, intensity: float,
        agent_state: Any, environment: Dict[str, Any]
    ) -> float:
        price_sensitivity = 0.5
        if hasattr(agent_state, "latent_state"):
            price_sensitivity = agent_state.latent_state.get("price_sensitivity", 0.5)

        base_cost_map = {
            "adopt": 0.3, "invest": 0.5, "increase": 0.2,
            "migrate": 0.7, "frequency": 0.15,
        }
        base_cost = base_cost_map.get(action_type, 0.1) * intensity

        price_mult = environment.get("price_multipliers", {}).get(
            "default", 1.0
        )
        return base_cost * price_mult * (0.5 + 0.5 * price_sensitivity)

    def _compute_social_approval(
        self, action_type: str, intensity: float,
        social_trait_fraction: float, agent_state: Any
    ) -> float:
        conformity_actions = {"adopt", "comply", "support", "increase"}
        nonconformity_actions = {"reject", "oppose", "protest", "decrease"}

        if action_type in conformity_actions:
            return social_trait_fraction * 2 - 1
        elif action_type in nonconformity_actions:
            return (1 - social_trait_fraction) * 2 - 1

        return 0.0

    def _compute_reward(
        self, action_type: str, intensity: float,
        agent_state: Any, cost: float, social_approval: float,
        rng: np.random.Generator,
    ) -> float:
        goal_alignment = 0.0
        gp = getattr(agent_state, "goal_profile", None)
        if gp:
            goals = getattr(gp, "goals", None) or getattr(gp, "active_goals", None)
            if goals and isinstance(goals, dict):
                for _, val in goals.items():
                    if isinstance(val, (int, float)):
                        goal_alignment += (1.0 - val) * 0.1

        noise = rng.normal(0, self.base_reward_noise)

        reward = (
            0.4 * (intensity - cost)
            + 0.3 * social_approval
            + 0.2 * goal_alignment
            + 0.1 * noise
        )
        return reward

    def apply_outcome(
        self,
        outcome: ActionOutcome,
        agent_state: Any,
        day: int = 0,
    ) -> None:
        """Apply reinforcement from outcome to agent state.

        Updates:
          1. Beliefs: positive reward reinforces decision-driving beliefs
          2. Latent state: cost pressure -> increase price_sensitivity
          3. Habits: repeated positive outcomes -> habit formation
          4. Memory: stores action-outcome pair
        """
        if not hasattr(agent_state, "latent_state"):
            return

        ls = agent_state.latent_state
        r = outcome.reward

        if r > 0:
            ls.set("novelty_seeking",
                    ls.get("novelty_seeking", 0.5) + 0.01 * r)
            ls.set("convenience_seeking",
                    ls.get("convenience_seeking", 0.5) + 0.005 * r)
        else:
            ls.set("risk_aversion",
                    ls.get("risk_aversion", 0.5) + 0.01 * abs(r))

        if outcome.cost > 0.3:
            ls.set("price_sensitivity",
                    ls.get("price_sensitivity", 0.5) + 0.01 * outcome.cost)

        if outcome.social_approval > 0.3:
            ls.set("social_influence_susceptibility",
                    ls.get("social_influence_susceptibility", 0.5) + 0.005)
        elif outcome.social_approval < -0.3:
            ls.set("social_influence_susceptibility",
                    ls.get("social_influence_susceptibility", 0.5) - 0.005)

        if hasattr(agent_state, "beliefs"):
            beliefs = agent_state.beliefs
            if r > 0.2:
                beliefs.set("innovation_curiosity",
                            beliefs.get("innovation_curiosity", 0.5) + 0.005 * r)
            if outcome.cost > 0.4:
                beliefs.set("price_consciousness",
                            beliefs.get("price_consciousness", 0.5) + 0.005)

        memory_store = getattr(agent_state, "memory_store", None)
        if memory_store and hasattr(memory_store, "add"):
            memory_text = (
                f"Day {day}: {outcome.action_type} {outcome.target} "
                f"(reward={outcome.reward:.2f}, cost={outcome.cost:.2f})"
            )
            try:
                memory_store.add(memory_text, memory_type="episodic")
            except Exception:
                pass


def temporal_discount(base_influence: float, days_since: int, decay_rate: float = _DEFAULT_DECAY_RATE) -> float:
    """Exponential decay: recent outcomes weigh more than distant ones."""
    return base_influence * math.exp(-decay_rate * days_since)


def process_outcomes_for_agents(
    agents: List[Dict[str, Any]],
    environment: Dict[str, Any],
    day: int,
) -> int:
    """Process action outcomes for all agents that took actions this step.

    Returns the number of outcomes applied.
    """
    engine = OutcomeEngine()
    count = 0
    rng = np.random.default_rng(day)

    for a in agents:
        state = a.get("state")
        if not state:
            continue

        recent_actions = getattr(state, "recent_actions", [])
        if not recent_actions:
            continue

        social_frac = a.get("social_trait_fraction", 0.0)

        for action_info in recent_actions:
            if isinstance(action_info, dict):
                at = action_info.get("type", "choose")
                tgt = action_info.get("target", "behavior")
                intensity = action_info.get("intensity", 0.5)
            else:
                continue

            outcome = engine.compute_outcome(
                action_type=at,
                target=tgt,
                intensity=intensity,
                agent_state=state,
                environment=environment,
                social_trait_fraction=social_frac,
                rng=rng,
            )
            outcome.day = day
            engine.apply_outcome(outcome, state, day=day)
            count += 1

        state.recent_actions = []

    return count
