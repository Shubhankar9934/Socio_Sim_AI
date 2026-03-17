"""
Intent Engine: generate proactive intentions from agent state, goals,
social pressure, and environmental signals.

Agents form intents autonomously during simulation (Step 6.5),
converting high-urgency intents into Actions via the factor graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """A proactive intention formed by an agent."""

    action_type: str       # "adopt", "switch", "protest", "seek_info", "comply"
    target: str            # what the intent is about
    urgency: float = 0.0   # 0-1, derived from activation + goal alignment
    source: str = ""       # "goal", "social", "event", "media", "life_event"
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentEngine:
    """Generate and resolve proactive intents for agents."""

    def __init__(
        self,
        urgency_threshold: float = 0.6,
        max_intents_per_agent: int = 3,
    ):
        self.urgency_threshold = urgency_threshold
        self.max_intents = max_intents_per_agent

    def generate_intents(
        self,
        agent_state: Any,
        environment: Dict[str, Any],
        day: int,
        social_trait_fraction: float = 0.0,
    ) -> List[Intent]:
        """Generate proactive intents from agent state and environment.

        Sources of intent:
          1. Goal profile: unmet goals create pursuit intents
          2. Activation level: high activation + media -> intent to act
          3. Social pressure: neighbors' behavior creates conformity intents
          4. Life events: recent events create adaptation intents
        """
        intents: List[Intent] = []

        intents.extend(self._goal_intents(agent_state))
        intents.extend(self._activation_intents(agent_state, environment))
        intents.extend(self._social_intents(agent_state, social_trait_fraction))
        intents.extend(self._life_event_intents(agent_state, day))

        intents.sort(key=lambda i: i.urgency, reverse=True)
        return intents[:self.max_intents]

    def _goal_intents(self, state: Any) -> List[Intent]:
        """Check goal_profile for unmet goals."""
        intents = []
        gp = getattr(state, "goal_profile", None)
        if gp is None:
            return intents

        goals = getattr(gp, "goals", None) or getattr(gp, "active_goals", None)
        if not goals:
            return intents

        if isinstance(goals, dict):
            for goal_name, goal_data in goals.items():
                progress = goal_data if isinstance(goal_data, (int, float)) else 0.0
                if progress < 0.5:
                    urgency = 0.3 + 0.4 * (1.0 - progress)
                    intents.append(Intent(
                        action_type="increase",
                        target=goal_name,
                        urgency=min(1.0, urgency),
                        source="goal",
                    ))
        return intents

    def _activation_intents(
        self, state: Any, environment: Dict[str, Any]
    ) -> List[Intent]:
        """High activation + media exposure -> intent to act publicly."""
        intents = []
        activation = getattr(state, "current_activation", 0.0)
        if activation < 0.5:
            return intents

        policies = environment.get("active_policies", [])
        if policies and activation > 0.6:
            trust = 0.5
            if hasattr(state, "beliefs"):
                trust = state.beliefs.get("government_trust", 0.5)

            if trust < 0.4:
                intents.append(Intent(
                    action_type="oppose",
                    target="policy",
                    urgency=min(1.0, activation * 0.8 + (0.5 - trust)),
                    source="media",
                    metadata={"trigger": "high_activation_low_trust"},
                ))
            else:
                intents.append(Intent(
                    action_type="comply",
                    target="policy",
                    urgency=activation * 0.5,
                    source="media",
                ))

        campaigns = environment.get("active_media_campaigns", [])
        if campaigns and activation > 0.5:
            intents.append(Intent(
                action_type="adopt",
                target="campaign_subject",
                urgency=activation * 0.6,
                source="media",
                metadata={"campaigns": len(campaigns)},
            ))

        return intents

    def _social_intents(
        self, state: Any, social_trait_fraction: float
    ) -> List[Intent]:
        """High neighbor adoption creates conformity pressure."""
        intents = []
        if social_trait_fraction < 0.4:
            return intents

        sus = 0.5
        if hasattr(state, "latent_state"):
            sus = state.latent_state.get("social_influence_susceptibility", 0.5)

        urgency = social_trait_fraction * sus
        if urgency > 0.3:
            intents.append(Intent(
                action_type="adopt",
                target="behavior",
                urgency=min(1.0, urgency),
                source="social",
                metadata={"peer_fraction": social_trait_fraction},
            ))
        return intents

    def _life_event_intents(self, state: Any, day: int) -> List[Intent]:
        """Recent life events trigger adaptation intents."""
        intents = []
        history = getattr(state, "life_event_history", None)
        if not history:
            return intents

        recent = [e for e in history if isinstance(e, dict) and day - e.get("day", 0) <= 7]
        for event in recent[-2:]:
            etype = event.get("type", "")
            if etype in ("promotion", "new_job"):
                intents.append(Intent(
                    action_type="increase",
                    target="lifestyle",
                    urgency=0.5,
                    source="life_event",
                    metadata={"event": etype},
                ))
            elif etype in ("job_loss", "divorce"):
                intents.append(Intent(
                    action_type="decrease",
                    target="spending",
                    urgency=0.7,
                    source="life_event",
                    metadata={"event": etype},
                ))
            elif etype == "relocation":
                intents.append(Intent(
                    action_type="adopt",
                    target="local_services",
                    urgency=0.4,
                    source="life_event",
                    metadata={"event": etype},
                ))
        return intents

    def resolve_intent(
        self, intent: Intent, agent_state: Any
    ) -> Optional[Dict[str, float]]:
        """Convert a high-urgency intent into state updates.

        Returns dimension deltas to apply, or None if urgency is below threshold.
        """
        if intent.urgency < self.urgency_threshold:
            return None

        from agents.actions import ACTION_STATE_IMPACTS

        impacts = ACTION_STATE_IMPACTS.get(intent.action_type, {})
        if not impacts:
            return None

        scaled = {dim: delta * intent.urgency for dim, delta in impacts.items()}
        return scaled


def process_intents_for_agents(
    agents: List[Dict[str, Any]],
    environment: Dict[str, Any],
    day: int,
) -> int:
    """Step 6.5: generate and resolve intents for all agents.

    Returns the number of intents that were resolved into state changes.
    """
    engine = IntentEngine()
    resolved_count = 0

    for a in agents:
        state = a.get("state")
        if not state:
            continue

        social_frac = a.get("social_trait_fraction", 0.0)
        intents = engine.generate_intents(state, environment, day, social_frac)

        for intent in intents:
            deltas = engine.resolve_intent(intent, state)
            if deltas and hasattr(state, "latent_state"):
                for dim, delta in deltas.items():
                    current = state.latent_state.get(dim, 0.5)
                    state.latent_state.set(dim, current + delta)
                resolved_count += 1

    return resolved_count
