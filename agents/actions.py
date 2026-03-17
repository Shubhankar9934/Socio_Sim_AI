"""
Universal Action Model: domain-agnostic representation of agent actions.

Actions wrap survey answers, proactive behaviors, and intent-driven decisions
into a single structured type that the simulation engine can process uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ACTION_TYPES = [
    "frequency",       # how often (usage, consumption)
    "adopt",           # start using / switch to
    "reject",          # stop using / decline
    "support",         # endorse / favor (policy, candidate)
    "oppose",          # disagree / reject (policy, candidate)
    "rate",            # satisfaction / NPS score
    "choose",          # pick one option (voting, preference)
    "increase",        # do more of something
    "decrease",        # do less of something
    "invest",          # allocate resources
    "migrate",         # move / relocate
    "comply",          # follow rule / norm
    "protest",         # resist / demonstrate
]

TARGET_CATEGORIES = [
    "service",
    "product",
    "policy",
    "candidate",
    "belief",
    "behavior",
    "location",
    "investment",
    "norm",
]


@dataclass
class Action:
    """Universal representation of an agent action."""

    type: str
    target: str
    intensity: float = 0.5
    option: str = ""
    agent_id: str = ""
    day: int = 0
    source: str = "survey"  # "survey", "intent", "event", "social"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "target": self.target,
            "intensity": self.intensity,
            "option": self.option,
            "agent_id": self.agent_id,
            "day": self.day,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_survey_answer(
        cls,
        agent_id: str,
        question: str,
        answer: str,
        answer_score: float,
        action_type: str = "choose",
        target: str = "behavior",
    ) -> Action:
        return cls(
            type=action_type,
            target=target,
            intensity=answer_score,
            option=answer,
            agent_id=agent_id,
            source="survey",
            metadata={"question": question},
        )


@dataclass
class ActionTemplate:
    """Describes how a question maps to actions."""

    action_type: str = "choose"
    target: str = "behavior"
    intensity_scale: str = "ordinal"  # "ordinal", "binary", "continuous"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "intensity_scale": self.intensity_scale,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ActionTemplate:
        return cls(
            action_type=d.get("action_type", "choose"),
            target=d.get("target", "behavior"),
            intensity_scale=d.get("intensity_scale", "ordinal"),
            description=d.get("description", ""),
        )


# Dimension impact rules: how action types feed back into agent state
ACTION_STATE_IMPACTS: Dict[str, Dict[str, float]] = {
    "adopt": {"novelty_seeking": 0.02, "technology_openness": 0.01},
    "reject": {"routine_stability": 0.01, "risk_aversion": 0.01},
    "support": {"institutional_trust": 0.01},
    "oppose": {"institutional_trust": -0.01},
    "invest": {"financial_confidence": 0.01, "risk_aversion": -0.01},
    "protest": {"institutional_trust": -0.02, "social_influence_susceptibility": 0.01},
    "comply": {"institutional_trust": 0.01, "routine_stability": 0.01},
}


def apply_action_to_state(action: Action, agent_state: Any) -> None:
    """Apply lightweight state feedback from an action."""
    impacts = ACTION_STATE_IMPACTS.get(action.type)
    if not impacts or not hasattr(agent_state, "latent_state"):
        return
    for dim, delta in impacts.items():
        scaled = delta * action.intensity
        current = agent_state.latent_state.get(dim, 0.5)
        agent_state.latent_state.set(dim, current + scaled)
