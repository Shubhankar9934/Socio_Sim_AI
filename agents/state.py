"""
Agent state: the core mutable representation of one agent.

The primary state is ``latent_state`` (BehavioralLatentState) -- 12 universal
dimensions that evolve via EMA updates, social influence, and macro feedback.
PersonalityTraits are only used **during initialization** and not stored.

All domain-specific fields have been removed. Use ``latent_state`` exclusively.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.behavior import BehavioralLatentState, init_from_persona as _init_latent
from agents.belief_network import BeliefNetwork, init_beliefs_from_persona as _init_beliefs
from agents.identity import IdentityState, init_identity_from_beliefs
from agents.realism import HabitProfile, derive_habit_profile
from agents.utility import GoalProfile, assign_initial_goals
from population.personas import Persona


@dataclass
class AgentState:
    """Mutable state for one agent.

    Core fields:
        latent_state            – 12-dimension behavioral vector (primary state)
        last_answers            – answer history for consistency checks
        current_day             – simulation time
        social_trait_fraction   – fraction of friends sharing primary service trait
    """

    agent_id: str
    latent_state: BehavioralLatentState = field(default_factory=BehavioralLatentState)
    beliefs: BeliefNetwork = field(default_factory=BeliefNetwork)
    identity: IdentityState = field(default_factory=IdentityState)
    goal_profile: GoalProfile = field(default_factory=GoalProfile)
    habit_profile: Optional[HabitProfile] = None
    last_answers: Dict[str, Any] = field(default_factory=dict)
    structured_memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    life_event_history: List[Dict[str, Any]] = field(default_factory=list)
    dialogue_summary: str = ""
    social_trait_fraction: float = 0.0
    current_day: int = 0

    # Bias engine fields
    base_malleability: float = 0.5
    calcification: float = 0.0
    knowledge_levels: Dict[str, float] = field(default_factory=dict)
    topic_importances: Dict[str, float] = field(default_factory=dict)

    # Activation / cascade fields
    current_activation: float = 0.0
    activation: Dict[str, float] = field(default_factory=dict)
    activation_prev: Dict[str, float] = field(default_factory=dict)
    cooldown_topics: Dict[str, int] = field(default_factory=dict)
    fatigue_history: List[Dict[str, Any]] = field(default_factory=list)

    # Media exposure
    media_exposure_history: List[Any] = field(default_factory=list)

    # Backward-compat aliases
    @property
    def friends_using_delivery(self) -> float:
        return self.social_trait_fraction

    @friends_using_delivery.setter
    def friends_using_delivery(self, value: float) -> None:
        self.social_trait_fraction = value

    @property
    def food_delivery_per_week(self) -> float:
        return self.latent_state.get("convenience_seeking", 0.5) * 5.0

    @property
    def baseline_delivery_frequency(self) -> float:
        return self.latent_state.get("convenience_seeking", 0.5) * 5.0

    @property
    def behavior_scores(self) -> Dict[str, float]:
        """Legacy accessor -- reads from latent_state."""
        return self.latent_state.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "latent_state": self.latent_state.to_dict(),
            "beliefs": self.beliefs.to_dict(),
            "identity": self.identity.to_dict(),
            "last_answers": self.last_answers,
            "structured_memory": dict(self.structured_memory),
            "dialogue_summary": self.dialogue_summary,
            "life_event_history": list(self.life_event_history),
            "social_trait_fraction": self.social_trait_fraction,
            "current_day": self.current_day,
            "behavior_scores": self.behavior_scores,
            "base_malleability": self.base_malleability,
            "calcification": self.calcification,
            "activation": dict(self.activation),
        }

    @classmethod
    def from_persona(cls, persona: Persona) -> "AgentState":
        """Initialize state from persona.

        PersonalityTraits are derived here and used only to seed the
        BehavioralLatentState; they are not stored on the state object.
        """
        from agents.personality import personality_from_persona

        traits = personality_from_persona(persona)
        latent = _init_latent(persona, traits)
        beliefs = _init_beliefs(persona, traits)
        identity = init_identity_from_beliefs(beliefs)
        goals = assign_initial_goals(persona)
        habits = derive_habit_profile(persona)

        openness = persona.lifestyle.tech_adoption
        malleability = max(0.1, min(0.9, 0.3 + 0.4 * openness))

        return cls(
            agent_id=persona.agent_id,
            latent_state=latent,
            beliefs=beliefs,
            identity=identity,
            goal_profile=goals,
            habit_profile=habits,
            base_malleability=malleability,
            social_trait_fraction=0.0,
        )

    def update_after_answer(
        self, question_id: str, answer: Any, semantic_key: str = "",
        max_last_answers: int = 10,
        max_structured_memory_keys: int = 20,
    ) -> None:
        """Record answer for consistency checks and structured memory.

        If *semantic_key* is provided (e.g. ``"delivery_frequency"``), the
        answer is also stored in ``structured_memory`` so that cross-question
        influence rules can bias future related questions.

        Older entries are evicted once the respective container exceeds its
        cap so that memory footprint stays bounded across long sessions.
        """
        self.last_answers[question_id] = answer
        if len(self.last_answers) > max_last_answers:
            oldest = list(self.last_answers.keys())[:-max_last_answers]
            for k in oldest:
                del self.last_answers[k]

        if semantic_key:
            self.structured_memory[semantic_key] = {
                "answer": answer,
                "question_id": question_id,
            }
            if len(self.structured_memory) > max_structured_memory_keys:
                oldest = list(self.structured_memory.keys())[:-max_structured_memory_keys]
                for k in oldest:
                    del self.structured_memory[k]

    def set_behavior_score(self, domain: str, score: float) -> None:
        """Set an arbitrary dimension on the latent state."""
        clamped = max(0.0, min(1.0, score))
        if hasattr(self.latent_state, domain):
            setattr(self.latent_state, domain, clamped)

    def get_behavior_score(self, domain: str, default: float = 0.5) -> float:
        return self.latent_state.get(domain, default)

    def set_social_trait_fraction(self, fraction: float) -> None:
        self.social_trait_fraction = max(0.0, min(1.0, fraction))

    def set_friends_using_delivery(self, fraction: float) -> None:
        """Backward-compat alias."""
        self.set_social_trait_fraction(fraction)

    def summarize_memory(
        self, max_entries: int = 10, max_summary_length: int = 200,
    ) -> None:
        """Compress last_answers into a short text summary.

        Called periodically between survey rounds to keep the dialogue context
        compact instead of forwarding the full conversation history to the LLM.
        The result is hard-capped at *max_summary_length* characters so that
        summaries cannot grow unboundedly across many rounds.
        """
        if len(self.last_answers) < 3:
            return
        items = list(self.last_answers.items())[-max_entries:]
        summary = "; ".join(f"{k}: {v}" for k, v in items)
        self.dialogue_summary = summary[:max_summary_length]

    def build_structured_context(self) -> Dict[str, Any]:
        """Return a compact dict of agent state for LLM prompt injection.

        Avoids context explosion by surfacing only key behavioural signals
        and the compressed dialogue summary rather than raw conversation
        history.
        """
        ctx: Dict[str, Any] = {}
        for sem_key, entry in self.structured_memory.items():
            ctx[sem_key] = entry.get("answer")
        ctx["price_sensitivity"] = self.latent_state.get("price_sensitivity")
        ctx["health_orientation"] = self.latent_state.get("health_orientation")
        ctx["convenience_seeking"] = self.latent_state.get("convenience_seeking")
        ctx["technology_openness"] = self.latent_state.get("technology_openness")
        belief_dict = self.beliefs.to_dict()
        ctx["belief_technology_optimism"] = belief_dict.get("technology_optimism")
        ctx["belief_price_consciousness"] = belief_dict.get("price_consciousness")
        if self.dialogue_summary:
            ctx["recent_answers_summary"] = self.dialogue_summary
        return ctx

    def delivery_drift_magnitude(self) -> float:
        """Backward-compat: returns 0 since domain fields are removed."""
        return 0.0

    def reset_behavior(self, blend: float = 0.7) -> None:
        """No-op: domain-specific fields removed."""
        pass
