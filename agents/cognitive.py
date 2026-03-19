"""
Agent cognitive pipeline with three modes:

  **Survey mode** (``think``): Perception -> Memory -> Decision -> LLM Reasoning -> Response
  **Simulation mode** (``decide_only``): Perception -> Decision (no LLM, no narrative)
  **Event mode** (``handle_event``): Dispatch incoming SimEvents to the appropriate handler

After each decision the agent's BehavioralLatentState and BeliefNetwork are
updated via EMA.  PersonalityTraits are derived once at construction and cached.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation.event_queue import SimEvent

from agents.decision import decide
from agents.perception import perceive, perceive_with_llm, Perception
from agents.personality import personality_from_persona, PersonalityTraits
from agents.state import AgentState
from config.belief_mappings import get_belief_dimensions
from config.question_models import get_behavioral_dimensions
from population.personas import Persona


Reasoner = Callable[
    [Persona, str, str, Dict[str, float], List[str]],
    Any,
]


class AgentCognitiveEngine:
    """One agent's brain: persona + state + cognitive pipeline."""

    def __init__(
        self,
        persona: Persona,
        state: Optional[AgentState] = None,
        memory_recall: Optional[Callable[[str, Perception], Any]] = None,
        reasoner: Optional[Reasoner] = None,
    ):
        self.persona = persona
        self.state = state or AgentState.from_persona(persona)
        self._memory_recall = memory_recall
        self._reasoner = reasoner
        self._traits: PersonalityTraits = personality_from_persona(persona)
        self._world_environment: Dict[str, Any] = {}

    @property
    def traits(self) -> PersonalityTraits:
        return self._traits

    def perceive(self, question: str) -> Perception:
        return perceive(question)

    async def recall(self, perception: Perception) -> List[str]:
        """Retrieve relevant memories (async if memory store is async)."""
        if self._memory_recall is None:
            return []
        out = self._memory_recall(self.persona.agent_id, perception)
        if hasattr(out, "__await__"):
            return await out
        return out if isinstance(out, list) else []

    def decide(
        self,
        perception: Perception,
        memories: List[str],
        friends_using: Optional[float] = None,
        location_quality: float = 0.5,
        environment: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Dict[str, float], str]:
        """Probabilistic decision; returns (distribution, sampled_answer).

        Merges any world-environment data stored on the agent (from the
        simulation kernel's event system) into the DecisionContext so
        factors can read price_multipliers, active_policies, beliefs, etc.
        Passes agent_state for cognitive dissonance adjustment.
        """
        friends_using = friends_using if friends_using is not None else self.state.social_trait_fraction
        env = dict(environment or {})
        env["latent_state"] = self.state.latent_state
        env.setdefault("beliefs", self.state.beliefs)
        if self.state.structured_memory:
            env.setdefault("structured_memory", self.state.structured_memory)
        if hasattr(self, "_world_environment") and self._world_environment:
            for k, v in self._world_environment.items():
                env[k] = v  # allow override e.g. temp_beliefs for survey-time media
        # Inject activation level (set by simulation kernel) for
        # temperature / bias modulation in the decision pipeline.
        activation = getattr(self.state, "current_activation", 0.0)
        env.setdefault("activation", activation)
        return decide(
            perception,
            self.persona,
            self.traits,
            friends_using=friends_using,
            location_quality=location_quality,
            memories=memories,
            environment=env,
            agent_state=self.state,
            rng=rng,
        )

    def set_world_environment(self, env: Dict[str, Any]) -> None:
        """Store event-driven world parameters for factor graph access."""
        self._world_environment = env

    def build_structured_context(
        self, research_ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compact agent context dict for LLM prompt injection.

        Surfaces key behavioural signals and the compressed dialogue summary
        instead of raw conversation history, avoiding token explosion in
        multi-question surveys.  Optionally merges shared research context.
        """
        ctx = self.state.build_structured_context()
        if research_ctx is not None:
            from research.context import enrich_archetype_context
            ctx = enrich_archetype_context(ctx, research_ctx)
        return ctx

    async def handle_event(self, event: SimEvent) -> Optional[Dict[str, Any]]:
        """Dispatch an event-queue ``SimEvent`` via the stateless EventDispatcher.

        Returns the response dict for survey_question events, None otherwise.
        """
        from simulation.dispatch import EventDispatcher
        return await EventDispatcher.dispatch(event, self)

    async def reason(
        self,
        question: str,
        sampled_answer: str,
        distribution: Dict[str, float],
        memories: List[str],
    ) -> str:
        """Generate narrative answer via LLM or deterministic fallback."""
        if self._reasoner is not None:
            result = self._reasoner(
                self.persona,
                question,
                sampled_answer,
                distribution,
                memories,
            )
            if hasattr(result, "__await__"):
                return await result
            return str(result)
        return sampled_answer

    def update_state(self, question_id: str, answer: Any) -> None:
        """Update agent state after answering (for consistency)."""
        self.state.update_after_answer(question_id, answer)

    def _update_state_after_answer(
        self, perception: Perception, distribution: Dict[str, float], sampled_answer: str
    ) -> None:
        """EMA-update latent behavioral dimensions, beliefs, habits, and structured memory."""
        scale = list(distribution.keys())
        if not scale:
            return
        idx = scale.index(sampled_answer) if sampled_answer in scale else len(scale) // 2
        answer_score = idx / max(1, len(scale) - 1)
        qm_key = perception.question_model_key if hasattr(perception, "question_model_key") else ""

        dim_weights = get_behavioral_dimensions(qm_key)
        self.state.latent_state.update_dimensions(dim_weights, answer_score)

        belief_weights = get_belief_dimensions(qm_key)
        self.state.beliefs.update_from_answer(belief_weights, answer_score)

        # Behavior inertia: EMA-update habit profile for longitudinal consistency
        if self.state.habit_profile is not None:
            from agents.realism import update_habits_after_answer
            update_habits_after_answer(self.state.habit_profile, sampled_answer)

        # Populate structured memory for cross-question consistency
        if qm_key:
            from agents.memory_rules import QUESTION_TO_SEMANTIC_KEY
            sem_key = QUESTION_TO_SEMANTIC_KEY.get(qm_key, "")
            if sem_key:
                self.state.structured_memory[sem_key] = {
                    "answer": sampled_answer,
                    "question_model_key": qm_key,
                    "answer_score": answer_score,
                }

    def decide_only(
        self,
        question: str,
        question_id: str = "",
        friends_using: Optional[float] = None,
        location_quality: float = 0.5,
    ) -> Dict[str, Any]:
        """Simulation mode: fast decision with no LLM call.

        Returns dict with sampled_option, distribution, and agent_id.
        Updates latent state via EMA but does not generate a narrative.
        """
        perception = self.perceive(question)
        distribution, sampled_answer = self.decide(
            perception, [], friends_using=friends_using, location_quality=location_quality
        )
        if question_id:
            self.update_state(question_id, sampled_answer)
        self._update_state_after_answer(perception, distribution, sampled_answer)
        return {
            "sampled_option": sampled_answer,
            "distribution": distribution,
            "agent_id": self.persona.agent_id,
        }

    async def think(
        self,
        question: str,
        question_id: str = "",
        friends_using: Optional[float] = None,
        location_quality: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Full cognitive pipeline (survey mode): perceive -> recall -> decide -> reason.
        Uses LLM fallback for question understanding on unknown topics.
        Returns dict with answer, probability, distribution, etc.
        """
        if self._reasoner is not None:
            perception = await perceive_with_llm(question)
        else:
            perception = self.perceive(question)
        memories = await self.recall(perception)
        distribution, sampled_answer = self.decide(
            perception, memories, friends_using=friends_using, location_quality=location_quality
        )
        narrative = await self.reason(question, sampled_answer, distribution, memories)
        if question_id:
            self.update_state(question_id, narrative)

        self._update_state_after_answer(perception, distribution, sampled_answer)

        pa = self.persona.personal_anchors
        meta = self.persona.meta
        return {
            "answer": narrative,
            "sampled_option": sampled_answer,
            "distribution": distribution,
            "agent_id": self.persona.agent_id,
            "perception_topic": perception.topic,
            "perception_domain": perception.domain,
            "demographics": {
                "age_group": self.persona.age,
                "nationality": self.persona.nationality,
                "income_band": self.persona.income,
                "location": self.persona.location,
                "occupation": self.persona.occupation,
                "household_size": self.persona.household_size,
                "family_children": self.persona.family.children,
                "has_spouse": self.persona.family.spouse,
            },
            "lifestyle": {
                "cuisine_preference": pa.cuisine_preference,
                "diet": pa.diet,
                "hobby": pa.hobby,
                "work_schedule": pa.work_schedule,
                "health_focus": pa.health_focus,
                "commute_method": pa.commute_method,
            },
            "persona_meta": {
                "persona_version": meta.persona_version,
                "synthesis_method": meta.synthesis_method,
                "generation_seed": meta.generation_seed,
                "archetype_id": meta.archetype_id,
                "persona_cluster": meta.persona_cluster,
            },
        }
