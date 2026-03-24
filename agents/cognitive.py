"""
Agent cognitive pipeline with three modes:

  **Survey mode** (``think``): Perception -> Memory -> Decision -> LLM Reasoning -> Response
  **Simulation mode** (``decide_only``): Perception -> Decision (no LLM, no narrative)
  **Event mode** (``handle_event``): Dispatch incoming SimEvents to the appropriate handler

After each decision the agent's BehavioralLatentState and BeliefNetwork are
updated via EMA.  PersonalityTraits are derived once at construction and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation.event_queue import SimEvent

from agents.decision import decide
from agents.intent_router import (
    IntentClass,
    classify_intent_class,
    resolve_reference,
    strip_survey_options_if_qualitative,
)
from agents.memory_manager import check_question_repetition, detect_contradiction, record_question
from agents.narrative import validate_narrative_consistency
from agents.perception import perceive, perceive_with_llm, Perception
from agents.personality import personality_from_persona, PersonalityTraits
from agents.state import AgentState
from config.belief_mappings import get_belief_dimensions
from config.option_space import canonicalize_option, get_option_space_key
from config.question_models import get_behavioral_dimensions
from agents.response_contract import build_response_contract, enforce_survey_response
from population.personas import Persona


@dataclass
class CognitiveTrace:
    """Structured trace of every decision point in the cognitive pipeline.

    Always returned in the result dict so every response is debuggable
    at scale without needing diagnostics_enabled.
    """
    agent_id: str = ""
    question: str = ""
    intent_class: str = ""
    parsed_topic: str = ""
    parsed_scale_type: str = ""
    retrieved_memories: List[str] = field(default_factory=list)
    contradiction_detected: Optional[Dict] = None
    repetition_detected: bool = False
    dominant_factor: Optional[str] = None
    conviction_profile: Optional[str] = None
    sampled_option: str = ""
    runner_up_option: Optional[str] = None
    confidence_band: str = ""
    tradeoff_guidance: Optional[str] = None
    belief_statements: List[str] = field(default_factory=list)
    personality_summary: str = ""
    tone_selected: str = ""
    emotional_state: str = "neutral"
    fatigue: float = 0.0
    turn_count: int = 0
    post_processing_applied: List[str] = field(default_factory=list)
    final_response: str = ""
    latent_state_snapshot: Dict[str, float] = field(default_factory=dict)
    belief_snapshot: Dict[str, float] = field(default_factory=dict)
    invariant_violations: List[str] = field(default_factory=list)
    decision_latency: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "question": self.question,
            "intent_class": self.intent_class,
            "parsed_topic": self.parsed_topic,
            "parsed_scale_type": self.parsed_scale_type,
            "retrieved_memories": self.retrieved_memories,
            "contradiction_detected": self.contradiction_detected,
            "repetition_detected": self.repetition_detected,
            "dominant_factor": self.dominant_factor,
            "conviction_profile": self.conviction_profile,
            "sampled_option": self.sampled_option,
            "runner_up_option": self.runner_up_option,
            "confidence_band": self.confidence_band,
            "tradeoff_guidance": self.tradeoff_guidance,
            "belief_statements": self.belief_statements,
            "personality_summary": self.personality_summary,
            "tone_selected": self.tone_selected,
            "emotional_state": self.emotional_state,
            "fatigue": self.fatigue,
            "turn_count": self.turn_count,
            "post_processing_applied": self.post_processing_applied,
            "final_response": self.final_response[:200],
            "latent_state_snapshot": self.latent_state_snapshot,
            "belief_snapshot": self.belief_snapshot,
            "invariant_violations": self.invariant_violations,
            "decision_latency": self.decision_latency,
        }


_GREETING_RESPONSES = [
    "Hey, what's up?", "Hi!", "Hello!", "Hey.", "Yo.",
    "Hey, how's it going?", "Hi there.", "Hey hey.",
]
_ACKNOWLEDGMENT_RESPONSES = [
    "Yeah.", "Mm hmm.", "Ok.", "Sure.", "Right.", "Got it.",
    "Yep.", "Alright.", "Cool.", "Haan.",
]

Reasoner = Callable[..., Any]


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
        self._last_decision_trace: Dict[str, Any] = {}
        self._last_response_contract: Dict[str, Any] = {}
        self._last_alignment_metadata: Dict[str, Any] = {}

    @property
    def traits(self) -> PersonalityTraits:
        return self._traits

    def perceive(self, question: str) -> Perception:
        return perceive(question, state=self.state)

    async def recall(self, perception: Perception) -> List[str]:
        """Retrieve relevant memories from external store + internal tiers."""
        memories: List[str] = []
        if self._memory_recall is not None:
            out = self._memory_recall(self.persona.agent_id, perception)
            if hasattr(out, "__await__"):
                out = await out
            if isinstance(out, list):
                memories.extend(out)

        from agents.memory_manager import get_relevant_memories
        topic = getattr(perception, "topic", "")
        tiered = get_relevant_memories(self.state, topic=topic, max_items=5)
        memories.extend(tiered)

        return memories

    def decide(
        self,
        perception: Perception,
        memories: List[str],
        friends_using: Optional[float] = None,
        location_quality: Optional[float] = None,
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
        from world.districts import resolve_location_quality

        lq = resolve_location_quality(self.persona.location, location_quality)
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
        dist, sampled = decide(
            perception,
            self.persona,
            self.traits,
            friends_using=friends_using,
            location_quality=lq,
            memories=memories,
            environment=env,
            agent_state=self.state,
            rng=rng,
        )
        self._last_decision_trace = env.get("__decision_trace", {})
        return dist, sampled

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
        response_contract: Optional[Dict[str, Any]] = None,
        turn_understanding: Optional[Dict[str, Any]] = None,
        diagnostics_enabled: bool = False,
        option_labels: Optional[List[str]] = None,
    ) -> str:
        """Generate narrative answer via LLM or deterministic fallback."""
        self._last_alignment_metadata = {}
        self._last_pp_log: List[str] = []
        if self._reasoner is not None:
            try:
                result = self._reasoner(
                    self.persona,
                    question,
                    sampled_answer,
                    distribution,
                    memories,
                    response_contract=response_contract,
                    turn_understanding=turn_understanding,
                    diagnostics_enabled=diagnostics_enabled,
                )
            except TypeError:
                result = self._reasoner(
                    self.persona,
                    question,
                    sampled_answer,
                    distribution,
                    memories,
                )
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict):
                self._last_alignment_metadata = dict(result.get("alignment") or {})
                self._last_pp_log = list(result.get("pp_log") or [])
                out = str(result.get("answer", ""))
            else:
                out = str(result)
        else:
            out = str(sampled_answer)
        tu = turn_understanding or {}
        st = str(tu.get("scale_type") or "categorical")
        if not distribution:
            st = "open_text"
        rc = response_contract or {}
        im = str(rc.get("interaction_mode") or "survey")
        return enforce_survey_response(
            out,
            scale_type=st,
            sampled_option=sampled_answer,
            option_labels=option_labels,
            interaction_mode=im,
        )

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
        self.state.latent_state.update_dimensions(
            dim_weights,
            answer_score,
            identity_anchor=self.state.identity_anchor,
            anchor_elasticity=self.state.anchor_elasticity,
        )

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

        # Temporal identity drift: measure how far latent state has moved from anchor
        _IDENTITY_SHIFT_THRESHOLD = 0.15
        if self.state.identity_anchor is not None:
            current_vec = self.state.latent_state.to_vector()
            anchor_vec = np.array(self.state.identity_anchor)
            if current_vec.shape == anchor_vec.shape:
                drift = float(np.linalg.norm(current_vec - anchor_vec))
                if drift > _IDENTITY_SHIFT_THRESHOLD:
                    self.state.identity_version += 1
                    self.state.identity_shift_log.append({
                        "turn": self.state.turn_count,
                        "drift": round(drift, 4),
                        "trigger": qm_key or "unknown",
                    })
                    # Slow anchor migration (90% old + 10% new)
                    self.state.identity_anchor = (0.9 * anchor_vec + 0.1 * current_vec).tolist()

    def decide_only(
        self,
        question: str,
        question_id: str = "",
        friends_using: Optional[float] = None,
        location_quality: Optional[float] = None,
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

    def _personality_summary(self) -> str:
        """Translate top personality traits into natural language."""
        trait_labels = {
            "risk_aversion": ("cautious about trying new things", "adventurous and willing to take risks"),
            "price_sensitivity": ("very price-conscious", "doesn't worry much about prices"),
            "convenience_preference": ("values convenience above most things", "doesn't mind extra effort"),
            "tech_adoption": ("enthusiastic about technology", "prefers traditional ways"),
            "health_consciousness": ("very health-focused", "relaxed about health"),
            "brand_loyalty": ("loyal to trusted brands", "willing to switch brands easily"),
            "time_pressure": ("always pressed for time", "has a relaxed schedule"),
            "social_activity": ("very social and outgoing", "prefers keeping to themselves"),
            "impulsivity": ("tends to act on impulse", "thinks carefully before deciding"),
        }
        traits = self._traits
        scored = []
        for attr, (high_text, low_text) in trait_labels.items():
            val = getattr(traits, attr, 0.5)
            distance = abs(val - 0.5)
            if distance >= 0.15:
                text = high_text if val > 0.5 else low_text
                scored.append((distance, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""
        return "You are someone who is " + ", ".join(t for _, t in scored[:3]) + "."

    def _update_emotional_state(self, perception) -> None:
        """Update agent emotional state based on fatigue, topic, and randomness."""
        import random as _random
        fatigue = getattr(self.state, "fatigue", 0.0)
        topic = getattr(perception, "topic", "general")

        sensitive_topics = {"cost_of_living", "housing", "policy"}
        if fatigue > 0.7:
            self.state.emotional_state = _random.choice(["bored", "annoyed"])
        elif fatigue > 0.4 and topic in sensitive_topics:
            self.state.emotional_state = _random.choice(["annoyed", "neutral", "neutral"])
        elif topic in sensitive_topics and _random.random() < 0.3:
            self.state.emotional_state = "annoyed"
        elif fatigue < 0.2 and _random.random() < 0.2:
            self.state.emotional_state = "enthusiastic"
        else:
            self.state.emotional_state = "neutral"

    def _demographics_dict(self) -> Dict[str, Any]:
        return {
            "age_group": self.persona.age,
            "nationality": self.persona.nationality,
            "income_band": self.persona.income,
            "location": self.persona.location,
            "occupation": self.persona.occupation,
            "household_size": self.persona.household_size,
            "family_children": self.persona.family.children,
            "has_spouse": self.persona.family.spouse,
        }

    def _lifestyle_dict(self) -> Dict[str, Any]:
        pa = self.persona.personal_anchors
        return {
            "cuisine_preference": pa.cuisine_preference,
            "diet": pa.diet,
            "hobby": pa.hobby,
            "work_schedule": pa.work_schedule,
            "health_focus": pa.health_focus,
            "commute_method": pa.commute_method,
        }

    def _meta_dict(self) -> Dict[str, Any]:
        meta = self.persona.meta
        return {
            "persona_version": meta.persona_version,
            "synthesis_method": meta.synthesis_method,
            "generation_seed": meta.generation_seed,
            "archetype_id": meta.archetype_id,
            "persona_cluster": meta.persona_cluster,
        }

    async def think(
        self,
        question: str,
        question_id: str = "",
        friends_using: Optional[float] = None,
        location_quality: Optional[float] = None,
        diagnostics_enabled: bool = False,
        option_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full cognitive pipeline (survey mode): perceive -> recall -> decide -> reason.
        Uses LLM fallback for question understanding on unknown topics.
        Handles greetings, acknowledgments, back-references, and repetitions
        as first-class intent classes before entering the full decision pipeline.
        Returns dict with answer, probability, distribution, etc.
        """
        import random as _random
        from config.settings import get_settings as _get_settings
        _settings = _get_settings()

        self.state.turn_count = getattr(self.state, "turn_count", 0) + 1

        trace = CognitiveTrace(
            agent_id=self.persona.agent_id,
            question=question,
            turn_count=self.state.turn_count,
        )

        if getattr(_settings, "enable_fatigue", True):
            fatigue_rate = 0.05
            if hasattr(self._traits, "impulsivity") and self._traits.impulsivity > 0.6:
                fatigue_rate = 0.08
            elif hasattr(self._traits, "conscientiousness") and self._traits.conscientiousness > 0.7:
                fatigue_rate = 0.03
            self.state.fatigue = min(1.0, getattr(self.state, "fatigue", 0.0) + fatigue_rate)
        trace.fatigue = self.state.fatigue

        intent = classify_intent_class(question, self.state)
        trace.intent_class = intent.value

        def _early_return(narrative: str, status: str, ic: str) -> Dict[str, Any]:
            trace.final_response = narrative
            trace.emotional_state = getattr(self.state, "emotional_state", "neutral")
            if hasattr(self.state, "record_interaction"):
                self.state.record_interaction(question, "conversation")
            return {
                "answer": narrative, "sampled_option": "", "sampled_option_canonical": "",
                "distribution": {}, "agent_id": self.persona.agent_id,
                "perception_topic": "general", "perception_domain": "general",
                "interaction_mode": "conversation", "question_model_key": "",
                "option_space_key": "", "turn_understanding": None,
                "decision_trace": {}, "narrative_alignment_status": status,
                "intent_class": ic,
                "cognitive_trace": trace.to_dict(),
                "demographics": self._demographics_dict(),
                "lifestyle": self._lifestyle_dict(),
                "persona_meta": self._meta_dict(),
            }

        if intent == IntentClass.GREETING:
            return _early_return(_random.choice(_GREETING_RESPONSES), "greeting", intent.value)

        if intent == IntentClass.ACKNOWLEDGMENT:
            return _early_return(_random.choice(_ACKNOWLEDGMENT_RESPONSES), "acknowledgment", intent.value)

        if getattr(_settings, "enable_fatigue", True) and getattr(self.state, "fatigue", 0.0) > 0.9 and _random.random() < 0.4:
            snippets = [
                "I already said what I think.", "Pass.", "I'm done with this honestly.",
                "Same thing I said before.", "Yeah whatever, next.",
            ]
            return _early_return(_random.choice(snippets), "fatigue_skip", "fatigue_skip")

        prior_answer = check_question_repetition(question, self.state)
        if prior_answer:
            trace.repetition_detected = True
            snippets = [
                f"Already told you na... {prior_answer}",
                f"Same as before -- {prior_answer}",
                f"I said this already, {prior_answer}",
                f"Like I mentioned, {prior_answer}",
            ]
            return _early_return(_random.choice(snippets), "repeated_question", "repetition")

        option_labels = strip_survey_options_if_qualitative(question, option_labels)

        self.state.nlu_question_id = (question_id or "").strip()
        if getattr(_settings, "clear_turn_understanding_cache_on_each_think", False):
            from agents.intent_router import clear_turn_understanding_cache

            clear_turn_understanding_cache()

        perception = await perceive_with_llm(
            question,
            state=self.state,
            options=option_labels,
            question_id=question_id or "",
        )
        memories = await self.recall(perception)
        trace.parsed_topic = getattr(perception, "topic", "")
        trace.parsed_scale_type = getattr(perception, "scale_type", "")
        trace.retrieved_memories = list(memories)

        ref_ctx = resolve_reference(question, self.state)
        if ref_ctx:
            memories.insert(0, f"[context] You were just asked: {ref_ctx['previous_question']} — you answered: {ref_ctx['previous_answer']}")

        distribution, sampled_answer = self.decide(
            perception, memories, friends_using=friends_using, location_quality=location_quality
        )
        trace.sampled_option = sampled_answer
        trace.conviction_profile = str(self._last_decision_trace.get("conviction_profile", ""))

        psummary = self._personality_summary()
        _rh_habit = getattr(
            self.persona.personal_anchors.narrative_style, "rhetorical_habit", "direct",
        ) or "direct"
        response_contract = build_response_contract(
            interaction_mode=getattr(perception, "interaction_mode", "survey"),
            scale_type=getattr(perception, "scale_type", ""),
            sampled_option=sampled_answer,
            distribution=distribution,
            ordered_options=list(distribution.keys()),
            latent_state=getattr(self.state, "latent_state", None),
            decision_trace=self._last_decision_trace,
            beliefs=getattr(self.state, "beliefs", None),
            personality_summary=psummary,
            rhetorical_habit=_rh_habit,
        )
        contract_dict = response_contract.to_dict()
        contract_dict["_turn_count"] = self.state.turn_count
        contract_dict["_fatigue"] = self.state.fatigue
        contract_dict["_emotional_state"] = getattr(self.state, "emotional_state", "neutral")
        self._last_response_contract = contract_dict

        trace.confidence_band = contract_dict.get("confidence_band", "")
        trace.tone_selected = contract_dict.get("tone_selected", "")
        trace.dominant_factor = contract_dict.get("dominant_factor")
        trace.runner_up_option = contract_dict.get("runner_up_option")
        trace.tradeoff_guidance = contract_dict.get("tradeoff_guidance")
        trace.belief_statements = contract_dict.get("belief_statements") or []
        trace.personality_summary = psummary
        trace.decision_latency = contract_dict.get("decision_latency", "normal")

        narrative = await self.reason(
            question,
            sampled_answer,
            distribution,
            memories,
            response_contract=self._last_response_contract,
            turn_understanding=getattr(perception, "turn_understanding", None),
            diagnostics_enabled=diagnostics_enabled,
            option_labels=option_labels,
        )
        if question_id:
            self.update_state(question_id, narrative)
        if hasattr(self.state, "record_interaction"):
            self.state.record_interaction(question, getattr(perception, "interaction_mode", "survey"))

        record_question(question, narrative if isinstance(narrative, str) else str(sampled_answer), self.state)

        if getattr(_settings, "enable_emotional_carryover", True):
            self._update_emotional_state(perception)
        self._update_state_after_answer(perception, distribution, sampled_answer)

        trace.emotional_state = getattr(self.state, "emotional_state", "neutral")
        trace.final_response = narrative if isinstance(narrative, str) else str(narrative)
        trace.latent_state_snapshot = self.state.latent_state.to_dict()
        trace.belief_snapshot = self.state.beliefs.to_dict()

        pp_log = []
        if isinstance(narrative, str) and hasattr(self, "_last_pp_log"):
            pp_log = self._last_pp_log
        trace.post_processing_applied = pp_log

        question_model_key = getattr(perception, "question_model_key", "")
        sampled_option_canonical = canonicalize_option(question_model_key, sampled_answer)
        option_space_key = get_option_space_key(question_model_key)
        if not distribution:
            alignment_status = "open_text_not_applicable"
        else:
            consistent, _ = validate_narrative_consistency(
                narrative,
                sampled_answer,
                list(distribution.keys()),
            )
            if self._last_alignment_metadata.get("hard_block_applied"):
                alignment_status = "contradiction_blocked"
            elif self._last_alignment_metadata.get("repaired"):
                alignment_status = "repaired"
            elif consistent:
                alignment_status = "aligned"
            else:
                alignment_status = "repaired"

        try:
            from evaluation.invariants import run_agent_invariants
            violations = run_agent_invariants(
                self.state, intent.value, bool(distribution), trace,
            )
            trace.invariant_violations = violations
        except ImportError:
            pass

        result = {
            "answer": narrative,
            "sampled_option": sampled_answer,
            "sampled_option_canonical": sampled_option_canonical,
            "distribution": distribution,
            "agent_id": self.persona.agent_id,
            "perception_topic": perception.topic,
            "perception_domain": perception.domain,
            "interaction_mode": getattr(perception, "interaction_mode", "survey"),
            "question_model_key": question_model_key,
            "option_space_key": option_space_key,
            "turn_understanding": getattr(perception, "turn_understanding", None),
            "decision_trace": self._last_decision_trace,
            "narrative_alignment_status": alignment_status,
            "intent_class": intent.value,
            "cognitive_trace": trace.to_dict(),
            "demographics": self._demographics_dict(),
            "lifestyle": self._lifestyle_dict(),
            "persona_meta": self._meta_dict(),
        }
        if diagnostics_enabled:
            result["response_diagnostics"] = {
                **self._last_response_contract,
                "narrative_alignment_status": alignment_status,
                "alignment": self._last_alignment_metadata,
            }
        return result
