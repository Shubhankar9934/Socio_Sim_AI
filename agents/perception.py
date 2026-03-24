"""
Perception layer: extract topic, domain, scale type, and question-model key
from survey questions.  Rule-based keyword matching with an optional
LLM-assisted classification fallback for unknown questions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from agents.intent_router import (
    build_turn_understanding_hybrid,
    build_turn_understanding_rules,
    detect_question_type as intent_router_detect_question_type,
    is_low_information_turn as intent_router_is_low_information_turn,
    looks_like_explicit_survey as intent_router_looks_like_explicit_survey,
    looks_like_qualitative_interview as intent_router_looks_like_qualitative_interview,
    resolve_model_key_rule,
)

from config.question_models import (
    GENERIC_DURATION,
    GENERIC_FALLBACK,
    GENERIC_FREQUENCY,
    GENERIC_LIKERT,
    GENERIC_OPEN_TEXT,
    QUESTION_MODELS,
    QuestionModel,
    get_question_model,
    load_generated_models_into_registry,
)

# ── Topic keywords for routing (loaded from domain config) ──────────────

def _load_keywords() -> tuple:
    """Load topic and domain keywords from domain config."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        tk = cfg.topic_keywords if cfg.topic_keywords else {}
        dk = cfg.domain_keywords if cfg.domain_keywords else {}
        return tk, dk
    except Exception:
        return {}, {}


_topic_kw, _domain_kw = _load_keywords()

TOPIC_KEYWORDS: dict[str, list[str]] = _topic_kw if _topic_kw else {
    "satisfaction": ["satisfied", "satisfaction", "happy", "experience", "rate", "rating"],
    "policy": ["policy", "regulation", "government", "law", "support", "oppose"],
    "general": [],
}

DOMAIN_KEYWORDS: dict[str, list[str]] = _domain_kw if _domain_kw else {
    "services": ["service", "government", "utility", "health"],
    "technology": ["app", "fintech", "digital", "tech", "online"],
}

# ── Scale-type detection keywords ───────────────────────────────────────

_FREQUENCY_CUES = frozenset({
    "how often", "frequency", "per week", "per month", "per day",
    "regularly", "times", "how many times",
})

_SATISFACTION_CUES = frozenset({
    "satisfied", "satisfaction", "rate", "rating", "how happy",
    "how would you rate", "score", "scale",
})

_LIKELIHOOD_CUES = frozenset({
    "likely", "likelihood", "would you", "will you", "probability",
    "chance", "intend to", "plan to", "how probable",
})

_POLICY_CUES = frozenset({
    "support", "oppose", "policy", "regulation", "government", "law",
    "congestion pricing", "subsidize", "subsidise",
})

_NPS_CUES = frozenset({
    "would you recommend", "recommend", "nps", "net promoter",
    "on a scale of 0 to 10", "0-10", "0 to 10",
})

_DURATION_CUES = frozenset({
    "how long", "how many years", "for how long", "since when",
    "how many months", "how long have you", "how long did you",
    "years have you",
})

_OPEN_ENDED_CUES = frozenset({
    "how do you", "tell me", "describe", "what do you think",
    "deal with", "your experience", "in your view", "in your opinion",
    "how would you describe", "what's your experience", "share your",
})

_GREETING_CUES = frozenset({
    "hi", "hello", "hey", "namaste", "good morning", "good afternoon",
    "good evening", "what's up", "whats up",
})

_CONVERSATION_CUES = frozenset({
    "thanks", "thank you", "okay", "ok", "alright", "sure", "fine",
    "got it", "cool", "great", "bye", "goodbye",
})

_FOLLOW_UP_CUES = frozenset({
    "and", "and?", "then", "then?", "what else", "what else?", "why", "why?",
    "how", "how?", "who", "who?", "which", "which?", "tell me",
    "go on", "continue", "hmm", "hmm?", "yes", "yes?", "no", "no?",
})

_QUALITATIVE_OPENERS = frozenset({
    "who", "what", "when", "where", "why", "how", "tell me", "describe",
    "share", "walk me through", "explain", "what do you think",
    "what comes to your mind", "what is your routine", "how do you feel",
    "if i ask", "suppose", "what do your friends say",
})

_TRANSPORT_CUES = frozenset({
    "uber", "careem", "ride-hailing", "ride hailing", "taxi", "commute",
    "metro", "bus", "transport", "parking", "driving",
})

_FOOD_CUES = frozenset({
    "food", "delivery", "restaurant", "dining", "order online", "talabat",
    "deliveroo", "zomato", "meal",
})

_HOUSING_CUES = frozenset({
    "housing", "rent", "apartment", "accommodation", "landlord",
})

_COST_OF_LIVING_CUES = frozenset({
    "cost of living",
    "living expenses",
    "affordability",
    "price of living",
    "grocery",
    "groceries",
    "grocery prices",
    "supermarket",
    "inflation",
    "food prices",
    "household budget",
    "utility bills",
    "utilities",
    "monthly bills",
})

_QUESTION_TYPE_TO_SCALE_TYPE: Dict[str, str] = {
    "frequency": "frequency",
    "likelihood": "likelihood",
    "likert": "likert",
    "policy_support": "policy_support",
    "nps": "nps",
    "duration": "duration",
    "open_text": "open_text",
    "categorical": "categorical",
}


@dataclass
class Perception:
    """Structured perception of a survey question."""

    topic: str
    domain: str
    location_related: bool
    keywords: List[str]
    raw_question: str
    scale_type: str = "categorical"
    question_type: str = "categorical"
    question_model_key: str = "generic"
    resolution_source: str = "rule"
    resolution_confidence: float = 0.8
    interaction_mode: str = "survey"
    adaptive_allowed: bool = True
    structured_response_expected: bool = True
    turn_understanding: Optional[Dict[str, Any]] = None


# ── Internal helpers ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.lower().strip()


def _normalized_tokens(text: str) -> str:
    return re.sub(r"\s+", " ", _normalize(text))


def _starts_with_any(text: str, cues: frozenset[str]) -> bool:
    return any(text.startswith(cue) for cue in cues)


def _is_low_information_turn(question: str) -> bool:
    return intent_router_is_low_information_turn(question)


def _extract_keywords(text: str, keyword_lists: dict[str, list[str]]) -> Set[str]:
    normalized = _normalize(text)
    found: Set[str] = set()
    for _, keywords in keyword_lists.items():
        for kw in keywords:
            if kw in normalized:
                found.add(kw)
    return found


def _detect_topic(question: str) -> str:
    normalized = _normalize(question)
    if any(cue in normalized for cue in _COST_OF_LIVING_CUES):
        return "cost_of_living"
    # Deterministic lexical overrides for high-impact domains.
    if any(cue in normalized for cue in _TRANSPORT_CUES):
        return "transport"
    if any(cue in normalized for cue in _FOOD_CUES):
        return "food_delivery"
    if any(cue in normalized for cue in _HOUSING_CUES):
        return "housing"

    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic == "general":
            continue
        for kw in keywords:
            if kw in normalized:
                return topic
    return "general"


def _detect_domain(question: str) -> str:
    normalized = _normalize(question)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                return domain
    return "general"


def _get_location_terms() -> list[str]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.location_terms:
            return cfg.location_terms
    except Exception:
        pass
    return ["area", "district", "neighborhood", "where", "location", "local", "nearby"]


def _is_location_related(question: str) -> bool:
    normalized = _normalize(question)
    return any(term in normalized for term in _get_location_terms())


def _detect_question_type(question: str) -> str:
    return intent_router_detect_question_type(question)


def _detect_scale_type(question: str) -> str:
    """Back-compat alias used by existing callers/tests."""
    return _QUESTION_TYPE_TO_SCALE_TYPE.get(_detect_question_type(question), "categorical")


def _looks_like_explicit_survey(question: str) -> bool:
    return intent_router_looks_like_explicit_survey(question)


def _looks_like_qualitative_interview(question: str) -> bool:
    return intent_router_looks_like_qualitative_interview(question)


def _classify_interaction_mode(question: str, state: Optional["AgentState"] = None) -> str:
    understanding = build_turn_understanding_rules(question, state=state)
    return str(understanding["interaction_mode"])


# ── Topic → QuestionModel key mapping ──────────────────────────────────

def _load_topic_to_model_key() -> dict[str, dict[str, str]]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.topic_to_model_key:
            return cfg.topic_to_model_key
    except Exception:
        pass
    return {}


_TOPIC_TO_MODEL_KEY: dict[str, dict[str, str]] = _load_topic_to_model_key()


def _resolve_model_key(topic: str, question_type: str) -> str:
    if question_type == "open_text":
        return "generic_open_text"
    if question_type == "duration":
        return "generic_duration"
    if question_type == "nps":
        return "nps_recommendation"
    if question_type == "policy_support":
        return "policy_support"
    if question_type == "likelihood":
        if topic in {"transport", "food_delivery", "housing"}:
            return "tech_adoption_likelihood"
        return "tech_adoption_likelihood"

    if topic == "cost_of_living":
        if question_type in {"frequency", "likert", "categorical"}:
            return "cost_of_living_satisfaction"
        return "cost_of_living_satisfaction"

    # Explicit deterministic correction for frequency transport questions.
    if question_type == "frequency" and topic == "transport":
        return "transport_usage_frequency"

    topic_map = _TOPIC_TO_MODEL_KEY.get(topic)
    if topic_map:
        resolved = topic_map.get(question_type) or topic_map.get("categorical")
        if resolved:
            return resolved
        return next(iter(topic_map.values()))

    if topic == "satisfaction" or question_type == "likert":
        return "generic_likert"
    if question_type == "frequency":
        return "generic_frequency"
    if question_type == "likelihood":
        return "tech_adoption_likelihood"
    if question_type == "policy_support":
        return "policy_support"
    if question_type == "nps":
        return "nps_recommendation"
    return "generic_likert"


# ── Public API ──────────────────────────────────────────────────────────

def perceive(question: str, state: Optional["AgentState"] = None) -> Perception:
    """Extract structured perception from a survey question."""
    rule_understanding = build_turn_understanding_rules(question, state=state)
    topic = str(rule_understanding["topic"])
    domain = str(rule_understanding["domain"])
    location_related = bool(rule_understanding["location_related"])
    keywords = list(rule_understanding["keywords"])
    question_type = str(rule_understanding["question_type"])
    scale_type = str(rule_understanding["scale_type"])
    model_key = str(rule_understanding["question_model_key_candidate"])
    resolution_source = "rule"
    resolution_confidence = float(rule_understanding["rule_confidence"])
    interaction_mode = str(rule_understanding["interaction_mode"])
    adaptive_allowed = interaction_mode == "survey"
    structured_response_expected = interaction_mode == "survey"

    if interaction_mode != "survey":
        resolution_source = interaction_mode

    if adaptive_allowed:
        try:
            from agents.adaptive_layer import get_cached_generated_model_key

            cached_key = get_cached_generated_model_key(question)
            if cached_key:
                model_key = cached_key
                resolution_source = "adaptive_cache"
                resolution_confidence = 0.86
        except Exception:
            pass

    return Perception(
        topic=topic,
        domain=domain,
        location_related=location_related,
        keywords=keywords,
        raw_question=question,
        scale_type=scale_type,
        question_type=question_type,
        question_model_key=model_key,
        resolution_source=resolution_source,
        resolution_confidence=resolution_confidence,
        interaction_mode=interaction_mode,
        adaptive_allowed=adaptive_allowed,
        structured_response_expected=structured_response_expected,
        turn_understanding=rule_understanding,
    )


def _preview_nlu_options(
    opts: List[str],
    *,
    max_options: int = 8,
    max_chars: int = 80,
) -> List[str]:
    out: List[str] = []
    for o in opts[:max_options]:
        s = str(o).strip()
        if len(s) > max_chars:
            s = s[: max_chars - 1] + "…"
        out.append(s)
    return out


def _resolve_effective_nlu_options(
    perception: Perception,
    options: Optional[List[str]],
    *,
    settings: Any,
) -> Tuple[List[str], str]:
    """Options list for hybrid NLU + source tag: caller | model_scale_fallback | none."""
    if options:
        cleaned = [str(o).strip() for o in options if str(o).strip()]
        if cleaned:
            return cleaned, "caller"
    if not perception.structured_response_expected:
        return [], "none"
    if not getattr(settings, "nlu_fallback_options_from_model_scale", True):
        return [], "none"
    if perception.question_type == "open_text" or perception.scale_type == "open_text":
        return [], "none"
    load_generated_models_into_registry()
    model = QUESTION_MODELS.get(perception.question_model_key)
    if model is None or not model.scale:
        return [], "none"
    scale = [str(x) for x in model.scale if str(x).strip()]
    if not scale:
        return [], "none"
    return scale, "model_scale_fallback"


def _enrich_turn_understanding_telemetry(
    tu: Dict[str, Any],
    *,
    interaction_mode: str,
    effective_options: List[str],
    options_source: str,
) -> None:
    """Stable top-level fields for dashboards (all intents). In-place update.

    ``rule_payload`` may contain ``provided_options`` from rules; these fields
    summarize what was actually passed into ``build_turn_understanding_hybrid``.
    """
    tu["nlu_interaction_mode"] = interaction_mode
    tu["nlu_provided_options_count"] = len(effective_options)
    tu["nlu_provided_options_preview"] = (
        _preview_nlu_options(effective_options) if effective_options else []
    )
    tu["nlu_options_source"] = options_source


def detect_question_model(perception: Perception) -> QuestionModel:
    """Map a Perception to the appropriate QuestionModel.

    Resolution order:
      1. Open-text scale type -> GENERIC_OPEN_TEXT.
      2. Exact key in QUESTION_MODELS registry.
      3. Generic frequency / likert based on scale_type.
      4. GENERIC_FALLBACK (always works).
    """
    qtype = getattr(perception, "question_type", getattr(perception, "scale_type", "categorical"))

    if qtype == "open_text" or perception.scale_type == "open_text":
        return GENERIC_OPEN_TEXT
    if qtype == "duration" or perception.scale_type == "duration":
        return GENERIC_DURATION
    load_generated_models_into_registry()
    model = QUESTION_MODELS.get(perception.question_model_key)
    if model is not None:
        return get_question_model(model.name)

    if qtype == "frequency" or perception.scale_type == "frequency":
        return get_question_model(GENERIC_FREQUENCY.name)
    if qtype in {"likert", "likelihood", "policy_support", "nps"} or perception.scale_type == "likert":
        return get_question_model(GENERIC_LIKERT.name)
    return get_question_model(GENERIC_FALLBACK.name)


async def classify_question_via_llm(question: str) -> Optional[str]:
    """LLM-assisted classification for questions that keyword matching cannot resolve.

    Returns a question_model_key string if the LLM succeeds, or None.
    Meant to be called only when keyword detection returns 'general' topic.
    """
    try:
        from config.settings import get_settings
        from llm.client import get_llm_client

        categories = ", ".join(QUESTION_MODELS.keys())
        prompt = (
            f"Classify the following survey question into exactly one of these categories:\n"
            f"{categories}\n\n"
            f"If none match, respond with: generic_likert\n\n"
            f"Question: \"{question}\"\n\n"
            f"Respond with ONLY the category name, nothing else."
        )
        client = get_llm_client()
        settings = get_settings()
        response = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30,
        )
        key = response.strip().lower().replace(" ", "_")
        if key in QUESTION_MODELS:
            return key
        return None
    except Exception:
        return None


# ── LLM-inferred dimension weights (persistent, hash-keyed cache) ──────

import hashlib
import json as _json
import pathlib

_CACHE_PATH = pathlib.Path("data/dimension_cache.json")

def _get_dimension_names_str() -> str:
    """Build dimension names string, including discovered extras."""
    try:
        from discovery.dimensions import get_active_dimension_names
        behavioral, belief = get_active_dimension_names()
        return ", ".join(behavioral + belief)
    except Exception:
        pass
    return ", ".join([
        "convenience_seeking", "price_sensitivity", "technology_openness",
        "risk_aversion", "health_orientation", "routine_stability",
        "novelty_seeking", "social_influence_susceptibility", "time_pressure",
        "financial_confidence", "environmental_consciousness", "institutional_trust",
    ])


def _cache_key(question: str) -> str:
    """Normalize and hash a question for stable cache lookup."""
    norm = " ".join(question.lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def _load_dimension_cache() -> dict[str, dict[str, float]]:
    if _CACHE_PATH.exists():
        try:
            return _json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_dimension_cache(cache: dict[str, dict[str, float]]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(_json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


_LLM_DIMENSION_CACHE: dict[str, dict[str, float]] = _load_dimension_cache()


async def infer_dimension_weights_via_llm(question: str) -> dict[str, float]:
    """Ask the LLM to assign behavioral dimension weights for an arbitrary question.

    Returns a dict of dimension_name -> weight (float in [-1, 1]).
    Results are cached by hash key both in-memory and on disk.
    Falls back to empty dict on any error.
    """
    key = _cache_key(question)
    if key in _LLM_DIMENSION_CACHE:
        return _LLM_DIMENSION_CACHE[key]

    try:
        from llm.client import get_llm_client

        prompt = (
            f"Given this survey question:\n\"{question}\"\n\n"
            f"Rate the relevance of each behavioral dimension on a scale from -1.0 to 1.0.\n"
            f"Positive means higher dimension value makes the person more likely to agree/choose higher.\n"
            f"Negative means higher dimension value makes the person less likely.\n"
            f"0.0 means irrelevant.\n\n"
            f"Dimensions: {_get_dimension_names_str()}\n\n"
            f"Respond with ONLY a JSON object mapping dimension names to float weights. "
            f"Example: {{\"convenience_seeking\": 0.6, \"price_sensitivity\": -0.3}}\n"
            f"Only include dimensions with |weight| >= 0.1."
        )
        client = get_llm_client()
        response = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        weights = _json.loads(text)
        result = {k: float(v) for k, v in weights.items() if abs(float(v)) >= 0.1}
        _LLM_DIMENSION_CACHE[key] = result
        _save_dimension_cache(_LLM_DIMENSION_CACHE)

        from config.question_models import QUESTION_DIMENSION_MAP
        p = perceive(question)
        qm_key = p.question_model_key or "generic_likert"
        QUESTION_DIMENSION_MAP.setdefault(qm_key, result)
        return result
    except Exception:
        return {}


async def perceive_with_llm(
    question: str,
    state: Optional["AgentState"] = None,
    *,
    options: Optional[List[str]] = None,
    question_id: str = "",
) -> Perception:
    """Perceive with LLM fallback for dimension weights on unknown questions.

    If keyword matching resolves to a 'general' topic, tries the LLM to
    infer behavioral dimension weights.  The weights are stored in the
    QUESTION_DIMENSION_MAP cache for subsequent calls.

    ``turn_understanding`` (after this call) includes hybrid fields plus
    telemetry: ``nlu_interaction_mode``, ``nlu_provided_options_count``,
    ``nlu_provided_options_preview``, ``nlu_options_source`` (caller |
    model_scale_fallback | none). Rule-layer ``provided_options`` lives under
    ``rule_payload`` when present.
    """
    from config.settings import get_settings

    settings = get_settings()
    perception = perceive(question, state=state)
    load_generated_models_into_registry()

    effective_options, options_source = _resolve_effective_nlu_options(
        perception, options, settings=settings,
    )
    opt_for_hybrid = effective_options if effective_options else None

    if (
        getattr(settings, "log_nlu_missing_options_for_structured_survey", False)
        and perception.structured_response_expected
        and not effective_options
    ):
        logger.warning(
            "NLU hybrid has no options for structured turn: question_id=%s model_key=%s q=%s",
            (question_id or "").strip() or getattr(state, "nlu_question_id", ""),
            perception.question_model_key,
            (question or "")[:120],
        )

    run_ns = ""
    qid = (question_id or "").strip()
    if state is not None:
        run_ns = str(getattr(state, "survey_run_id", "") or "")
        if not qid:
            qid = str(getattr(state, "nlu_question_id", "") or "").strip()
    understanding = await build_turn_understanding_hybrid(
        question,
        state=state,
        options=opt_for_hybrid,
        survey_run_id=run_ns,
        question_id=qid,
    )
    tu = understanding.to_dict()
    _enrich_turn_understanding_telemetry(
        tu,
        interaction_mode=understanding.interaction_mode,
        effective_options=effective_options,
        options_source=options_source,
    )
    perception.turn_understanding = tu
    perception.interaction_mode = understanding.interaction_mode
    perception.adaptive_allowed = understanding.interaction_mode == "survey"
    perception.structured_response_expected = understanding.interaction_mode == "survey"
    perception.topic = understanding.topic
    perception.domain = understanding.domain
    perception.location_related = understanding.location_related
    perception.keywords = list(understanding.keywords)
    perception.question_type = understanding.question_type
    perception.scale_type = understanding.scale_type
    perception.question_model_key = understanding.question_model_key_candidate
    perception.resolution_confidence = understanding.final_confidence
    perception.resolution_source = f"hybrid:{understanding.fusion_reason}"

    if not perception.adaptive_allowed:
        return perception

    def _update_types_from_model_key(model_key: str, source: str, confidence: float) -> None:
        perception.question_model_key = model_key
        perception.resolution_source = source
        perception.resolution_confidence = confidence
        model = QUESTION_MODELS.get(model_key)
        if model is None:
            return
        if not model.scale:
            perception.question_type = "open_text"
            perception.scale_type = "open_text"
            return
        if all(str(v).isdigit() for v in model.scale):
            if len(model.scale) >= 10:
                perception.question_type = "nps"
                perception.scale_type = "nps"
            else:
                perception.question_type = "likert"
                perception.scale_type = "likert"
            return
        lowered = {str(v).lower() for v in model.scale}
        if {"never", "rarely", "sometimes", "often", "very often"}.issubset(lowered):
            perception.question_type = "frequency"
            perception.scale_type = "frequency"
        elif any("support" in v for v in lowered):
            perception.question_type = "policy_support"
            perception.scale_type = "policy_support"
        elif any("likely" in v for v in lowered):
            perception.question_type = "likelihood"
            perception.scale_type = "likelihood"

    try:
        from agents.adaptive_layer import adaptive_generate_and_register

        adaptive_key = await adaptive_generate_and_register(
            question,
            fallback_model_key=perception.question_model_key if perception.question_model_key in QUESTION_MODELS else None,
        )
        if adaptive_key:
            load_generated_models_into_registry(force=True)
            _update_types_from_model_key(adaptive_key, source="adaptive", confidence=0.72)
    except Exception:
        pass

    if perception.question_model_key not in QUESTION_MODELS:
        model_key = await classify_question_via_llm(question)
        if model_key:
            _update_types_from_model_key(model_key, source="llm", confidence=0.65)
        weights = await infer_dimension_weights_via_llm(question)
        if weights:
            from config.question_models import QUESTION_DIMENSION_MAP
            QUESTION_DIMENSION_MAP[perception.question_model_key] = weights
    return perception
