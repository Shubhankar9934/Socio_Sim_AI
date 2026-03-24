"""Shared hybrid understanding for survey turns and prompt routing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from llm.client import get_llm_client

if TYPE_CHECKING:
    from agents.state import AgentState


class IntentClass(str, Enum):
    """Fine-grained intent classification for incoming turns."""
    GREETING = "greeting"
    ACKNOWLEDGMENT = "acknowledgment"
    BACK_REFERENCE = "back_reference"
    CLARIFICATION_REQUEST = "clarification_request"
    IMPLICIT_OPINION = "implicit_opinion"
    SURVEY = "survey"
    CONVERSATION = "conversation"
    QUALITATIVE = "qualitative"


INTERACTION_MODES = frozenset({
    "conversation",
    "qualitative_interview",
    "survey",
})

QUESTION_TYPES = frozenset({
    "duration",
    "open_text",
    "nps",
    "policy_support",
    "frequency",
    "likelihood",
    "likert",
    "categorical",
})

SCALE_TYPES = frozenset({
    "duration",
    "open_text",
    "nps",
    "policy_support",
    "frequency",
    "likelihood",
    "likert",
    "categorical",
    "numeric",
})

TOPIC_NAMES = frozenset({
    "general",
    "satisfaction",
    "policy",
    "transport",
    "food_delivery",
    "housing",
    "cost_of_living",
})

DOMAIN_NAMES = frozenset({
    "general",
    "services",
    "technology",
})

ACTION_TYPES = frozenset({
    "frequency",
    "adopt",
    "reject",
    "support",
    "oppose",
    "rate",
    "choose",
    "increase",
    "decrease",
    "invest",
    "migrate",
    "comply",
    "protest",
})

TARGET_TYPES = frozenset({
    "service",
    "product",
    "policy",
    "candidate",
    "belief",
    "behavior",
    "location",
    "investment",
    "norm",
    "experience",
})

INTENSITY_SCALES = frozenset({"ordinal", "binary", "continuous"})

QUESTION_TYPE_TO_SCALE_TYPE: Dict[str, str] = {
    "frequency": "frequency",
    "likelihood": "likelihood",
    "likert": "likert",
    "policy_support": "policy_support",
    "nps": "nps",
    "duration": "duration",
    "open_text": "open_text",
    "categorical": "categorical",
}

GREETING_CUES = frozenset({
    "hi", "hello", "hey", "namaste", "good morning", "good afternoon",
    "good evening", "what's up", "whats up",
})

CONVERSATION_CUES = frozenset({
    "thanks", "thank you", "okay", "ok", "alright", "sure", "fine",
    "got it", "cool", "great", "bye", "goodbye",
})

FOLLOW_UP_CUES = frozenset({
    "and", "and?", "then", "then?", "what else", "what else?", "why", "why?",
    "how", "how?", "who", "who?", "which", "which?", "tell me",
    "go on", "continue", "hmm", "hmm?", "yes", "yes?", "no", "no?",
})

ACKNOWLEDGMENT_CUES = frozenset({
    "hmm ok", "ok", "okay", "alright", "sure", "fine", "got it",
    "cool", "nice", "right", "i see", "ah ok", "oh ok", "hmm",
    "hm", "uh huh", "yep", "yeah", "ya", "haan", "achha", "theek hai",
})

BACK_REFERENCE_CUES = frozenset({
    "same as before", "same thing", "what i said", "like i said",
    "already told you", "same answer", "as i mentioned",
    "i already said", "told you already",
})

ANAPHORIC_CUES = frozenset({
    "why", "why?", "how come", "how come?", "really", "really?",
    "what do you mean", "what do you mean?", "how so", "how so?",
    "in what way", "elaborate", "can you explain",
})

IMPLICIT_OPINION_MARKERS = frozenset({
    "these days", "nowadays", "too expensive", "too cheap",
    "getting worse", "getting better", "increasing", "decreasing",
    "so annoying", "so good", "so bad", "not fair", "unfair",
    "love it", "hate it", "overpriced", "underrated",
})


def classify_intent_class(
    question: str,
    state: Optional["AgentState"] = None,
) -> IntentClass:
    """Classify the fine-grained intent of an incoming turn."""
    normalized = normalize_turn(question)
    if not normalized:
        return IntentClass.GREETING

    if normalized in GREETING_CUES:
        return IntentClass.GREETING

    if normalized in ACKNOWLEDGMENT_CUES or (
        len(normalized.split()) <= 3 and any(
            cue in normalized for cue in ACKNOWLEDGMENT_CUES
        )
    ):
        return IntentClass.ACKNOWLEDGMENT

    if any(cue in normalized for cue in BACK_REFERENCE_CUES):
        return IntentClass.BACK_REFERENCE

    if normalized in ANAPHORIC_CUES or (
        len(normalized.split()) <= 4 and any(
            cue in normalized for cue in ANAPHORIC_CUES
        )
    ):
        return IntentClass.CLARIFICATION_REQUEST

    if not normalized.endswith("?") and any(
        marker in normalized for marker in IMPLICIT_OPINION_MARKERS
    ):
        return IntentClass.IMPLICIT_OPINION

    if looks_like_explicit_survey(question):
        return IntentClass.SURVEY
    if looks_like_qualitative_interview(question):
        return IntentClass.QUALITATIVE
    if is_low_information_turn(question):
        return IntentClass.CONVERSATION

    return IntentClass.SURVEY


def resolve_reference(
    question: str,
    state: Optional["AgentState"] = None,
) -> Optional[Dict[str, str]]:
    """Resolve anaphoric/back-reference questions to prior context.

    Returns a dict with 'previous_question' and 'previous_answer' if
    the turn is a back-reference or clarification request and context
    is available.  Returns None otherwise.
    """
    intent = classify_intent_class(question, state)
    if intent not in (
        IntentClass.BACK_REFERENCE,
        IntentClass.CLARIFICATION_REQUEST,
    ):
        return None
    if state is None:
        return None

    prev_q = ""
    prev_a = ""
    if state.recent_utterances:
        prev_q = state.recent_utterances[-1]
    if state.last_answers:
        last_key = list(state.last_answers.keys())[-1]
        prev_a = str(state.last_answers[last_key])

    if not prev_q and not prev_a:
        return None

    return {"previous_question": prev_q, "previous_answer": prev_a}


def question_hash(question: str) -> str:
    """Stable hash for question dedup."""
    return hashlib.sha256(normalize_turn(question).encode()).hexdigest()[:16]


QUALITATIVE_OPENERS = frozenset({
    "who", "what", "when", "where", "why", "how", "tell me", "describe",
    "share", "walk me through", "explain", "what do you think",
    "what comes to your mind", "what is your routine", "how do you feel",
    "if i ask", "suppose", "what do your friends say",
})

SURVEY_PATTERNS = (
    r"^how\s+\w+\s+do you feel\b",
    r"^how\s+\w+\s+are you\b",
    r"^how important\b",
    r"^how concerned\b",
    r"^how comfortable\b",
    r"^how safe\b",
    r"^how affordable\b",
    r"^to what extent\b",
    r"^how much do you agree\b",
    r"^how much do you support\b",
)

FREQUENCY_CUES = frozenset({
    "how often", "frequency", "per week", "per month", "per day",
    "regularly", "times", "how many times",
})

SATISFACTION_CUES = frozenset({
    "satisfied", "satisfaction", "rate", "rating", "how happy",
    "how would you rate", "score", "scale",
})

LIKELIHOOD_CUES = frozenset({
    "likely", "likelihood", "would you", "will you", "probability",
    "chance", "intend to", "plan to", "how probable",
})

POLICY_CUES = frozenset({
    "support", "oppose", "policy", "regulation", "government", "law",
    "congestion pricing", "subsidize", "subsidise",
})

NPS_CUES = frozenset({
    "would you recommend", "recommend", "nps", "net promoter",
    "on a scale of 0 to 10", "0-10", "0 to 10",
})

DURATION_CUES = frozenset({
    "how long", "how many years", "for how long", "since when",
    "how many months", "how long have you", "how long did you",
    "years have you",
})

OPEN_ENDED_CUES = frozenset({
    "how do you", "tell me", "describe", "what do you think",
    "deal with", "your experience", "in your view", "in your opinion",
    "how would you describe", "what's your experience", "share your",
})

TRANSPORT_CUES = frozenset({
    "uber", "careem", "ride-hailing", "ride hailing", "taxi", "commute",
    "metro", "bus", "transport", "parking", "driving",
})

FOOD_CUES = frozenset({
    "food", "delivery", "restaurant", "dining", "order online", "talabat",
    "deliveroo", "zomato", "meal",
})

HOUSING_CUES = frozenset({
    "housing", "rent", "apartment", "accommodation", "landlord",
})

COST_OF_LIVING_CUES = frozenset({
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

LIFESTYLE_KEYWORDS = frozenset({
    "food", "diet", "exercise", "shopping", "travel", "media",
})

ANCHOR_EXCLUDE_KEYWORDS = frozenset({
    "network international", "payment provider", "bank", "fintech",
    "transaction", "payfast", "pay fast",
})

ACTION_PATTERN_RULES = [
    (["how often", "how frequently", "per week", "per month", "times"], ("frequency", "behavior", "ordinal")),
    (["do you support", "in favor", "should the government", "agree or disagree"], ("support", "policy", "ordinal")),
    (["would you vote", "who would you", "which candidate"], ("choose", "candidate", "binary")),
    (["rate your", "how satisfied", "satisfaction", "how happy", "nps", "score"], ("rate", "experience", "continuous")),
    (["would you switch", "would you try", "willing to adopt", "start using"], ("adopt", "product", "binary")),
    (["would you stop", "quit", "cancel", "give up"], ("reject", "service", "binary")),
    (["how much would you invest", "allocate", "budget"], ("invest", "investment", "continuous")),
    (["would you move", "relocate", "migrate"], ("migrate", "location", "binary")),
    (["would you protest", "demonstrate", "rally"], ("protest", "policy", "binary")),
    (["would you comply", "follow the rule", "obey"], ("comply", "norm", "binary")),
]

_TURN_UNDERSTANDING_CACHE: Dict[str, "TurnUnderstandingResult"] = {}


@dataclass(frozen=True)
class IntentRouterResult:
    rule_mode: str
    llm_mode: str
    final_mode: str
    rule_confidence: float
    llm_confidence: float
    reason: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurnUnderstandingResult:
    question: str
    normalized_question: str
    interaction_mode: str
    question_type: str
    scale_type: str
    topic: str
    domain: str
    location_related: bool
    question_model_key_candidate: str
    persona_anchor_allowed: bool
    action_type_candidate: Optional[str]
    target_candidate: Optional[str]
    intensity_scale_candidate: Optional[str]
    normalization_candidates: List[str]
    keywords: List[str]
    rule_confidence: float
    llm_confidence: float
    final_confidence: float
    fusion_reason: str
    rule_payload: Dict[str, Any]
    llm_payload: Dict[str, Any]
    intent_class: str = "survey"
    resolved_reference: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_keywords() -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    try:
        from config.domain import get_domain_config

        cfg = get_domain_config()
        return cfg.topic_keywords or {}, cfg.domain_keywords or {}
    except Exception:
        return {}, {}


_topic_kw, _domain_kw = _load_keywords()

TOPIC_KEYWORDS: Dict[str, List[str]] = _topic_kw if _topic_kw else {
    "satisfaction": ["satisfied", "satisfaction", "happy", "experience", "rate", "rating"],
    "policy": ["policy", "regulation", "government", "law", "support", "oppose"],
    "general": [],
}

DOMAIN_KEYWORDS: Dict[str, List[str]] = _domain_kw if _domain_kw else {
    "services": ["service", "government", "utility", "health"],
    "technology": ["app", "fintech", "digital", "tech", "online"],
}


def normalize_turn(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def starts_with_any(text: str, cues: frozenset[str]) -> bool:
    return any(text.startswith(cue) for cue in cues)


def _normalize_token(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def detect_question_type(question: str) -> str:
    normalized = normalize_turn(question)
    if any(cue in normalized for cue in DURATION_CUES):
        return "duration"
    if any(cue in normalized for cue in OPEN_ENDED_CUES):
        return "open_text"
    if any(cue in normalized for cue in NPS_CUES):
        return "nps"
    if any(cue in normalized for cue in POLICY_CUES):
        return "policy_support"
    if any(cue in normalized for cue in FREQUENCY_CUES):
        return "frequency"
    if any(cue in normalized for cue in LIKELIHOOD_CUES):
        return "likelihood"
    if any(cue in normalized for cue in SATISFACTION_CUES):
        return "likert"
    return "categorical"


def infer_scale_type_from_options(options: Optional[List[str]]) -> str:
    if not options:
        return "open_text"
    normalized = [_normalize_token(opt) for opt in options if str(opt or "").strip()]
    if normalized and all(opt.isdigit() for opt in normalized):
        return "numeric"
    frequency_terms = {"never", "rarely", "sometimes", "often", "daily", "weekly", "very often"}
    frequency_phrases = {"1-2 per week", "3-4 per week", "multiple per day", "1-2 per month", "2-3 per week"}
    if any(opt in frequency_terms or opt in frequency_phrases for opt in normalized):
        return "frequency"
    if len(normalized) == 2:
        return "categorical"
    return "likert"


def is_low_information_turn(question: str) -> bool:
    normalized = normalize_turn(question)
    if not normalized:
        return True
    if normalized in GREETING_CUES or normalized in CONVERSATION_CUES or normalized in FOLLOW_UP_CUES:
        return True
    word_count = len(normalized.split())
    if word_count <= 3 and any(cue in normalized for cue in (GREETING_CUES | CONVERSATION_CUES)):
        return True
    return False


def looks_like_explicit_survey(question: str) -> bool:
    normalized = normalize_turn(question)
    qtype = detect_question_type(question)
    if qtype in {"duration", "frequency", "likelihood", "likert", "policy_support", "nps"}:
        return True
    if any(re.search(pattern, normalized) for pattern in SURVEY_PATTERNS):
        return True
    if "on a scale" in normalized or "rate" in normalized or "rating" in normalized:
        return True
    if re.search(r"\b\d+\s*[:=]\s*.+", normalized):
        return True
    return False


def looks_like_qualitative_interview(question: str) -> bool:
    normalized = normalize_turn(question)
    if not normalized:
        return False
    if looks_like_explicit_survey(question) or is_low_information_turn(question):
        return False
    if starts_with_any(normalized, QUALITATIVE_OPENERS):
        return True
    if normalized.endswith("?") and len(normalized.split()) >= 3:
        return True
    if any(cue in normalized for cue in OPEN_ENDED_CUES):
        return True
    return False


def _survey_options_have_labels(options: Optional[List[str]]) -> bool:
    if not options:
        return False
    return any(str(o).strip() for o in options)


def strip_survey_options_if_qualitative(
    question: str, options: Optional[List[str]],
) -> Optional[List[str]]:
    """Route discrete options: caller-supplied labels always win over question shape.

    If the client sends at least one non-blank option string, it is passed through
    unchanged so structured NLU and choice math run. Only when there are no usable
    labels (missing, empty list, or all blank) do we clear options for qualitative
    interview wording so open turns stay uncluttered.
    """
    if _survey_options_have_labels(options):
        return options
    if not options:
        return options
    if looks_like_qualitative_interview(question):
        return None
    return options


def _previous_mode(state: Optional["AgentState"]) -> str:
    if state is None:
        return ""
    if hasattr(state, "recent_interaction_mode"):
        try:
            return str(state.recent_interaction_mode(""))
        except Exception:
            return ""
    return ""


def _extract_keywords(text: str, keyword_lists: Dict[str, List[str]]) -> List[str]:
    normalized = normalize_turn(text)
    found = set()
    for keywords in keyword_lists.values():
        for kw in keywords:
            if kw in normalized:
                found.add(kw)
    return sorted(found)


def detect_topic_rule(question: str) -> str:
    normalized = normalize_turn(question)
    if any(cue in normalized for cue in COST_OF_LIVING_CUES):
        return "cost_of_living"
    if any(cue in normalized for cue in TRANSPORT_CUES):
        return "transport"
    if any(cue in normalized for cue in FOOD_CUES):
        return "food_delivery"
    if any(cue in normalized for cue in HOUSING_CUES):
        return "housing"
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic == "general":
            continue
        if any(kw in normalized for kw in keywords):
            return topic
    return "general"


def detect_domain_rule(question: str) -> str:
    normalized = normalize_turn(question)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            return domain
    return "general"


def _get_location_terms() -> List[str]:
    try:
        from config.domain import get_domain_config

        cfg = get_domain_config()
        if cfg.location_terms:
            return list(cfg.location_terms)
    except Exception:
        pass
    return ["area", "district", "neighborhood", "where", "location", "local", "nearby"]


def is_location_related_rule(question: str) -> bool:
    normalized = normalize_turn(question)
    return any(term in normalized for term in _get_location_terms())


def _load_topic_to_model_key() -> Dict[str, Dict[str, str]]:
    try:
        from config.domain import get_domain_config

        cfg = get_domain_config()
        if cfg.topic_to_model_key:
            return cfg.topic_to_model_key
    except Exception:
        pass
    return {}


_TOPIC_TO_MODEL_KEY: Dict[str, Dict[str, str]] = _load_topic_to_model_key()


def resolve_model_key_rule(topic: str, question_type: str) -> str:
    if question_type == "open_text":
        return "generic_open_text"
    if question_type == "duration":
        return "generic_duration"
    if question_type == "nps":
        return "nps_recommendation"
    if question_type == "policy_support":
        return "policy_support"
    if question_type == "likelihood":
        return "tech_adoption_likelihood"
    if topic == "cost_of_living":
        return "cost_of_living_satisfaction"
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


def allow_persona_anchor_rule(question: str, topic: Optional[str] = None, domain: Optional[str] = None) -> bool:
    normalized = normalize_turn(question)
    if any(keyword in normalized for keyword in ANCHOR_EXCLUDE_KEYWORDS):
        return False
    if topic in {"transport", "food_delivery"}:
        return True
    if topic in {"housing", "cost_of_living"}:
        return any(keyword in normalized for keyword in LIFESTYLE_KEYWORDS)
    if domain == "technology":
        return False
    return any(keyword in normalized for keyword in LIFESTYLE_KEYWORDS)


def infer_action_template_rule(
    question: str,
    question_type: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    normalized = normalize_turn(question)
    for patterns, template in ACTION_PATTERN_RULES:
        if any(pattern in normalized for pattern in patterns):
            return template
    qtype = question_type or detect_question_type(question)
    if qtype == "frequency":
        return "frequency", "behavior", "ordinal"
    if qtype == "policy_support":
        return "support", "policy", "ordinal"
    if qtype in {"likert", "nps"}:
        return "rate", "experience", "continuous"
    if qtype == "likelihood":
        return "adopt", "product", "binary"
    return "choose", "behavior", "ordinal"


def build_intent_router_payload(question: str, state: Optional["AgentState"] = None) -> Dict[str, Any]:
    normalized = normalize_turn(question)
    previous_mode = _previous_mode(state)
    recent_utterances: List[str] = []
    recent_modes: List[str] = []
    if state is not None:
        recent_utterances = list(getattr(state, "recent_utterances", [])[-4:])
        recent_modes = list(getattr(state, "recent_interaction_modes", [])[-4:])
    return {
        "question": question,
        "normalized_question": normalized,
        "previous_mode": previous_mode or "none",
        "recent_utterances": recent_utterances,
        "recent_modes": recent_modes,
        "signals": {
            "low_information": is_low_information_turn(question),
            "explicit_survey": looks_like_explicit_survey(question),
            "qualitative_interview": looks_like_qualitative_interview(question),
            "question_type": detect_question_type(question),
            "word_count": len(normalized.split()) if normalized else 0,
        },
    }


def classify_interaction_mode_rules(question: str, state: Optional["AgentState"] = None) -> tuple[str, float, str]:
    normalized = normalize_turn(question)
    previous_mode = _previous_mode(state)

    if looks_like_explicit_survey(question):
        return "survey", 0.94, "explicit_survey_pattern"
    if normalized in GREETING_CUES or normalized in CONVERSATION_CUES:
        return "conversation", 0.97, "direct_conversation_cue"
    if normalized in FOLLOW_UP_CUES:
        if previous_mode in {"qualitative_interview", "conversation"}:
            return previous_mode, 0.78, "follow_up_inherits_previous_mode"
        return "conversation", 0.72, "follow_up_defaults_to_conversation"
    if looks_like_qualitative_interview(question):
        return "qualitative_interview", 0.86, "qualitative_interview_pattern"
    if previous_mode in {"qualitative_interview", "conversation"} and not looks_like_explicit_survey(question):
        return previous_mode, 0.68, "inherits_previous_mode"
    if detect_question_type(question) == "open_text":
        return "qualitative_interview", 0.70, "open_text_question_type"
    return "survey", 0.55, "survey_fallback"


def build_turn_understanding_rules(
    question: str,
    state: Optional["AgentState"] = None,
    options: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized = normalize_turn(question)
    question_type = detect_question_type(question)
    topic = detect_topic_rule(question)
    domain = detect_domain_rule(question)
    interaction_mode, interaction_confidence, interaction_reason = classify_interaction_mode_rules(question, state)
    scale_type = QUESTION_TYPE_TO_SCALE_TYPE.get(question_type, "categorical")
    if options:
        option_scale_type = infer_scale_type_from_options(options)
        if option_scale_type != "open_text":
            scale_type = option_scale_type
    question_model_key_candidate = resolve_model_key_rule(topic, question_type)
    if interaction_mode != "survey":
        question_type = "open_text"
        scale_type = "open_text"
        question_model_key_candidate = "generic_open_text"
    action_type, target, intensity_scale = infer_action_template_rule(question, question_type)
    keywords = sorted(set(_extract_keywords(question, {**TOPIC_KEYWORDS, **DOMAIN_KEYWORDS})))
    payload = build_intent_router_payload(question, state)
    payload["rule_reason"] = interaction_reason
    payload["rule_topic"] = topic
    payload["rule_domain"] = domain
    payload["rule_model_key"] = question_model_key_candidate
    payload["rule_scale_type"] = scale_type
    payload["provided_options"] = list(options or [])
    return {
        "question": question,
        "normalized_question": normalized,
        "interaction_mode": interaction_mode,
        "question_type": question_type,
        "scale_type": scale_type,
        "topic": topic,
        "domain": domain,
        "location_related": is_location_related_rule(question),
        "question_model_key_candidate": question_model_key_candidate,
        "persona_anchor_allowed": allow_persona_anchor_rule(question, topic=topic, domain=domain),
        "action_type_candidate": action_type,
        "target_candidate": target,
        "intensity_scale_candidate": intensity_scale,
        "normalization_candidates": [],
        "keywords": keywords,
        "rule_confidence": interaction_confidence,
        "payload": payload,
    }


def _strip_fence(text: str) -> str:
    content = str(text or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return content.strip()


def _sanitize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _sanitize_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _sanitize_understanding_payload(parsed: Dict[str, Any], rule_understanding: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from config.question_models import QUESTION_MODELS, load_generated_models_into_registry

        load_generated_models_into_registry()
        valid_model_keys = set(QUESTION_MODELS)
    except Exception:
        valid_model_keys = set()

    llm_question_type = str(parsed.get("question_type", rule_understanding["question_type"])).strip().lower()
    if llm_question_type not in QUESTION_TYPES:
        llm_question_type = rule_understanding["question_type"]

    llm_scale_type = str(parsed.get("scale_type", rule_understanding["scale_type"])).strip().lower()
    if llm_scale_type not in SCALE_TYPES:
        llm_scale_type = QUESTION_TYPE_TO_SCALE_TYPE.get(llm_question_type, rule_understanding["scale_type"])

    llm_interaction_mode = str(parsed.get("interaction_mode", rule_understanding["interaction_mode"])).strip().lower()
    if llm_interaction_mode not in INTERACTION_MODES:
        llm_interaction_mode = rule_understanding["interaction_mode"]

    llm_topic = str(parsed.get("topic", rule_understanding["topic"])).strip().lower()
    if llm_topic not in TOPIC_NAMES:
        llm_topic = rule_understanding["topic"]

    llm_domain = str(parsed.get("domain", rule_understanding["domain"])).strip().lower()
    if llm_domain not in DOMAIN_NAMES:
        llm_domain = rule_understanding["domain"]

    model_key = str(parsed.get("question_model_key_candidate", rule_understanding["question_model_key_candidate"])).strip()
    if valid_model_keys and model_key and model_key not in valid_model_keys:
        model_key = rule_understanding["question_model_key_candidate"]

    action_type = str(parsed.get("action_type_candidate", rule_understanding["action_type_candidate"] or "")).strip().lower() or None
    if action_type not in ACTION_TYPES:
        action_type = rule_understanding["action_type_candidate"]

    target = str(parsed.get("target_candidate", rule_understanding["target_candidate"] or "")).strip().lower() or None
    if target not in TARGET_TYPES:
        target = rule_understanding["target_candidate"]

    intensity_scale = str(parsed.get("intensity_scale_candidate", rule_understanding["intensity_scale_candidate"] or "")).strip().lower() or None
    if intensity_scale not in INTENSITY_SCALES:
        intensity_scale = rule_understanding["intensity_scale_candidate"]

    confidence = float(parsed.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "interaction_mode": llm_interaction_mode,
        "question_type": llm_question_type,
        "scale_type": llm_scale_type,
        "topic": llm_topic,
        "domain": llm_domain,
        "location_related": _sanitize_bool(parsed.get("location_related"), rule_understanding["location_related"]),
        "question_model_key_candidate": model_key,
        "persona_anchor_allowed": _sanitize_bool(parsed.get("persona_anchor_allowed"), rule_understanding["persona_anchor_allowed"]),
        "action_type_candidate": action_type,
        "target_candidate": target,
        "intensity_scale_candidate": intensity_scale,
        "normalization_candidates": _sanitize_list(parsed.get("normalization_candidates")),
        "confidence": confidence,
        "reason": str(parsed.get("reason", "")).strip() or "llm_understanding",
    }


def _truncate_provided_options_for_llm(rule_understanding: Dict[str, Any], max_each: int = 400) -> Dict[str, Any]:
    """Shrink long survey options in the LLM payload only (rules already used full text)."""
    ru = {**rule_understanding}
    opts = list(ru.get("provided_options") or [])
    ru["provided_options"] = [
        o if len(o) <= max_each else (o[: max(1, max_each - 1)] + "…")
        for o in opts
    ]
    pl = ru.get("payload")
    if isinstance(pl, dict):
        pl2 = {**pl, "provided_options": ru["provided_options"]}
        ru["payload"] = pl2
    return ru


async def classify_turn_understanding_llm(
    question: str,
    state: Optional["AgentState"] = None,
    options: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rule_understanding = build_turn_understanding_rules(question, state=state, options=options)
    for_llm = _truncate_provided_options_for_llm(rule_understanding)
    prompt = (
        "Classify this user input using the provided rule signals.\n"
        "Return ONLY strict JSON with these keys:\n"
        "- interaction_mode\n"
        "- question_type\n"
        "- scale_type\n"
        "- topic\n"
        "- domain\n"
        "- location_related\n"
        "- question_model_key_candidate\n"
        "- persona_anchor_allowed\n"
        "- action_type_candidate\n"
        "- target_candidate\n"
        "- intensity_scale_candidate\n"
        "- normalization_candidates\n"
        "- confidence\n"
        "- reason\n\n"
        "Allowed interaction_mode: conversation, qualitative_interview, survey\n"
        "Allowed question_type: duration, open_text, nps, policy_support, frequency, likelihood, likert, categorical\n"
        "Allowed scale_type: duration, open_text, nps, policy_support, frequency, likelihood, likert, categorical, numeric\n"
        "If the turn is not a structured survey question, prefer open_text/generic_open_text semantics.\n\n"
        f"Payload:\n{json.dumps(for_llm, ensure_ascii=False)}"
    )
    try:
        client = get_llm_client()
        response = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
        )
        parsed = json.loads(_strip_fence(response))
        return _sanitize_understanding_payload(parsed, rule_understanding)
    except Exception:
        return {
            "interaction_mode": rule_understanding["interaction_mode"],
            "question_type": rule_understanding["question_type"],
            "scale_type": rule_understanding["scale_type"],
            "topic": rule_understanding["topic"],
            "domain": rule_understanding["domain"],
            "location_related": rule_understanding["location_related"],
            "question_model_key_candidate": rule_understanding["question_model_key_candidate"],
            "persona_anchor_allowed": rule_understanding["persona_anchor_allowed"],
            "action_type_candidate": rule_understanding["action_type_candidate"],
            "target_candidate": rule_understanding["target_candidate"],
            "intensity_scale_candidate": rule_understanding["intensity_scale_candidate"],
            "normalization_candidates": [],
            "confidence": 0.0,
            "reason": "llm_understanding_failed",
        }


def _fuse_value(
    *,
    rule_value: Any,
    llm_value: Any,
    rule_confidence: float,
    llm_confidence: float,
    sticky_rule_values: Optional[set[Any]] = None,
) -> Any:
    if sticky_rule_values and rule_value in sticky_rule_values:
        return rule_value
    if llm_value in (None, "", [], {}):
        return rule_value
    if rule_value == llm_value:
        return rule_value
    if llm_confidence >= 0.88:
        return llm_value
    if rule_confidence >= 0.82:
        return rule_value
    if llm_confidence > rule_confidence + 0.1:
        return llm_value
    return rule_value


def combine_intent_modes(
    *,
    question: str,
    rule_mode: str,
    rule_confidence: float,
    llm_mode: str,
    llm_confidence: float,
    previous_mode: str = "",
) -> tuple[str, str]:
    normalized = normalize_turn(question)
    if normalized in GREETING_CUES or normalized in CONVERSATION_CUES:
        return "conversation", "hard_conversation_override"
    if looks_like_explicit_survey(question) and rule_confidence >= 0.9:
        return "survey", "hard_survey_override"
    if rule_mode == llm_mode:
        return rule_mode, "rule_llm_agree"
    if llm_confidence >= 0.85 and llm_mode in INTERACTION_MODES:
        return llm_mode, "llm_high_confidence_override"
    if rule_confidence >= 0.8:
        return rule_mode, "rule_high_confidence_override"
    if previous_mode in {"conversation", "qualitative_interview"} and llm_mode == previous_mode:
        return llm_mode, "llm_matches_session_mode"
    if llm_confidence > rule_confidence + 0.1 and llm_mode in INTERACTION_MODES:
        return llm_mode, "llm_weighted_win"
    return rule_mode, "rule_weighted_win"


_OPTION_KEY_HASH_LEN = 2400


def _options_cache_segment(options: Optional[List[str]]) -> str:
    parts = [_normalize_token(opt) for opt in (options or [])]
    raw = "|".join(parts)
    if len(raw) > _OPTION_KEY_HASH_LEN:
        digest = hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"h:{digest}"
    return raw


def _build_cache_key(
    question: str,
    state: Optional["AgentState"],
    options: Optional[List[str]],
    survey_run_id: str = "",
    question_id: str = "",
) -> str:
    previous_mode = _previous_mode(state) or "none"
    option_key = _options_cache_segment(options)
    ns = (survey_run_id or "").strip()
    if not ns and state is not None:
        ns = str(getattr(state, "survey_run_id", "") or "").strip()
    qid = (question_id or "").strip()
    if not qid and state is not None:
        qid = str(getattr(state, "nlu_question_id", "") or "").strip()
    return f"{normalize_turn(question)}::{previous_mode}::{option_key}::{ns}::{qid}"


def clear_turn_understanding_cache() -> None:
    """Clear hybrid turn-understanding cache (tests, notebooks, multi-survey processes).

    Cache keys include question text, options (or hash), survey_run_id, and nlu_question_id.
    Enable clear_turn_understanding_cache_on_survey_start or pass fresh question_id when
    reusing identical wording across batches in one process.
    """
    _TURN_UNDERSTANDING_CACHE.clear()


async def build_turn_understanding_hybrid(
    question: str,
    state: Optional["AgentState"] = None,
    options: Optional[List[str]] = None,
    *,
    use_cache: bool = True,
    survey_run_id: str = "",
    question_id: str = "",
) -> TurnUnderstandingResult:
    cache_key = _build_cache_key(
        question, state, options, survey_run_id=survey_run_id, question_id=question_id,
    )
    if use_cache and cache_key in _TURN_UNDERSTANDING_CACHE:
        return _TURN_UNDERSTANDING_CACHE[cache_key]

    rule_understanding = build_turn_understanding_rules(question, state=state, options=options)
    llm_understanding = await classify_turn_understanding_llm(question, state=state, options=options)
    final_mode, mode_reason = combine_intent_modes(
        question=question,
        rule_mode=str(rule_understanding["interaction_mode"]),
        rule_confidence=float(rule_understanding["rule_confidence"]),
        llm_mode=str(llm_understanding["interaction_mode"]),
        llm_confidence=float(llm_understanding["confidence"]),
        previous_mode=rule_understanding["payload"].get("previous_mode", ""),
    )
    final_question_type = _fuse_value(
        rule_value=rule_understanding["question_type"],
        llm_value=llm_understanding["question_type"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
        sticky_rule_values={"duration", "nps", "policy_support", "frequency", "likelihood", "likert"},
    )
    final_scale_type = _fuse_value(
        rule_value=rule_understanding["scale_type"],
        llm_value=llm_understanding["scale_type"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
        sticky_rule_values={"duration", "nps", "policy_support", "frequency", "likelihood", "likert", "numeric"},
    )
    final_topic = _fuse_value(
        rule_value=rule_understanding["topic"],
        llm_value=llm_understanding["topic"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    )
    final_domain = _fuse_value(
        rule_value=rule_understanding["domain"],
        llm_value=llm_understanding["domain"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    )
    final_model_key = _fuse_value(
        rule_value=rule_understanding["question_model_key_candidate"],
        llm_value=llm_understanding["question_model_key_candidate"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    )
    final_anchor_allowed = bool(_fuse_value(
        rule_value=rule_understanding["persona_anchor_allowed"],
        llm_value=llm_understanding["persona_anchor_allowed"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    ))
    final_action = _fuse_value(
        rule_value=rule_understanding["action_type_candidate"],
        llm_value=llm_understanding["action_type_candidate"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    )
    final_target = _fuse_value(
        rule_value=rule_understanding["target_candidate"],
        llm_value=llm_understanding["target_candidate"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    )
    final_intensity = _fuse_value(
        rule_value=rule_understanding["intensity_scale_candidate"],
        llm_value=llm_understanding["intensity_scale_candidate"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    )
    final_location_related = bool(_fuse_value(
        rule_value=rule_understanding["location_related"],
        llm_value=llm_understanding["location_related"],
        rule_confidence=rule_understanding["rule_confidence"],
        llm_confidence=llm_understanding["confidence"],
    ))
    normalization_candidates = list(llm_understanding.get("normalization_candidates") or [])

    if final_mode != "survey":
        final_question_type = "open_text"
        final_scale_type = "open_text"
        final_model_key = "generic_open_text"

    intent_cls = classify_intent_class(question, state)
    ref_ctx = resolve_reference(question, state)

    result = TurnUnderstandingResult(
        question=question,
        normalized_question=rule_understanding["normalized_question"],
        interaction_mode=final_mode,
        question_type=str(final_question_type),
        scale_type=str(final_scale_type),
        topic=str(final_topic),
        domain=str(final_domain),
        location_related=final_location_related,
        question_model_key_candidate=str(final_model_key),
        persona_anchor_allowed=final_anchor_allowed,
        action_type_candidate=str(final_action) if final_action else None,
        target_candidate=str(final_target) if final_target else None,
        intensity_scale_candidate=str(final_intensity) if final_intensity else None,
        normalization_candidates=normalization_candidates,
        keywords=list(rule_understanding["keywords"]),
        rule_confidence=float(rule_understanding["rule_confidence"]),
        llm_confidence=float(llm_understanding["confidence"]),
        final_confidence=max(float(rule_understanding["rule_confidence"]), float(llm_understanding["confidence"]), 0.70),
        fusion_reason=mode_reason if final_mode != rule_understanding["interaction_mode"] else "shared_hybrid_understanding",
        rule_payload=rule_understanding["payload"],
        llm_payload=llm_understanding,
        intent_class=intent_cls.value,
        resolved_reference=ref_ctx,
    )
    if use_cache:
        _TURN_UNDERSTANDING_CACHE[cache_key] = result
    return result


async def classify_interaction_mode_hybrid(
    question: str,
    state: Optional["AgentState"] = None,
    *,
    options: Optional[List[str]] = None,
    survey_run_id: str = "",
    question_id: str = "",
) -> IntentRouterResult:
    understanding = await build_turn_understanding_hybrid(
        question,
        state=state,
        options=options,
        survey_run_id=survey_run_id,
        question_id=question_id,
    )
    rule_understanding = build_turn_understanding_rules(question, state=state, options=options)
    return IntentRouterResult(
        rule_mode=str(rule_understanding["interaction_mode"]),
        llm_mode=str(understanding.llm_payload.get("interaction_mode", understanding.interaction_mode)),
        final_mode=understanding.interaction_mode,
        rule_confidence=understanding.rule_confidence,
        llm_confidence=understanding.llm_confidence,
        reason=understanding.fusion_reason,
        payload={
            **understanding.rule_payload,
            "topic": understanding.topic,
            "domain": understanding.domain,
            "question_type": understanding.question_type,
            "scale_type": understanding.scale_type,
            "question_model_key_candidate": understanding.question_model_key_candidate,
        },
    )
