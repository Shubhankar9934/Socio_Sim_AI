"""
Perception layer: extract topic, domain, scale type, and question-model key
from survey questions.  Rule-based keyword matching with an optional
LLM-assisted classification fallback for unknown questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from config.question_models import (
    GENERIC_DURATION,
    GENERIC_FALLBACK,
    GENERIC_FREQUENCY,
    GENERIC_LIKERT,
    GENERIC_OPEN_TEXT,
    QUESTION_MODELS,
    QuestionModel,
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
    "chance", "support", "oppose",
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


@dataclass
class Perception:
    """Structured perception of a survey question."""

    topic: str
    domain: str
    location_related: bool
    keywords: List[str]
    raw_question: str
    scale_type: str = "categorical"
    question_model_key: str = "generic"


# ── Internal helpers ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.lower().strip()


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


def _detect_scale_type(question: str) -> str:
    normalized = _normalize(question)
    if any(cue in normalized for cue in _DURATION_CUES):
        return "duration"
    if any(cue in normalized for cue in _FREQUENCY_CUES):
        return "frequency"
    if any(cue in normalized for cue in _SATISFACTION_CUES):
        return "likert"
    if any(cue in normalized for cue in _LIKELIHOOD_CUES):
        return "likert"
    if any(cue in normalized for cue in _OPEN_ENDED_CUES):
        return "open_text"
    return "categorical"


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


def _resolve_model_key(topic: str, scale_type: str) -> str:
    if scale_type == "open_text":
        return "generic_open_text"
    if scale_type == "duration":
        return "generic_duration"
    topic_map = _TOPIC_TO_MODEL_KEY.get(topic)
    if topic_map:
        return topic_map.get(scale_type, next(iter(topic_map.values())))

    if topic == "satisfaction" or scale_type == "likert":
        return "generic_likert"
    if scale_type == "frequency":
        return "generic_frequency"
    return "generic_likert"


# ── Public API ──────────────────────────────────────────────────────────

def perceive(question: str) -> Perception:
    """Extract structured perception from a survey question."""
    topic = _detect_topic(question)
    domain = _detect_domain(question)
    location_related = _is_location_related(question)
    all_keywords = {**TOPIC_KEYWORDS, **DOMAIN_KEYWORDS}
    keywords = list(_extract_keywords(question, all_keywords))
    scale_type = _detect_scale_type(question)
    model_key = _resolve_model_key(topic, scale_type)

    return Perception(
        topic=topic,
        domain=domain,
        location_related=location_related,
        keywords=keywords,
        raw_question=question,
        scale_type=scale_type,
        question_model_key=model_key,
    )


def detect_question_model(perception: Perception) -> QuestionModel:
    """Map a Perception to the appropriate QuestionModel.

    Resolution order:
      1. Open-text scale type -> GENERIC_OPEN_TEXT.
      2. Exact key in QUESTION_MODELS registry.
      3. Generic frequency / likert based on scale_type.
      4. GENERIC_FALLBACK (always works).
    """
    if perception.scale_type == "open_text":
        return GENERIC_OPEN_TEXT
    if perception.scale_type == "duration":
        return GENERIC_DURATION
    model = QUESTION_MODELS.get(perception.question_model_key)
    if model is not None:
        return model

    if perception.scale_type == "frequency":
        return GENERIC_FREQUENCY
    if perception.scale_type == "likert":
        return GENERIC_LIKERT
    return GENERIC_FALLBACK


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
        QUESTION_DIMENSION_MAP.setdefault(question, result)
        return result
    except Exception:
        return {}


async def perceive_with_llm(question: str) -> Perception:
    """Perceive with LLM fallback for dimension weights on unknown questions.

    If keyword matching resolves to a 'general' topic, tries the LLM to
    infer behavioral dimension weights.  The weights are stored in the
    QUESTION_DIMENSION_MAP cache for subsequent calls.
    """
    perception = perceive(question)
    if perception.topic == "general":
        model_key = await classify_question_via_llm(question)
        if model_key:
            perception.question_model_key = model_key
        weights = await infer_dimension_weights_via_llm(question)
        if weights:
            from config.question_models import QUESTION_DIMENSION_MAP
            QUESTION_DIMENSION_MAP[perception.question_model_key] = weights
    return perception
