"""Adaptive question-structure generation for survey questions."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config.generated_registry import (
    get_cached_model_key_for_question,
    load_generated_registry,
    save_generated_registry,
)
from config.question_models import QUESTION_MODELS, QuestionModel, load_generated_models_into_registry
from llm.client import get_llm_client

_ALLOWED_FACTORS = ("personality", "income", "social", "location", "memory", "behavioral", "belief")
_ALLOWED_SEMANTIC_PROFILES = (
    "economic_pressure",
    "health_behavior",
    "policy_opinion",
    "safety_perception",
    "social_trust",
    "lifestyle_frequency",
    "generic_attitude",
)
_ALLOWED_GENERATION_CLASSES = (
    "survey_model",
    "not_a_survey_question",
    "conversation_only",
    "qualitative_only",
)
_VALID_FILTER_KEYS = {
    "age",
    "nationality",
    "income",
    "location",
    "occupation",
    "household_size",
    "spouse",
    "children",
    "car",
    "metro_usage",
    "cuisine_preference",
    "diet",
    "hobby",
    "work_schedule",
    "typical_dinner_time",
    "commute_method",
    "health_focus",
    "archetype",
}


def question_hash(question: str) -> str:
    import hashlib

    normalized = " ".join(question.lower().strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _strip_fence(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return content.strip()


def _normalize_scale(scale: Any) -> List[str]:
    if not isinstance(scale, list):
        return []
    out = [str(v).strip() for v in scale if str(v).strip()]
    return out


def _normalize_reference(reference: Any, scale: List[str]) -> Dict[str, float]:
    if not isinstance(reference, dict) or not scale:
        return {}
    vals = {k: float(reference.get(k, 0.0)) for k in scale}
    vals = {k: (v if v > 0 else 0.0) for k, v in vals.items()}
    total = sum(vals.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in vals.items()}


def _is_non_uniform(reference: Dict[str, float]) -> bool:
    if len(reference) < 2:
        return False
    n = len(reference)
    uniform = 1.0 / n
    max_dev = max(abs(v - uniform) for v in reference.values())
    return max_dev >= 0.03


def _factor_weights_from_names(factors: Any) -> Dict[str, float]:
    base = {
        "personality": 0.32,
        "income": 0.12,
        "social": 0.10,
        "location": 0.10,
        "memory": 0.08,
        "behavioral": 0.16,
        "belief": 0.12,
    }
    if not isinstance(factors, list):
        return base
    selected = [str(f).strip().lower() for f in factors if str(f).strip().lower() in _ALLOWED_FACTORS]
    if not selected:
        return base
    weighted = {k: 0.04 for k in _ALLOWED_FACTORS}
    pool = 1.0 - sum(weighted.values())
    step = pool / len(selected)
    for key in selected:
        weighted[key] += step
    total = sum(weighted.values())
    return {k: v / total for k, v in weighted.items()}


def _normalize_dominant_factors(factors: Any) -> List[str]:
    if not isinstance(factors, list):
        return []
    out = []
    for factor in factors:
        name = str(factor).strip().lower()
        if name in _ALLOWED_FACTORS and name not in out:
            out.append(name)
    return out


def _normalize_semantic_profile(value: Any, question: str) -> str:
    profile = str(value or "").strip().lower()
    if profile in _ALLOWED_SEMANTIC_PROFILES:
        return profile
    normalized = question.lower()
    if any(token in normalized for token in ("cost of living", "afford", "expense", "price")):
        return "economic_pressure"
    if any(token in normalized for token in ("exercise", "work out", "gym", "fitness", "physical activity")):
        return "health_behavior"
    if any(token in normalized for token in ("support", "oppose", "policy", "government", "law", "trust")):
        return "policy_opinion"
    if any(token in normalized for token in ("safe", "safety", "crime", "walking alone")):
        return "safety_perception"
    return "generic_attitude"


def _normalize_narrative_guidance(value: Any, scale: List[str]) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for key, guidance in value.items():
        k = str(key)
        if k in scale and isinstance(guidance, str) and guidance.strip():
            out[k] = guidance.strip()
    return out


def _normalize_generation_classification(value: Any) -> str:
    classification = str(value or "").strip().lower()
    if classification in _ALLOWED_GENERATION_CLASSES:
        return classification
    return "survey_model"


def _rebalance_reference(reference: Dict[str, float], scale: List[str], semantic_profile: str) -> Dict[str, float]:
    if not reference or len(scale) != 5:
        return reference
    middle = scale[len(scale) // 2]
    middle_val = float(reference.get(middle, 0.0))
    if semantic_profile == "safety_perception" and middle_val > 0.22:
        excess = middle_val - 0.22
        updated = dict(reference)
        updated[middle] = 0.22
        updated[scale[0]] = updated.get(scale[0], 0.0) + excess * 0.35
        updated[scale[1]] = updated.get(scale[1], 0.0) + excess * 0.15
        updated[scale[3]] = updated.get(scale[3], 0.0) + excess * 0.25
        updated[scale[4]] = updated.get(scale[4], 0.0) + excess * 0.25
        return _normalize_reference(updated, scale)
    return reference


def _expand_constraints(raw: Any, model_key: str, scale: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    valid_options = set(scale)
    for item in raw:
        if not isinstance(item, dict):
            continue
        filters = item.get("if", {})
        disallow = item.get("disallow", [])
        if not isinstance(filters, dict) or not isinstance(disallow, list):
            continue
        clean_filters = {}
        for key, value in filters.items():
            k = str(key)
            if k in _VALID_FILTER_KEYS:
                clean_filters[k] = value
        if not clean_filters:
            continue
        for option in disallow:
            opt = str(option)
            if opt in valid_options:
                out.append(
                    {
                        "question_model_key": model_key,
                        "filters": clean_filters,
                        "option": opt,
                        "severity": "hard",
                        "source": "adaptive",
                    }
                )
    return out


def _build_llm_prompt(question: str) -> str:
    return f"""
You are designing a deterministic survey model configuration for a synthetic population engine.
Return ONLY strict JSON.

If the input is casual conversation, a greeting, a follow-up fragment, or a qualitative interview turn rather than a real survey question, return:
- classification

Allowed classification values:
- survey_model
- not_a_survey_question
- conversation_only
- qualitative_only

If classification is survey_model, include these keys:
- model_key (snake_case string)
- scale (array of answer options, at least 2 items; use strings for all values)
- reference_distribution (object from scale option -> probability; must sum to 1.0)
- factors (array using only: personality,income,social,location,memory,behavioral,belief)
- dominant_factors (array using only the same factor names, ordered strongest to weakest)
- semantic_profile (one of: economic_pressure,health_behavior,policy_opinion,safety_perception,social_trust,lifestyle_frequency,generic_attitude)
- narrative_guidance (object of option -> short tone/stance hint)
- constraints (array of objects with keys: if (object of persona filters), disallow (array of options from scale))

Question: "{question}"

Rules:
- If this is not a true survey question, do NOT invent a scale.
- Keep distribution realistic and non-uniform.
- Use a 5-point numeric scale if the question is satisfaction/opinion without explicit options.
- Constraints must only use these filter keys if needed:
  age,nationality,income,location,occupation,household_size,spouse,children,car,metro_usage,cuisine_preference,diet,hobby,work_schedule,typical_dinner_time,commute_method,health_focus,archetype
- Do not output explanations, markdown, or code fences. JSON only.
""".strip()


def get_cached_generated_model_key(question: str) -> str:
    load_generated_models_into_registry()
    mapped = get_cached_model_key_for_question(question_hash(question))
    return mapped if mapped in QUESTION_MODELS else ""


async def adaptive_generate_and_register(
    question: str,
    *,
    fallback_model_key: Optional[str] = None,
) -> Optional[str]:
    """Generate a model for a question and persist it for reuse."""
    qh = question_hash(question)
    load_generated_models_into_registry()
    mapped = get_cached_generated_model_key(question)
    if mapped:
        return mapped

    client = get_llm_client()
    response = await client.chat(
        [{"role": "user", "content": _build_llm_prompt(question)}],
        temperature=0.0,
        max_tokens=900,
    )
    try:
        parsed = json.loads(_strip_fence(response))
    except Exception:
        return None

    classification = _normalize_generation_classification(parsed.get("classification"))
    if classification != "survey_model":
        return None

    model_key = str(parsed.get("model_key", "")).strip()
    scale = _normalize_scale(parsed.get("scale"))
    if not model_key or len(scale) < 2:
        return fallback_model_key
    model_key = model_key.lower().replace(" ", "_")
    reference = _normalize_reference(parsed.get("reference_distribution"), scale)
    if not reference or not _is_non_uniform(reference):
        return fallback_model_key

    factors = _normalize_dominant_factors(parsed.get("factors", []))
    dominant_factors = _normalize_dominant_factors(parsed.get("dominant_factors", factors))
    factor_weights = _factor_weights_from_names(factors)
    semantic_profile = _normalize_semantic_profile(parsed.get("semantic_profile"), question)
    reference = _rebalance_reference(reference, scale, semantic_profile)
    narrative_guidance = _normalize_narrative_guidance(parsed.get("narrative_guidance"), scale)
    model_payload = {
        "name": model_key,
        "scale": scale,
        "dimension_weights": {},
        "factor_weights": factor_weights,
        "temperature": 1.0,
        "source": "adaptive",
        "dominant_factors": dominant_factors,
        "semantic_profile": semantic_profile,
        "narrative_guidance": narrative_guidance,
    }

    # Register in-memory immediately.
    QUESTION_MODELS[model_key] = QuestionModel(
        name=model_key,
        scale=scale,
        dimension_weights={},
        factor_weights=factor_weights,
        temperature=1.0,
    )

    constraints = _expand_constraints(parsed.get("constraints", []), model_key, scale)

    reg = load_generated_registry()
    reg.setdefault("models", {})[model_key] = model_payload
    reg.setdefault("references", {})[model_key] = reference
    reg.setdefault("constraints", {})[model_key] = constraints
    reg.setdefault("question_hash_map", {})[qh] = model_key
    save_generated_registry(reg)
    return model_key

