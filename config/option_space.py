"""Canonical option-space registry and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

from llm.client import get_llm_client


@dataclass(frozen=True)
class OptionSpace:
    key: str
    canonical_options: List[str]
    aliases: Dict[str, str]


_FREQUENCY_CANONICAL = ["rarely", "1-2 per week", "3-4 per week", "daily", "multiple per day"]
_GENERIC_FREQUENCY_CANONICAL = ["never", "rarely", "sometimes", "often", "very often"]

OPTION_SPACES: Dict[str, OptionSpace] = {
    "food_delivery_frequency": OptionSpace(
        key="food_delivery_frequency",
        canonical_options=_FREQUENCY_CANONICAL,
        aliases={
            "never": "rarely",
            "very often": "multiple per day",
            "often": "daily",
            "sometimes": "3-4 per week",
            "rarely": "rarely",
            "1-2/week": "1-2 per week",
            "3-4/week": "3-4 per week",
            "multiple/day": "multiple per day",
        },
    ),
    "generic_frequency": OptionSpace(
        key="generic_frequency",
        canonical_options=_GENERIC_FREQUENCY_CANONICAL,
        aliases={
            "multiple per day": "very often",
            "daily": "often",
            "3-4 per week": "sometimes",
            "1-2 per week": "rarely",
            "rarely": "rarely",
            "never": "never",
        },
    ),
}

_HYBRID_ALIAS_CACHE_PATH = Path("data/option_space_cache.json")


def _load_hybrid_alias_cache() -> Dict[str, Dict[str, str]]:
    if _HYBRID_ALIAS_CACHE_PATH.exists():
        try:
            return json.loads(_HYBRID_ALIAS_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_hybrid_alias_cache(cache: Dict[str, Dict[str, str]]) -> None:
    try:
        _HYBRID_ALIAS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HYBRID_ALIAS_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


_HYBRID_ALIAS_CACHE: Dict[str, Dict[str, str]] = _load_hybrid_alias_cache()


def _normalize_token(v: str) -> str:
    return " ".join((v or "").strip().lower().split())


def get_option_space_key(question_model_key: str) -> str:
    """Return canonical option-space key for a question model key."""
    if question_model_key in OPTION_SPACES:
        return question_model_key
    return question_model_key


def canonicalize_option(question_model_key: str, option: str) -> str:
    """Canonicalize one option value for a question model."""
    space = OPTION_SPACES.get(get_option_space_key(question_model_key))
    if not space:
        return option
    normalized = _normalize_token(option)
    for canonical in space.canonical_options:
        if _normalize_token(canonical) == normalized:
            return canonical
    dynamic_aliases = _HYBRID_ALIAS_CACHE.get(space.key, {})
    mapped = dynamic_aliases.get(normalized)
    if mapped is None:
        mapped = space.aliases.get(normalized)
    return mapped if mapped is not None else option


def canonicalize_distribution(question_model_key: str, dist: Dict[str, float]) -> Dict[str, float]:
    """Canonicalize and merge distribution entries by mapped option labels."""
    if not dist:
        return {}
    out: Dict[str, float] = {}
    for k, v in dist.items():
        ck = canonicalize_option(question_model_key, k)
        out[ck] = out.get(ck, 0.0) + float(v)
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def canonicalize_options(question_model_key: str, options: List[str]) -> List[str]:
    """Canonicalize provided option list while preserving order uniqueness."""
    seen = set()
    result: List[str] = []
    for opt in options or []:
        c = canonicalize_option(question_model_key, opt)
        key = _normalize_token(c)
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
    return result


def canonicalize_observed_and_reference(
    question_model_key: str,
    observed: Dict[str, float],
    reference: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Canonicalize both observed and reference distributions."""
    return (
        canonicalize_distribution(question_model_key, observed),
        canonicalize_distribution(question_model_key, reference),
    )


def validate_option_compatibility(
    question_model_key: str,
    provided_options: List[str],
    expected_scale: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """Validate provided options against expected scale after canonicalization.

    Returns:
      (is_compatible, normalized_options, warnings)
    """
    normalized = canonicalize_options(question_model_key, provided_options or [])
    if not expected_scale:
        # Open-text/duration questions are not option-constrained.
        return True, normalized, []

    expected_tokens = {_normalize_token(v) for v in (expected_scale or [])}
    provided_tokens = {_normalize_token(v) for v in normalized}
    warnings: List[str] = []

    if not normalized:
        warnings.append("provided_options_empty_for_structured_question")
        return False, normalized, warnings

    if provided_tokens == expected_tokens:
        if normalized != expected_scale:
            warnings.append("provided_options_reordered_or_aliased")
        return True, normalized, warnings

    missing = sorted(expected_tokens - provided_tokens)
    extras = sorted(provided_tokens - expected_tokens)
    if missing:
        warnings.append("missing_options:" + ",".join(missing))
    if extras:
        warnings.append("unexpected_options:" + ",".join(extras))
    return False, normalized, warnings


async def _suggest_alias_mapping_via_llm(
    question_model_key: str,
    provided_options: List[str],
    expected_scale: List[str],
) -> Dict[str, str]:
    prompt = (
        "Map provided answer options to the closest expected canonical options.\n"
        "Return ONLY strict JSON as an object mapping each provided option string to exactly one expected canonical option.\n"
        "If an option cannot be mapped safely, omit it.\n\n"
        f"Question model key: {question_model_key}\n"
        f"Provided options: {provided_options}\n"
        f"Expected canonical options: {expected_scale}\n"
    )
    try:
        client = get_llm_client()
        response = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=180,
        )
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return {}
        expected_tokens = {_normalize_token(opt): opt for opt in expected_scale}
        out: Dict[str, str] = {}
        for src, dst in parsed.items():
            src_key = _normalize_token(str(src))
            dst_key = _normalize_token(str(dst))
            if src_key and dst_key in expected_tokens:
                out[src_key] = expected_tokens[dst_key]
        return out
    except Exception:
        return {}


async def validate_option_compatibility_hybrid(
    question_model_key: str,
    provided_options: List[str],
    expected_scale: List[str],
) -> Tuple[bool, List[str], List[str]]:
    compatible, normalized, warnings = validate_option_compatibility(
        question_model_key,
        provided_options,
        expected_scale,
    )
    if compatible or not expected_scale or not provided_options:
        return compatible, normalized, warnings

    suggested_aliases = await _suggest_alias_mapping_via_llm(
        question_model_key,
        provided_options,
        expected_scale,
    )
    if not suggested_aliases:
        return compatible, normalized, warnings

    option_space_key = get_option_space_key(question_model_key)
    cache_bucket = _HYBRID_ALIAS_CACHE.setdefault(option_space_key, {})
    cache_bucket.update(suggested_aliases)
    _save_hybrid_alias_cache(_HYBRID_ALIAS_CACHE)

    compatible_after, normalized_after, warnings_after = validate_option_compatibility(
        question_model_key,
        provided_options,
        expected_scale,
    )
    if compatible_after:
        warnings_after = list(warnings_after) + ["hybrid_alias_mapping_applied"]
    return compatible_after, normalized_after, warnings_after

