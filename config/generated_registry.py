"""Persistent registry for LLM-generated question structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_REGISTRY_PATH = Path(__file__).resolve().parent / "generated_models.json"

_DEFAULT_REGISTRY: Dict[str, Any] = {
    "models": {},
    "references": {},
    "constraints": {},
    "question_hash_map": {},
}


def _normalized_registry(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(_DEFAULT_REGISTRY)
    for key in out:
        value = payload.get(key, out[key]) if isinstance(payload, dict) else out[key]
        out[key] = value if isinstance(value, dict) else {}
    return out


def load_generated_registry() -> Dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return dict(_DEFAULT_REGISTRY)
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(_DEFAULT_REGISTRY)
    return _normalized_registry(raw)


def save_generated_registry(registry: Dict[str, Any]) -> None:
    safe = _normalized_registry(registry)
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")


def get_generated_model_payload(model_key: str) -> Dict[str, Any]:
    registry = load_generated_registry()
    payload = registry.get("models", {}).get(model_key, {})
    return payload if isinstance(payload, dict) else {}


def get_cached_model_key_for_question(question_hash: str) -> str:
    registry = load_generated_registry()
    mapped = registry.get("question_hash_map", {}).get(question_hash, "")
    return mapped if isinstance(mapped, str) else ""

