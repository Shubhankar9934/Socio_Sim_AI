"""Runtime storage for calibrated factor-weight overrides."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

_CALIBRATED_WEIGHTS: Dict[str, Dict[str, float]] = {}


def set_calibrated_weights(question_model_key: str, weights: Dict[str, float]) -> None:
    _CALIBRATED_WEIGHTS[question_model_key] = dict(weights)


def get_calibrated_weights(question_model_key: str) -> Optional[Dict[str, float]]:
    weights = _CALIBRATED_WEIGHTS.get(question_model_key)
    if weights is None:
        return None
    return deepcopy(weights)


def get_all_calibrated_weights() -> Dict[str, Dict[str, float]]:
    return deepcopy(_CALIBRATED_WEIGHTS)

