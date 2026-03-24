"""Ensure curated docs example JSON still matches Pydantic API models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.schemas import (
    EvaluateRequest,
    EvaluationReportResponse,
    SurveyRequest,
    SurveyResponseItem,
    SurveyResult,
)

_DOCS_EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"


def _load(name: str) -> dict:
    path = _DOCS_EXAMPLES / name
    assert path.is_file(), f"missing {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_survey_request_minimal() -> None:
    SurveyRequest.model_validate(_load("survey-request-minimal.json"))


def test_survey_request_diagnostics() -> None:
    SurveyRequest.model_validate(_load("survey-request-diagnostics.json"))


def test_survey_response_trimmed() -> None:
    raw = _load("survey-response-trimmed.json")
    SurveyResult.model_validate(raw)
    for item in raw.get("responses", []):
        SurveyResponseItem.model_validate(item)


def test_evaluate_request_with_reference() -> None:
    EvaluateRequest.model_validate(_load("evaluate-request-with-reference.json"))


def test_evaluate_report_trimmed() -> None:
    EvaluationReportResponse.model_validate(_load("evaluate-report-trimmed.json"))
