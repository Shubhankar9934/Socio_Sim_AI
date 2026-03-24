# Example payloads

Small, versioned JSON snippets for documentation and Postman. **Full** captured I/O remains in [`api_details_input_output.txt`](../../api_details_input_output.txt) at repo root.

| File | Use |
|------|-----|
| [survey-request-minimal.json](survey-request-minimal.json) | `POST /survey` body — smallest useful shape |
| [survey-request-diagnostics.json](survey-request-diagnostics.json) | Enable per-response debug fields |
| [survey-response-trimmed.json](survey-response-trimmed.json) | Example `SurveyResult` with one agent |
| [evaluate-request-with-reference.json](evaluate-request-with-reference.json) | `POST /evaluate/{survey_id}` with explicit histogram |
| [evaluate-report-trimmed.json](evaluate-report-trimmed.json) | Shaped like `EvaluationReportResponse` (illustrative) |

Linked from [Survey API](../jadu-api/survey.md), [Evaluation API](../jadu-api/evaluation.md), and [Vision and goals](../vision-and-goals.md).

**CI-style check:** [`tests/test_docs_examples.py`](../../tests/test_docs_examples.py) validates these files against Pydantic models (`pytest tests/test_docs_examples.py`).
