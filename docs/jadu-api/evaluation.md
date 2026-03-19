# Evaluation API

Source collection: **JADU_Full_API** → folder `evaluation`.

**Prerequisites:** `agents_store` populated + `survey_id` present in `survey_results` (from `POST /survey`).

---

## POST `/evaluate/{survey_id}`

Runs the **quality / diagnostics** bundle on the stored survey responses and current population.

### Path

| Param | Purpose |
|-------|---------|
| `survey_id` | Must exist in `api.state.survey_results`. |

### Body (`EvaluateRequest`)

| Field | Type | Purpose |
|-------|------|---------|
| `run_judge` | bool | If true, runs LLM-as-judge on a sample (cost + latency). |
| `judge_sample` | int | Max responses to judge when `run_judge` is true. |
| `realism_threshold` | float | Threshold for population realism pass/fail in report. |
| `drift_threshold` | float | Per-agent drift classification cutoff. |
| `run_similarity` | bool | Compute narrative duplicate / similarity stats. |
| `similarity_threshold` | float | Flag pairs above this similarity. |

### Response (high level)

| Section | Source module | Meaning |
|---------|---------------|---------|
| `population_realism` | [evaluation/realism.py](../../evaluation/realism.py) | Personas vs reference marginals (similar to post-generate validation). |
| `drift` | [evaluation/drift.py](../../evaluation/drift.py) | Agents whose recent answers diverge from inferred “stable” behavior. |
| `consistency_score` | [evaluation/consistency.py](../../evaluation/consistency.py) | Cross-question alignment when ≥2 distinct `question_id`s exist in histories. |
| `distribution_validation` | [evaluation/distribution_validation.py](../../evaluation/distribution_validation.py) | Observed `sampled_option` histogram vs **reference** distribution. |
| `narrative_similarity` | [evaluation/similarity.py](../../evaluation/similarity.py) | Embedding/text similarity between narrative answers. |
| `llm_judge` | [evaluation/judge.py](../../evaluation/judge.py) | Optional mean score + details. |
| `dashboard` / `quantitative_metrics` / `summary` | [evaluation/report.py](../../evaluation/report.py) | Aggregated pass/fail vs `QUALITY_TARGETS`. |

### Code flow

1. [api/routes/evaluation.py](../../api/routes/evaluation.py) `evaluate_survey`
2. Loads `responses` from `survey_results[survey_id]`
3. `await run_evaluation(...)` in [evaluation/report.py](../../evaluation/report.py)
4. `export_evaluation_report(..., output_path=f"evaluation_report_{survey_id}.json", ...)`

### Distribution validation detail

`validate_survey_distribution(survey_responses, reference=...)`:

- If `reference` is **not** passed (current API behavior), `compare_to_reference` uses `DEFAULT_REFERENCE` from [evaluation/distribution_validation.py](../../evaluation/distribution_validation.py), which resolves via `get_reference_distribution("generic_frequency")` — scale **never / rarely / sometimes / often / very often**.

If your survey’s `sampled_option` values use a **different** scale (e.g. `food_delivery_frequency` buckets), the JS similarity will look **falsely poor** unless you extend the API to pass `reference_distribution` or `question_model_key`. This is **configuration / wiring**, not a hardcoded single answer.

---

## GET `/evaluate/{evaluation_id}/report`

**Current behavior (stub):** If `evaluation_id` exists as a key in `survey_results`, returns:

```json
{ "survey_id": "<id>", "message": "Run POST /evaluate/{survey_id} to get report." }
```

It does **not** return the last computed report from memory. The POST handler writes `evaluation_report_{survey_id}.json` to disk instead.

**404** if id not in `survey_results`.

---

## Errors

- **404** — survey not found.
- **400** — no population loaded.
