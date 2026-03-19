# Analytics API

Source collection: **JADU_Full_API** → folder `analytics`.

---

## GET `/analytics/{survey_id}`

Segmented breakdown of **survey responses** for one stored survey.

### Path

| Param | Purpose |
|-------|---------|
| `survey_id` | From `POST /survey` response (or compatible id from multi-survey round storage). |

### Query

| Param | Default | Allowed | Purpose |
|-------|---------|---------|---------|
| `segment_by` | `location` | `location`, `income`, `nationality`, `age` | Which persona field defines segments. |

### Response (200)

| Field | Purpose |
|-------|---------|
| `survey_id` | Echo. |
| `segment_by` | Echo. |
| `aggregated` | `segment_value → { answer_text → proportion }` |
| `insights` | Human-readable strings from [analytics/insights.py](../../analytics/insights.py) |

### How `aggregated` is computed

1. [api/routes/analytics.py](../../api/routes/analytics.py) loads `survey_results[survey_id]["responses"]`.
2. [analytics/aggregator.py](../../analytics/aggregator.py) `aggregate_with_personas`:
   - Joins each response to `persona` by `agent_id`.
   - Copies `location`, `income`, `nationality`, `age` from persona onto each row.
   - Groups by `segment_by` and computes **value counts of the `answer` field** (verbatim narrative), normalized to proportions.

So keys inside each segment are **full answer strings**, not `sampled_option` labels.

### How `insights` are computed

1. `generate_insights(aggregated, segment_name=segment_by)` — lists top answers per segment by proportion.
2. `delivery_frequency_insight(aggregated)` — calls `high_frequency_insight` with default `high_keys` like `"3-4 per week"`, `"daily"`, `"multiple per day"`, `"often"`, `"very often"`.

**Important:** Because `aggregated` uses **verbatim** `answer` text, those `high_keys` usually **do not appear** as dict keys → the “high frequency” summary can show **0%** or nonsensical segment comparisons. This is a **known mismatch** between aggregation key and insight logic, not a single hardcoded survey answer.

### Code path

`get_analytics` → `aggregate_with_personas` → `generate_insights` + `delivery_frequency_insight`.

### Errors

- **404** if `survey_id` not in `survey_results`.
- **Empty** `aggregated` if no personas in `agents_store` (route still returns structure).

### Possible improvements (platform-general)

- Add `aggregate_by=sampled_option` query param to aggregate on discrete options for any question type.
- Make `high_keys` configurable per domain or derive from the survey’s `distribution` keys.
