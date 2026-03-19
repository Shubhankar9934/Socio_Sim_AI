# Survey API

Source collection: **JADU_Full_API** → folder `survey`.

Surveys run against the **current** `agents_store`. Results are stored in `api.state.survey_results` keyed by `survey_id` (single survey) or tracked by `session_id` (multi).

---

## 1. POST `/survey` — single question (synchronous)

### Request body (`SurveyRequest`)

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `question` | string | yes | Full question text shown to the cognitive/narrative pipeline. |
| `question_id` | string | no | Stable id for consistency / history; default UUID if empty. |
| `use_archetypes` | bool | no (default false) | Whether to use archetype shortcuts in orchestration. |
| `options` | string[] | no | If provided, intended to constrain answer options (see engine behavior in `simulation/orchestrator.py` / decision path). |
| `current_events` | object[] | no | Real-time framing: temporary belief adjustments from media/events at survey time. |

### Response (200) — `SurveyResult`

| Field | Purpose |
|-------|---------|
| `survey_id` | New UUID; use for `/survey/{id}/results`, `/analytics/{id}`, `/evaluate/{id}`. |
| `question` | Echo of request question. |
| `responses` | List of `SurveyResponseItem` per agent. |
| `n_total` | Count of responses. |

Each **response item** includes: `agent_id`, `answer` (narrative), `sampled_option`, `distribution` (option → probability), `demographics`, `lifestyle`, `error`.

### Code flow

1. [api/routes/survey.py](../../api/routes/survey.py) `run_survey_endpoint`
2. Validates `agents_store` non-empty
3. `get_llm_client().reset_survey_stats()`
4. **`await run_survey(...)`** → [simulation/orchestrator.py](../../simulation/orchestrator.py)
   - Per agent: perception → question model → factor graph → sampling → narrative generation (LLM)
   - `think_fn=None` enables orchestrator `default_think` (memory, social context, style)
5. Builds `SurveyResponseItem` list; stores raw `responses` + pydantic `items` in `survey_results[survey_id]`
6. Appends to `response_histories[agent_id]` for drift / consistency

### Key modules

- **Perception / question model:** [agents/perception.py](../../agents/perception.py), [config/question_models.py](../../config/question_models.py)
- **Decision / distribution:** [agents/decision.py](../../agents/decision.py)
- **Narrative:** [agents/narrative.py](../../agents/narrative.py)
- **LLM:** [llm/client.py](../../llm/client.py)

---

## 2. GET `/survey/{survey_id}/results`

Returns the same payload as POST `/survey` for a stored `survey_id`.

- **404** if unknown `survey_id`.
- Reads `survey_results[survey_id]` → returns `items` as `responses`.

---

## 3. POST `/survey/multi` — multi-question (async)

### Request body (`MultiSurveyRequest`)

| Field | Type | Purpose |
|-------|------|---------|
| `questions` | `{ question, question_id?, options? }[]` | Sequence of rounds. |
| `use_archetypes` | bool | Passed to engine. |
| `social_influence_between_rounds` | bool | Social updates between questions. |
| `summarize_every` | int (1–50) | How often to compress dialogue / memory. |

### Response (200)

| Field | Purpose |
|-------|---------|
| `session_id` | Poll progress and fetch results. |
| `current_round`, `total_rounds`, `status`, `completed_questions` | Progress snapshot. |

### Code flow

1. `run_multi_survey_endpoint` registers background task `_run_multi_survey_task`
2. [simulation/survey_engine.py](../../simulation/survey_engine.py) `SurveyEngine` runs rounds; optional WebSocket broadcasts
3. [storage/writer.py](../../storage/writer.py) `JSONLWriter` may write session file under `data/sessions/`
4. On completion, aggregate results stored in `survey_sessions[session_id]`

---

## 4. GET `/survey/session/{session_id}/progress`

Returns current `status` (`running` | `completed` | `failed`), round index, completed question ids.

---

## 5. GET `/survey/session/{session_id}/results`

Full multi-survey payload when `status == completed`.

- **409** (or detail message) if still running — see route implementation in [api/routes/survey.py](../../api/routes/survey.py).

---

## 6. GET `/survey/session/{session_id}/round/{round_idx}`

Results for one round (0-based), including `survey_id` alias like `{session_id}_r{round}` for compatibility with analytics in some flows.

---

## Design note (general vs “one kind of response”)

The engine selects a **question model** (and thus a **discrete option scale**) from the question text + domain config. It is **not** returning a single hardcoded answer; it returns a **distribution** per agent and a **sampled** option. To align scales with your wording (e.g. explicit Likert labels), use `options` in the API and/or extend domain `topic_to_model_key` and `QUESTION_MODELS` ([config/question_models.py](../../config/question_models.py)).
