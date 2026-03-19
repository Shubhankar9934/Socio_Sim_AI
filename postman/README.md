# Testing JADU API with Postman

**JADU** (v0.1.0) is a synthetic society simulation platform with survey orchestration, analytics, and evaluation. This folder contains Postman assets to test **all** API endpoints.

---

## Overview – Detailed API reference

Use this section to see **purpose**, **flow**, **input parameters**, and **output** for every endpoint. The same details are embedded in each request and folder in **JADU_Full_API.postman_collection.json**: when you select a request or folder in Postman, the **Overview** (right panel) shows that request’s description with Purpose, Flow, Input, and Output. You can also copy any block below into a request’s Overview in Postman if you prefer to edit there.

---

### Population

#### POST /population/generate

**Description**  
Generates a synthetic population of agents (personas), builds a Barabási–Albert social graph, and initializes agent state (including social influence fractions). Must be run **before** any survey, simulation, or analytics.

**Flow**  
1. Call once at the start of a session (or after changing domain).  
2. Then use **GET /agents**, **POST /survey**, **POST /simulation**, etc.

**Input (body, JSON)**

| Parameter   | Type   | Required | Constraints / Notes |
|------------|--------|----------|----------------------|
| `n`        | int    | No       | Default `500`. Min `10`, max `10000`. Number of agents. |
| `method`   | string | No       | Default `"bayesian"`. One of: `monte_carlo`, `bayesian`, `ipf`. Synthesis method. |
| `id_prefix`| string | No       | Default `"DXB"`. Prefix for agent IDs (e.g. `DXB_001`). |
| `seed`     | int    | No       | Optional. Random seed for reproducibility. |

**Output (200)**  
JSON object:

| Field                 | Type  | Description |
|-----------------------|-------|-------------|
| `n`                   | int   | Number of agents created. |
| `method`              | string| Synthesis method used. |
| `realism_passed`       | bool  | Whether population passed realism validation. |
| `realism_score`        | float | Overall realism score. |
| `per_attribute`        | object| Per-attribute validation details. |
| `segment_distribution` | object| Counts per population segment. |

---

### Agents

#### GET /agents

**Description**  
Returns a paginated list of agent summaries (demographics only). Optional filters by location and nationality.

**Flow**  
Call after **POST /population/generate**. Use to pick an `agent_id` for **GET /agents/:agent_id**.

**Input**

| Source | Parameter    | Type   | Required | Description |
|--------|--------------|--------|----------|-------------|
| Query  | `location`   | string | No       | Filter by location (exact match). |
| Query  | `nationality`| string | No       | Filter by nationality (exact match). |
| Query  | `limit`      | int    | No       | Default `100`, min 1, max 1000. Page size. |
| Query  | `offset`     | int    | No       | Default `0`, min 0. Pagination offset. |

**Output (200)**  
Array of objects:

| Field         | Type   | Description |
|---------------|--------|-------------|
| `agent_id`    | string | Unique agent ID. |
| `age`         | string | Age group. |
| `nationality` | string | Nationality. |
| `income`      | string | Income band. |
| `location`    | string | Location. |
| `occupation`  | string | Occupation. |

---

#### GET /agents/:agent_id

**Description**  
Returns one agent’s full record: persona (all attributes) and current state (beliefs, memory, etc.).

**Flow**  
Use after **GET /agents**; set path parameter `agent_id` (e.g. from list or env var `agent_id`).

**Input**

| Source | Parameter | Type   | Required | Description |
|--------|------------|--------|----------|-------------|
| Path   | `agent_id` | string | Yes      | Agent ID (e.g. `DXB_001`). |

**Output (200)**  
JSON object:

| Field     | Type  | Description |
|-----------|-------|-------------|
| `agent_id`| string| Same as path. |
| `persona` | object| Full persona (demographics, lifestyle, meta, etc.). |
| `state`   | object\|null | Current agent state (beliefs, narrative, etc.); null if not available. |

**Output (404)**  
`{"detail": "Agent not found"}`

---

### Survey

#### POST /survey (single question)

**Description**  
Asks the current population one survey question and returns the full set of responses immediately (synchronous). Use for one-question-at-a-time flows.

**Flow**  
1. Ensure population exists (**POST /population/generate**).  
2. Call **POST /survey** with one question.  
3. From the response, copy `survey_id` into env for **GET /survey/:survey_id/results**, **GET /analytics/:survey_id**, **POST /evaluate/:survey_id**.

**Input (body, JSON)**

| Parameter        | Type   | Required | Description |
|------------------|--------|----------|-------------|
| `question`       | string | Yes      | Question text. Can include options in text (e.g. "Options: A, B, C"). |
| `question_id`    | string | No       | Default `""`. Your identifier for this question. |
| `use_archetypes` | bool   | No       | Default `false`. Use archetype-based sampling. |
| `options`        | string[] | No     | Explicit options; if null/empty, treated as open-ended. |
| `current_events` | object[] | No     | Optional real-time context (e.g. temp_beliefs). |

**Output (200)**  
JSON object (SurveyResult):

| Field       | Type   | Description |
|-------------|--------|-------------|
| `survey_id` | string | Unique survey ID; use for results/analytics/evaluation. |
| `question`  | string | Echo of question. |
| `responses` | array | List of per-agent response items. |
| `n_total`   | int    | Total number of responses. |

Each element of `responses`:

| Field           | Type   | Description |
|-----------------|--------|-------------|
| `agent_id`      | string | Agent ID. |
| `answer`        | string | Answer text. |
| `sampled_option`| string | Option chosen if multiple choice. |
| `distribution`  | object | Option probabilities if applicable. |
| `demographics`  | object | Agent demographics. |
| `lifestyle`     | object | Agent lifestyle. |
| `error`         | string | Error message if this agent failed. |

---

#### GET /survey/:survey_id/results

**Description**  
Returns the same result payload as **POST /survey** for a given `survey_id` (e.g. from an earlier **POST /survey** call).

**Flow**  
Call after **POST /survey**; use the `survey_id` from that response (or set in env).

**Input**

| Source | Parameter   | Type   | Required | Description |
|--------|-------------|--------|----------|-------------|
| Path   | `survey_id`  | string | Yes      | Survey ID from **POST /survey** response. |

**Output (200)**  
Same structure as **POST /survey** response (SurveyResult): `survey_id`, `question`, `responses`, `n_total`.

**Output (404)**  
If survey not found.

---

#### POST /survey/multi

**Description**  
Starts a multi-question survey in the background. Returns a `session_id` immediately; results are obtained by polling progress and then fetching session results.

**Flow**  
1. **POST /survey/multi** → get `session_id` (collection script can set `session_id` in env).  
2. **GET /survey/session/:session_id/progress** until `status` is `"completed"` (or `"failed"`).  
3. **GET /survey/session/:session_id/results** to get all round results.

**Input (body, JSON)**

| Parameter                       | Type  | Required | Description |
|--------------------------------|-------|----------|-------------|
| `questions`                    | array | Yes      | List of `{ "question": string, "question_id": string, "options": string[]? }`. |
| `use_archetypes`               | bool  | No       | Default `false`. |
| `social_influence_between_rounds` | bool | No    | Default `true`. Apply social influence between rounds. |
| `summarize_every`              | int   | No       | Default `5`. Summarize agent memory every N rounds (1–50). |

**Output (200)**  
JSON object (MultiSurveyProgress):

| Field          | Type   | Description |
|----------------|--------|-------------|
| `session_id`   | string | Use for progress and results endpoints. |
| `current_round`| int    | Current round index. |
| `total_rounds` | int    | Total number of questions. |
| `status`       | string | `"running"` \| `"completed"` \| `"failed"`. |
| `completed_questions` | array | List of completed question_ids. |

---

#### GET /survey/session/:session_id/progress

**Description**  
Returns current progress of a multi-question survey session.

**Flow**  
Poll after **POST /survey/multi** until `status` is `"completed"` (or `"failed"`), then call **GET /survey/session/:session_id/results**.

**Input**

| Source | Parameter   | Type   | Required | Description |
|--------|-------------|--------|----------|-------------|
| Path   | `session_id`| string | Yes      | From **POST /survey/multi** response. |

**Output (200)**  
Same as **POST /survey/multi** response: `session_id`, `current_round`, `total_rounds`, `status`, `completed_questions`.

---

#### GET /survey/session/:session_id/results

**Description**  
Returns full results for a completed multi-survey session (all rounds and responses).

**Flow**  
Call when **GET /survey/session/:session_id/progress** shows `status: "completed"`.

**Input**

| Source | Parameter   | Type   | Required | Description |
|--------|-------------|--------|----------|-------------|
| Path   | `session_id`| string | Yes      | Session ID. |

**Output (200)**  
JSON object (SurveySessionResult):

| Field             | Type   | Description |
|-------------------|--------|-------------|
| `session_id`      | string | Session ID. |
| `questions`       | array  | List of question strings. |
| `rounds`          | array  | Per-round results (round_idx, question, question_id, responses, n_total, elapsed_seconds). |
| `total_responses` | int    | Total response count. |
| `elapsed_seconds` | float  | Total time. |
| `status`          | string | e.g. `"completed"`. |

**Output (409)**  
If session is still running.

---

#### GET /survey/session/:session_id/round/:round_idx

**Description**  
Returns results for a single round of a multi-survey session.

**Flow**  
Use after session is completed; `round_idx` is 0-based.

**Input**

| Source | Parameter   | Type | Required | Description |
|--------|-------------|------|----------|-------------|
| Path   | `session_id`| string | Yes | Session ID. |
| Path   | `round_idx` | int  | Yes | 0-based round index. |

**Output (200)**  
Same structure as one round in `SurveySessionResult.rounds`: `round_idx`, `question`, `question_id`, `responses`, `n_total`, `elapsed_seconds`.

---

### Simulation

#### POST /simulation

**Description**  
Runs N days of simulation: processes scheduled events, applies social influence, and updates agent state. Mutates global agent state.

**Flow**  
Requires population. Optionally schedule events with **POST /simulation/events** first; then call **POST /simulation**. Check **GET /simulation/status** or **GET /simulation/events** as needed.

**Input (body, JSON)**

| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| `days`    | int  | No       | Default `30`, min 1, max 365. |

**Output (200)**  
`{"status": "ok", "days": <int>, "n_agents": <int>}`

**Output (400)**  
If no population loaded.

---

#### POST /simulation/events

**Description**  
Schedules a world event to be applied during a future **POST /simulation** run (e.g. price change, policy, infrastructure).

**Flow**  
Call before or between **POST /simulation** runs. Events are processed in day order when simulation runs.

**Input (body, JSON)**

| Parameter  | Type   | Required | Description |
|------------|--------|----------|-------------|
| `day`      | int    | Yes      | Day index (≥ 0) when event triggers. |
| `type`     | string | Yes      | One of: `price_change`, `policy`, `infrastructure`, `market`, `new_service`, `new_metro_station`. |
| `payload`  | object | No       | Event-specific data (e.g. `{"change_pct": -10}`). |
| `district` | string | No       | Optional district scope. |

**Output (200)**  
`{"status": "scheduled", "event_type": "<type>", "day": <int>, "pending_events": <count>}`

---

#### GET /simulation/events

**Description**  
Lists all pending events and current global parameters (no auth).

**Input**  
None.

**Output (200)**  
`{"pending_events": [{"day", "type", "district", "payload"}, ...], "global_params": {...}}`

---

#### GET /simulation/status

**Description**  
Returns current simulation status: population size, whether social graph is loaded, and count of pending events.

**Input**  
None.

**Output (200)**  
`{"population_size": <int>, "social_graph_loaded": <bool>, "pending_events": <int>}`

---

#### POST /simulation/scenario

**Description**  
Runs a named scenario in isolation (deep-copy of agents). Does **not** mutate global state. Returns dimension means after the run.

**Flow**  
Use for “what-if” runs. Requires population. Compare two scenarios with **POST /simulation/scenario/compare**.

**Input (body, JSON)**

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `name`    | string | No       | Default `"unnamed"`. Scenario name. |
| `days`    | int    | No       | Default `30`, min 1. |
| `seed`    | int    | No       | Optional random seed. |
| `events`  | array  | No       | List of `{ "day": int, "type": string, "payload": object, "district": string? }`. |

**Output (200)**  
`{"name", "days", "seed", "population_size", "dimension_means": {...}}`

---

#### POST /simulation/scenario/compare

**Description**  
Runs two scenarios and returns the difference of their macro metrics (no global state change).

**Input (body, JSON)**  
`scenario_a`, `scenario_b`: same shape as **POST /simulation/scenario** body.

**Output (200)**  
Comparison object (diff of dimension means and any other macro metrics).

---

#### POST /simulation/scenario/run-with-survey

**Description**  
Runs one scenario, then runs a survey on the post-scenario population. Returns scenario results plus survey results and timeline.

**Input (body, JSON)**  
`scenario`: same as **POST /simulation/scenario**. `questions`: array of question strings (min length 1).

**Output (200)**  
`{"name", "days", "population_size", "dimension_means", "belief_means", "survey_results", "timeline"}`

---

#### POST /simulation/scenario/compare-with-survey

**Description**  
Runs two scenarios, runs the same survey on both post-scenario populations, and returns comparison.

**Input (body, JSON)**  
`scenario_a`, `scenario_b`, `questions` (array of strings, min 1).

**Output (200)**  
Comparison including survey results for both scenarios.

---

#### GET /simulation/causal/graph

**Description**  
Returns the default causal graph structure (nodes and edges).

**Input**  
None.

**Output (200)**  
Graph object (structure depends on implementation: nodes, edges, etc.).

---

#### POST /simulation/causal/do-intervention

**Description**  
Runs a counterfactual do-intervention on the causal graph (set variables and compute downstream effects).

**Input (body, JSON)**  
`intervention`: object (variable → value). `observational`: optional object.

**Output (200)**  
`{"counterfactual_values": {...}}`

---

#### POST /simulation/causal/ate

**Description**  
Estimates the Average Treatment Effect for a treatment and outcome with optional confounders.

**Input (body, JSON)**

| Parameter         | Type   | Required | Description |
|-------------------|--------|----------|-------------|
| `treatment`       | string | Yes      | Treatment variable name. |
| `outcome`         | string | Yes      | Outcome variable name. |
| `confounders`     | array  | No       | List of confounder variable names. |
| `treatment_value` | float  | No       | Default `1.0`. |
| `control_value`   | float  | No       | Default `0.0`. |

**Output (200)**  
`{"treatment", "outcome", "ate": <float>}`

---

#### POST /simulation/causal/learn

**Description**  
Learns causal structure from simulation timeline data.

**Input (body, JSON)**  
`timeline`: array of time-step data (structure defined by implementation).

**Output (200)**  
Graph object (learned structure).

---

### Analytics

#### GET /analytics/:survey_id

**Description**  
Returns segmented analytics and insights for a survey: aggregated response distributions by segment (e.g. location, income) and generated insight text.

**Flow**  
Call after at least one **POST /survey** (or completed multi-survey). Use the same `survey_id` in path and, optionally, env.

**Input**

| Source | Parameter    | Type   | Required | Description |
|--------|--------------|--------|----------|-------------|
| Path   | `survey_id`  | string | Yes      | Survey ID from **POST /survey** (or stored from multi). |
| Query  | `segment_by` | string | No       | Default `"location"`. One of: `location`, `income`, `nationality`, `age`. |

**Output (200)**  
JSON object:

| Field        | Type   | Description |
|--------------|--------|-------------|
| `survey_id`  | string | Echo. |
| `segment_by` | string | Segment dimension used. |
| `aggregated` | object | Map of segment value → { option → proportion }. |
| `insights`   | array  | List of insight strings. |

**Output (404)**  
If survey not found.

---

### Evaluation

#### POST /evaluate/:survey_id

**Description**  
Runs the evaluation framework on a survey’s responses: population realism, drift, consistency, distribution validation, narrative similarity, and optional LLM judge. Returns a report and can export a file.

**Flow**  
Requires population and a survey that has results (e.g. from **POST /survey**). Set `survey_id` in path (e.g. from env). Optionally call **GET /evaluate/:evaluation_id/report** afterward if report is stored by id.

**Input**

| Source | Parameter   | Type   | Required | Description |
|--------|-------------|--------|----------|-------------|
| Path   | `survey_id` | string | Yes      | Survey ID with stored results. |

**Input (body, JSON)**

| Parameter             | Type  | Required | Description |
|-----------------------|-------|----------|-------------|
| `run_judge`           | bool  | No       | Default `false`. Run LLM judge. |
| `judge_sample`        | int   | No       | Default `20`. Sample size for judge. |
| `realism_threshold`   | float | No       | Default `0.85`. |
| `drift_threshold`     | float | No       | Default `0.3`. |
| `run_similarity`       | bool  | No       | Default `true`. Run narrative similarity. |
| `similarity_threshold`| float | No       | Default `0.9`. |

**Output (200)**  
JSON object (EvaluationReportResponse):

| Field                     | Type   | Description |
|---------------------------|--------|-------------|
| `population_realism`     | object | Realism metrics. |
| `drift`                  | object | Drift metrics. |
| `consistency_score`      | float  | Consistency score. |
| `distribution_validation`| object | Distribution checks. |
| `narrative_similarity`   | object | Similarity metrics. |
| `llm_judge`              | object | Judge results if run. |
| `dashboard`              | object | DashboardMetrics (e.g. duplicate_narrative_rate, persona_realism_score, distribution_similarity, consistency_score, drift_rate, mean_judge_score). |
| `quantitative_metrics`   | object | Extra metrics. |
| `summary`                | object | Text summary. |

**Output (400)**  
If no population or survey not found.

---

#### GET /evaluate/:evaluation_id/report

**Description**  
Returns a stored evaluation report by ID. Implementation may map `evaluation_id` to a stored report (e.g. by survey_id).

**Input**

| Source | Parameter       | Type   | Required | Description |
|--------|-----------------|--------|----------|-------------|
| Path   | `evaluation_id` | string | Yes      | Often same as `survey_id` if stored by survey. |

**Output (200)**  
Report object (structure as implemented; may include `survey_id`, `message`, or full report).

**Output (404)**  
If not found.

---

### Discovery

#### POST /discovery/domains/auto-setup

**Description**  
Creates a new domain configuration (files under `data/domains/<domain_id>/`) from name, description, sample questions, and optional city/currency/reference data.

**Flow**  
Use when setting up a new domain. No dependency on population. Use returned `domain_id` for dimension discovery with `save: true` if needed.

**Input (body, JSON)**

| Parameter         | Type   | Required | Description |
|-------------------|--------|----------|-------------|
| `domain_name`     | string | Yes      | Domain name (e.g. `food_delivery`). |
| `description`     | string | No       | Default `""`. |
| `sample_questions`| array  | No       | List of sample question strings. |
| `city_name`       | string | No       | Default `""`. |
| `currency`        | string | No       | Default `"USD"`. |
| `reference_data`  | object | No       | Optional reference data. |

**Output (200)**  
`{"domain_id": "<id>", "message": "Domain '...' created at data/domains/.../"}`

---

#### POST /discovery/dimensions

**Description**  
Discovers behavioral and belief dimensions from a list of questions (e.g. via LLM). Optionally saves results to a domain config.

**Flow**  
Can be used standalone or after **POST /discovery/domains/auto-setup** with `domain_id` and `save: true` to persist.

**Input (body, JSON)**

| Parameter   | Type   | Required | Constraints / Description |
|-------------|--------|----------|----------------------------|
| `questions` | array  | Yes      | Min 1. List of question strings. |
| `n_behavioral` | int  | No       | Default `12`, min 1, max 50. |
| `n_belief`  | int    | No       | Default `7`, min 1, max 30. |
| `domain_id` | string | No       | If set and `save: true`, persist to this domain. |
| `save`      | bool   | No       | Default `false`. Persist to `domain_id`. |

**Output (200)**  
JSON object:

| Field                    | Type   | Description |
|--------------------------|--------|-------------|
| `behavioral`             | array  | List of `{ name, description, representative_questions }`. |
| `belief`                 | array  | Same structure. |
| `question_to_dimension`  | object | Map question → dimension. |
| `saved`                  | bool   | Whether results were saved. |

---

### Calibration

#### POST /calibration/auto-weights

**Description**  
Optimizes factor weights so that simulated survey responses match reference distributions for multiple questions. Requires a loaded population.

**Flow**  
1. Optionally get reference distributions from **POST /calibration/upload-data** (real data) or define manually.  
2. **POST /population/generate** if not already done.  
3. Call **POST /calibration/auto-weights**.  
4. Use learned weights in your pipeline (implementation-specific).

**Input (body, JSON)**

| Parameter                  | Type   | Required | Description |
|----------------------------|--------|----------|-------------|
| `questions`                | array  | Yes      | Min 1. List of question strings. |
| `reference_distributions`  | object | Yes      | Map question string → { option → proportion }. |
| `n_iterations`             | int    | No       | Default `50`, min 5, max 500. |
| `seed`                    | int    | No       | Optional. Default `42`. |

**Output (200)**  
`{"overall_loss": <float>, "results": [{"question", "learned_weights", "best_loss", "converged": bool}, ...]}`

**Output (400)**  
If no population loaded.

---

#### POST /calibration/fit

**Description**  
Runs calibration for a **single** question against one reference distribution. Requires population.

**Input (body, JSON)**

| Parameter                 | Type   | Required | Description |
|---------------------------|--------|----------|-------------|
| `question`                | string | Yes      | Question text. |
| `reference_distribution`  | object | Yes      | Map option string → proportion. |
| `demographics_cols`       | array  | No       | Optional demographics columns. |
| `n_iterations`            | int    | No       | Default `50`, min 5. |

**Output (200)**  
`{"question", "learned_weights", "best_loss", "converged", "n_iterations"}`

**Output (400)**  
If no population.

---

#### POST /calibration/upload-data

**Description**  
Uploads real survey data (question + list of responses, optional demographics). Returns computed reference distribution for use in **POST /calibration/fit** or **POST /calibration/auto-weights**.

**Flow**  
Call first to get `reference_distribution`, then use that in calibration endpoints. No population required for this call.

**Input (body, JSON)**

| Parameter      | Type   | Required | Description |
|----------------|--------|----------|-------------|
| `question`     | string | Yes      | Question text. |
| `responses`    | array  | Yes      | List of response strings. |
| `demographics` | array  | No       | Optional list of demographic objects (key-value per respondent). |

**Output (200)**  
`{"question", "n_responses": <int>, "reference_distribution": { "<option>": <proportion>, ... }}`

---

## API reference (OpenAPI / Swagger)

| Resource | URL |
|----------|-----|
| **OpenAPI 3.1 spec** | `http://localhost:8000/openapi.json` |
| **Swagger UI** | `http://localhost:8000/docs` |
| **ReDoc** | `http://localhost:8000/redoc` |

**Import from OpenAPI into Postman:**  
In Postman: **Import** → **Link** → paste `http://localhost:8000/openapi.json` (server must be running). This generates a collection from the spec. For a ready-made collection with example bodies and variables, use the files below.

---

## Setup

1. **Start the API server**
   ```bash
   python main.py run
   ```
   API base URL: `http://localhost:8000`.

2. **Import into Postman**
   - **Collection:** `JADU_Full_API.postman_collection.json` (all endpoints)  
     or `Socio_Sim_AI_Surveys.postman_collection.json` (survey-only flow).
   - **Environment:** `Socio_Sim_AI.postman_environment.json`
   - In Postman: **File → Import** → select the collection and environment → set active environment to **Socio Sim AI - Local**.

3. **Environment variables** (optional to edit)
   - `base_url` — default `http://localhost:8000`
   - `session_id` — set automatically by **POST /survey/multi** (for multi-survey polling)
   - `survey_id` — set from **POST /survey** response when testing analytics/evaluation
   - `agent_id` — use any agent id from **GET /agents** when testing **GET /agents/:agent_id**

---

## Testing all APIs – recommended order

Run in this order so dependent endpoints have data.

### 1. Population

| Request | Method | Purpose |
|---------|--------|--------|
| **POST /population/generate** | POST | Create synthetic population (e.g. 50 agents). **Run first** before survey/simulation/analytics. |

Body: `n` (10–10000), `method` (`monte_carlo` \| `bayesian` \| `ipf`), `id_prefix`, optional `seed`.

---

### 2. Agents

| Request | Method | Purpose |
|---------|--------|--------|
| **GET /agents** | GET | List agents (optional query: `location`, `nationality`, `limit`, `offset`). |
| **GET /agents/:agent_id** | GET | Get one agent (persona + state). Set `agent_id` from list. |

---

### 3. Survey

| Request | Method | Purpose |
|---------|--------|--------|
| **POST /survey** | POST | Single question → immediate full response. Use for one-question-at-a-time. |
| **GET /survey/:survey_id/results** | GET | Get results by `survey_id` from POST /survey response. |
| **POST /survey/multi** | POST | Multi-question survey; returns `session_id`. |
| **GET /survey/session/:session_id/progress** | GET | Poll multi-survey progress. |
| **GET /survey/session/:session_id/results** | GET | Get multi-survey results when completed. |
| **GET /survey/session/:session_id/round/:round_idx** | GET | Get one round’s results (0-indexed). |

**Tip:** After **POST /survey**, copy `survey_id` from the response into the environment variable `survey_id` to use **GET /survey/:survey_id/results**, **GET /analytics/:survey_id**, and **POST /evaluate/:survey_id**.

---

### 4. Simulation

| Request | Method | Purpose |
|---------|--------|--------|
| **POST /simulation** | POST | Run N days of simulation (requires population). |
| **POST /simulation/events** | POST | Schedule event (e.g. `price_change`, `policy`, `infrastructure`). |
| **GET /simulation/events** | GET | List pending events and global params. |
| **GET /simulation/status** | GET | Population size, graph loaded, pending events. |
| **POST /simulation/scenario** | POST | Run named scenario (no global state change). |
| **POST /simulation/scenario/compare** | POST | Compare two scenarios (macro diff). |
| **POST /simulation/scenario/run-with-survey** | POST | Run scenario then survey post-scenario population. |
| **POST /simulation/scenario/compare-with-survey** | POST | Run two scenarios, survey both, compare. |
| **GET /simulation/causal/graph** | GET | Default causal graph. |
| **POST /simulation/causal/do-intervention** | POST | Do-intervention (counterfactual). |
| **POST /simulation/causal/ate** | POST | Estimate Average Treatment Effect. |
| **POST /simulation/causal/learn** | POST | Learn causal structure from timeline. |

---

### 5. Analytics

| Request | Method | Purpose |
|---------|--------|--------|
| **GET /analytics/:survey_id** | GET | Segmented analytics. Query: `segment_by` = `location` \| `income` \| `nationality` \| `age`. |

Requires a `survey_id` from a completed survey (e.g. POST /survey).

---

### 6. Evaluation

| Request | Method | Purpose |
|---------|--------|--------|
| **POST /evaluate/:survey_id** | POST | Run evaluation on survey responses (realism, drift, judge, similarity). |
| **GET /evaluate/:evaluation_id/report** | GET | Get evaluation report (e.g. use `survey_id` as `evaluation_id` if stored). |

Requires population and a `survey_id` that has results.

---

### 7. Discovery

| Request | Method | Purpose |
|---------|--------|--------|
| **POST /discovery/domains/auto-setup** | POST | Create domain config (e.g. `domain_name`, `description`, `sample_questions`, `city_name`). |
| **POST /discovery/dimensions** | POST | Discover behavioral/belief dimensions from a list of questions. |

---

### 8. Calibration

| Request | Method | Purpose |
|---------|--------|--------|
| **POST /calibration/auto-weights** | POST | Optimize factor weights for multiple questions and reference distributions. Requires population. |
| **POST /calibration/fit** | POST | Calibrate one question to a reference distribution. |
| **POST /calibration/upload-data** | POST | Upload real responses; returns reference distribution. |

---

## Quick flow: one question at a time (survey)

1. **POST /population/generate** — e.g. `{"n": 50, "method": "bayesian", "id_prefix": "DXB"}`.
2. **POST /survey** — e.g. `{"question": "How often do you use delivery? Options: Never, Rarely, Often", "question_id": "q1", "use_archetypes": false}`.
3. Copy `survey_id` from response → set env `survey_id`.
4. **GET /survey/{{survey_id}}/results** — same results.
5. **GET /analytics/{{survey_id}}?segment_by=location** — segmented analytics.
6. **POST /evaluate/{{survey_id}}** — run evaluation (body optional; defaults for judge/similarity).

---

## Collections in this folder

| File | Description |
|------|-------------|
| **JADU_Full_API.postman_collection.json** | All endpoints: population, agents, survey, simulation, analytics, evaluation, discovery, calibration. |
| **Socio_Sim_AI_Surveys.postman_collection.json** | Survey-focused: setup, single questions 1–5, multi-survey + poll + results. |
| **Socio_Sim_AI.postman_environment.json** | Environment with `base_url`, `session_id`, `survey_id`, `agent_id`. |

Use **JADU_Full_API** to test every API; use **Socio_Sim_AI_Surveys** for a short survey-only workflow.
