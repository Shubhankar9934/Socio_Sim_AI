# Simulation API

Source collection: **JADU_Full_API** → folder `simulation`.

**Prerequisite:** `POST /population/generate` (agents + `social_graph` in `api.state`).

---

## POST `/simulation`

Runs the **main** time-stepping loop on the **live** `agents_store` (mutates global state).

### Body (`SimulateRequest`)

| Field | Type | Default | Range | Purpose |
|-------|------|---------|-------|---------|
| `days` | int | 30 | 1–365 | Number of simulation days to advance. |

### Response

`{ "status": "ok", "days": <int>, "n_agents": <int> }`

### Flow

1. [api/routes/simulation.py](../../api/routes/simulation.py) → `run_simulation(agents_store, days, social_graph, scheduler)`  
2. [simulation/engine.py](../../simulation/engine.py) — applies scheduler events, social influence, state updates per day.

---

## POST `/simulation/events`

Schedules a **future** event (does not run simulation by itself).

### Body (`EventInjectRequest`)

| Field | Purpose |
|-------|---------|
| `day` | Simulation day index when the event fires (≥ 0). |
| `type` | e.g. `price_change`, `policy`, `infrastructure`, `market`, `new_service`, `new_metro_station`. |
| `payload` | Type-specific parameters (e.g. `{ "service": "delivery", "change_pct": -10 }`). |
| `district` | Optional geographic scope. |

### Response

`status`, `event_type`, `day`, `pending_events` (count).

### Flow

Builds [world/events.py](../../world/events.py) `SimulationEvent` → `app_state.event_scheduler.add(event)`.

---

## GET `/simulation/events`

Lists `pending_events` and `global_params` from the scheduler.

---

## GET `/simulation/status`

`population_size`, `social_graph_loaded`, `pending_events` count.

---

## POST `/simulation/scenario`

Runs a **copy** of the population through a scenario **without** mutating the global `agents_store`.

### Body (`ScenarioRunRequest`)

| Field | Purpose |
|-------|---------|
| `name` | Label for the scenario. |
| `days` | Length of run. |
| `seed` | Optional RNG seed for the isolated run. |
| `events` | List of timed `ScenarioEvent` objects (`day`, `type`, `payload`, `district`). |

### Response

Includes `dimension_means` (and related aggregates from [simulation/scenario.py](../../simulation/scenario.py) `run_scenario`).

### Flow

`run_scenario(agents_store, scenario, social_graph)` — deep copy agents, simulate, return metrics.

---

## POST `/simulation/scenario/compare`

### Body (`ScenarioCompareRequest`)

- `scenario_a`, `scenario_b`: same shape as single scenario.

### Flow

`compare_scenarios(...)` — runs both, returns `dimension_diff_b_minus_a`, `causal_attribution_b`, timelines, etc. ([simulation/scenario.py](../../simulation/scenario.py)).

---

## POST `/simulation/scenario/run-with-survey`

### Body (`ScenarioWithSurveyRequest`)

| Field | Purpose |
|-------|---------|
| `scenario` | `ScenarioRunRequest` shape. |
| `questions` | List of question strings (min 1). After scenario, each question is run as a survey on the **post-scenario** clone. |

### Response

Scenario summary + `belief_means`, `survey_results` (per question), `timeline`.

### Flow

`run_scenario_with_survey` in [simulation/scenario.py](../../simulation/scenario.py) → internal calls to survey/orchestrator pipeline per question.

---

## POST `/simulation/scenario/compare-with-survey`

### Body (`ScenarioCompareWithSurveyRequest`)

`scenario_a`, `scenario_b`, `questions[]`.

### Flow

`compare_scenarios_with_survey` — two isolated runs + surveys + statistical comparison of distributions ([simulation/scenario.py](../../simulation/scenario.py)).

---

## GET `/simulation/causal/graph`

Returns default **nodes** and **edges** from the causal module (static graph structure for documentation / UI).

**Implementation:** [api/routes/simulation.py](../../api/routes/simulation.py) → [causal/graph.py](../../causal/graph.py) `build_default_causal_graph()` → `g.to_dict()`.

---

## POST `/simulation/causal/do-intervention`

### Body

- `intervention`: map variable → clamped value (e.g. `{ "price": 0.8 }`).
- `observational`: optional baseline context.

### Response

`counterfactual_values` — propagated latent / outcome values under the structural model.

**Flow:** `build_default_causal_graph()` then `g.do(intervention, observational)` ([causal/graph.py](../../causal/graph.py)).

---

## POST `/simulation/causal/ate`

### Body

`treatment`, `outcome`, `confounders[]`, `treatment_value`, `control_value`.

### Response

`{ "treatment", "outcome", "ate" }` — from `g.estimate_ate(...)` on the default graph (may be `0.0` if the outcome variable is not connected / implemented in that estimator).

---

## POST `/simulation/causal/learn`

### Body

`timeline` — array of observations for structure learning.

### Response

Learned `nodes` / `edges` (or default graph if timeline empty — see your sample).

**Flow:** [causal/learner.py](../../causal/learner.py) `CausalLearner.learn_from_timeline(timeline)` → `g.to_dict()`.

---

## Summary: global vs isolated simulation

| Endpoint | Mutates `agents_store`? |
|----------|-------------------------|
| `POST /simulation` | **Yes** |
| `POST /simulation/events` | No (queues only) |
| `POST /simulation/scenario*` | **No** (uses copies) |
| Causal POSTs | Typically **No** (analytical / counterfactual) |
