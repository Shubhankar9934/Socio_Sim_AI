# API Module

FastAPI application, shared state, schemas, WebSocket manager, and route handlers for population, agents, survey, simulation, analytics, evaluation, discovery, calibration, and WebSocket streaming.

## app.py

**Purpose**: Create the FastAPI app with CORS, lifespan, and all routers.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `lifespan(app)` | Async context manager for startup/shutdown. | Yields once; startup can optionally pre-generate a default population (commented out). No shutdown logic. |
| `create_app()` | Build and return the FastAPI application. | Instantiates FastAPI with title "JADU", version, lifespan; adds CORSMiddleware (allow_origins=["*"], credentials, all methods/headers); includes routers for population, agents, survey, simulation, analytics, evaluation, discovery, calibration, websocket; returns app. |

### Routers (order of inclusion)

- `population` — `/population`
- `agents` — `/agents`
- `survey` — `/survey`
- `simulation` — `/simulation`
- `analytics` — `/analytics`
- `evaluation` — `/evaluate`
- `discovery` — `/discovery`
- `calibration` — `/calibration`
- `websocket` — (prefix `/ws` on route paths)

---

## state.py

**Purpose**: Module-level shared state used by routes and simulation.

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `agents_store` | List[Dict] | List of {persona, state, social_trait_fraction, location_quality}. Each entry is one agent. |
| `survey_results` | Dict[str, Dict] | survey_id → {survey_id, question, question_id, responses, items}. |
| `social_graph` | Any (NetworkX) | Barabasi-Albert social graph; set after population generate. |
| `event_scheduler` | EventScheduler | World event scheduler (world.events). |
| `event_driven_scheduler` | EventDrivenScheduler | Priority-queue event-driven scheduler for simulations. |
| `response_histories` | Dict[str, List] | agent_id → list of {question_id, survey_id, answer, sampled_option}. |
| `survey_sessions` | Dict | session_id → {session_id, total_rounds, current_round, status, completed_questions, result, error}. |

---

## schemas.py

**Purpose**: Pydantic request/response models for all API endpoints.

### Request models

| Model | Fields | Description |
|-------|--------|-------------|
| `GeneratePopulationRequest` | n, method, seed, id_prefix | Population size 10–10000, method monte_carlo/bayesian/ipf. |
| `SurveyRequest` | question, question_id, use_archetypes, options | Single question; options = None treated as open text. |
| `SurveyQuestionItem` | question, question_id, options | One item in multi-survey questions list. |
| `MultiSurveyRequest` | questions, use_archetypes, social_influence_between_rounds, summarize_every | Multi-question survey config. |
| `SimulateRequest` | days | 1–365. |
| `EventInjectRequest` | day, type, payload, district | Event types: price_change, policy, infrastructure, market, new_service, new_metro_station. |
| `ScenarioEventRequest` | day, type, payload, district | One event in a scenario. |
| `ScenarioRunRequest` | name, days, seed, events | Named scenario run. |
| `ScenarioCompareRequest` | scenario_a, scenario_b | Compare two scenarios. |
| `ScenarioWithSurveyRequest` | scenario, questions | Run scenario then survey. |
| `ScenarioCompareWithSurveyRequest` | scenario_a, scenario_b, questions | Compare two scenarios with surveys. |
| `EvaluateRequest` | run_judge, judge_sample, realism_threshold, drift_threshold, run_similarity, similarity_threshold | Evaluation options. |

### Response / nested models

| Model | Description |
|-------|-------------|
| `AgentSummary` | agent_id, age, nationality, income, location, occupation. |
| `AgentDetail` | agent_id, persona (dict), state (dict or None). |
| `AgentDemographics` | age_group, nationality, income_band, location, occupation, household_size, family_children, has_spouse. |
| `AgentLifestyle` | cuisine_preference, diet, hobby, work_schedule, health_focus, commute_method. |
| `SurveyResponseItem` | agent_id, answer, sampled_option, distribution, demographics, lifestyle, error. |
| `SurveyResult` | survey_id, question, responses (list of SurveyResponseItem), n_total. |
| `SurveyQuestionItem` | question, question_id, options. |
| `MultiSurveyProgress` | session_id, current_round, total_rounds, status, completed_questions. |
| `RoundResultItem` | round_idx, question, question_id, responses, n_total, elapsed_seconds. |
| `SurveySessionResult` | session_id, questions, rounds, total_responses, elapsed_seconds, status. |
| `AnalyticsResponse` | survey_id, segment_by, aggregated, insights. |
| `EvaluateRequest` | run_judge, judge_sample, realism_threshold, drift_threshold, run_similarity, similarity_threshold. |
| `DashboardMetrics` | duplicate_narrative_rate, persona_realism_score, distribution_similarity, consistency_score, drift_rate, mean_judge_score. |
| `EvaluationReportResponse` | population_realism, drift, consistency_score, distribution_validation, narrative_similarity, llm_judge, dashboard, quantitative_metrics, summary. |

---

## websocket.py (api/websocket.py)

**Purpose**: Connection manager for WebSocket channels (e.g. survey session, simulation). Multiple clients can subscribe to the same channel and receive broadcast messages.

### Class: ConnectionManager

| Method | Description | How |
|--------|-------------|-----|
| `connect(channel, ws)` | Register a WebSocket on a channel. | Accepts connection; appends ws to _active[channel]. |
| `disconnect(channel, ws)` | Remove WebSocket from channel. | Filters ws out of _active[channel]; deletes key if empty. |
| `broadcast(channel, data)` | Send JSON message to all clients on channel. | Serializes data with json.dumps(default=str); sends to each ws; removes stale connections on failure. |
| `send_personal(ws, data)` | Send JSON to a single client. | Same serialization; single ws.send_text. |
| `channels` (property) | List of active channel names. | Returns list(_active.keys()). |
| `subscriber_count(channel)` | Number of connections on channel. | len(_active.get(channel, [])). |

### Module variable

| Variable | Description |
|----------|-------------|
| `ws_manager` | Singleton ConnectionManager instance. |

---

## routes/

### population.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/population/generate` | POST | Generate synthetic population. | Validates n ≤ 10000; calls generate_population then validate_population; for each persona builds AgentState, location_quality; builds social graph; computes social_trait_fraction per agent via fraction_friends_with_trait; clears and fills agents_store; returns n, method, realism_passed, realism_score, per_attribute, segment_distribution. |

### agents.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/agents` | GET | List agents with optional filters. | Query: location, nationality, limit, offset. Slices agents_store; filters by location/nationality if provided; maps to AgentSummary. |
| `/agents/{agent_id}` | GET | Get one agent by id. | Looks up persona by agent_id; returns AgentDetail(persona.model_dump(), state.to_dict()); 404 if not found. |

### survey.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/survey` | POST | Run single-question survey. | Requires agents_store. Gets LLM client, resets survey stats; calls run_survey(agents_store, question, question_id, options, think_fn=None, use_archetypes); stores result in survey_results and response_histories; returns SurveyResult with new survey_id. |
| `/survey/{survey_id}/results` | GET | Get stored survey results. | Returns SurveyResult from survey_results[survey_id]; 404 if missing. |
| `/survey/multi` | POST | Start multi-question survey. | Creates session in survey_sessions; starts background task _run_multi_survey_task; returns MultiSurveyProgress immediately. Progress and completion stream via WebSocket channel survey:{session_id}. |
| `/survey/session/{session_id}/progress` | GET | Poll multi-survey progress. | Returns current_round, status, completed_questions from survey_sessions. |
| `/survey/session/{session_id}/results` | GET | Full multi-survey results. | Requires status != running; returns SurveySessionResult from session result. |
| `/survey/session/{session_id}/round/{round_idx}` | GET | One round's results. | Looks up survey_results["{session_id}_r{round_idx}"]; returns SurveyResult. |

### simulation.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/simulation` | POST | Run N days of simulation. | Calls run_simulation(agents_store, days, social_graph, event_scheduler); returns status, days, n_agents. |
| `/simulation/events` | POST | Schedule a world event. | Builds SimulationEvent; adds to event_scheduler; returns status, event_type, day, pending_events count. |
| `/simulation/events` | GET | List pending events. | Returns pending_events list and global_params from event_scheduler. |
| `/simulation/status` | GET | Current status. | Returns population_size, social_graph_loaded, pending_events. |
| `/simulation/scenario` | POST | Run named scenario. | Converts body to ScenarioConfig; calls run_scenario; returns name, days, seed, population_size, dimension_means. |
| `/simulation/scenario/compare` | POST | Compare two scenarios. | Runs both scenarios; returns compare_scenarios result. |
| `/simulation/scenario/run-with-survey` | POST | Run scenario then survey. | run_scenario_with_survey; returns scenario result plus survey_results, timeline. |
| `/simulation/scenario/compare-with-survey` | POST | Compare two scenarios with surveys. | compare_scenarios_with_survey; returns comparison with survey results. |
| `/simulation/causal/graph` | GET | Default causal graph. | build_default_causal_graph().to_dict(). |
| `/simulation/causal/do-intervention` | POST | Do-intervention query. | Request: intervention, observational. Builds graph; returns counterfactual_values. |
| `/simulation/causal/ate` | POST | Estimate ATE. | Request: treatment, outcome, confounders, treatment_value, control_value. Returns ate. |
| `/simulation/causal/learn` | POST | Learn causal graph from timeline. | Request: timeline. CausalLearner.learn_from_timeline; returns graph dict. |

### analytics.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/analytics/{survey_id}` | GET | Segmented analytics. | Query: segment_by (location, income, nationality, age). Loads responses and personas from survey_results and agents_store; aggregate_with_personas; generate_insights; appends delivery_frequency_insight if possible; returns survey_id, segment_by, aggregated, insights. 404 if survey not found. |

### evaluation.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/evaluate/{survey_id}` | POST | Run evaluation on survey. | Loads survey responses and personas; calls run_evaluation with body params; export_evaluation_report to JSON file; returns EvaluationReportResponse. |
| `/evaluate/{evaluation_id}/report` | GET | Placeholder. | If evaluation_id in survey_results returns message; else 404. |

### discovery.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/discovery/domains/auto-setup` | POST | Create new domain. | DomainAutoSetup.setup_domain(domain_name, description, sample_questions, city_name, currency, reference_data); returns domain_id and message. |
| `/discovery/dimensions` | POST | Discover dimensions from questions. | DimensionDiscovery.discover_dimensions(questions, n_behavioral, n_belief); optionally save_discovered_dimensions(domain_id, result) if save and domain_id; returns behavioral, belief, question_to_dimension, saved. |

### calibration.py

| Endpoint | Method | Description | How |
|----------|--------|-------------|-----|
| `/calibration/auto-weights` | POST | Factor weight optimization. | FactorWeightLearner.learn_weights(questions, reference_distributions, agents_store); returns overall_loss and results per question (learned_weights, best_loss, converged). |
| `/calibration/fit` | POST | Single-question calibration. | FactorWeightLearner.learn_weights_for_question(question, reference_distribution, agents_store); returns question, learned_weights, best_loss, converged, n_iterations. |
| `/calibration/upload-data` | POST | Upload real data, get reference distribution. | RealSurveyData.from_raw(question, responses, demographics); returns question, n_responses, reference_distribution. |

### websocket.py (routes)

| Endpoint | Description | How |
|----------|-------------|-----|
| `/ws/survey/{session_id}` | Stream survey progress for a session. | Connect to channel "survey:{session_id}"; server pushes round_complete and session_complete events; client can send any text (e.g. ping); on disconnect, unregister. |
| `/ws/simulation` | Bidirectional simulation channel. | Connect to channel "simulation". Server can push step updates; client can send JSON: action "inject_event" (day, type, payload, district), "pause", "resume", or "status". inject_event adds event to event_scheduler; status returns population_size. |
