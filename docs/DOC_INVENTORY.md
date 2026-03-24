# Documentation ↔ code inventory

Living map of HTTP routes, primary doc pages, and tests. Use when refreshing docs after code changes.

## HTTP routes

| Method | Path | Doc page | Route module | Notes |
|--------|------|----------|--------------|-------|
| POST | `/population/generate` | [jadu-api/population.md](jadu-api/population.md) | `api/routes/population.py` | |
| GET | `/agents` | [jadu-api/agents.md](jadu-api/agents.md) | `api/routes/agents.py` | |
| GET | `/agents/{agent_id}` | [jadu-api/agents.md](jadu-api/agents.md) | `api/routes/agents.py` | `?debug=true` → `decision_profile` |
| POST | `/survey` | [jadu-api/survey.md](jadu-api/survey.md) | `api/routes/survey.py` | `diagnostics` → extra fields |
| GET | `/survey/{survey_id}/results` | [jadu-api/survey.md](jadu-api/survey.md) | `api/routes/survey.py` | |
| POST | `/survey/multi` | [jadu-api/survey.md](jadu-api/survey.md) | `api/routes/survey.py` | WebSocket `survey:{session_id}` |
| GET | `/survey/session/{session_id}/progress` | [jadu-api/survey.md](jadu-api/survey.md) | `api/routes/survey.py` | |
| GET | `/survey/session/{session_id}/results` | [jadu-api/survey.md](jadu-api/survey.md) | `api/routes/survey.py` | |
| GET | `/survey/session/{session_id}/round/{round_idx}` | [jadu-api/survey.md](jadu-api/survey.md) | `api/routes/survey.py` | |
| POST | `/simulation` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/events` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| GET | `/simulation/events` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| GET | `/simulation/status` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/scenario` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/scenario/compare` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/scenario/run-with-survey` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/scenario/compare-with-survey` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| GET | `/simulation/causal/graph` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/causal/do-intervention` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/causal/ate` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| POST | `/simulation/causal/learn` | [jadu-api/simulation.md](jadu-api/simulation.md) | `api/routes/simulation.py` | |
| GET | `/analytics/{survey_id}` | [jadu-api/analytics.md](jadu-api/analytics.md) | `api/routes/analytics.py` | Query: `segment_by`, `answer_key` (default `sampled_option_canonical`); response includes `verbatim_examples` |
| POST | `/evaluate/{survey_id}` | [jadu-api/evaluation.md](jadu-api/evaluation.md) | `api/routes/evaluation.py` | |
| GET | `/evaluate/{evaluation_id}/report` | [jadu-api/evaluation.md](jadu-api/evaluation.md) | `api/routes/evaluation.py` | Reads `evaluation_report_{id}.json`; 404 if absent |
| POST | `/discovery/domains/auto-setup` | [jadu-api/discovery.md](jadu-api/discovery.md) | `api/routes/discovery.py` | |
| POST | `/discovery/dimensions` | [jadu-api/discovery.md](jadu-api/discovery.md) | `api/routes/discovery.py` | |
| POST | `/calibration/auto-weights` | [jadu-api/calibration.md](jadu-api/calibration.md) | `api/routes/calibration.py` | |
| POST | `/calibration/fit` | [jadu-api/calibration.md](jadu-api/calibration.md) | `api/routes/calibration.py` | |
| POST | `/calibration/upload-data` | [jadu-api/calibration.md](jadu-api/calibration.md) | `api/routes/calibration.py` | |
| WS | `/ws/survey/{session_id}` | [jadu-api/websockets.md](jadu-api/websockets.md) | `api/routes/websocket.py` | |
| WS | `/ws/simulation` | [jadu-api/websockets.md](jadu-api/websockets.md) | `api/routes/websocket.py` | |

**Schemas:** [`api/schemas.py`](../api/schemas.py) — keep jadu-api field ledgers in sync.

## Module reference coverage (high-signal files)

| Area | Module doc | Key source files |
|------|------------|------------------|
| Agents | [modules/agents.md](modules/agents.md) | `agents/*.py` including `intent_router`, `response_contract`, `context_relevance`, `memory_manager`, `adaptive_layer`, `behavior_controller` |
| API | [modules/api.md](modules/api.md) | `api/app.py`, `api/state.py`, `api/schemas.py`, `api/websocket.py`, `api/routes/*` |
| Core | [modules/core.md](modules/core.md) | `core/rng.py` |
| Simulation | [modules/simulation.md](modules/simulation.md) | `simulation/orchestrator.py`, `survey_engine.py`, `coordinator.py`, `dispatch.py`, `engine.py` |
| Population | [modules/population.md](modules/population.md) | `synthesis.py`, `personas.py`, `constraints.py`, `lazy_store.py`, `life_path.py` |
| Config | [modules/config.md](modules/config.md) | `settings.py`, `domain.py`, `option_space.py`, `generated_registry.py`, `calibrated_weights.py` |
| Evaluation | [modules/evaluation.md](modules/evaluation.md) | `report.py`, `invariants.py`, `runtime_metrics.py` |

## Suggested tests for traceability

| Concern | Tests |
|---------|--------|
| Docs `examples/*.json` vs `api/schemas.py` | `tests/test_docs_examples.py` |
| Response contract / options | `tests/test_response_contract.py`, `tests/test_option_space.py` |
| RNG / seeds | `tests/test_rng_policy.py` |
| System invariants | `tests/test_system_invariants.py` |
| Survey / orchestration | `tests/test_general_decision_engine.py`, `tests/test_narrative_alignment_guard.py` |

## Recurring route audit

Run [`scripts/doc_route_audit.py`](../scripts/doc_route_audit.py) from the repo root and compare the table above (OpenAPI `/docs` routes are expected extras).

## Large sample payloads

Full captured I/O: [`api_details_input_output.txt`](../api_details_input_output.txt). Curated snippets: [examples/README.md](examples/README.md).
