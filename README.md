# JADU – Synthetic Society Simulation Platform

A research-grade **synthetic population simulation** platform combining agent-based modeling, LLM reasoning, behavioral economics, and probabilistic population synthesis. The system simulates a Dubai-like population that answers surveys, is influenced by social networks, and can be validated and evaluated.

## Features

- **Synthetic population generation**: Monte Carlo, IPF (Iterative Proportional Fitting), and Bayesian conditional sampling to match real Dubai demographics (age, nationality, income, location, occupation).
- **Agent cognitive architecture**: Perception → Memory → Personality → Decision (probabilistic) → LLM reasoning. Hybrid model: statistics from probability, narrative from LLM.
- **Social network**: Barabasi-Albert graph with friend/coworker/neighbor edges; opinion diffusion and adoption cascades.
- **World model**: Dubai districts (metro access, parking, restaurant density); economy (budget allocation); events (e.g. new metro station).
- **Memory**: Episodic, semantic, and behavioral memory with ChromaDB (or in-memory fallback).
- **Survey orchestration**: Async execution across agents; optional archetype compression to reduce LLM calls (e.g. 500 agents → ~80 LLM calls).
- **Analytics**: Aggregation by segment (location, income, nationality, age); automated insights and visualization.
- **Evaluation**: Population realism (JS divergence), drift detection, cross-question consistency, LLM-as-judge (realism, persona consistency, cultural plausibility).

## Quick start

### 1. Install

```bash
cd JADU
pip install -r requirements.txt
```

### 2. Environment

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY` – required for agent reasoning and optional LLM judge.

### 3. Run API

```bash
python main.py run
```

Server: `http://0.0.0.0:8000`. Docs: `http://localhost:8000/docs`.

### 4. Typical flow

1. **Generate population**  
   `POST /population/generate` with `{"n": 500, "method": "bayesian"}`.

2. **Optional: run simulation**  
   `POST /simulation` with `{"days": 30}` to update social influence and state.

3. **Run survey**  
   `POST /survey` with `{"question": "How often do you order food delivery?", "diagnostics": false}`. Set `"diagnostics": true` for extra per-agent debug fields. Returns `survey_id`.

4. **Get analytics**  
   `GET /analytics/{survey_id}?segment_by=location`.

5. **Evaluate**  
   `POST /evaluate/{survey_id}` with `{"run_judge": false}` (or `true` for LLM judge).

### NLU cache and multi-survey runs

Hybrid question understanding (`build_turn_understanding_hybrid`) caches results in-process. Each `POST /survey` call gets a fresh `survey_run_id` so batches do not reuse understanding from an earlier API request. For long-lived Python processes (notebooks, custom scripts), pass `survey_run_id=` into [`run_survey`](simulation/orchestrator.py) or call `clear_turn_understanding_cache()` from [`agents.intent_router`](agents/intent_router.py) between studies.

## Deep-dive API docs (JADU_Full_API)

**Vision and scope:** [docs/vision-and-goals.md](docs/vision-and-goals.md). **Curated JSON examples:** [docs/examples/](docs/examples/).

Per-area documentation with request/response fields, code flow, and how outputs are computed: **[docs/jadu-api/](docs/jadu-api/)** (also in the **MkDocs** site / GitHub Pages under *JADU API reference (Postman)*). A short pointer remains at [docs2/README.md](docs2/README.md).

## API summary

| Method | Path | Description |
|--------|------|-------------|
| POST | /discovery/domains/auto-setup | Create new domain (domain_name, sample_questions, etc.) |
| POST | /discovery/dimensions | Discover behavioral/belief dimensions from questions |
| POST | /calibration/auto-weights | Optimize factor weights against reference distributions |
| POST | /calibration/fit | Calibrate single question |
| POST | /calibration/upload-data | Upload real survey data, get reference distribution |
| POST | /population/generate | Generate synthetic population (n, method, seed) |
| GET | /agents | List agents (filter by location, nationality) |
| GET | /agents/{id} | Agent detail (persona + state); `?debug=true` adds `decision_profile` |
| POST | /survey | Run survey; optional `diagnostics`, `use_archetypes`, `current_events` |
| GET | /survey/{id}/results | Get survey results by id |
| POST | /survey/multi | Multi-question session; progress via `WS /ws/survey/{session_id}` |
| WS | /ws/survey/{session_id} | Round/session events during multi-survey (see [docs/jadu-api/websockets.md](docs/jadu-api/websockets.md)) |
| WS | /ws/simulation | Simulation commands (inject_event, status); see websockets doc |
| POST | /simulation | Run N days of simulation |
| GET | /simulation/status | Population size and graph status |
| GET | /analytics/{survey_id} | Segmented analytics and insights |
| POST | /evaluate/{survey_id} | Evaluation; optional `reference_distribution`, `question_model_key` in body |
| GET | /evaluate/{id}/report | Placeholder for stored report |

## Project layout

- `config/` – Settings, domain config (domain.py), demographics from JSON (demographics.py), question models, belief mappings.
- `data/domains/` – Domain configs per city (domain.json, demographics.json, reference_distributions.json).
- `discovery/` – Dimension discovery, domain auto-setup, action inference.
- `causal/` – Causal learner and graph.
- `population/` – Synthesis (Monte Carlo, IPF, Bayesian), personas, validator, constraints, lazy store, life paths.
- `core/` – Seeded RNG helpers for reproducible stochastic behavior.
- `agents/` – Cognitive pipeline (perception, intent routing, response contract, personality, decision, state, memory tiers).
- `llm/` – OpenAI client (rate-limited), prompts, reasoner.
- `social/` – Network (Barabasi-Albert), influence (opinion diffusion).
- `world/` – City graph, districts, economy, events.
- `memory/` – Types (episodic, semantic, behavioral), ChromaDB store.
- `simulation/` – Engine (daily loop), orchestrator (survey), archetypes, timeline.
- `analytics/` – Aggregation, insights, visualization.
- `evaluation/` – Realism, consistency, drift, LLM judge, report.
- `calibration/` – Factor weight learning, real data loader, calibration pipeline.
- `api/` – FastAPI app, schemas, routes.

## CLI

- `python main.py run` – Start API server.
- `python main.py generate [N]` – Generate N agents (default 100) and print realism score.

## Documentation

Full documentation is in `docs/`:

- **Vision** – [docs/vision-and-goals.md](docs/vision-and-goals.md)
- **Overview** – [docs/index.md](docs/index.md), [docs/overview.md](docs/overview.md), [docs/architecture.md](docs/architecture.md)
- **Maintainers** – [docs/DOC_INVENTORY.md](docs/DOC_INVENTORY.md) (routes ↔ docs ↔ tests)
- **MkDocs** – Build the docs site with `mkdocs serve` (see `mkdocs.yml`). Install dependencies with `pip install -r requirements-docs.txt` if present, then open http://127.0.0.1:8000
- **Module reference** – `docs/modules/` (one file per module: agents, api, calibration, config, discovery, etc.)

## Scaling and cost

- **Archetypes**: For 500+ agents, set `use_archetypes: true` in the survey request to route LLM calls through cluster representatives (~80 calls instead of 500).
- **Memory**: Set `CHROMA_PERSIST_DIR` to persist agent memory; otherwise in-memory store is used.
- **Validation**: Adjust `POPULATION_REALISM_THRESHOLD` and `DRIFT_THRESHOLD` in `.env` or config.

## License

MIT.