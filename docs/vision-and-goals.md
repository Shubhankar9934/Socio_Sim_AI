# Vision, goals, and scope

JADU (this repository) is a **synthetic population + cognitive survey** stack: you instantiate thousands of statistically grounded personas, optionally run them through a social and media-rich world model, ask survey questions, and obtain **structured distributions** (probabilities over options) plus **LLM narratives** that read as individual answers. The same machinery supports **calibration** to real survey histograms, **scenario and causal** experiments, and **quality evaluation** so outputs are not blindly trusted.

## Problems this stack is meant to address

- **Pre-fielding surveys:** Estimate response distributions and segment splits before spending on panels; test question wording and option sets.
- **Policy and product “what-if”:** Combine world events, media frames, and social diffusion to stress-test how attitudes might shift under shocks (with explicit limits on external validity).
- **Method research:** A testbed for bounded-rationality models, bias engines, memory tiers, archetype compression, and hybrid probabilistic + LLM reasoning.

## What is working end-to-end today

- **HTTP API** ([`api/app.py`](../api/app.py)): population generation, agent listing, single and multi-round surveys, simulation days, scenarios, causal helpers, analytics, evaluation, discovery, calibration — plus **WebSockets** for multi-survey progress ([`docs/jadu-api/websockets.md`](jadu-api/websockets.md)).
- **Hybrid cognition:** Perception and question models drive a **factor-graph decision** ([`agents/decision.py`](../agents/decision.py)); the LLM primarily **expresses** the drawn option as natural language, guided by a **response contract** ([`agents/response_contract.py`](../agents/response_contract.py)) and **context relevance** filtering ([`agents/context_relevance.py`](../agents/context_relevance.py)).
- **Cost controls:** Archetype clustering reduces LLM calls on large populations; diagnostics are opt-in per request.
- **Quality gates:** Population realism (marginals), drift, cross-question consistency, distribution fit vs reference, narrative deduplication, optional LLM judge — see [`evaluation/report.py`](../evaluation/report.py).
- **Determinism hooks:** Seeded RNG packs and agent-level seeds ([`core/rng.py`](../core/rng.py)) support reproducible runs when configured.

## Full potential (where this can grow)

- **Tighter calibration loops:** Persist learned weights per domain question; wire `reference_distribution` / `question_model_key` on evaluation consistently in client UIs.
- **Larger-scale deployment:** `LazyPopulationStore` ([`population/lazy_store.py`](../population/lazy_store.py)) sketches on-demand materialization for very large N with bounded memory.
- **Richer longitudinal panels:** Multi-session surveys with tiered memory ([`agents/memory_manager.py`](../agents/memory_manager.py)) and coordinator-level health checks ([`simulation/coordinator.py`](../simulation/coordinator.py)).
- **Causal and scenario APIs:** Already exposed under `/simulation/causal/*` and scenario compare endpoints; docs and examples can be expanded for teaching and replication.

## Honest limits (non-goals)

- **Not a substitute for real humans:** Synthetic agents encode assumptions; calibrated distributions still reflect model + data choices, not ground truth.
- **LLM cost and latency:** Full surveys at scale require archetypes, concurrency limits, or smaller N; see [`config/settings.py`](../config/settings.py).
- **Domain grounding:** Quality depends on `data/domains/{id}/` JSON, discovery outputs, and reference distributions; garbage in → misleadingly crisp histograms out.
- **WebSocket simulation channel:** The `/ws/simulation` handler implements a subset of actions (e.g. `inject_event`, `status`); pause/resume may be partial — verify against [`api/routes/websocket.py`](../api/routes/websocket.py) before relying on them in production.

## How to read the rest of the docs

| Doc | Use when you need |
|-----|-------------------|
| [Overview](overview.md) | Features, quick start, CLI |
| [Architecture](architecture.md) | Diagrams, pipelines, data flow |
| [JADU API reference](jadu-api/README.md) | HTTP + WS contracts, traces |
| [Module reference](modules/agents.md) | Per-file behavior |
| [DOC_INVENTORY](DOC_INVENTORY.md) | Route ↔ doc ↔ test mapping |

For small copy-paste JSON, see [`examples/README.md`](examples/README.md).
