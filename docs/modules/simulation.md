# Simulation Module

Simulation kernel, survey orchestration, archetypes, cascade detection, and event scheduling.

## engine.py

**Purpose**: Daily simulation kernel with 13-step causally-correct pipeline.

### Constants

| Constant | Description |
|----------|-------------|
| `VECTORIZE_THRESHOLD` | 200. Use vectorized path when N ≥ 200. |

### Functions

| Function | Description |
|----------|-------------|
| `run_simulation(agents, days, social_graph, scheduler, seed, config, enable_social, enable_macro, world_state)` | Run simulation for N days. Pre-computes sparse adjacency. |
| `run_daily_step(agents, social_graph, scheduler, day, ...)` | One day: events, research, media frames, exposure, attention, life events, cultural influence, cognitive processing, alignment, activation, social diffusion, cascade detection, macro feedback. |

### Pipeline Steps

1. Scheduled events
2. Research context
3. Media frames
4. Raw selective exposure
5. Adaptive attention
6. Life events
7. Cultural influence
8. Cognitive processing (media → beliefs)
9. Alignment computation
10. Activation update
11. Social diffusion
12. Cascade detection, emergent events, fatigue
13. Macro feedback, world feedback

---

## orchestrator.py

**Purpose**: Survey distribution, social warmup, archetype compression, narrative adaptation.

### Functions

| Function | Description |
|----------|-------------|
| `run_survey(agents, question, question_id, think_fn, use_archetypes, max_concurrent)` | Run survey: social warmup, neighbor latent means, default_think (or custom), archetype representatives + adapted narratives. |
| `_social_warmup(agents, social_graph, warmup_steps)` | Run diffusion steps before survey. |
| `_compute_neighbor_latent_means(agents, social_graph, sample_k)` | Per-agent neighbor mean latent vectors. |
| `_adapt_narrative(template, source_persona, target_persona, sampled_option)` | Fuzzy token substitution for archetype adaptation. |

---

## survey_engine.py

**Purpose**: Multi-question survey with persistent state, event-driven scheduler.

### Classes

| Class | Description |
|-------|-------------|
| `SurveyEngineConfig` | use_archetypes, social_influence_between_rounds, summarize_every, social_warmup_steps, recluster_every, etc. |
| `RoundResult` | round_idx, question, question_id, responses, elapsed_seconds. |
| `SurveyEngine` | Run multi-question rounds; optional archetype mode; social diffusion between rounds; memory summarization. |

### Methods

| Method | Description |
|--------|-------------|
| `run(questions)` | Schedule rounds on EventDrivenScheduler; process all; return SurveySessionResult. |
| `on_progress(callback)` | Register async callback after each round. |
| `get_progress()` | Current session progress snapshot. |

### Coordinator integration

[`SurveyEngine`](../../simulation/survey_engine.py) constructs a [`SimulationCoordinator`](../../simulation/coordinator.py) by default. After each round it calls `enforce_distribution_health` on responses; after all rounds it may call `compute_population_health` for aggregate checks. This surfaces entropy / collapse / correlation issues without replacing per-agent decision logic.

---

## coordinator.py

**Purpose**: **Population-level enforcement** between survey rounds — distribution health (entropy, dominant share, optional JS vs reference), and hooks into [`evaluation/invariants.check_population_invariants`](../../evaluation/invariants.py).

### Classes

| Class | Description |
|-------|-------------|
| `PopulationHealth` | `entropy_status`, `collapse_status`, `correlation_status`, `invariant_violations`. |
| `SimulationCoordinator` | `enforce_distribution_health`, `compute_population_health`, round history. |

---

## archetypes.py

**Purpose**: KMeans clustering for LLM cost reduction.

### Functions

| Function | Description |
|----------|-------------|
| `build_archetype_map(personas, archetype_count)` | KMeans cluster; return (rep_indices, labels). |
| `build_archetype_states(agents, archetype_count, ...)` | Aggregate agent states per cluster (median/trimmed_mean/mean). |
| `build_archetype_graph(social_graph, labels, n_clusters)` | Coarsened graph over archetypes. |
| `refresh_archetype_states(archetype_states, agents, labels, aggregation)` | Update archetype states from current agent states. |

---

## archetype_runner.py

**Purpose**: Archetype-based survey round execution.

### Functions

| Function | Description |
|----------|-------------|
| `run_archetype_round(archetype_states, labels, agents, question, question_id, ...)` | Run survey using archetype representatives; expand to members with noise; narrative budget for non-representatives. |

---

## cascade_detector.py

**Purpose**: Activation dynamics, cluster detection, emergent events, fatigue.

### Functions

| Function | Description |
|----------|-------------|
| `update_activation(activation, exposure_agg, emotion_agg, topic_importance, alignment, neighbor_act, susceptibility, ...)` | Activation update with decay, validation/outrage weights. |
| `compute_neighbor_activation(activation, adj_norm)` | Mean neighbor activation. |
| `detect_activation_clusters(activation, sparse_adj, activation_threshold, min_size_absolute, min_size_fraction, min_density)` | Sparse connected-component clusters above threshold. |
| `generate_emergent_event(cluster, agent_states_dict, total_population)` | Generate emergent event from cluster (e.g. protest, movement). |
| `apply_fatigue(activation, cluster, cooldown_topics, fatigue_factor, cooldown_days)` | Reduce activation in cluster; set cooldown. |
| `tick_cooldowns(cooldown_topics)` | Decrement cooldown counters. |

---

## event_queue.py

**Purpose**: Priority-queue event-driven scheduler.

### Classes

| Class | Description |
|-------|-------------|
| `EventDrivenScheduler` | Schedule SimEvents by time; process_all_async; register handlers by event_type. |
| `SimEvent` | time, agent_id, event_type, payload. |

---

## scenario.py

**Purpose**: Scenario A/B comparison.

### Functions

| Function | Description |
|----------|-------------|
| `run_scenario(agents, scenario_config, social_graph)` | Deep-copy agents, run scenario, return result (dimension_means, etc.). |
| `compare_scenarios(agents, scenario_a, scenario_b, social_graph)` | Run both; return diff of macro metrics. |

### Classes

| Class | Description |
|-------|-------------|
| `ScenarioConfig` | name, days, seed, events. |
| `ScenarioEvent` | day, type, payload, district. |

---

## macro.py

**Purpose**: Population-level aggregation and influence.

### Functions

| Function | Description |
|----------|-------------|
| `compute_macro_metrics(agents)` | Population means per dimension, population_size. |
| `macro_influence(macro)` | Convert macro to influence signals for apply_macro_influence. |

---

## dispatch.py

**Purpose**: Event routing to agents.

### Classes

| Class | Description |
|-------|-------------|
| `EventDispatcher` | Routes SimEvents to agent handlers. `dispatch(event, agent)` for survey_question events. |

---

## world_feedback.py

**Purpose**: World state feedback loop — demand-driven event generation.

### Classes

| Class | Description |
|-------|-------------|
| `WorldState` | apply_demand_feedback(macro, day, max_events_per_step) — generates events from macro demand signals. |

---

## config.py

**Purpose**: Simulation configuration.

### Classes

| Class | Description |
|-------|-------------|
| `SimulationConfig` | master_seed, days, vectorize_threshold. make_rng(), derive_child_seed(). |

---

## timeline.py

**Purpose**: Timeline utilities for time handling.
