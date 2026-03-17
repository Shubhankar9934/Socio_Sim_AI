# Root Scripts

Entry point and standalone scripts at the project root. These are run directly (e.g. `python main.py run`) and call into the API, population, simulation, and evaluation layers.

## main.py

**Purpose**: CLI entry point for running the API server or one-off population generation.

### Usage

| Command | Description |
|---------|-------------|
| `python main.py run` | Start the FastAPI server (uvicorn) on `0.0.0.0:8000`. Loads `api.app:app`. |
| `python main.py generate [N]` | Generate N synthetic agents (default 100), validate realism, and print score and per-attribute breakdown. |

### How it works

- **run**: Imports uvicorn and runs `uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)`.
- **generate**: Imports `generate_population` from population.synthesis and `validate_population` from population.validator; calls `generate_population(n=N, method="bayesian")` then `validate_population(personas)`; prints realism score and per_attribute; does not start the API or persist agents.

---

## regenerate_survey.py

**Purpose**: Regenerate the 500-agent survey dataset with full realism stack (segmentation, structured memory, narrative styles, social warmup) and write results to `survey_response.json`. No API server required.

### Usage

```bash
python regenerate_survey.py [N]
```

Default N=500. Output: `survey_response.json`.

### How it works

1. **Generate population**: `generate_population(n, method="bayesian", seed=42)`.
2. **Report segments and narrative verbosity**: Counter over population_segment and narrative_style.verbosity.
3. **Build agents**: For each persona, `AgentState.from_persona(p)`; each agent dict has persona, state, social_trait_fraction, location_quality.
4. **Social graph**: `build_social_network(personas, seed=42)`; stored in `api.state.social_graph`.
5. **Social trait fraction**: `fraction_friends_with_trait` and `location_quality_for_satisfaction` per agent; `state.set_social_trait_fraction(frac)`.
6. **Run survey**: Single question (e.g. "How often do you order food delivery?"); `run_survey(agents, question, question_id, think_fn=None, use_archetypes=...)` via simulation.orchestrator.
7. **Write output**: Responses and metadata written to `survey_response.json`.

Uses asyncio: `asyncio.run(regenerate(n))`.

---

## benchmark_scale.py

**Purpose**: Profile performance of activation update, cluster detection, and bias pipeline at 10K, 50K, and 100K agents. No API or LLM; uses synthetic sparse graphs and random vectors.

### Usage

```bash
python benchmark_scale.py
```

### How it works

- **Sparse graph**: `_build_sparse_ba_graph(n, m=5)` builds a Barabasi-Albert-like sparse adjacency matrix (scipy.sparse). `_normalize_adj` row-normalizes for diffusion.
- **benchmark_activation(n, n_frames)**: Random activation, exposure, emotion, alignment, susceptibility; `compute_neighbor_activation` then `update_activation` from simulation.cascade_detector; returns elapsed time.
- **benchmark_cluster_detection(n)**: Random activation (top 10% set high); `detect_activation_clusters`; returns elapsed time.
- **benchmark_bias_pipeline(n)**: For each of n “agents” (mock state/context), calls `apply_all_biases` from agents.biases; returns total time (sequential).
- **Main**: Runs benchmarks at 10_000, 50_000, 100_000; prints table of seconds per size. Requires scipy for sparse matrices.

---

## validate_realism.py

**Purpose**: Demonstrate realism improvements without LLM calls. Generates a population and runs checks for lifestyle/hobby diversity, archetype distribution, cultural priors, conviction-profile diversity, demographic plausibility, and banned-pattern detection.

### Usage

```bash
python validate_realism.py
```

### How it works

1. **Generate population**: `generate_population(n=500, method="bayesian", seed=42)`.
2. **Lifestyle field realism**: Counter over hobbies and health_focus; print top hobbies and health distribution.
3. **Archetype distribution**: Counter over personal_anchors.archetype; print distribution.
4. **Cultural priors**: For sample nationalities (Western, Indian, Pakistani, Emirati, Filipino), `get_cultural_prior(persona)` and print prior over options.
5. **Conviction profiles**: Perceive food-delivery question; for a sample of personas, `assign_conviction_profile`, build DecisionContext, `compute_distribution`, `sample_from_distribution`; show distribution shape diversity (e.g. bimodal vs diffuse).
6. **Demographic plausibility**: Sample responses; `validate_demographic_plausibility(persona, sampled_option)`; report implausible combos.
7. **Banned patterns**: Check `is_banned_pattern(text)` on sample strings.
8. **Vague-answer logic**: `should_use_vague_answer` and `pick_vague_answer` for sample persona/question.

Uses agents.decision, agents.factor_graph, agents.realism, agents.narrative, agents.perception, agents.personality, config.question_models, population.synthesis. No API, no network.
