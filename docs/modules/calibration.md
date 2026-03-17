# Calibration Module

Gradient-free calibration to match simulated survey distributions to real-world data: factor weight optimization (differential evolution), real data loading, parameter space, optimizer, eval-feedback composite loss, and end-to-end pipeline.

**API**: See [API module](api.md) for routes: `POST /calibration/auto-weights`, `POST /calibration/fit`, `POST /calibration/upload-data`.

---

## __init__.py

Package marker; exports calibration engine description. No public symbols listed.

---

## auto_weights.py

**Purpose**: Optimize factor weights per question against reference distributions using scipy differential evolution. Targets factor_weights (personality, income, social, location, memory, behavioral, belief) to minimize Jensen-Shannon divergence between simulated and reference distributions.

### Constants

| Constant | Description |
|----------|-------------|
| `FACTOR_NAMES` | ["personality", "income", "social", "location", "memory", "behavioral", "belief"]. |

### Dataclasses

| Class | Description |
|-------|-------------|
| `WeightLearningResult` | question, learned_weights (dict), best_loss, n_iterations, converged. |
| `AutoWeightsResult` | results (list of WeightLearningResult), overall_loss. |

### Helper functions

| Function | Description | How |
|----------|-------------|-----|
| `_weights_to_vector(weights)` | Map factor name → float dict to fixed-order vector. | np.array([weights.get(f, 0.1) for f in FACTOR_NAMES]). |
| `_vector_to_weights(vec)` | Map vector back to factor name → float dict. | {FACTOR_NAMES[i]: round(vec[i], 4)}. |
| `_weight_bounds()` | Bounds for DE. | [(-0.5, 1.0)] * len(FACTOR_NAMES). |

### Class: FactorWeightLearner

| Method | Description | How |
|--------|-------------|-----|
| `__init__(n_iterations, seed)` | Store iteration count and RNG seed. | — |
| `learn_weights_for_question(question, reference_distribution, agents, simulate_fn)` | Optimize factor weights for one question. | If simulate_fn is None, builds default simulator that: perceives question, gets question_model, patches factor_weights, samples up to 100 agents with compute_distribution + sample_from_distribution, returns empirical distribution. Objective: JS(simulated, reference). Runs differential_evolution with _weight_bounds(), maxiter=n_iterations, popsize=10; returns WeightLearningResult. |
| `learn_weights(questions, reference_distributions, agents)` | Optimize for multiple questions. | For each question with a reference distribution, calls learn_weights_for_question; accumulates results and average loss; returns AutoWeightsResult. |
| `_build_default_simulator(question, agents)` | Build (factor_weights -> distribution) callable. | Returns a function that patches question_model.factor_weights, builds DecisionContext per agent, runs compute_distribution and sample_from_distribution, aggregates counts into proportion dict. |

---

## data_loader.py

**Purpose**: Ingest real survey data (raw lists, CSV, or JSON), compute reference and segmented distributions, and support holdout splits for train/test calibration.

### Class: RealSurveyData

| Attribute | Description |
|-----------|-------------|
| question | str |
| responses | List[str] |
| demographics | List[Dict[str, str]] |

| Method / Property | Description | How |
|-------------------|-------------|-----|
| `n_responses` | len(responses). | Property. |
| `to_reference_distribution()` | Aggregate into option → proportion. | Count each response; normalize by total; return dict with rounded proportions. |
| `to_segmented_distributions(segment_by)` | Per-segment distributions. | Group responses by demographics[i][segment_by]; per segment count and normalize. |
| `holdout_split(train_fraction, seed)` | Train/test split. | Shuffle indices with rng; split at train_fraction; return two RealSurveyData instances (train, test). |
| `from_raw(question, responses, demographics)` | Factory. | cls(question=question, responses=responses, demographics=demographics or []). |

### Class: RealDataLoader

| Method | Description | How |
|--------|-------------|-----|
| `load_csv(path, question_col, answer_col, demographics_cols)` | Load from CSV file. | DictReader; group by question column; collect answers and optional demographic row dicts; return question → RealSurveyData. |
| `load_json(path)` | Load from JSON file. | Expects {question: {responses: [...], demographics: [...]}} or {question: [responses]}; returns question → RealSurveyData. |
| `load_csv_text(text, question_col, answer_col, demographics_cols)` | Same as load_csv but from string. | Same logic as load_csv using StringIO. |

---

## eval_feedback.py

**Purpose**: Closed-loop feedback: combine evaluation metrics (realism, drift, consistency, distribution fit, narrative quality) into a composite loss so calibration can minimize a multi-objective score via the same optimizer.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `evaluation_composite_loss(eval_report, simulated_dist, real_dist, w_distribution, w_realism, w_drift, w_consistency, w_narrative)` | Single scalar loss (lower = better). | JS via calibration_loss(simulated, real); realism_loss = 1 - population_realism_score; drift_loss = clip(drift rate); consistency_loss = 1 - consistency_score; narrative_loss = clip(duplicate_rate). Composite = weighted sum of these; clip to [0, 1]. |
| `make_eval_aware_simulator(run_fn, personas, real_distribution)` | Wrap run_fn so optimizer sees composite loss. | Returns simulator(params) that: calls run_fn(params) for distribution and survey_responses; runs run_evaluation(personas, survey_responses) (async handled with get_event_loop or ThreadPoolExecutor); computes evaluation_composite_loss; injects __composite_loss__ into returned dict so downstream can use it. |
| `closed_loop_calibrate(run_fn, personas, real_distribution, parameter_space, n_iterations, seed)` | End-to-end calibration with full evaluation. | Builds eval-aware simulator; calls calibrate(real_distribution, parameter_space, simulator, n_iterations, seed). |

---

## optimizer.py

**Purpose**: Gradient-free calibration by minimizing Jensen-Shannon divergence between simulated and real distributions using scipy differential_evolution.

### Class: CalibrationResult

| Attribute | Description |
|-----------|-------------|
| best_params | SimulationParameters |
| best_loss | float |
| loss_history | List[float] |
| n_iterations | int |
| converged | bool |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `calibration_loss(simulated, real)` | Jensen-Shannon divergence. | Union of keys; normalize both dicts to probabilities; scipy jensenshannon(p, q). |
| `calibration_report(result)` | Summarise for logging/export. | Dict with best_loss, n_iterations, converged, best_params (personality_weights, factor_weights, temperature), loss_curve (last 20). |
| `calibrate(real_distribution, parameter_space, simulator, n_iterations, population_size, seed)` | Run differential evolution. | Bounds from parameter_space.bounds(). Objective: from_vector(vec) -> params, simulator(params) -> sim_dist, calibration_loss(sim_dist, real). If simulator None, uses uniform over keys. Appends loss to loss_history. Returns CalibrationResult(best_params=from_vector(result.x), best_loss=result.fun, ...). |

---

## parameter_space.py

**Purpose**: Wrap all tunable simulation parameters (personality weights, factor weights, temperature) into a single vector interface for scipy optimizers.

### Class: SimulationParameters

| Attribute | Description |
|-----------|-------------|
| personality_weights | Dict (e.g. convenience_preference, price_sensitivity, primary_service_preference, dining_out). |
| factor_weights | Dict (personality, income, social, location, memory, behavioral). |
| temperature | float |

| Method / Property | Description | How |
|--------------------|-------------|-----|
| `__post_init__` | Set key order for vectorization. | _key_order_personality = sorted(personality_weights); _key_order_factors = sorted(factor_weights). |
| `n_params` | Total scalar count. | len(personality) + len(factors) + 1. |
| `to_vector()` | Flatten to numpy array. | [personality_weights[k] for k in order] + [factor_weights[k] for k in order] + [temperature]. |
| `from_vector(vec)` | Reconstruct from flat vector. | Read personality, then factor, then temperature; return new SimulationParameters; temperature clamped to min 0.1. |
| `bounds()` | (lower, upper) per parameter. | (-1, 1) per personality, (0, 1) per factor, (0.1, 5.0) for temperature. |

---

## pipeline.py

**Purpose**: End-to-end calibration: real data → holdout split per question → factor weight learning on train → validation on test with JS divergence.

### Class: CalibrationReport

| Attribute | Description |
|-----------|-------------|
| weights_result | AutoWeightsResult or None |
| train_js_divergence | Dict[str, float] per question |
| test_js_divergence | Dict[str, float] per question |
| overall_train_loss | float |
| overall_test_loss | float |
| n_questions | int |
| converged | bool |

| Method | Description | How |
|--------|-------------|-----|
| `to_dict()` | Export for logging. | Dict with n_questions, overall_train_loss, overall_test_loss, converged, train_js_divergence, test_js_divergence, learned_weights (question -> weights from weights_result.results). |

### Class: CalibrationPipeline

| Method | Description | How |
|--------|-------------|-----|
| `__init__(n_iterations, holdout_fraction, seed)` | Store config. | — |
| `fit(real_data, agents)` | Full pipeline. | Skip questions with &lt;10 responses. Per question: holdout_split; build train_refs, test_refs (to_reference_distribution). FactorWeightLearner.learn_weights(questions, train_refs, agents). For each WeightLearningResult: train_js from best_loss; test_js by running _build_default_simulator with learned_weights and computing JS vs test_refs. overall_train_loss = mean(train_js); overall_test_loss = mean(test_js). Return CalibrationReport. |
