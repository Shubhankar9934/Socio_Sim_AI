# Evaluation Module

Population realism, behavioral drift, cross-question consistency, distribution validation, narrative similarity, and LLM-as-judge. Produces a unified report and dashboard for quality assessment.

## report.py

**Purpose**: Orchestrate all evaluation steps and export a standalone JSON report.

### Constants

| Constant | Description |
|----------|-------------|
| QUALITY_TARGETS | duplicate_narrative_rate (<0.05), persona_realism_score (>0.90), distribution_similarity (>0.85), consistency_score (>0.90), drift_rate (<0.10), mean_judge_score (>3.50). Each has target string, threshold, and direction (above/below). |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `run_evaluation(..., reference_distribution, question_model_key)` | Build full report. | Same pipeline as before; distribution step uses explicit `reference_distribution` or resolves reference via `question_model_key` when set. Sets `consistency_valid` from cross-question data availability. Returns report dict. |
| `export_evaluation_report(report, output_path, system_info)` | Write JSON file. | Builds export with timestamp, system_info, quantitative_metrics, dashboard, distribution_validation, narrative_similarity (no flagged_pairs), population_realism, summary; json.dump to output_path. Returns absolute path. |

---

## realism.py

**Purpose**: Population realism — aggregate score and report vs target demographic distributions.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `compute_realism_score(personas)` | Single score in [0, 1]. | Delegates to population.validator.population_realism_score(personas). |
| `compute_realism_report(personas, threshold)` | Full report. | Calls validate_population(personas, realism_threshold); returns population_realism_score, passed, threshold, per_attribute. |

---

## drift.py

**Purpose**: Detect agents whose survey answers have drifted from persona baseline (e.g. primary_service_preference); optional auto-reset.

### Constants

| Constant | Description |
|----------|-------------|
| _ANSWER_TO_BEHAVIOR | Maps answer keywords ("multiple", "daily", "3-4", "1-2", "rarely", "never") to 0–1 behavioral value. |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `drift_score(initial_behavior, current_behavior)` | Absolute difference. | abs(initial - current). |
| `infer_current_behavior(response_history)` | Map response history to 0–1. | Iterate responses; for each answer match keywords in _ANSWER_TO_BEHAVIOR and update current value; default 0.5 if empty. |
| `detect_drift(agent_id, persona, response_history, threshold)` | (is_drifted, magnitude). | Baseline = persona.lifestyle.primary_service_preference; current = infer_current_behavior(hist); magnitude = drift_score(baseline, current); return (magnitude > threshold, magnitude). |
| `drift_report(personas, response_histories, threshold, agent_states, auto_reset)` | Per-agent drift. | For each persona, detect_drift; collect drifted ids and per_agent magnitude; if auto_reset and agent_states, call state.reset_behavior(blend=0.7) for drifted. Returns drifted_agent_ids, count, rate, threshold, per_agent_magnitude, auto_reset_count. |

---

## consistency.py

**Purpose**: Cross-question logical consistency (e.g. "rarely" delivery vs "5 times last week" = inconsistent).

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `check_frequency_consistency(q1_answers, q2_answers, agent_ids)` | (rate, inconsistent_ids). | For each agent, if q1 is "rarely"/"never"/"0" and q2 contains high-frequency cues ("3","4","5","week","times"), mark inconsistent. rate = 1 - (inconsistent / checked). |
| `consistency_score_from_responses(response_sets, agent_ids)` | Average pairwise consistency. | For each pair of question response sets, extract agent_id → answer maps and call check_frequency_consistency; average rates. |

---

## distribution_validation.py

**Purpose**: Compare observed survey distribution to reference (JS divergence, chi-square, per-option diff); pass/fail for calibration.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `aggregate_survey_distribution(responses, key)` | Observed proportions. | Count responses by key (default sampled_option); normalize to proportions. |
| `compare_to_reference(observed, reference, significance)` | Full comparison. | Union keys; normalize both; jensenshannon for JS; 1 - JS = js_similarity; chi-square test (scaled); per_option observed/reference/diff; passed = js_similarity >= 0.85. |
| `validate_survey_distribution(responses, reference, key, question_model_key)` | End-to-end. | If reference None and question_model_key set, get_reference_distribution(question_model_key); aggregate_survey_distribution; compare_to_reference; add observed_distribution, n_responses, question_model_key. |

---

## similarity.py

**Purpose**: Detect duplicate or near-duplicate narrative answers using sentence embeddings and cosine similarity.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `compute_narrative_similarity(narratives, threshold, model_name)` | Duplicate rate and stats. | SentenceTransformer encode; cosine_similarity matrix; count pairs above threshold in upper triangle; duplicate_rate = duplicate_pairs / total_pairs; return duplicate_rate, duplicate_pairs, total_pairs, threshold, mean_similarity, max_similarity, flagged_pairs (up to 20). Target: duplicate_rate < 0.05. |

---

## judge.py

**Purpose**: LLM-as-judge: score responses for realism, persona consistency, cultural plausibility (1–5).

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `judge_response(persona, question, response, model)` | Single response scores. | persona.to_compressed_summary(); build_judge_prompt(summary, question, response); chat with JUDGE_SYSTEM and user prompt; parse JSON from response (regex \{...\}); return realism, persona_consistency, cultural_plausibility (1–5); default 3 on parse failure. |
| `judge_responses_batch(personas, questions, responses, sample_size)` | Batch with optional sampling. | If sample_size set, random.sample indices; for each (persona, question, response) call judge_response; aggregate average realism, persona_consistency, cultural_plausibility. Returns scores list, average dict, n_judged. |

---

## invariants.py

**Purpose**: **Formal invariants** after each agent response and **population-level** checks for survey rounds.

### Agent-level

| Function | Role |
|----------|------|
| `run_agent_invariants` | Entry from [`AgentCognitiveEngine.think`](../../agents/cognitive.py) — consistency vs structured memory, distribution sums, etc. |
| `invariant_consistency` | No large jumps vs prior answer for same semantic key. |

### Population-level

| Function | Role |
|----------|------|
| `check_population_invariants` | Aggregate checks used by [`SimulationCoordinator`](../../simulation/coordinator.py). |

**Tests:** [`tests/test_system_invariants.py`](../../tests/test_system_invariants.py).

---

## runtime_metrics.py

**Purpose**: **Live session metrics** — `MetricsCollector` ingests per-response trace dicts (`record_response`, `finalize_round`) and exposes `SessionMetrics` (diversity entropy, duplicate rate, invariant violation counts, intent distribution, mean confidence, word-count stats). Complements post-hoc `run_evaluation` for long multi-round runs.

---

## __init__.py

Package marker; may re-export evaluation symbols.
