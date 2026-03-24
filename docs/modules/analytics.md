# Analytics Module

Response aggregation by segment, automated insights, visualization, and per-step telemetry for simulation runs.

## aggregator.py

**Purpose**: Aggregate survey responses by segment (location, income, nationality, age) for analytics and comparison.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `aggregate_responses(responses, segment_by, answer_key, agent_id_key)` | Group responses by segment and return segment → {answer_value: proportion}. | Builds a DataFrame from responses; if `segment_by` is missing, pulls from nested `persona`; drops NaN; groups by segment and normalizes value_counts per group. |
| `aggregate_with_personas(responses, personas, segment_by, answer_key)` | Join responses with personas to get segment, then aggregate. | Builds persona_by_id map; enriches each response with persona and explicit location/income/nationality/age; calls `aggregate_responses`. Default `answer_key` in code is `sampled_option_canonical`. |
| `verbatim_examples_by_segment(responses, personas, segment_by, limit_per_segment)` | Segment → list of example **`answer`** strings (not keyed by option). | Used by [`GET /analytics`](../../api/routes/analytics.py). |
| `frequency_distribution_by_segment(responses, personas, segment_by)` | Convenience wrapper for `aggregate_with_personas` with `answer_key='answer'`. | Delegates to `aggregate_with_personas`. |

---

## insights.py

**Purpose**: Generate short textual insights from aggregated survey results for dashboards and reports.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `generate_insights(aggregated, segment_name, top_n)` | Produce one insight string per segment (e.g. "segment X: option1: 40%, option2: 30%"). | For each segment, takes top_n answer values by proportion and formats as "value: pct". |
| `compare_segments(aggregated, metric_answer, segment_name)` | Compare one metric (answer value) across all segments. | For each segment, gets proportion for `metric_answer` and returns a list of "{segment_name} {seg}: {metric_answer} = {pct}". |
| `high_frequency_insight(aggregated, high_keys)` | Single summary sentence for high-frequency answers by segment. | Sums proportions for keys in `high_keys` (default delivery-frequency style: "3-4 per week", "daily", etc.) per segment; returns best vs worst segment. |
| `delivery_frequency_insight(aggregated)` | Alias for `high_frequency_insight` (delivery-focused). | Same as `high_frequency_insight` with default high_keys. |

---

## telemetry.py

**Purpose**: Per-step telemetry for simulation runs: lightweight daily snapshots of population-level belief/activation trajectories, event counts, and drift without storing full agent state.

### Classes

| Class | Description |
|-------|-------------|
| `DailySnapshot` | Dataclass: day, activation_mean, activation_std, belief_means, belief_variances, latent_means, polarization, belief_entropy, event_count, cascade_count, population_size, regime, consensus_index, instability. |

| Class | Description |
|-------|-------------|
| `TelemetryCollector` | Accumulates per-day snapshots during a simulation. |

### Methods (TelemetryCollector)

| Method | Description | How |
|--------|-------------|-----|
| `record(agents, day, activation_state, event_count, cascade_count)` | Append one daily snapshot. | Extracts activation array; builds belief/latent vectors from agent states; computes means/variances per dimension, polarization (mean variance), belief entropy; optionally calls `compute_regime_metrics` from cascade_detector for regime/consensus/instability; appends DailySnapshot to list. |
| `to_dicts()` | Export snapshots as list of plain dicts. | Uses dataclasses.asdict for each snapshot. |

---

## visualization.py

**Purpose**: Charts for survey analytics using matplotlib (grouped bar charts, distribution plots).

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `bar_chart_by_segment(aggregated, title, xlabel, output_path)` | Grouped bar chart by segment. | Collects all answer values across segments; plots one bar series per answer with segment on x-axis; saves to file if output_path given; uses Agg backend. |
| `distribution_plot(values, title, output_path)` | Simple histogram or value-count bar plot. | Counts values with Counter; single bar chart of counts; saves to file if output_path given. |

---

## __init__.py

Package marker; no public exports.
