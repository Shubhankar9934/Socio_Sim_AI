# Causal Module

Causal structure learning from simulation time-series data and a lightweight structural causal model (SCM) over dimensions, beliefs, and actions. Used for attribution and counterfactual analysis.

## learner.py

**Purpose**: Learn causal structure from simulation timeline data using Granger-causality-inspired analysis on dimension trajectories.

### Classes

| Class | Description |
|-------|-------------|
| `CausalLearner` | significance_threshold, lag. learn_from_timeline(timeline, dimension_names): learns causal graph from timeline of dimension_means snapshots; uses lagged cross-correlation as proxy for Granger causality. Returns CausalGraph. |

### Timeline format

Each timeline entry: `{"day": int, "dimension_means": dict}`. dimension_names default from first snapshot keys.

---

## graph.py

**Purpose**: Structural causal graph supporting do-intervention, Average Treatment Effect (ATE) with confounding adjustment, topological propagation, and counterfactual queries.

### Classes

| Class | Description |
|-------|-------------|
| `CausalEdge` | cause, effect, weight, mechanism (optional). |
| `CausalGraph` | nodes, edges (cause, effect) → CausalEdge. add_node(), add_edge(), remove_edge(). Child/parent indexing for propagation. |

### Functions

| Function | Description |
|----------|-------------|
| `build_default_causal_graph()` | Return a default graph when timeline is insufficient. |

### Capabilities

- **Do-intervention**: Set a variable and propagate effects.
- **ATE**: Average Treatment Effect estimation with confounding adjustment.
- **Topological propagation**: Causal effects along graph.
- **Counterfactual queries**: What-if analysis.
