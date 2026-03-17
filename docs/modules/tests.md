# Tests Module

Unit and integration tests for the JADU platform. **Note:** At present there is no `tests/` directory in the repository; this page describes the intended test layout and how to run tests once test files are added.

## Intended Test Files

When a test suite is added, the following modules are expected to be covered:

| File | Validates |
|------|-----------|
| `test_archetype_adaptation.py` | Archetype narrative adaptation, token substitution |
| `test_macro_feedback.py` | Macro influence, population-level aggregation |
| `test_factor_graph_integration.py` | Factor graph, decision pipeline integration |
| `test_architect_refinements_v2.py` | Architecture refinements (v2) |
| `test_cognitive_pipeline.py` | Full cognitive pipeline (perceive, recall, decide, reason) |
| `test_narrative_diversity.py` | Narrative styles, banned patterns, consistency |
| `test_events_expanded.py` | Event system, SimEvent handling |
| `test_belief_network.py` | BeliefNetwork, init, update, diffusion |
| `test_dissonance.py` | Cognitive dissonance, consistency score |
| `test_architecture_refinements.py` | Architecture refinements |
| `test_evaluation_pipeline.py` | Evaluation report, realism, drift, consistency |
| `test_stability_corrections.py` | Stability guarantees, bias pipeline |
| `test_behavioral_state.py` | BehavioralLatentState, EMA, social influence |
| `test_risk_fixes.py` | Risk mitigations (archetype aggregation, etc.) |
| `test_calibration.py` | Calibration optimizer, parameter space |
| `test_cultural_fields.py` | Cultural influence, district norms |
| `test_life_events.py` | Life events sampling and application |

## Running Tests

Once tests exist under `tests/`:

```bash
pytest tests/ -v
```

For a specific file:

```bash
pytest tests/test_cognitive_pipeline.py -v
```

## Test Configuration

Use `config.domain.reset_domain_cache()` and `config.demographics.reset_demographics_cache()` in fixtures if tests need a clean config state. Factor graph cache can be cleared via `agents.factor_graph.clear_graph_cache()` for isolated decision tests.
