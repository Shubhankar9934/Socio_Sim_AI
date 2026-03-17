# JADU — Computational Model of Collective Human Cognition

A research-grade socio-simulation platform that models how large populations form opinions, react to information, and generate emergent social phenomena under media pressure.

## Documentation

- [Project Overview](overview.md) — What the project does, how pieces connect, features, quick start, scaling, CLI
- [Architecture](architecture.md) — System flow, API and cognitive pipeline, simulation loop, calibration/discovery/evaluation flows, end-to-end flow
- [Root Scripts](root-scripts.md) — main.py, regenerate_survey.py, benchmark_scale.py, validate_realism.py
- **Module Reference** — Per-folder documentation (every file: classes, functions, what and how):
  - [Agents](modules/agents.md) | [Analytics](modules/analytics.md) | [API](modules/api.md) | [Calibration](modules/calibration.md) | [Causal](modules/causal.md) | [Config](modules/config.md) | [Data](modules/data.md) | [Discovery](modules/discovery.md)
  - [Evaluation](modules/evaluation.md) | [LLM](modules/llm.md) | [Media](modules/media.md) | [Memory](modules/memory.md) | [Population](modules/population.md)
  - [Postman](modules/postman.md) | [Research](modules/research.md) | [Simulation](modules/simulation.md) | [Social](modules/social.md) | [Storage](modules/storage.md) | [World](modules/world.md) | [Tests](modules/tests.md)

## Architecture Overview

JADU is a layered system stack:

| Layer | Module | Description |
|-------|--------|-------------|
| 0 | `config/domain.py`, `config/demographics.py` | Domain config and demographics from `data/domains/{id}/` |
| 0b | `discovery/` | Dimension discovery, domain auto-setup, action inference |
| 0c | `calibration/` | Factor weight optimization, real data loader, calibration pipeline |
| 1 | `population/` | Monte Carlo / Bayesian / IPF synthesis of 100K+ agents |
| 2 | `simulation/archetypes.py` | KMeans compression for shared LLM reasoning |
| 3 | `research/` | Web search + fact extraction for shared factual grounding |
| 4 | `media/` | Narrative framing, selective exposure, echo chambers |
| 5 | `media/attention.py` | Adaptive attention (emotion-perception feedback loop) |
| 6 | `agents/biases.py` | Bounded-rational bias engine with residual mixing |
| 7 | `simulation/cascade_detector.py` | Activation dynamics (outrage/validation asymmetry) |
| 8 | `social/` | Barabasi-Albert graph, sparse matrix diffusion |
| 9 | `simulation/cascade_detector.py` | Sparse connected-component cluster detection |
| 10 | `simulation/engine.py` | 13-step causally-correct daily loop |

## Core Mathematical Models

### Bias Stabilization Pipeline

```
D_final = Normalize( (1 - ε) · [γ · F(D₀) + (1 - γ) · D₀] + ε · U )
```

- **γ** (bias susceptibility): `effective_malleability × (1 - topic_importance)^1.5 × (1 - calcification)`, clipped to [0.05, 0.95]
- **ε** (entropy factor): `0.05 + 0.1×(1 - knowledge) + 0.15×media_conflict`, clipped to [0.01, 0.3]
- **F**: Confirmation → Loss aversion → Anchoring → Bandwagon → Availability

### Activation Dynamics

```
A_{t+1} = clip( decay · A_t + (1 - A_t) · (media_term + social_term), 0, 1 )
```

- **media_term** = exposure × emotion × importance × (0.3×validation + 0.6×outrage)
- **social_term** = 0.2 × susceptibility × mean_neighbor_activation
- Outrage is weighted 2× heavier than validation

### Adaptive Attention

```
sharpness = 1 + 5.0 · A^2.0
weights = softmax(sharpness · salience) with entropy floor 0.05
```

High activation → tunnel vision. Low activation → broad processing.

### Gated Peak Alignment

```
alignment = 0.85 · peak_alignment + 0.15 · weighted_mean_alignment
```

Peak is gated by exposure: extreme but unseen frames don't dominate.

## Stability Guarantees

1. **Bias explosion prevention**: Residual mixing anchors output to original distribution
2. **Media over-dominance prevention**: Prior belief weight (0.70) > media (0.15) + social (0.15)
3. **Cascade runaway prevention**: Saturation term `(1-A_t)` + fatigue (70% drop) + 5-day cooldown
4. **Temporal coupling fix**: Beliefs update before activation (cognition before emotion)
5. **Attention collapse prevention**: Entropy floor ensures minimum 5% attention to all frames

## Daily Simulation Loop (13 Steps)

1. Process scheduled events
2. Build shared research context
3. Generate media frames (1 LLM call per source per event)
4. Compute raw selective exposure (subscription filter)
5. Apply adaptive attention (emotion-gated reweighting)
6. Process life events (marriage, job change, etc.)
7. Apply cultural influence (district norms)
8. Cognitive processing — bias engine updates beliefs from media
9. Compute alignment (updated beliefs vs media frames)
10. Update activation (emotional layer via outrage/validation)
11. Social diffusion (sparse matrix belief + trait propagation)
12. Cascade detection → event generation → fatigue application
13. Macro feedback + world state updates

## Emergent Behaviors (Not Hardcoded)

- Echo chamber formation
- Outrage spirals
- Sudden polarization jumps
- Misinformation persistence
- Movement formation
- Burnout cycles
- Cognitive calcification

## Configuration

All parameters are environment-variable configurable via `config/settings.py`. See `.env.example` for the full list.
