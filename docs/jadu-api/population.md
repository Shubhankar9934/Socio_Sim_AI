# Population API

Source collection: **JADU_Full_API** → folder `population` → `POST /population/generate`.

---

## Endpoint

| Method | Path | Description |
|--------|------|-------------|
| **POST** | `/population/generate` | Create `n` synthetic personas, validate realism, build `AgentState` per agent, build social graph, replace in-memory population. |

---

## Request body (`GeneratePopulationRequest`)

Defined in [api/schemas.py](../../api/schemas.py) as `GeneratePopulationRequest`.

| Field | Type | Required | Constraints | Purpose |
|-------|------|----------|-------------|---------|
| `n` | int | no (default 500) | 10–10000 in schema; route also rejects `n > 10000` | Number of synthetic agents to create. |
| `method` | string | no (default `"bayesian"`) | One of `"monte_carlo"`, `"bayesian"`, `"ipf"` | Which synthesis algorithm to use (see below). |
| `id_prefix` | string | no (default `"DXB"`) | any string | Prefix for `agent_id` (e.g. `DXB_0000`). Lets you tag runs or cities. |
| `seed` | int or null | no | optional | RNG seed for reproducible draws (personas, graph, noisy sampling). `null` = non-deterministic where supported. |

### Example

```json
{
  "n": 50,
  "method": "bayesian",
  "id_prefix": "DXB",
  "seed": null
}
```

---

## Response body (200)

| Field | Type | Meaning |
|-------|------|---------|
| `n` | int | `len(personas)` after generation (should match requested `n` if no internal cap). |
| `method` | string | Echo of request `method`. |
| `realism_passed` | bool | `true` if aggregate realism score ≥ threshold from settings. |
| `realism_score` | float | Mean of per-demographic “1 − JS divergence” scores (see below). **Excludes** `multimodality` and `segment_entropy` from the mean. |
| `per_attribute` | object | Per-attribute diagnostics: marginal fit + extras. |
| `segment_distribution` | object | Count of agents per `population_segment` (not proportions). |

### Example response (annotated)

```json
{
  "n": 50,
  "method": "bayesian",
  "realism_passed": true,
  "realism_score": 0.9118,
  "per_attribute": {
    "age": 0.9177,
    "nationality": 0.8807,
    "income": 0.9048,
    "location": 0.8748,
    "household_size": 0.9535,
    "occupation": 0.9393,
    "multimodality": 0.0,
    "segment_entropy": 1.6697
  },
  "segment_distribution": {
    "health_premium": 6,
    "budget_worker": 15,
    "convenience_maximizer": 3,
    "family_homemaker": 11,
    "young_professional": 10,
    "student_explorer": 5
  }
}
```

---

## How each output value is calculated

### `per_attribute` (marginal attributes)

For each demographic attribute in `get_demographics().get_all_marginals()` ([population/validator.py](../../population/validator.py) + [config/demographics.py](../../config/demographics.py)):

1. **Target distribution** `target_dist`: from domain demographics (marginals for age, nationality, income, location, household_size, occupation).
2. **Empirical distribution** `empirical`: share of each category across generated `Persona` list (`_distribution_from_personas`).
3. **Jensen–Shannon divergence** `JS(target, empirical)` between aligned probability vectors (`jensen_shannon_divergence`).
4. **Stored value** for that key: `1.0 - JS` (higher = closer match to target). Same as “similarity” to the reference marginal.

So `per_attribute.age` ≈ 0.92 means the synthetic age histogram is close to the configured age marginal.

### `multimodality`

- From `multimodality_score(personas)` in [population/validator.py](../../population/validator.py).
- Builds latent trait vectors per persona (personality + `init_from_persona`), stacks to matrix `X`.
- If `len(personas) < 30` or `sklearn` missing → **0.0**.
- Otherwise fits two Gaussian Mixture Models (1 vs `k` components, `k` derived from segment count), compares BIC; returns a score in `[0,1]` indicating how much a multi-modal latent structure is favored.
- **Not** included in `realism_score` mean (diagnostic only).

### `segment_entropy`

- `segment_distribution(personas)` → empirical probabilities over `Persona.meta.population_segment`.
- `segment_entropy` = Shannon entropy of that probability vector (`scipy.stats.entropy`), in nats.
- Higher = more spread across segments; lower = concentrated in few segments.
- **Not** included in `realism_score` mean.

### `realism_score`

```text
realism_score = mean(per_attribute[k] for k in marginals only)
              = mean of (1 - JS) over age, nationality, income, location, household_size, occupation
```

Threshold: `settings.population_realism_threshold` ([config/settings.py](../../config/settings.py)), compared in [api/routes/population.py](../../api/routes/population.py).

### `realism_passed`

`realism_passed = (realism_score >= population_realism_threshold)`.

### `segment_distribution`

- `Counter(p.meta.population_segment for p in personas)` in [api/routes/population.py](../../api/routes/population.py).
- **Counts**, not fractions. Segment labels come from `_stamp_segments` during synthesis (domain segments config).

---

## End-to-end code flow (trigger → response)

High-level call graph:

1. **FastAPI** → [api/routes/population.py](../../api/routes/population.py) `generate_population_endpoint(body)`
2. **`get_settings()`** → realism threshold
3. **`generate_population(...)`** → [population/synthesis.py](../../population/synthesis.py)
   - Branches: `generate_monte_carlo` | `generate_bayesian` | `generate_ipf` (same file)
   - Post-steps (all methods):
     - `_stamp_archetypes(personas)`
     - `_stamp_segments(personas, seed=...)`
     - `_stamp_narrative_styles(personas, seed=...)`
     - `_stamp_media_subscriptions(personas, seed=...)`
4. **`validate_population(personas, realism_threshold=...)`** → [population/validator.py](../../population/validator.py)  
   Returns `(passed, score, per_attr)`.
5. **Per persona** → [agents/state.py](../../agents/state.py) `AgentState.from_persona(p)` → attached dict `{ persona, state, social_trait_fraction, location_quality }`
6. **`build_social_network(personas, seed=...)`** → [social/network.py](../../social/network.py) (Barabási–Albert style graph); stored in `api.state.social_graph`
7. **`fraction_friends_with_trait`** → [social/influence.py](../../social/influence.py) using trait = `primary_service_preference >= 0.5`; updates `social_trait_fraction` on agent dict and `state.set_social_trait_fraction`
8. **`app_state.agents_store`** cleared and filled with agent dicts
9. **Response JSON** built from `validate_population` outputs + segment counts

### Classes / types involved

- **`Persona`**, **`PersonaMeta`**, **`LifestyleCoefficients`**, etc. → [population/personas.py](../../population/personas.py)
- **`AgentState`** → [agents/state.py](../../agents/state.py)
- **Graph** → network structure in `social` package; referenced as `app_state.social_graph`

### Data sources (why it’s not “one hardcoded population”)

- **Demographics / marginals / conditionals**: loaded for the active **domain** via `get_demographics()` / `get_domain_config()` → typically `data/domains/<domain_id>/demographics.json` and related domain files ([config/domain.py](../../config/domain.py)).

---

## Errors

- **400** if `n > 10000` (route-level check).

---

## Prerequisites / side effects

- **Replaces** the entire in-memory `agents_store` and `social_graph`.
- Must be called **before** surveys, analytics, evaluation, and most calibration flows that need `agents_store`.
