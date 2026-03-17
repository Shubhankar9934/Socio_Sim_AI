# Population Module

Synthetic population synthesis, personas, segments, and validation.

## synthesis.py

**Purpose**: Monte Carlo, IPF, and Bayesian conditional sampling for Dubai demographics.

### Functions

| Function | Description |
|----------|-------------|
| `generate_population(n, method, seed, id_prefix)` | Unified entry point. method: monte_carlo, bayesian, ipf. Stamps archetypes, segments, narrative styles, media subscriptions. |
| `generate_monte_carlo(n, seed, id_prefix)` | Independent weighted sampling from marginals. |
| `generate_bayesian(n, seed, id_prefix)` | Conditional chain: nationality → income → location → occupation. Age-aware household and family. |
| `generate_ipf(n, seed, id_prefix)` | IPF for age × nationality joint; then Bayesian for rest. |

### Internal Helpers

- `_lifestyle_from_demographics` — Derive lifestyle coefficients with noise
- `_mobility_from_location` — Car and metro usage
- `_personal_anchors_from_demographics` — Cuisine, diet, hobby, work schedule, commute, health
- `_family_from_household` — Conditional probability table for spouse/children
- `_stamp_archetypes`, `_stamp_segments`, `_stamp_narrative_styles`, `_stamp_media_subscriptions`

---

## personas.py

**Purpose**: Persona dataclasses and schema.

### Classes

| Class | Description |
|-------|-------------|
| `Persona` | agent_id, age, nationality, income, location, occupation, household_size, family, mobility, lifestyle, personal_anchors, meta. |
| `FamilyStructure` | spouse, children. |
| `LifestyleCoefficients` | luxury_preference, tech_adoption, dining_out, convenience_preference, price_sensitivity, food_delivery_preference. |
| `MobilityProfile` | car, metro_usage. |
| `PersonalAnchors` | cuisine_preference, diet, hobby, work_schedule, typical_dinner_time, commute_method, health_focus, archetype, narrative_style. |
| `PersonaMeta` | synthesis_method, generation_seed, population_segment, persona_cluster, archetype_id. |

### Methods

| Method | Description |
|--------|-------------|
| `Persona.model_dump()` | Serialize to dict. |
| `Persona.to_compressed_summary()` | Natural language summary. |

---

## segments.py

**Purpose**: Population segment clustering for multimodal behavioral priors.

### Functions

| Function | Description |
|----------|-------------|
| `assign_segment(age, income, location, rng)` | Assign segment label (e.g. young_urban_professional). |
| `sample_latent_from_segment(segment_name, rng)` | Sample latent dimensions from segment Gaussian priors. |

---

## validator.py

**Purpose**: Population realism validation.

### Functions

| Function | Description |
|----------|-------------|
| `validate_population(personas, realism_threshold)` | Returns (passed, score, per_attribute). Compares marginals to target via JS divergence. |
| `population_realism_score(personas)` | Aggregate score. |
| `get_all_marginals(personas)` | Per-attribute distributions. |
