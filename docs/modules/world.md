# World Module

Dubai city model, districts, economy, events, life events, and culture.

## model.py

**Purpose**: Dubai as graph of districts and connections.

### Functions

| Function | Description |
|----------|-------------|
| `build_city_graph()` | NetworkX graph: nodes = districts with properties; edges = connections with distance_km. |
| `get_city_graph()` | Module-level singleton. |
| `districts_with_metro(G)` | List districts with metro access. |

---

## districts.py

**Purpose**: District properties and location quality.

### Classes

| Class | Description |
|-------|-------------|
| `DistrictProperties` | metro_access, parking_availability, restaurant_density, etc. |
| `DEFAULT_DISTRICT_PROPERTIES` | Registry of district configs. |

### Functions

| Function | Description |
|----------|-------------|
| `get_district(name)` | Return DistrictProperties for district. |
| `location_quality_for_satisfaction(location)` | 0–1 quality score for satisfaction questions (parking, transport). |

---

## economy.py

**Purpose**: Budget allocation for food delivery.

### Functions

| Function | Description |
|----------|-------------|
| `food_delivery_budget_share(income, luxury_preference, household_size)` | Fraction of food budget plausible for delivery (0–1). Used by income_factor. |

---

## events.py

**Purpose**: World event scheduling.

### Classes

| Class | Description |
|-------|-------------|
| `SimulationEvent` | day, type, payload, district. |
| `EventScheduler` | add(event), process_until(day), get_environment(). Maintains global_params, event_dimension_impacts, event_belief_impacts. |

---

## life_events.py

**Purpose**: Life events (marriage, job change, etc.).

### Functions

| Function | Description |
|----------|-------------|
| `sample_life_events(persona, state, rng, social_graph, agents)` | Sample triggered life events for agent. |
| `apply_life_event(persona, state, event, rng)` | Apply behavioral_impacts, belief_impacts, demographic_changes. |

---

## culture.py

**Purpose**: District cultural norms and influence.

### Functions

| Function | Description |
|----------|-------------|
| `get_effective_cultural_field(location)` | Cultural field for district. |
| `apply_cultural_influence(latent_state, cultural_field)` | Nudge latent dimensions toward district norms. |
| `build_cultural_matrix(agents)` | (N × 12) cultural target matrix. |
| `vectorized_cultural_influence(latent_matrix, cultural_matrix)` | Batch cultural influence. |
