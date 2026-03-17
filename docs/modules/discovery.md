# Discovery Module

Dimension discovery and domain auto-setup for new cities/regions. Enables adding new domains without hardcoding demographics or question models; dimensions can be discovered from sample questions (embedding + clustering + LLM) and persisted to `data/domains/{domain_id}/`.

**API**: See [API module](api.md) for routes: `POST /discovery/domains/auto-setup`, `POST /discovery/dimensions`.

---

## dimensions.py

**Purpose**: Discover behavioral and belief dimensions from a set of survey questions using sentence-transformers, KMeans, and LLM. The 12 behavioral + 7 belief core dimensions remain the default; discovery can extend them with domain-specific extras.

### Classes

| Class | Description |
|-------|-------------|
| `DiscoveredDimension` | name, description, kind ("behavioral" or "belief"), representative_questions. |
| `DiscoveredDimensions` | behavioral (list of DiscoveredDimension), belief, question_to_dimension (question → dimension weights). Properties: behavioral_names, belief_names. Methods: to_dict(), from_dict(). |
| `DimensionDiscovery` | Discover dimensions from questions. Uses embedding model (default all-MiniLM-L6-v2), clusters, and LLM to name dimensions. |

### Methods (DimensionDiscovery)

| Method | Description |
|--------|-------------|
| `discover_dimensions(questions, n_behavioral, n_belief, domain_id)` | Async. Embed questions, cluster, assign to dimensions, name via LLM. Returns DiscoveredDimensions. |

### Functions

| Function | Description |
|----------|-------------|
| `load_discovered_dimensions(domain_id)` | Load from `data/domains/{domain_id}/discovered_dimensions.json`; returns None if missing. |
| `save_discovered_dimensions(domain_id, dims)` | Persist DiscoveredDimensions to same path. |
| `get_active_dimension_names(domain_id)` | Return (behavioral_names, belief_names) including core + any discovered extras for the domain. |

---

## domain_setup.py

**Purpose**: Auto-generate a complete domain config from a name, description, and sample questions (LLM + dimension discovery).

### Classes

| Class | Description |
|-------|-------------|
| `DomainAutoSetup` | Creates a new domain under `data/domains/{domain_id}/`. |

### Methods

| Method | Description |
|--------|-------------|
| `setup_domain(domain_name, description, sample_questions, city_name, currency, reference_data)` | Async. Create domain_id from name; generate domain.json via LLM; run dimension discovery and save; generate question model overrides; write demographics.json stub if missing; optionally save reference_data as reference_distributions.json. Returns domain_id. |

---

## dimension_monitor.py

**Purpose**: Detect when existing dimensions are insufficient and suggest new dimensions at epoch boundaries. Runs between simulation epochs (not mid-step) to avoid breaking the vectorized N×12 / N×7 pipeline.

### Classes

| Class | Description |
|-------|-------------|
| `AdequacyReport` | day, adequacy_score (0–1), residual_variance, suggested_new_dimensions, event_signals, needs_extension. to_dict(). |
| `DimensionEvolutionMonitor` | Check dimension adequacy across simulation. variance_threshold, min_agents. |

### Methods (DimensionEvolutionMonitor)

| Method | Description | How |
|--------|-------------|-----|
| `check_adequacy(agents, events_log, day)` | Analyze whether current dimensions adequately capture behavior. | Collect latent_state.to_vector() from agents; if &lt; min_agents return AdequacyReport(adequacy_score=1, residual_variance=0); else compute total variance, residual variance (e.g. unexplained spread); adequacy_score = 1 - residual; optionally suggest new dimensions from event_signals; needs_extension = adequacy_score &lt; (1 - variance_threshold). Append to _history; return AdequacyReport. |

### Functions (dimensions.py)

| Function | Description | How |
|----------|-------------|-----|
| `load_discovered_dimensions(domain_id)` | Load from data/domains/{id}/discovered_dimensions.json. | Read JSON; DiscoveredDimensions.from_dict(data); None if missing or error. |
| `save_discovered_dimensions(domain_id, dims)` | Persist to same path. | dims.to_dict(); mkdir parents; write JSON. |
| `get_active_dimension_names(domain_id)` | (behavioral_names, belief_names) including core + discovered. | load_discovered_dimensions; merge with core 12 behavioral + 7 belief. |

---

## action_inference.py

**Purpose**: Classify survey questions into universal action types (frequency, support, rate, adopt, etc.) using rule-based patterns with LLM fallback. Caches results in `data/action_template_cache.json`.

### Constants / Helpers

| Item | Description |
|------|-------------|
| _PATTERN_RULES | List of (keyword_list, ActionTemplate): e.g. ("how often", "per week") → frequency/behavior/ordinal; "do you support" → support/policy; "rate your", "satisfaction" → rate/experience; etc. |
| _cache_key(question) | SHA256 of normalized question text (first 16 chars). |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `infer_action_type_rule(question)` | Rule-based match. | Normalize to lowercase; for each (patterns, template) if any pattern in question return template; else None. |

### Class: ActionModelBuilder

| Method | Description | How |
|--------|-------------|-----|
| `infer_action_type(question, options)` | Cache → rule → LLM. | Look up _ACTION_CACHE[cache_key(question)]; if hit return ActionTemplate from dict. Else infer_action_type_rule(question); if None and LLM available, call LLM to classify action_type, target_category, scale_type; cache and return. |
