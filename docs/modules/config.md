# Config Module

Application settings (Pydantic from env), domain configuration from JSON, question models, belief mappings, reference distributions, and demographic data (Dubai or `data/domains/{domain_id}/`).

## settings.py

**Purpose**: Pydantic settings loaded from `.env`; single source for all configurable parameters.

### Class: Settings

Uses `pydantic_settings.BaseSettings` with `env_file=".env"`, `extra="ignore"`.

| Category | Fields | Description |
|----------|--------|-------------|
| OpenAI | openai_api_key, openai_agent_model, openai_judge_model | API key and model names for agent and judge. |
| Domain | domain_id, demographics_path | Domain to load; override path for demographics. |
| Simulation | population_size, max_concurrent_llm_calls, simulation_days_default | Defaults for runs. |
| Archetypes | archetype_count, use_archetypes_above_agents | When to use archetype compression. |
| LLM | llm_temperature, llm_top_p | Generation diversity. |
| Validation | population_realism_threshold, drift_threshold | Pass/fail thresholds. |
| ChromaDB | chroma_persist_dir | Persistence dir; empty = in-memory. |
| Multi-survey | max_survey_questions, summarize_memory_every, social_influence_between_rounds, jsonl_output_dir, jsonl_buffer_size, jsonl_flush_interval, jsonl_max_file_size_mb, jsonl_max_file_age_hours | Survey and JSONL writer. |
| Memory | max_summary_length, max_last_answers, max_structured_memory_keys | Agent state caps. |
| Social | social_neighbor_sample_k | Neighbors sampled during diffusion. |
| Archetype execution | archetype_noise_std, narrative_budget, archetype_aggregation, narrative_templates_per_archetype, recluster_every | Risk mitigations. |
| State | vectorize_state_above | Enable StateMatrix above this population size. |
| Scheduler | event_batch_size | Events per scheduler tick. |
| World | enable_world_feedback, max_feedback_events_per_step | World feedback loop. |
| Bias | bias_gamma_floor, bias_gamma_ceiling, bias_epsilon_base, bias_epsilon_floor, bias_epsilon_ceiling, calcification_rate | Bias engine. |
| Media | media_prior_weight, media_influence_weight, social_influence_weight, attention_sharpness_k, attention_sharpness_p, attention_entropy_floor, alignment_beta | Media and attention. |
| Cascade | activation_decay, activation_threshold, cascade_min_size, cascade_min_fraction, cascade_min_density, fatigue_factor, cooldown_days, cooldown_decay, outrage_weight, validation_weight, social_lambda | Cascade detection. |
| Research | research_api_provider, research_cache_path | Web search and cache path. |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `get_settings()` | Cached Settings instance. | `@lru_cache`; first call loads from env. |

---

## domain.py

**Purpose**: Load city/region-specific configuration from `data/domains/{domain_id}/`. All domain content (demographics, cultural priors, prompts, topic keywords, strategic actors) is loaded once via `get_domain_config()`.

### Class: DomainConfig

Dataclass with: city_id, city_name, currency, id_prefix, districts, nationalities, premium_areas; topic_keywords, domain_keywords, topic_to_model_key, location_terms; services, price_levels; cultural_priors, family_modifiers, income_modifiers, implausible_combos, answer_habit_scores, vague_answers; system_prompts, archetype_hints, cultural_hints, frequency_interpretation, lifestyle_keywords; strategic_actors; memory_rules, question_to_semantic_key; reference_distributions, question_belief_map; segments; cuisine_by_nationality, diet_pool, cultural_family_multiplier, late_dinner_nationalities.

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `_load_json(path)` | Load JSON file or return {}. | path.read_text + json.loads if path.exists(). |
| `load_domain_config(domain_id)` | Load full config from directory. | Reads domain.json, demographics.json, reference_distributions.json, segments.json; builds DomainConfig from domain_data and demo_data (districts/nationalities from demographics); merges ref_data into reference_distributions. |
| `get_domain_config(domain_id)` | Cached config. | If domain_id None, uses get_settings().domain_id; returns cached config if same domain_id; else load_domain_config and cache. |
| `reset_domain_cache()` | Clear module-level cache. | Sets _cached_config and _cached_domain_id to None. |

---

## demographics.py

**Purpose**: Load demographic marginal and conditional distributions from `data/domains/{domain_id}/demographics.json`. Replaces/complements hardcoded dubai_data for any city.

### Class: DemographicData

Dataclass: age, nationality, income, location, household_size, occupation (marginals); income_given_nationality, location_given_income, car_given_location, metro_access_by_location, occupation_given_nationality (conditionals).

| Method | Description | How |
|--------|-------------|-----|
| `get_all_marginals()` | Dict of all marginal distributions. | Returns {age, nationality, income, location, household_size, occupation}. |
| `get_nationality_keys()`, `get_age_keys()`, `get_income_keys()`, `get_location_keys()` | Key lists. | list(dict.keys()). |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `get_demographics(domain_id)` | Load demographics (cached). | If domain_id None, uses get_settings().domain_id; returns cache if same domain; else reads demographics.json, builds DemographicData, caches. |
| `reset_demographics_cache()` | Clear cache. | Sets _cached_demo and _cached_demo_id to None. |

---

## question_models.py

**Purpose**: Declarative registry mapping survey domains to scale, dimension_weights (personality traits), and factor_weights. Adding a new survey type = adding one QuestionModel; no engine code changes.

### Class: QuestionModel

Frozen dataclass: name, scale (list of ordered options), dimension_weights (trait name → weight), factor_weights (factor name → weight), temperature.

### Constants

| Constant | Description |
|----------|-------------|
| QUESTION_MODELS | Registry: food_delivery_frequency, parking_satisfaction, transport_satisfaction, shopping_frequency, housing_satisfaction, nps_recommendation, tech_adoption_likelihood, policy_support, etc. |
| GENERIC_LIKERT, GENERIC_FREQUENCY, GENERIC_OPEN_TEXT, GENERIC_DURATION, GENERIC_FALLBACK | Fallback models for unknown questions. |
| QUESTION_DIMENSION_MAP | Question model key → behavioral dimension weights. |
| GENERIC_DIMENSION_WEIGHTS | Default behavioral dimension weights. |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `get_question_model(key)` | Look up model. | QUESTION_MODELS.get(key, GENERIC_FALLBACK). |
| `get_behavioral_dimensions(key)` | Behavioral dimension weights for question. | From QUESTION_DIMENSION_MAP or GENERIC_DIMENSION_WEIGHTS. |

---

## belief_mappings.py

**Purpose**: Map question model name to belief dimension relevance weights (BeliefNetwork layer). Positive weight: high answer score reinforces belief; negative: reduces it.

### Constants

| Constant | Description |
|----------|-------------|
| QUESTION_BELIEF_MAP | question_model_name → {belief_dimension: weight}. Covers food_delivery_frequency, parking_satisfaction, transport_satisfaction, shopping_frequency, housing_satisfaction, tech_adoption_likelihood, policy_support. |
| GENERIC_BELIEF_WEIGHTS | Default belief weights when question not in map. |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `get_belief_dimensions(question_model_key)` | Belief weights for question. | QUESTION_BELIEF_MAP.get(question_model_key, GENERIC_BELIEF_WEIGHTS). |

---

## reference_distributions.py

**Purpose**: Population-level prior distributions per question model for calibration and distribution validation. Includes dynamic fallback for unknown question models.

### Constants

| Constant | Description |
|----------|-------------|
| REFERENCE_DISTRIBUTIONS | question_model_name → {option: proportion}. Entries for food_delivery_frequency, parking_satisfaction, transport_satisfaction, housing_satisfaction, shopping_frequency, nps_recommendation, etc. |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `get_reference_distribution(question_name, scale)` | Prior distribution for question. | REFERENCE_DISTRIBUTIONS.get(question_name) or build fallback from scale (e.g. uniform or generic_frequency). |

---

## dubai_data.py

**Purpose**: Real Dubai demographic marginals and conditionals (Dubai Statistics Center, UAE census, urban research). Used for synthesis when domain uses Dubai; for new cities use `data/domains/{id}/demographics.json` via demographics.py.

### Constants

Marginals: AGE_DISTRIBUTION, NATIONALITY_DISTRIBUTION, INCOME_DISTRIBUTION, LOCATION_DISTRIBUTION, OCCUPATION_DISTRIBUTION, HOUSEHOLD_SIZE_DISTRIBUTION.

Conditionals: INCOME_GIVEN_NATIONALITY, LOCATION_GIVEN_INCOME, OCCUPATION_GIVEN_NATIONALITY, CAR_GIVEN_LOCATION (and similar).

### Functions

| Function | Description | How |
|----------|-------------|-----|
| `get_age_keys()`, `get_nationality_keys()`, `get_income_keys()`, `get_location_keys()` | Key lists for IPF/sampling. | list(dict.keys()) on corresponding constant. |

---

## __init__.py

Package marker; may re-export config symbols. No required public API listed.
