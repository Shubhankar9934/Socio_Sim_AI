# Data Module

Static data files and domain directories consumed by config, discovery, calibration, and evaluation. No Python package under `data/`; only JSON (and optional discovered/segment files).

## data/ (root level)

| File | Schema / Purpose | Consumed by |
|------|------------------|-------------|
| dimension_cache.json | question_hash → LLM-inferred dimension_weights (behavioral). | agents/perception.py when `infer_dimension_weights_via_llm` is used; written on first inference for unknown questions. |
| dubai_demographics.json | Marginal and conditional demographic distributions (age, nationality, income, location, occupation). | Legacy or fallback; primary source for Dubai is data/domains/dubai/demographics.json via config.demographics. |
| research_cache.json | Cached research context (e.g. web search / fact extraction). | research/engine.py (path from config.settings.research_cache_path). |

## data/domains/{domain_id}/

Used by `config/domain.py`, `config/demographics.py`, [discovery](discovery.md), and [calibration](calibration.md). To add a new city: create `data/domains/{city_id}/` with at least `domain.json` and `demographics.json` (see `data/domains/dubai/` as template).

| File | Schema / Purpose | Consumed by |
|------|------------------|-------------|
| domain.json | city_id, city_name, currency, id_prefix, districts, premium_areas, topic_keywords, domain_keywords, topic_to_model_key, location_terms, services, price_levels, cultural_priors, family_modifiers, income_modifiers, implausible_combos, answer_habit_scores, vague_answers, system_prompts, archetype_hints, cultural_hints, frequency_interpretation, lifestyle_keywords, strategic_actors, memory_rules, question_to_semantic_key, question_belief_map, cuisine_by_nationality, diet_pool, cultural_family_multiplier, late_dinner_nationalities. | config.domain.load_domain_config; discovery when saving domain; domain_setup when creating new domain. |
| demographics.json | age, nationality, income, location, household_size, occupation (marginals); income_given_nationality, location_given_income, car_given_location, metro_access_by_location, occupation_given_nationality (conditionals). | config.demographics.get_demographics; population synthesis. |
| reference_distributions.json | question_model_key or question text → { answer_option: proportion }. | config.domain (merged into DomainConfig.reference_distributions); evaluation.distribution_validation; calibration (target distributions). |
| discovered_dimensions.json | Optional. behavioral: [{name, description, representative_questions}], belief: [...], question_to_dimension: {question: {dim: weight}}. | discovery.dimensions (save_discovered_dimensions, load_discovered_dimensions); get_active_dimension_names. |
| segments.json | Optional. Population segment definitions. | config.domain (DomainConfig.segments). |

## Data flow summary

- **Domain config**: `get_domain_config(domain_id)` loads domain.json + demographics.json + reference_distributions.json + segments.json once and caches.
- **Demographics**: `get_demographics(domain_id)` loads demographics.json and caches; used by population synthesis.
- **Dimension cache**: Written by perception when LLM infers dimension weights; read to avoid re-calling LLM for same question.
- **Research cache**: Written/read by research engine for shared factual grounding.
- **Reference distributions**: Loaded from domain or config.reference_distributions; used for distribution validation and calibration targets.
