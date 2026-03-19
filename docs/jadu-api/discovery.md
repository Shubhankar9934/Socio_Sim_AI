# Discovery API

Source collection: **JADU_Full_API** → folder `discovery`.

**No population required** — operates on questions and file system under `data/domains/`.

---

## POST `/discovery/domains/auto-setup`

Creates or updates a **domain folder** for a new vertical / city use case.

### Body (`DomainAutoSetupRequest`)

| Field | Required | Purpose |
|-------|----------|---------|
| `domain_name` | yes | Human name; slugified to `domain_id` (e.g. `food_delivery`). |
| `description` | no | Free text for LLM prompts when generating domain.json. |
| `sample_questions` | no | Used to bootstrap dimension discovery and prompts. |
| `city_name` | no | Metadata for generated config. |
| `currency` | no | Default `"USD"`. |
| `reference_data` | no | Optional dict written to `reference_distributions.json` if provided. |

### Response

| Field | Meaning |
|-------|---------|
| `domain_id` | Slug / folder name under `data/domains/`. |
| `message` | Confirmation path. |

### Code flow

1. [api/routes/discovery.py](../../api/routes/discovery.py) `auto_setup_domain`
2. [discovery/domain_setup.py](../../discovery/domain_setup.py) `DomainAutoSetup.setup_domain(...)`  
   - May call LLM to draft `domain.json`  
   - Dimension discovery on sample questions  
   - Writes `data/domains/{domain_id}/` files (`domain.json`, stubs, optional `reference_distributions.json`)

### Why use this

Lets you add a **new market or topic** without editing Python: new keywords, services, and prompts live in JSON under the domain directory.

---

## POST `/discovery/dimensions`

Discovers **behavioral** and **belief** dimensions from a list of survey questions (typically LLM-assisted).

### Body (`DimensionDiscoveryRequest`)

| Field | Type | Purpose |
|-------|------|---------|
| `questions` | string[] (min 1) | Input questions to analyze. |
| `n_behavioral` | int (1–50, default 12) | How many behavioral dimensions to propose. |
| `n_belief` | int (1–30, default 7) | How many belief dimensions to propose. |
| `domain_id` | string or null | If `save` is true, target domain for persistence. |
| `save` | bool (default false) | Persist to `discovered_dimensions.json` for `domain_id`. |

### Response

| Field | Purpose |
|-------|---------|
| `behavioral` | `{ name, description, representative_questions }[]` |
| `belief` | Same shape |
| `question_to_dimension` | Map question text → dimension relevance weights |
| `saved` | Whether written to disk |

### Code flow

1. [api/routes/discovery.py](../../api/routes/discovery.py) `discover_dimensions`
2. [discovery/dimensions.py](../../discovery/dimensions.py) `DimensionDiscovery.discover_dimensions`
3. Optional `save_discovered_dimensions(domain_id, result)`

### Why use this

Supports **general** research design: infer latent constructs from your questionnaire before calibrating or running large simulations.
