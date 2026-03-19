# JADU API reference (Postman collection)

This folder documents every HTTP endpoint from [postman/JADU_Full_API.postman_collection.json](../../postman/JADU_Full_API.postman_collection.json), with **request/response fields**, **why each parameter exists**, and **code flow** (routes → functions → key modules).

| Document | Scope |
|----------|--------|
| [population.md](population.md) | `POST /population/generate` |
| [agents.md](agents.md) | `GET /agents`, `GET /agents/:agent_id` |
| [survey.md](survey.md) | Single survey, results, multi-survey session |
| [simulation.md](simulation.md) | Days run, events, scenarios, causal |
| [analytics.md](analytics.md) | `GET /analytics/:survey_id` |
| [evaluation.md](evaluation.md) | `POST /evaluate/:survey_id`, report stub |
| [discovery.md](discovery.md) | Domain auto-setup, dimension discovery |
| [calibration.md](calibration.md) | Auto-weights, fit, upload-data |

---

## Hardcoding vs configuration (verification summary)

JADU is **not** a single hardcoded survey engine: behavior is driven by **domain config**, **demographics JSON**, **question models**, and **reference distributions**. The following are important nuances:

**Configured / data-driven (general platform behavior)**

- **Population marginals** (age, nationality, income, location, household, occupation): loaded via `config.demographics.get_demographics()` from the active domain’s `demographics.json` (see `config/domain.py` → `get_domain_config()`).
- **Synthesis method**: `monte_carlo`, `bayesian`, `ipf` branch in `population/synthesis.py` → `generate_population()`.
- **Survey routing**: `agents/perception.py` maps questions to `question_model_key` using domain `topic_keywords`, `topic_to_model_key`, and `QUESTION_MODELS` in `config/question_models.py`.
- **Decision scales & factor graphs**: declared per question model in `config/question_models.py` and `agents/factors/`.
- **Reference distributions for calibration / optional validation**: `config/reference_distributions.py` + per-domain `reference_distributions.json`.

**Fixed logic or defaults (not “one response type”, but structural choices)**

- **Fallback keyword lists** in `agents/perception.py` if domain config does not define `topic_keywords` / `domain_keywords` (still generic categories, not a single hardcoded answer).
- **Insight helper** in `analytics/insights.py` uses a **fixed list of “high frequency” option strings** for `delivery_frequency_insight` — this is a known limitation when aggregations use verbatim `answer` text instead of `sampled_option` (see [analytics.md](analytics.md)).
- **Lifestyle heuristics** in `population/synthesis.py` reference label strings like `"Western"`, `"Emirati"` and income bands — these align with **category labels in your demographics file**, not arbitrary literals for final survey answers.
- **Evaluation default reference** when no question-specific reference is passed: see [evaluation.md](evaluation.md) — uses a generic frequency prior unless extended.

**Takeaway:** The platform is built to support **multiple domains and question types** through config and registries. Remaining “hardcoded” pieces are mostly **defaults**, **fallback maps**, or **heuristic thresholds** that you can generalize further (e.g. domain-driven insight keys, explicit `options` on every survey request, evaluation keyed by question model).

For endpoint-level detail, open the linked `.md` files above.
