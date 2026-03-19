# Calibration API

Source collection: **JADU_Full_API** → folder `calibration`.

Aligns **simulated** response distributions with **real** or **target** distributions by adjusting factor weights in the decision pipeline.

---

## POST `/calibration/auto-weights`

**Requires:** `agents_store` non-empty (`POST /population/generate` first).

### Body (`AutoWeightsRequest`)

| Field | Type | Purpose |
|-------|------|---------|
| `questions` | string[] (min 1) | Questions to calibrate (same strings used in simulation path). |
| `reference_distributions` | map: question → { option → proportion } | Target histogram per question; option keys must match what the simulator emits (or you accept remapping work). |
| `n_iterations` | int (5–500, default 50) | Optimization budget for the learner. |
| `seed` | int or null (default 42) | Reproducibility for the search. |

### Response

| Field | Purpose |
|-------|---------|
| `overall_loss` | Aggregate objective (e.g. mean JS divergence). |
| `results` | Per question: `learned_weights`, `best_loss`, `converged`. |

### Code flow

1. [api/routes/calibration.py](../../api/routes/calibration.py) `auto_weights`
2. [calibration/auto_weights.py](../../calibration/auto_weights.py) `FactorWeightLearner.learn_weights(...)`

### Why use this

Makes the platform **data-driven**: match synthetic population responses to **your** survey marginals, not a fixed demo curve.

---

## POST `/calibration/fit`

Single-question variant of auto-weights.

### Body (`FitRequest`)

| Field | Purpose |
|-------|---------|
| `question` | One question string. |
| `reference_distribution` | `{ option: proportion }`. |
| `demographics_cols` | Optional list for stratified fitting (if supported by learner). |
| `n_iterations` | Min 5. |

### Response

`question`, `learned_weights`, `best_loss`, `converged`, `n_iterations`.

### Code flow

`FactorWeightLearner.learn_weights_for_question(...)`.

---

## POST `/calibration/upload-data`

**No population required.**

### Body (`UploadDataRequest`)

| Field | Purpose |
|-------|---------|
| `question` | Label stored with the dataset. |
| `responses` | List of observed answer strings (required). |
| `demographics` | Optional parallel list of demographic dicts per response. |

### Response

| Field | Purpose |
|-------|---------|
| `question` | Echo. |
| `n_responses` | Count. |
| `reference_distribution` | Normalized empirical histogram over `responses`. |

### Code flow

1. [calibration/data_loader.py](../../calibration/data_loader.py) `RealSurveyData.from_raw(...)`
2. `to_reference_distribution()` — counts / total

### Why use this

Typical pipeline: **upload real data** → get `reference_distribution` → pass into `/calibration/fit` or `/calibration/auto-weights`.

---

## Errors

- **400** on `/auto-weights` and `/fit` if `agents_store` is empty.

---

## Relation to “no hardcoding”

Reference distributions are **you-supplied** or **upload-derived**. Built-in priors in [config/reference_distributions.py](../../config/reference_distributions.py) are **defaults** for validation and priors; production workflows should override with domain JSON or API-uploaded refs.
