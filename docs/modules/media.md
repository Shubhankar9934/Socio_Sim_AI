# Media Module

Media framing, selective exposure, adaptive attention, and belief updates from media.

## exposure.py

**Purpose**: Selective exposure and belief update from media frames.

### Functions

| Function | Description |
|----------|-------------|
| `compute_exposure_matrices(agents, frames)` | Raw exposure, alignment, emotion matrices (N × num_frames). Uses agent media subscriptions and frame alignment. |
| `update_beliefs_from_media(agents, adjusted_exposure, frames, w_prior, w_media)` | Blend prior beliefs with media-influenced beliefs. In-place update of agent BeliefNetwork. |
| `compute_alignment(alignment_matrix, adjusted_exposure, beta)` | Gated peak alignment: 0.85 × peak + 0.15 × weighted mean. |

---

## attention.py

**Purpose**: Emotion-gated adaptive attention.

### Functions

| Function | Description |
|----------|-------------|
| `adaptive_attention(activation, raw_exposure, emotion_matrix, k, p, min_attention)` | Reweight exposure by activation-sharpened salience. High activation → tunnel vision. Entropy floor prevents collapse. |

---

## framing.py

**Purpose**: Event → narrative frames per media source.

### Functions

| Function | Description |
|----------|-------------|
| `generate_frames(events)` | Generate media frames (narrative per source) from world events. |

---

## sources.py

**Purpose**: Media diet assignment based on belief alignment.

### Functions

| Function | Description |
|----------|-------------|
| `assign_media_diet(belief_vector, rng)` | Assign media subscriptions (sources) based on belief-aligned homophilic selection. |

---

## strategic.py

**Purpose**: **Strategic media actors** — goal-driven injectors that add targeted [`MediaFrame`](../../media/framing.py) narratives into the ecosystem (`StrategicActor`, `FramingPolicy`, `get_active_actors`, `inject_strategic_frames`). Used from the simulation loop to model campaigns with budgets and demographic targeting; uses seeded RNG via [`core/rng.py`](../core.md).

---

## __init__.py

Package marker.
