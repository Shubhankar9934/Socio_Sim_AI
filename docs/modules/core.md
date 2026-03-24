# Core module

Small shared utilities used across population, simulation, and agents.

## rng.py

**Purpose:** Central **seeded RNG** helpers for deterministic stochastic behavior. Aligns child streams with `Settings.master_seed` and optional per-run overrides.

### Functions

| Function | Description |
|----------|-------------|
| `stable_seed_from_key(key, base_seed=None)` | 32-bit seed from SHA-256(`master_seed` or `base_seed`, `key`). |
| `make_rng_pack(key, base_seed=None)` | Returns `RngPack` with paired `np.random.Generator` and `random.Random`. |
| `ensure_np_rng(rng, key)` | Use provided NumPy RNG, or deterministic fallback unless `allow_unseeded_rng`. |
| `ensure_py_rng(rng, key)` | Same for Python `random.Random`. |
| `agent_seed_from_id(agent_id, base_seed=None)` | Per-agent independent stream seed from `agent:{seed}:{agent_id}`. |
| `agent_rng_pack(agent_id, base_seed=None)` | `RngPack` for one agent id. |

### Class

| Class | Description |
|-------|-------------|
| `RngPack` | Frozen dataclass: `seed`, `np_rng`, `py_rng`. |

### Configuration

[`config/settings.py`](../../config/settings.py) (see [Config module](config.md)): `master_seed`, `allow_unseeded_rng` — when `allow_unseeded_rng` is true, missing explicit RNGs fall back to **non-deterministic** defaults.

### Relations

- **Population:** [`GeneratePopulationRequest.seed`](../../api/schemas.py) flows into synthesis where applicable.
- **Constraints / life path:** [`population/constraints.py`](../population/constraints.py), [`population/life_path.py`](../population/life_path.py) use `agent_rng_pack` / `agent_seed_from_id`.
- **Lazy store:** [`population/lazy_store.py`](../population/lazy_store.py) uses per-agent seeds for on-demand personas.

**Tests:** [`tests/test_rng_policy.py`](../../tests/test_rng_policy.py).
