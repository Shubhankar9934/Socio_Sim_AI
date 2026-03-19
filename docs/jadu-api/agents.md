# Agents API

Source collection: **JADU_Full_API** → folder `agents`.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/agents` | Paginated list of agent summaries (demographics only). |
| **GET** | `/agents/{agent_id}` | Full persona JSON + serialized `AgentState`. |

**Prerequisite:** `POST /population/generate` must have run so `api.state.agents_store` is non-empty.

---

## GET `/agents`

### Query parameters

| Param | Type | Default | Range | Purpose |
|-------|------|---------|-------|---------|
| `location` | string | omitted | exact match on `persona.location` | Filter agents by district / area. |
| `nationality` | string | omitted | exact match on `persona.nationality` | Filter by nationality label from demographics. |
| `limit` | int | 100 | 1–1000 | Max rows returned (slice length). |
| `offset` | int | 0 | ≥ 0 | Skip first `offset` entries in `agents_store` **before** filtering. |

**Note:** Filtering applies **after** the slice `agents_store[offset : offset + limit]`, so pagination interacts with filters (see [api/routes/agents.py](../../api/routes/agents.py)).

### Response (200)

Array of `AgentSummary`:

| Field | Source |
|-------|--------|
| `agent_id` | `persona.agent_id` |
| `age`, `nationality`, `income`, `location`, `occupation` | `Persona` fields |

Household and lifestyle are **not** included in the list view (only in detail).

### Code flow

1. [api/routes/agents.py](../../api/routes/agents.py) `list_agents(...)`
2. Iterates `agents_store[offset : offset + limit]`
3. Builds [api/schemas.py](../../api/schemas.py) `AgentSummary` per matching persona

---

## GET `/agents/{agent_id}`

### Path parameters

| Param | Purpose |
|-------|---------|
| `agent_id` | Must match `persona.agent_id` (e.g. `DXB_0042`). |

### Response (200)

`AgentDetail`:

| Field | Content |
|-------|---------|
| `agent_id` | string |
| `persona` | `Persona.model_dump()` — full persona (demographics, lifestyle, anchors, meta, media_subscriptions, …) |
| `state` | `AgentState.to_dict()` if present — latent state, beliefs, memory fields, `behavior_scores`, etc. |

### Errors

- **404** if no persona matches `agent_id`.

### Code flow

1. [api/routes/agents.py](../../api/routes/agents.py) `get_agent(agent_id)`
2. Linear search in `agents_store`
3. `state.to_dict()` from [agents/state.py](../../agents/state.py)

---

## In-memory store shape

Each entry in `agents_store` is a dict like:

```python
{
  "persona": Persona,
  "state": AgentState,
  "social_trait_fraction": float,
  "location_quality": float,  # from world.districts.location_quality_for_satisfaction
}
```

Populated entirely by `POST /population/generate` ([api/routes/population.py](../../api/routes/population.py)).
