# Memory Module

Agent memory types (episodic, semantic, behavioral) and vector store with ChromaDB or in-memory fallback. Supports add, recall by similarity, get_recent, and optional exponential decay of weights.

## types.py

**Purpose**: Pydantic models and enum for memory kinds used by the store.

### Enum

| Name | Description |
|------|-------------|
| MemoryType | EPISODIC, SEMANTIC, BEHAVIORAL. |

### Classes

| Class | Description |
|-------|-------------|
| EpisodicMemory | agent_id, content, timestamp, weight, metadata. Past experience (e.g. "I complained about parking last year"). |
| SemanticMemory | agent_id, content, timestamp, weight, metadata. World knowledge (e.g. "Dubai Marina parking is difficult"). |
| BehavioralMemory | agent_id, content, timestamp, weight, value (optional), metadata. Habit pattern (e.g. "orders food 3x/week"). |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| memory_to_text(mem) | Single-line text for embedding/retrieval. | Returns mem.content. |

---

## store.py

**Purpose**: Per-agent memory store. Uses ChromaDB when available and persist_dir (or CHROMA_PERSIST_DIR) is set; otherwise in-memory list with keyword-based recall.

### Class: MemoryStore

| Method | Description | How |
|--------|-------------|-----|
| `__init__(persist_dir, collection_prefix)` | Initialize store. | persist_dir from arg or CHROMA_PERSIST_DIR; _use_chroma = HAS_CHROMADB and persist_dir; if _use_chroma create chromadb.PersistentClient. |
| `_collection_name(agent_id)` | Safe Chroma collection name. | prefix + alphanumeric/underscore only agent_id. |
| `_get_or_create_collection(agent_id)` | Get or create Chroma collection for agent. | Cached in _collections; get_or_create_collection(name, metadata agent_id). |
| `add_memory(agent_id, content, memory_type, value, metadata, initial_weight)` | Add one memory. | Build meta with memory_type, value, timestamp, weight; if Chroma: coll.add(ids=uuid, documents=[content], metadatas=[meta]); else append to _in_memory[agent_id]. |
| `recall(agent_id, query, top_k)` | Top-k relevant memories. | Chroma: coll.query(query_texts=[query], n_results=min(top_k,50)); return documents[0]. In-memory: score each memory by keyword overlap × weight; sort by score desc; return top_k with score > 0. |
| `get_recent(agent_id, n)` | n most recent by timestamp. | Chroma: coll.get(include documents, metadatas); sort by metadata timestamp desc; return first n. In-memory: sort _in_memory[agent_id] by timestamp desc; return first n contents. |
| `decay_all(lambda_, prune_threshold)` | Exponential decay of in-memory weights. | factor = exp(-lambda_); for each agent, multiply each memory weight by factor; remove memories with weight < prune_threshold. Call once per simulation day. Chroma not modified. |

### Functions

| Function | Description | How |
|----------|-------------|-----|
| get_memory_store(persist_dir) | Module-level lazy singleton. | If _default_store is None, create MemoryStore(persist_dir); return it. |

---

## __init__.py

Package marker; may re-export MemoryType, MemoryStore, get_memory_store, etc.
