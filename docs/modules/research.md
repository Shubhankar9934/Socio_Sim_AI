# Research Module

Web search and factual grounding for shared context.

## engine.py

**Purpose**: Research engine for factual context from web search.

### Classes

| Class | Description |
|-------|-------------|
| `ResearchEngine` | Provider (tavily, serpapi, none), cache_path, openai_api_key. |

### Methods

| Method | Description |
|--------|-------------|
| `research_question(question)` | Fetch and cache research context for question. Returns context object. |
| `build_context(events)` | Build shared factual context from world events. |

---

## context.py

**Purpose**: Research context dataclass and enrichment.

### Functions

| Function | Description |
|----------|-------------|
| `enrich_archetype_context(ctx, research_ctx)` | Merge research context into agent context for LLM prompt. |
