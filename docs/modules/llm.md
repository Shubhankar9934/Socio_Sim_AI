# LLM Module

Async OpenAI client and prompt templates for agent reasoning and evaluation.

## client.py

**Purpose**: Async OpenAI client with rate limiting (semaphore), retries, and token tracking.

### Classes

| Class | Description |
|-------|-------------|
| `LLMClient` | Wraps AsyncOpenAI with concurrency limit and token counters. |

### Key Methods

| Method | Description | How |
|--------|-------------|-----|
| `chat(messages, model, temperature, max_tokens)` | Send chat completion with semaphore. | Acquire _semaphore; _chat_impl: client.chat.completions.create; update total and session token counters and call_count; return choices[0].message.content. |
| `complete(prompt, system, model, temperature, max_tokens)` | Single completion. | Build messages list (optional system, then user prompt); call chat(). |
| `reset_survey_stats()` | Reset per-survey counters. | Set _session_call_count, _session_prompt_tokens, _session_completion_tokens to 0. |

### Properties

| Property | Description |
|----------|-------------|
| `total_prompt_tokens`, `total_completion_tokens` | Lifetime totals. |
| `session_call_count`, `session_prompt_tokens`, `session_completion_tokens` | Per-survey totals. |

### Functions

| Function | Description |
|----------|-------------|
| `get_llm_client()` | Module-level lazy singleton. |

---

## prompts.py

**Purpose**: Agent reasoning prompts with narrative diversity, persona compression, and consistency checks.

### Key Functions

| Function | Description | How |
|----------|-------------|-----|
| `build_agent_prompt(persona, question, sampled_option, distribution, memories, ...)` | Full agent prompt with response contract support. | Builds persona/memory/style blocks, then applies Decision->Expression contract hints (`expression_mode`, `confidence_band`, `latent_stance`) so wording follows LPFG output without changing the selected answer. |
| `reasoner_via_llm(persona, question, sampled_answer, distribution, memories, ...)` | Async narrative generation under LPFG authority. | Uses seeded prompt generation, retry-on-contradiction checks, and confidence-to-tone mapping from the response contract; narrative is constrained to explain the selected option (or open latent stance) rather than decide it. |
| `build_judge_prompt(persona_summary, question, response)` | Judge prompt. | Asks LLM to score realism, persona_consistency, cultural_plausibility 1–5; JSON output. |
| `compress_persona_for_display(persona)` | Short summary. | Natural language from persona attributes. |
| `infer_scale_type(options)` | Scale type from options. | No options → open_text; all digits → numeric; frequency terms → frequency; 2 options → categorical; else likert. |
| `allow_persona_anchor(question)` | Whether to inject lifestyle anchors. | Question-domain heuristics. |

### Decision->Expression Contract

- Decision authority remains in LPFG (`distribution`, `sampled_option`).
- Prompting consumes a response contract from cognitive layer:
  - `expression_mode`: `structured_expression` or `open_expression`
  - `confidence_band`: `high|medium|low`
  - `tone_selected`: wording style only
  - `expected_score` / `latent_stance`: open-text grounding signal
- API diagnostics exposure is opt-in (`diagnostics=true`); default responses do not include response diagnostics.

### Constants

| Constant | Description |
|----------|-------------|
| `_SYSTEM_PROMPTS` | Pool of system-prompt variants (rotated per agent). |
| `ARCHETYPE_HINTS` | Archetype-specific narrative guidance. |
| `CULTURAL_BEHAVIOR_HINTS` | Nationality-specific behavior hints. |
