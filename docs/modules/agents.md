# Agents Module

The agent cognitive engine: perception, decision-making, personality, beliefs, biases, and narrative generation.

## state.py

**Purpose**: Mutable agent state — the core representation of one agent.

### Classes

| Class | Description |
|-------|-------------|
| `AgentState` | Mutable state for one agent. Core fields: `latent_state` (12-dim behavioral vector), `beliefs`, `habit_profile`, `last_answers`, `structured_memory`, `life_event_history`, `friends_using_delivery`, `current_day`. |

### Key Methods

| Method | Description |
|--------|-------------|
| `from_persona(persona)` | Initialize state from persona. Derives BehavioralLatentState, BeliefNetwork, HabitProfile. |
| `update_after_answer(question_id, answer, semantic_key)` | Record answer for consistency; store in structured_memory if semantic_key provided. |
| `build_structured_context()` | Compact dict for LLM prompt injection (behavioral signals, dialogue summary). |
| `summarize_memory()` | Compress last_answers into short text summary. |
| `set_behavior_score(domain, score)` | Set dimension on latent state. |
| `get_behavior_score(domain)` | Read dimension from latent state. |

---

## cognitive.py

**Purpose**: Full cognitive pipeline — perceive, recall, decide, reason.

### Classes

| Class | Description |
|-------|-------------|
| `AgentCognitiveEngine` | One agent's brain: persona + state + cognitive pipeline. |

### Key Methods

| Method | Description |
|--------|-------------|
| `perceive(question)` | Extract structured perception (topic, domain, scale). |
| `recall(perception)` | Retrieve relevant memories (async if store is async). |
| `decide(perception, memories, friends_using, location_quality)` | Probabilistic decision; returns (distribution, sampled_answer). |
| `think(question, question_id, ...)` | Full pipeline: perceive → recall → decide → reason. Uses LLM. |
| `decide_only(question, question_id, ...)` | Simulation mode: fast decision, no LLM. |
| `handle_event(event)` | Dispatch SimEvent via EventDispatcher. |
| `set_world_environment(env)` | Store event-driven world params for factor graph. |
| `build_structured_context(research_ctx)` | Compact context for LLM injection. |

---

## decision.py

**Purpose**: Generic probabilistic decision engine — P(response | persona, context, factors).

### Functions

| Function | Description |
|----------|-------------|
| `decide(perception, persona, traits, ...)` | Compute probability distribution and sampled answer. Resolves QuestionModel from perception. |
| `compute_distribution(question_model, context, agent_state, ...)` | 12-stage pipeline: factor graph → logit scoring → reference prior → habit bias → cultural prior → softmax → Dirichlet noise → conviction shaping → dissonance → memory rules → biases. |
| `sample_from_distribution(dist)` | Nucleus (top-p) sampling with relative floor and resample guard. |

### Pipeline Stages

1. Factor graph inference with weight perturbation
2. Trait-vector logit scoring
3. Reference prior blend
4. Habit profile bias
5. Cultural behavior prior
6. Per-option noise, conviction spike
7. Per-agent softmax temperature
8. Dirichlet noise
9. Conviction profile shaping
10. Cognitive dissonance
11. Memory rules
12. Bounded-rational biases

---

## perception.py

**Purpose**: Extract topic, domain, scale type, and question-model key from survey questions.

### Classes

| Class | Description |
|-------|-------------|
| `Perception` | Dataclass: topic, domain, location_related, keywords, raw_question, scale_type, question_model_key. |

### Functions

| Function | Description |
|----------|-------------|
| `perceive(question)` | Rule-based keyword matching → Perception. |
| `perceive_with_llm(question)` | Perceive with LLM fallback for unknown questions (dimension weights). |
| `detect_question_model(perception)` | Map Perception to QuestionModel (scale, dimension_weights, factor_weights). |
| `classify_question_via_llm(question)` | LLM-assisted category classification. |
| `infer_dimension_weights_via_llm(question)` | LLM assigns behavioral dimension weights; cached on disk. |

---

## factor_graph.py

**Purpose**: Composable behavioral factors for agent decisions.

### Classes

| Class | Description |
|-------|-------------|
| `DecisionContext` | Inputs for factors: persona, traits, perception, friends_using, location_quality, memories, environment. |
| `FactorGraph` | Weighted collection of (FactorFn, weight). Computes normalised weighted average. |

### Functions

| Function | Description |
|----------|-------------|
| `get_or_build_graph(model_name, builder)` | Cached FactorGraph per QuestionModel name. |
| `clear_graph_cache()` | Reset cache (tests). |

---

## factors/ (submodule)

**Purpose**: Per-question factor implementations. Each returns float in [0, 1].

| File | Function | Description |
|------|----------|-------------|
| `income.py` | `income_factor(ctx)` | Budget share for delivery via world.economy. |
| `personality.py` | `personality_factor(ctx)` | Weighted sum of traits from dimension_weights. |
| `social.py` | `social_factor(ctx)` | friends_using + optional neighbor latent similarity. |
| `location.py` | `location_factor(ctx)` | location_quality (0–1). |
| `memory.py` | `memory_factor(ctx)` | Structured memory + keyword sentiment. |
| `behavioral.py` | `behavioral_factor(ctx)` | BehavioralLatentState.behavioral_score(dim_weights). |
| `belief.py` | `belief_factor(ctx)` | BeliefNetwork.belief_score(dim_weights). |

`build_factor_graph(model)` assembles graph from QuestionModel.factor_weights.

---

## behavior.py

**Purpose**: 12 universal behavioral dimensions driving decisions.

### Classes

| Class | Description |
|-------|-------------|
| `BehavioralLatentState` | 12 dimensions: convenience_seeking, price_sensitivity, technology_openness, risk_aversion, health_orientation, routine_stability, novelty_seeking, social_influence_susceptibility, time_pressure, financial_confidence, environmental_consciousness, institutional_trust. |

### Key Methods

| Method | Description |
|--------|-------------|
| `update_dimensions(weights, answer_score)` | EMA update toward answer. |
| `apply_social_influence(neighbor_mean)` | Nudge toward neighbor mean. |
| `apply_macro_influence(signals)` | Population-level trends. |
| `apply_event_impact(impacts)` | Direct shifts from events. |
| `behavioral_score(weights)` | Weighted combination → 0–1. |

### Functions

| Function | Description |
|----------|-------------|
| `init_from_persona(persona, traits)` | Derive initial state; optionally blend with segment priors. |

---

## belief_network.py

**Purpose**: 7 high-level belief dimensions (attitudes, not behaviors).

### Classes

| Class | Description |
|-------|-------------|
| `BeliefNetwork` | technology_optimism, brand_loyalty, environmental_concern, health_priority, government_trust, price_consciousness, innovation_curiosity. |

### Key Methods

| Method | Description |
|--------|-------------|
| `update_from_answer(weights, answer_score)` | Slow EMA update. |
| `apply_social_diffusion(neighbor_mean)` | Nudge toward neighbors. |
| `apply_event_impact(impacts)` | Direct belief shifts. |
| `belief_score(weights)` | Weighted combination → 0–1. |

### Functions

| Function | Description |
|----------|-------------|
| `init_beliefs_from_persona(persona, traits)` | Derive initial beliefs. |

---

## personality.py

**Purpose**: Trait derivation from persona for decision model.

### Classes

| Class | Description |
|-------|-------------|
| `PersonalityTraits` | risk_aversion, convenience_preference, price_sensitivity, tech_adoption, luxury_preference, food_delivery_preference, dining_out, social_activity, health_consciousness, mobility_dependence, time_pressure, brand_loyalty. |

### Functions

| Function | Description |
|----------|-------------|
| `personality_from_persona(persona)` | Build traits from lifestyle and anchors. |

---

## biases.py

**Purpose**: Bounded-rational cognitive bias engine.

### Functions

| Function | Description |
|----------|-------------|
| `apply_all_biases(dist, scale, agent_state, context, ...)` | Full pipeline: confirmation → loss aversion → anchoring → bandwagon → availability → residual mixing → entropy injection. |
| `compute_gamma(agent_state, context)` | Bias susceptibility (malleability × topic × calcification). |
| `compute_epsilon(agent_state, context)` | Entropy factor (ignorance + media conflict). |
| `apply_confirmation_bias`, `apply_loss_aversion`, `apply_anchoring`, `apply_bandwagon_effect`, `apply_availability_heuristic` | Individual bias transforms. |

---

## dissonance.py

**Purpose**: Cognitive dissonance — adjust distribution toward belief/behavior consistency.

### Functions

| Function | Description |
|----------|-------------|
| `apply_cognitive_dissonance(dist, consistency_score, scale)` | Re-weight options toward consistency anchor. |
| `compute_consistency_score(agent_state, question_model)` | Blend behavioral + belief scores. |
| `dissonance_penalty(old_score, new_score)` | Exponential penalty for deviation. |

---

## memory_rules.py

**Purpose**: Cross-question memory influence rules.

### Classes

| Class | Description |
|-------|-------------|
| `MemoryRule` | source_key, target_model_keys, condition, bias (option → multiplicative weight). |

### Constants

| Constant | Description |
|----------|-------------|
| `QUESTION_TO_SEMANTIC_KEY` | QuestionModel name → semantic key in structured_memory. |
| `CROSS_QUESTION_RULES` | List of MemoryRule for delivery_frequency, parking_satisfaction, etc. |

### Functions

| Function | Description |
|----------|-------------|
| `apply_memory_rules(distribution, question_model_key, structured_memory)` | Apply matching rules to distribution. |

---

## realism.py

**Purpose**: Stratified human imperfection engine.

### Classes / Enums

| Name | Description |
|------|-------------|
| `HabitProfile` | delivery_tendency, cooking_tendency, budget_consciousness, health_strictness, tech_comfort. |
| `ConvictionProfile` | CERTAIN, LEANING, DIFFUSE, BIMODAL, ANCHORED. |

### Functions

| Function | Description |
|----------|-------------|
| `derive_habit_profile(persona)` | Deterministic habit vector from persona. |
| `apply_habit_bias(raw_scores, scale, habit_profile)` | Matrix-based habit influence. |
| `update_habits_after_answer(habit_profile, sampled_option)` | EMA habit update. |
| `assign_conviction_profile(persona)` | Archetype-weighted profile assignment. |
| `apply_conviction_shaping(probs, profile)` | Reshape distribution (peaky, bimodal, diffuse). |
| `get_cultural_prior(persona, scale)` | Nationality × family × income prior. |
| `validate_demographic_plausibility(persona, sampled_option)` | Check implausible combos. |
| `suggest_plausible_resampling(persona, dist, sampled)` | Resample if implausible. |
| `maybe_add_hedging`, `maybe_fragmentize`, `degrade_polish` | Response texture (hedging, fragments, casual phrasing). |

---

## narrative.py

**Purpose**: Narrative diversity — styles, openings, banned patterns.

### Classes

| Class | Description |
|-------|-------------|
| `NarrativeStyleProfile` | verbosity, preferred_tone, preferred_style, slang_level, grammar_quality. |

### Functions

| Function | Description |
|----------|-------------|
| `derive_narrative_style_profile(age, income, occupation, nationality, rng)` | Demographics → style profile. |
| `build_style_instruction(style, structure, opening, ...)` | LLM instruction block. |
| `pick_style_from_profile`, `pick_tone_from_profile`, `pick_length_from_profile` | Profile-biased selection. |
| `pick_opening`, `pick_opening_deduplicated` | Sentence opening patterns. |
| `validate_narrative_consistency(narrative, sampled_option, scale)` | Check contradiction. |
| `is_banned_pattern(text)` | AI-style pattern detection. |

---

## vectorized.py

**Purpose**: Numpy batch operations for 100k+ agents.

### Functions

| Function | Description |
|----------|-------------|
| `build_trait_matrix(agents)` | (N × 12) latent dimensions. |
| `write_trait_matrix(agents, mat)` | Write back to AgentState. |
| `build_belief_matrix`, `write_belief_matrix` | Same for beliefs. |
| `vectorized_decide(trait_matrix, weight_vector, n_options)` | Batch softmax. |
| `vectorized_sample(distributions)` | Sample per agent. |
| `vectorized_social_influence(trait_matrix, adj_norm)` | Sparse diffusion. |
| `vectorized_belief_diffusion(belief_matrix, adj_norm)` | Belief diffusion. |
| `vectorized_behavior_ema(...)` | EMA update. |
| `vectorized_macro_aggregation(trait_matrix)` | Population means. |

### Classes

| Class | Description |
|-------|-------------|
| `StateMatrix` | Columnar store: latent, beliefs, habits. sync_from_agents, sync_to_agents. |

---

## actions.py

**Purpose**: Domain-agnostic representation of agent actions. Wraps survey answers, proactive behaviors, and intent-driven decisions for uniform processing by the simulation engine.

### Constants

| Constant | Description |
|----------|-------------|
| `ACTION_TYPES` | frequency, adopt, reject, support, oppose, rate, choose, increase, decrease, invest, migrate, comply, protest. |
| `TARGET_CATEGORIES` | service, product, policy, candidate, belief, behavior, location, investment, norm. |

### Classes

| Class | Description |
|-------|-------------|
| `Action` | type, target, intensity, option, metadata. From survey answer or intent. |
| `ActionTemplate` | action_type, target_category, scale_type. Used by discovery/action_inference. |

---

## intent.py

**Purpose**: Generate proactive intentions from agent state, goals, social pressure, and environmental signals. Agents form intents autonomously during simulation; high-urgency intents convert to Actions via the factor graph.

### Classes

| Class | Description |
|-------|-------------|
| `Intent` | action_type, target, urgency, source ("goal", "social", "event", "media", "life_event"), metadata. |
| `IntentEngine` | urgency_threshold, max_intents_per_agent. Generate and resolve intents. |

---

## outcome.py

**Purpose**: Compute action outcomes and apply reinforcement signals. After agents take actions (survey, intents, events), the outcome engine computes rewards/costs and feeds them back into beliefs and latent state (closed learning loop).

### Classes

| Class | Description |
|-------|-------------|
| `ActionOutcome` | action_type, target, reward, cost, social_approval, day, metadata. |
| `OutcomeEngine` | Compute outcomes for actions and apply reinforcement to agent state. |

---

## identity.py

**Purpose**: Stable self-concept that evolves slower than beliefs. IdentityState captures core values that resist rapid change; when beliefs diverge from identity, cognitive dissonance rises (opinion inertia, consistent long-horizon personas).

### Classes

| Class | Description |
|-------|-------------|
| `IdentityState` | core_values (belief-sized vector), identity_strength. Update: core_values += identity_lr * (beliefs - core_values). |

---

## utility.py

**Purpose**: Goal-directed utility layer: intentional, utility-maximising behavior blended into the reactive factor-graph pipeline. Agents hold active goals (e.g. "save money", "eat healthier") that bias decisions.

### Classes

| Class | Description |
|-------|-------------|
| `AgentGoal` | name, dimension_weights, priority, ttl_days. |
| `GoalProfile` | goals. add_goal(), tick() (decrement TTL). |
| (utility functions) | final_score = alpha * factor_graph_score + (1 - alpha) * utility_score; alpha typically 0.7–0.8. |
