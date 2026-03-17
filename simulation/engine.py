"""
Simulation kernel: daily time steps with a causally-correct 13-step pipeline.

Pipeline order (perception -> interpretation -> emotion -> action):
  1. Scheduled events
  2. Research context (shared factual grounding)
  3. Media frames (event -> narrative per source)
  4. Raw selective exposure (subscription filter)
  5. Adaptive attention (emotion-gated exposure reweighting)
  6. Life events
  7. Cultural influence
  8. Cognitive processing (bias engine updates beliefs via media)
  9. Alignment computation (updated beliefs vs media)
 10. Activation update (emotional layer)
 11. Social diffusion (behavioral + beliefs)
 12. Cascade detection + event generation + fatigue
 13. Macro feedback + world feedback + scheduler update

Two execution paths:
  - **Vectorized** (>= VECTORIZE_THRESHOLD): matrix operations, sparse graph.
  - **Scalar** (< threshold): consolidated single pass through agents.

The kernel never calls the LLM.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from config.settings import get_settings
from simulation.config import SimulationConfig
from social.influence import fraction_friends_with_trait
from social.network import (
    build_social_network,
    neighbors,
    normalize_adjacency,
    to_sparse_adjacency,
)
from simulation.macro import compute_macro_metrics, macro_influence
from world.districts import location_quality_for_satisfaction
from world.events import EventScheduler

VECTORIZE_THRESHOLD = 2


def _compute_neighbor_means_scalar(agents, social_graph):
    """Per-agent neighbor mean latent vectors and belief vectors (scalar path).

    Returns (latent_means, belief_means) dicts keyed by agent_id.
    """
    from social.network import agent_id_to_node, node_to_agent_id

    id_to_idx = {}
    for i, a in enumerate(agents):
        p = a.get("persona")
        if p:
            id_to_idx[p.agent_id] = i

    latent_means: Dict[str, np.ndarray] = {}
    belief_means: Dict[str, np.ndarray] = {}
    for i, a in enumerate(agents):
        p = a.get("persona")
        if not p:
            continue
        node = agent_id_to_node(social_graph, p.agent_id)
        if node is None:
            continue
        latent_vecs = []
        belief_vecs = []
        for n_node in social_graph.neighbors(node):
            nid = node_to_agent_id(social_graph, n_node)
            j = id_to_idx.get(nid)
            if j is not None:
                s = agents[j].get("state")
                if s and hasattr(s, "latent_state"):
                    latent_vecs.append(s.latent_state.to_vector())
                if s and hasattr(s, "beliefs"):
                    belief_vecs.append(s.beliefs.to_vector())
        if latent_vecs:
            latent_means[p.agent_id] = np.mean(latent_vecs, axis=0)
        if belief_vecs:
            belief_means[p.agent_id] = np.mean(belief_vecs, axis=0)
    return latent_means, belief_means


def _process_life_events(
    agents: List[Dict[str, Any]],
    day: int,
    life_rng: Optional[np.random.Generator] = None,
    social_graph: Optional[Any] = None,
) -> None:
    """Sample and apply life events for each agent (used in both paths)."""
    from world.life_events import sample_life_events, apply_life_event, cascade_to_neighbors

    if life_rng is None:
        return

    agents_by_id = {
        a.get("persona").agent_id: a
        for a in agents
        if a.get("persona") is not None
    }

    for a in agents:
        persona = a.get("persona")
        state = a.get("state")
        if persona is None or state is None:
            continue
        triggered = sample_life_events(
            persona, state, life_rng,
            social_graph=social_graph, agents=agents,
        )
        for evt in triggered:
            apply_life_event(persona, state, evt, rng=life_rng)
            state.life_event_history.append({
                "day": day,
                "event": evt.name,
                "behavioral_impacts": dict(evt.behavioral_impacts),
                "belief_impacts": dict(evt.belief_impacts),
                "demographic_changes": dict(evt.demographic_changes),
            })
            cascade_to_neighbors(persona.agent_id, evt, social_graph, agents_by_id)


def _run_common_daily_tail(
    agents: List[Dict[str, Any]],
    day: int,
    activation_state: Optional[Dict[str, np.ndarray]] = None,
    telemetry: Optional[Any] = None,
) -> None:
    """Post-processing shared by both vectorized and scalar paths.

    Handles goal expiration, memory decay, habit-latent sync, and telemetry.
    """
    for a in agents:
        state = a.get("state")
        if state is None:
            continue

        # SYNC-A: Tick goal profiles so goals expire
        if hasattr(state, "goal_profile"):
            state.goal_profile.tick()

        # SYNC-D: Lightweight habit-latent sync (daily EMA alignment)
        if hasattr(state, "habit_profile") and hasattr(state, "latent_state"):
            hp = state.habit_profile
            ls = state.latent_state
            ema_alpha = 0.02
            for dim in ("routine_stability", "health_orientation", "price_sensitivity",
                        "novelty_seeking", "social_influence_susceptibility"):
                if hasattr(hp, dim) and hasattr(ls, dim):
                    habit_val = getattr(hp, dim)
                    latent_val = getattr(ls, dim)
                    setattr(hp, dim, habit_val + ema_alpha * (latent_val - habit_val))

    # SYNC-B: Memory decay once per simulation day
    try:
        from memory.store import get_memory_store
        store = get_memory_store()
        store.decay_all()
    except Exception:
        pass

    # GAP-1: Record telemetry if collector is provided
    if telemetry is not None and activation_state is not None:
        try:
            telemetry.record(agents, day, activation_state)
        except Exception:
            logger.exception("Telemetry recording failed for day %d", day)


def run_daily_step(
    agents: List[Dict[str, Any]],
    social_graph: Optional[Any],
    scheduler: EventScheduler,
    day: int,
    adj_norm: Optional[Any] = None,
    sparse_adj: Optional[Any] = None,
    enable_social: bool = True,
    enable_macro: bool = True,
    life_rng: Optional[np.random.Generator] = None,
    world_state: Optional[Any] = None,
    research_engine: Optional[Any] = None,
    activation_state: Optional[Dict[str, np.ndarray]] = None,
    cooldown_topics: Optional[Dict[str, Dict[str, int]]] = None,
    telemetry: Optional[Any] = None,
) -> None:
    """One day of the simulation kernel — causally-correct 13-step pipeline.

    Parameters
    ----------
    adj_norm : pre-normalized sparse adjacency (rows sum to 1).
    sparse_adj : raw sparse adjacency (for cascade detection).
    enable_social / enable_macro : toggles for causal attribution runs.
    life_rng : seeded Generator for deterministic life event sampling.
    research_engine : ResearchEngine instance (or None to skip).
    activation_state : shared mutable dict with 'activation' and 'activation_prev'
        arrays of shape (N,). Created on first call if None.
    cooldown_topics : per-agent cooldown tracking dict (mutated in-place).
    """
    from agents.behavior import DIMENSION_NAMES
    from agents.belief_network import BELIEF_DIMENSIONS
    from agents.vectorized import (
        build_belief_matrix,
        build_trait_matrix,
        vectorized_belief_diffusion,
        vectorized_social_influence,
        write_belief_matrix,
        write_trait_matrix,
    )
    from world.culture import (
        apply_cultural_influence,
        build_cultural_matrix,
        get_effective_cultural_field,
        vectorized_cultural_influence,
    )

    if not agents:
        return

    N = len(agents)
    use_vectorized = N >= VECTORIZE_THRESHOLD

    # Ensure activation_state arrays exist
    if activation_state is None:
        activation_state = {}
    if "activation" not in activation_state:
        activation_state["activation"] = np.zeros(N)
        activation_state["activation_prev"] = np.zeros(N)

    if cooldown_topics is None:
        cooldown_topics = {}

    # ── Step 1: Scheduled events ──────────────────────────────────────
    scheduler.process_until(day)
    env = scheduler.get_environment()
    event_impacts = env.get("event_dimension_impacts", {})
    belief_impacts = env.get("event_belief_impacts", {})

    raw_events = env.get("active_events", [])
    if not isinstance(raw_events, list):
        raw_events = []

    # ── Step 2: Research context (shared factual grounding) ───────────
    research_ctx = None
    if research_engine is not None and raw_events:
        try:
            research_ctx = research_engine.build_context(raw_events)
        except Exception:
            logger.exception("Step 2: research context build failed")

    # Inject research context into every agent's environment so that
    # downstream factor-graph decisions and media processing can access it.
    if research_ctx is not None:
        try:
            ctx_text = research_ctx.to_prompt_text() if hasattr(research_ctx, "to_prompt_text") else str(research_ctx)
            for a in agents:
                a.setdefault("environment", {})["research_context"] = ctx_text
        except Exception:
            logger.exception("Step 2b: research context injection failed")

    # ── Step 3: Media frames ─────────────────────────────────────────
    frames = []
    try:
        from media.framing import generate_frames
        if raw_events:
            frames = generate_frames(raw_events)
    except Exception:
        logger.exception("Step 3: media frame generation failed")

    # Step 3b: Strategic actor frames (targeted narrative injection)
    try:
        from media.strategic import get_active_actors, inject_strategic_frames
        active_actors = get_active_actors(day)
        if active_actors:
            strategic_frames = inject_strategic_frames(active_actors, agents, day)
            frames.extend(strategic_frames)
    except Exception:
        logger.exception("Step 3b: strategic media frame injection failed")

    # ── Step 4: Raw selective exposure ───────────────────────────────
    raw_exposure = np.zeros((N, max(len(frames), 1)))
    alignment_matrix = np.zeros((N, max(len(frames), 1)))
    emotion_matrix = np.zeros((N, max(len(frames), 1)))

    if frames:
        try:
            from media.exposure import compute_exposure_matrices
            raw_exposure, alignment_matrix, emotion_matrix = compute_exposure_matrices(
                agents, frames,
            )
        except Exception:
            logger.exception("Step 4: exposure matrix computation failed")

    # ── Step 5: Adaptive attention (emotion -> perception feedback) ──
    adjusted_exposure = raw_exposure
    if frames and raw_exposure.sum() > 0:
        try:
            from media.attention import adaptive_attention
            from config.settings import get_settings as _attn_settings
            _as = _attn_settings()
            adjusted_exposure = adaptive_attention(
                activation_state["activation"],
                raw_exposure,
                emotion_matrix,
                k=_as.attention_sharpness_k,
                p=_as.attention_sharpness_p,
                min_attention=_as.attention_entropy_floor,
            )
        except Exception:
            logger.exception("Step 5: adaptive attention failed")

    # ── SYNC-C: Populate media_exposure_history from consumed frames ──
    if frames and adjusted_exposure.sum() > 0:
        for i, a in enumerate(agents):
            state = a.get("state")
            if state and hasattr(state, "media_exposure_history"):
                row = adjusted_exposure[i]
                consumed = [
                    {"source": f.source_type if hasattr(f, "source_type") else "unknown",
                     "topic": f.topic if hasattr(f, "topic") else "general",
                     "headline": getattr(f, "headline", ""),
                     "weight": float(row[j])}
                    for j, f in enumerate(frames) if j < len(row) and row[j] > 0.01
                ]
                if consumed:
                    state.media_exposure_history.append({
                        "day": day,
                        "frames": consumed[:5],
                    })
                    if len(state.media_exposure_history) > 30:
                        state.media_exposure_history = state.media_exposure_history[-30:]

    # --- Legacy per-agent metadata (backward compat) ---
    if social_graph is not None:
        trait_by_agent: Dict[str, bool] = {}
        for a in agents:
            persona = a.get("persona")
            if not persona:
                continue
            aid = persona.agent_id
            state = a.get("state")
            if persona.lifestyle.primary_service_preference >= 0.5:
                trait_by_agent[aid] = True
            else:
                trait_by_agent[aid] = False

        for a in agents:
            persona = a.get("persona")
            if not persona:
                continue
            aid = persona.agent_id
            frac = fraction_friends_with_trait(social_graph, aid, trait_by_agent)
            a["social_trait_fraction"] = frac
            a["location_quality"] = location_quality_for_satisfaction(persona.location)
            a["environment"] = env
            state = a.get("state")
            if state:
                state.set_social_trait_fraction(frac)
                state.current_day = day

    # ── Step 6: Life events ──────────────────────────────────────────
    _process_life_events(agents, day, life_rng, social_graph)

    # ── Step 6.5: Intent generation (proactive agent behavior) ──────
    try:
        from agents.intent import process_intents_for_agents
        env_for_intents = scheduler.get_environment() if scheduler else {}
        process_intents_for_agents(agents, env_for_intents, day)
    except Exception:
        pass

    # ── Step 6.6: Outcome processing (reinforcement from past actions) ──
    try:
        from agents.outcome import process_outcomes_for_agents
        env_for_outcomes = scheduler.get_environment() if scheduler else {}
        process_outcomes_for_agents(agents, env_for_outcomes, day)
    except Exception:
        pass

    # ── Vectorized pipeline ──────────────────────────────────────────
    if use_vectorized:
        mat = build_trait_matrix(agents)
        bmat = build_belief_matrix(agents)

        # Event impacts (behavioral + beliefs)
        if event_impacts:
            for dim, shift in event_impacts.items():
                if dim in DIMENSION_NAMES:
                    idx = DIMENSION_NAMES.index(dim)
                    mat[:, idx] += shift
        if belief_impacts:
            for dim, shift in belief_impacts.items():
                if dim in BELIEF_DIMENSIONS:
                    idx = BELIEF_DIMENSIONS.index(dim)
                    bmat[:, idx] += shift

        # ── Step 7: Cultural influence (every 7 days) ─────────────────
        if day % 7 == 0:
            cmat = build_cultural_matrix(agents)
            mat = vectorized_cultural_influence(mat, cmat)

        # ── Step 8: Cognitive processing (media -> belief update) ────
        if frames and adjusted_exposure.sum() > 0:
            try:
                from media.exposure import update_beliefs_from_media
                from config.settings import get_settings as _media_settings
                _ms = _media_settings()
                update_beliefs_from_media(
                    agents, adjusted_exposure, frames,
                    w_prior=_ms.media_prior_weight,
                    w_media=_ms.media_influence_weight,
                )
                bmat = build_belief_matrix(agents)
            except Exception:
                logger.exception("Step 8: media belief update failed")

        # Step 8b: Cross-belief coupling propagation (every 3 days)
        if day % 3 == 0:
            for a in agents:
                state = a.get("state")
                if state and hasattr(state, "beliefs"):
                    state.beliefs.propagate_coupling()
            bmat = build_belief_matrix(agents)

        # Step 8c: Identity slow-update (every 15 days)
        if day % 15 == 0:
            for a in agents:
                state = a.get("state")
                if state and hasattr(state, "identity") and hasattr(state, "beliefs"):
                    state.identity.update(state.beliefs.to_vector())

        # ── Step 9: Alignment computation (updated beliefs vs media) ─
        alignment_final = np.zeros(N)
        if frames and adjusted_exposure.sum() > 0:
            try:
                from media.exposure import compute_alignment
                from config.settings import get_settings as _align_settings
                alignment_final = compute_alignment(
                    alignment_matrix, adjusted_exposure,
                    beta=_align_settings().alignment_beta,
                )
            except Exception:
                logger.exception("Step 9: alignment computation failed")

        # ── Step 10: Activation update (emotional layer) ─────────────
        try:
            from simulation.cascade_detector import (
                compute_neighbor_activation,
                update_activation,
            )
            from config.settings import get_settings as _cas_settings
            _cs = _cas_settings()

            neighbor_act = compute_neighbor_activation(
                activation_state["activation"], adj_norm,
            )

            sus_idx = DIMENSION_NAMES.index("social_influence_susceptibility")
            susceptibility = mat[:, sus_idx]

            exposure_agg = adjusted_exposure.max(axis=1) if adjusted_exposure.shape[1] > 0 else np.zeros(N)
            emotion_agg = emotion_matrix.max(axis=1) if emotion_matrix.shape[1] > 0 else np.zeros(N)

            topic_imp_idx = DIMENSION_NAMES.index("price_sensitivity")
            topic_importance = mat[:, topic_imp_idx]

            activation_state["activation_prev"] = activation_state["activation"].copy()
            activation_state["activation"] = update_activation(
                activation_state["activation"],
                exposure_agg,
                emotion_agg,
                topic_importance,
                alignment_final,
                neighbor_act,
                susceptibility,
                decay=_cs.activation_decay,
                w_val=_cs.validation_weight,
                w_out=_cs.outrage_weight,
                lambda_social=_cs.social_lambda,
            )
        except Exception:
            logger.exception("Step 10: activation update failed")

        # Sync activation back to per-agent state for survey/cognitive use
        for i, a in enumerate(agents):
            state = a.get("state")
            if state and hasattr(state, "current_activation"):
                state.current_activation = float(activation_state["activation"][i])

        # ── Step 11: Social diffusion (behavioral + beliefs) ─────────
        if enable_social and adj_norm is not None:
            mat = vectorized_social_influence(mat, adj_norm)
            sus_idx = DIMENSION_NAMES.index("social_influence_susceptibility")
            sus_vec = mat[:, sus_idx]
            bmat = vectorized_belief_diffusion(bmat, adj_norm, susceptibility=sus_vec)

        # Macro feedback (behavioral only) via macro.py
        if enable_macro:
            macro_metrics = compute_macro_metrics(agents)
            if macro_metrics.population_size > 0:
                signals = macro_influence(macro_metrics)
                for a in agents:
                    state = a.get("state")
                    if state and hasattr(state, "latent_state"):
                        state.latent_state.apply_macro_influence(signals)
                mat = build_trait_matrix(agents)

        mat = np.clip(mat, 0.0, 1.0)
        bmat = np.clip(bmat, 0.0, 1.0)
        write_trait_matrix(agents, mat)
        write_belief_matrix(agents, bmat)

        # ── Step 12: Cascade detection + event generation + fatigue ──
        if sparse_adj is not None:
            try:
                from simulation.cascade_detector import (
                    apply_fatigue,
                    detect_activation_clusters,
                    generate_emergent_event,
                    tick_cooldowns,
                )
                from config.settings import get_settings as _cd_settings
                _cds = _cd_settings()

                clusters = detect_activation_clusters(
                    activation_state["activation"],
                    sparse_adj,
                    activation_threshold=_cds.activation_threshold,
                    min_size_absolute=_cds.cascade_min_size,
                    min_size_fraction=_cds.cascade_min_fraction,
                    min_density=_cds.cascade_min_density,
                )

                trust_idx = BELIEF_DIMENSIONS.index("government_trust")
                agent_states_dict = {
                    "activation": activation_state["activation"],
                    "activation_prev": activation_state["activation_prev"],
                    "government_trust": bmat[:, trust_idx],
                    "beliefs": bmat,
                    "social_influence_susceptibility": mat[:, DIMENSION_NAMES.index("social_influence_susceptibility")],
                    "topic_importance": topic_importance,
                }

                for cluster in clusters:
                    event = generate_emergent_event(
                        cluster, agent_states_dict, total_population=N,
                    )
                    if event:
                        from world.events import SimulationEvent
                        sim_event = SimulationEvent(
                            day=day + 1,
                            type=f"emergent_{event['type']}",
                            payload=event,
                        )
                        scheduler.add(sim_event)
                        apply_fatigue(
                            activation_state["activation"],
                            cluster,
                            cooldown_topics,
                            fatigue_factor=_cds.fatigue_factor,
                            cooldown_days=_cds.cooldown_days,
                        )

                tick_cooldowns(cooldown_topics)
            except Exception:
                logger.exception("Step 12: cascade detection failed")

        # Increment calcification for all agents
        from config.settings import get_settings as _calc_settings
        _calc_rate = _calc_settings().calcification_rate
        for a in agents:
            state = a.get("state")
            if state and hasattr(state, "calcification"):
                state.calcification = min(1.0, state.calcification + _calc_rate)

        # ── Step 13: World feedback loop (opt-in) ────────────────────
        if world_state is not None:
            from config.settings import get_settings as _get_ws_settings
            macro = compute_macro_metrics(agents)
            feedback_events = world_state.apply_demand_feedback(
                macro, day,
                max_events_per_step=_get_ws_settings().max_feedback_events_per_step,
            )
            for evt in feedback_events:
                scheduler.add(evt)

        # Emergent cultural norms: agent behavior -> district culture (every 7 days)
        if day % 7 == 0:
            from world.culture import update_emergent_norms
            update_emergent_norms(agents)

        _run_common_daily_tail(agents, day, activation_state, telemetry)
        return

    # ── Scalar path (single consolidated pass) ───────────────────────
    # Event impacts (behavioral + beliefs)
    for a in agents:
        state = a.get("state")
        if not state:
            continue
        if event_impacts and hasattr(state, "latent_state"):
            state.latent_state.apply_event_impact(event_impacts)
        if belief_impacts and hasattr(state, "beliefs"):
            state.beliefs.apply_event_impact(belief_impacts)

    # Cultural field influence (scalar, every 7 days to match vectorized path)
    if day % 7 == 0:
        for a in agents:
            persona = a.get("persona")
            state = a.get("state")
            if persona is None or state is None:
                continue
            cf = get_effective_cultural_field(persona.location)
            apply_cultural_influence(state.latent_state, cf)

    # Step 8 (scalar): Media -> belief update
    if frames and adjusted_exposure.sum() > 0:
        try:
            from media.exposure import update_beliefs_from_media
            from config.settings import get_settings as _media_settings_s
            _ms_s = _media_settings_s()
            update_beliefs_from_media(
                agents, adjusted_exposure, frames,
                w_prior=_ms_s.media_prior_weight,
                w_media=_ms_s.media_influence_weight,
            )
        except Exception:
            logger.exception("Scalar path: media belief update failed")

    # Step 8b (scalar): Cross-belief coupling propagation (every 3 days)
    if day % 3 == 0:
        for a in agents:
            state = a.get("state")
            if state and hasattr(state, "beliefs"):
                state.beliefs.propagate_coupling()

    # Step 8c (scalar): Identity slow-update (every 15 days)
    if day % 15 == 0:
        for a in agents:
            state = a.get("state")
            if state and hasattr(state, "identity") and hasattr(state, "beliefs"):
                state.identity.update(state.beliefs.to_vector())

    # Step 9 (scalar): Alignment computation
    alignment_final_s = np.zeros(N)
    if frames and adjusted_exposure.sum() > 0:
        try:
            from media.exposure import compute_alignment
            from config.settings import get_settings as _align_settings_s
            alignment_final_s = compute_alignment(
                alignment_matrix, adjusted_exposure,
                beta=_align_settings_s().alignment_beta,
            )
        except Exception:
            logger.exception("Scalar path: alignment computation failed")

    # Pre-compute adjacency for scalar path (used in Steps 10 and 12)
    scalar_adj_norm = None
    scalar_sparse_adj = None
    if social_graph is not None:
        scalar_sparse_adj = to_sparse_adjacency(social_graph)
        scalar_adj_norm = normalize_adjacency(scalar_sparse_adj)

    sus_vec_s = np.array([
        a.get("state").latent_state.social_influence_susceptibility
        if a.get("state") and hasattr(a.get("state"), "latent_state") else 0.5
        for a in agents
    ])
    topic_importance_s = np.array([
        a.get("state").latent_state.price_sensitivity
        if a.get("state") and hasattr(a.get("state"), "latent_state") else 0.5
        for a in agents
    ])

    # Step 10 (scalar): Activation update
    try:
        from simulation.cascade_detector import (
            compute_neighbor_activation,
            update_activation,
        )
        from config.settings import get_settings as _cas_settings_s
        _cs_s = _cas_settings_s()

        neighbor_act_s = compute_neighbor_activation(
            activation_state["activation"], scalar_adj_norm,
        )
        exposure_agg_s = adjusted_exposure.max(axis=1) if adjusted_exposure.shape[1] > 0 else np.zeros(N)
        emotion_agg_s = emotion_matrix.max(axis=1) if emotion_matrix.shape[1] > 0 else np.zeros(N)

        activation_state["activation_prev"] = activation_state["activation"].copy()
        activation_state["activation"] = update_activation(
            activation_state["activation"],
            exposure_agg_s, emotion_agg_s, topic_importance_s,
            alignment_final_s, neighbor_act_s, sus_vec_s,
            decay=_cs_s.activation_decay,
            w_val=_cs_s.validation_weight,
            w_out=_cs_s.outrage_weight,
            lambda_social=_cs_s.social_lambda,
        )
    except Exception:
        logger.exception("Scalar path: activation update failed")

    # Sync activation back to per-agent state (scalar path)
    for i, a in enumerate(agents):
        state = a.get("state")
        if state and hasattr(state, "current_activation"):
            state.current_activation = float(activation_state["activation"][i])

    # Social diffusion (behavioral + beliefs) -- requires social_graph
    if enable_social and social_graph is not None:
        latent_means, belief_nbr_means = _compute_neighbor_means_scalar(agents, social_graph)
        for a in agents:
            p = a.get("persona")
            if not p:
                continue
            state = a.get("state")
            if not state:
                continue
            if hasattr(state, "latent_state"):
                nm = latent_means.get(p.agent_id)
                if nm is not None:
                    state.latent_state.apply_social_influence(nm)
            if hasattr(state, "beliefs"):
                bm = belief_nbr_means.get(p.agent_id)
                if bm is not None:
                    state.beliefs.apply_social_diffusion(bm)

    # Macro feedback (behavioral only)
    if enable_macro:
        macro = compute_macro_metrics(agents)
        if macro.population_size > 0:
            signals = macro_influence(macro)
            for a in agents:
                state = a.get("state")
                if state and hasattr(state, "latent_state"):
                    state.latent_state.apply_macro_influence(signals)

    # Calcification increment (scalar path)
    from config.settings import get_settings as _calc_settings_s
    _calc_rate_s = _calc_settings_s().calcification_rate
    for a in agents:
        state = a.get("state")
        if state and hasattr(state, "calcification"):
            state.calcification = min(1.0, state.calcification + _calc_rate_s)

    # ── Step 12 (scalar): Cascade detection + emergent events + fatigue ──
    if scalar_sparse_adj is not None:
        try:
            from simulation.cascade_detector import (
                apply_fatigue,
                detect_activation_clusters,
                generate_emergent_event,
                tick_cooldowns,
            )
            from agents.belief_network import BELIEF_DIMENSIONS as _BELIEF_DIMS_S
            from config.settings import get_settings as _cd_settings_s
            _cds_s = _cd_settings_s()

            clusters_s = detect_activation_clusters(
                activation_state["activation"],
                scalar_sparse_adj,
                activation_threshold=_cds_s.activation_threshold,
                min_size_absolute=_cds_s.cascade_min_size,
                min_size_fraction=_cds_s.cascade_min_fraction,
                min_density=_cds_s.cascade_min_density,
            )

            if clusters_s:
                bmat_s = np.array([
                    a.get("state").beliefs.to_vector()
                    if a.get("state") and hasattr(a.get("state"), "beliefs")
                    else np.full(len(_BELIEF_DIMS_S), 0.5)
                    for a in agents
                ])
                trust_idx_s = _BELIEF_DIMS_S.index("government_trust")
                agent_states_dict_s = {
                    "activation": activation_state["activation"],
                    "activation_prev": activation_state["activation_prev"],
                    "government_trust": bmat_s[:, trust_idx_s],
                    "beliefs": bmat_s,
                    "social_influence_susceptibility": sus_vec_s,
                    "topic_importance": topic_importance_s,
                }

                for cluster in clusters_s:
                    event = generate_emergent_event(
                        cluster, agent_states_dict_s, total_population=N,
                    )
                    if event:
                        from world.events import SimulationEvent as _SE_s
                        sim_event = _SE_s(
                            day=day + 1,
                            type=f"emergent_{event['type']}",
                            payload=event,
                        )
                        scheduler.add(sim_event)
                        apply_fatigue(
                            activation_state["activation"],
                            cluster,
                            cooldown_topics,
                            fatigue_factor=_cds_s.fatigue_factor,
                            cooldown_days=_cds_s.cooldown_days,
                        )

            tick_cooldowns(cooldown_topics)
        except Exception:
            logger.exception("Scalar path: cascade detection failed")

    # World feedback loop (opt-in)
    if world_state is not None:
        from config.settings import get_settings as _get_ws_settings2
        _macro = compute_macro_metrics(agents)
        feedback_events = world_state.apply_demand_feedback(
            _macro, day,
            max_events_per_step=_get_ws_settings2().max_feedback_events_per_step,
        )
        for evt in feedback_events:
            scheduler.add(evt)

    # Emergent cultural norms (scalar path, every 7 days)
    if day % 7 == 0:
        from world.culture import update_emergent_norms
        update_emergent_norms(agents)

    _run_common_daily_tail(agents, day, activation_state, telemetry)


def run_simulation(
    agents: List[Dict[str, Any]],
    days: int,
    social_graph: Optional[Any] = None,
    scheduler: Optional[EventScheduler] = None,
    seed: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
    enable_social: bool = True,
    enable_macro: bool = True,
    world_state: Optional[Any] = None,
    collect_telemetry: bool = False,
) -> List[Dict[str, Any]]:
    """Run simulation for ``days`` days.

    A row-normalized sparse adjacency matrix is pre-computed once from
    the social graph for vectorized social diffusion.

    ``enable_social`` / ``enable_macro`` can be disabled for causal
    attribution (isolate the contribution of each mechanism).

    If ``collect_telemetry`` is True, a TelemetryCollector is attached
    and returned agents will have a ``_telemetry`` key in the first
    agent dict containing the collected snapshots.
    """
    if config is None:
        config = SimulationConfig(master_seed=seed, days=days)

    rng = config.make_rng()

    if social_graph is None and agents:
        personas = [a["persona"] for a in agents]
        social_graph = build_social_network(personas, seed=config.master_seed)
    if scheduler is None:
        scheduler = EventScheduler()

    adj_norm = None
    sparse_adj_matrix = None
    if social_graph is not None:
        sparse_adj_matrix = to_sparse_adjacency(social_graph)
        adj_norm = normalize_adjacency(sparse_adj_matrix)

    life_seed = config.derive_child_seed("life_events")
    life_rng = np.random.default_rng(life_seed)

    activation_state: Dict[str, np.ndarray] = {
        "activation": np.zeros(len(agents)),
        "activation_prev": np.zeros(len(agents)),
    }
    cooldown_topics: Dict[str, Dict[str, int]] = {}

    # Auto-instantiate WorldState when world feedback is enabled.
    if world_state is None and get_settings().enable_world_feedback:
        try:
            from simulation.world_feedback import WorldState
            world_state = WorldState()
        except Exception:
            logger.exception("Could not auto-instantiate WorldState")

    research_engine_instance = None
    try:
        from research.engine import ResearchEngine
        from config.settings import get_settings as _re_settings
        _rs = _re_settings()
        research_engine_instance = ResearchEngine(
            provider=_rs.research_api_provider,
            cache_path=_rs.research_cache_path,
            openai_api_key=_rs.openai_api_key,
        )
    except Exception:
        logger.exception("Could not initialize ResearchEngine")

    telemetry_collector = None
    if collect_telemetry:
        from analytics.telemetry import TelemetryCollector
        telemetry_collector = TelemetryCollector()

    dimension_monitor = None
    epoch_size = 30
    try:
        from discovery.dimension_monitor import DimensionEvolutionMonitor
        dimension_monitor = DimensionEvolutionMonitor()
    except Exception:
        pass

    for day in range(1, config.days + 1):
        run_daily_step(
            agents, social_graph, scheduler, day,
            adj_norm=adj_norm,
            sparse_adj=sparse_adj_matrix,
            enable_social=enable_social,
            enable_macro=enable_macro,
            life_rng=life_rng,
            world_state=world_state,
            research_engine=research_engine_instance,
            activation_state=activation_state,
            cooldown_topics=cooldown_topics,
            telemetry=telemetry_collector,
        )

        if dimension_monitor and day % epoch_size == 0:
            try:
                report = dimension_monitor.check_adequacy(agents, day=day)
                if report.needs_extension and report.suggested_new_dimensions:
                    dimension_monitor.extend_dimensions(
                        report.suggested_new_dimensions, agents, kind="behavioral"
                    )
            except Exception:
                logger.debug("Dimension monitor check failed at day %d", day)

    if telemetry_collector is not None and agents:
        agents[0]["_telemetry"] = telemetry_collector.to_dicts()

    return agents
