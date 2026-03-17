"""
Survey orchestrator: distribute questions to agents async, collect responses.
Supports archetype compression with narrative template sharing for cost reduction.

Before running a survey, the orchestrator performs a social warmup diffusion
pass so that agents' latent states reflect peer influence.  Neighbor latent
means are also computed and injected into each agent's environment dict for
the social_factor to consume during decision-making.
"""

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config.settings import get_settings
from population.personas import Persona
from simulation.archetypes import build_archetype_map


def _fuzzy_replace(text: str, src_val: str, tgt_val: str) -> str:
    """Replace src_val in text, trying word-boundary exact match first then partial."""
    if not src_val or not tgt_val or src_val == tgt_val:
        return text
    # Word-boundary-aware exact match to avoid replacing substrings (e.g. "car" in "career")
    exact_pattern = r'(?<!\w)' + re.escape(src_val) + r'(?!\w)'
    replaced = re.sub(exact_pattern, tgt_val, text, flags=re.IGNORECASE)
    if replaced != text:
        return replaced
    # Partial: try matching the first significant word (3+ chars) from src_val
    words = [w for w in src_val.split() if len(w) >= 3]
    for word in words:
        pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
        if re.search(pattern, text, flags=re.IGNORECASE):
            replaced = re.sub(pattern, tgt_val, text, count=1, flags=re.IGNORECASE)
            return replaced
    return text


def _adapt_narrative(
    template_narrative: str,
    source_persona: Persona,
    target_persona: Persona,
    sampled_option: str,
) -> str:
    """Adapt a representative's LLM narrative for a non-representative agent.

    Uses fuzzy replacement: tries exact match first, then falls back to
    word-boundary partial match on significant tokens.
    """
    text = template_narrative

    swap_pairs = [
        (source_persona.location, target_persona.location),
        (source_persona.personal_anchors.cuisine_preference, target_persona.personal_anchors.cuisine_preference),
        (source_persona.personal_anchors.hobby, target_persona.personal_anchors.hobby),
        (source_persona.personal_anchors.work_schedule, target_persona.personal_anchors.work_schedule),
        (source_persona.personal_anchors.typical_dinner_time, target_persona.personal_anchors.typical_dinner_time),
        (source_persona.personal_anchors.diet, target_persona.personal_anchors.diet),
        (source_persona.personal_anchors.commute_method, target_persona.personal_anchors.commute_method),
        (source_persona.occupation, target_persona.occupation),
    ]
    for src_val, tgt_val in swap_pairs:
        text = _fuzzy_replace(text, src_val, tgt_val)

    return text


def _social_warmup(
    agents: List[Dict[str, Any]],
    social_graph: Optional[Any],
    warmup_steps: int = 3,
) -> None:
    """Run a few social diffusion steps before a survey so latent states
    reflect peer influence.  Operates in-place on agent states.
    """
    if social_graph is None or not agents:
        return
    from agents.vectorized import (
        build_belief_matrix,
        build_trait_matrix,
        vectorized_belief_diffusion,
        vectorized_social_influence,
        write_belief_matrix,
        write_trait_matrix,
    )
    from social.network import normalize_adjacency, to_sparse_adjacency

    sparse_adj = to_sparse_adjacency(social_graph)
    adj_norm = normalize_adjacency(sparse_adj)

    for _ in range(warmup_steps):
        mat = build_trait_matrix(agents)
        bmat = build_belief_matrix(agents)
        mat = vectorized_social_influence(mat, adj_norm)
        bmat = vectorized_belief_diffusion(bmat, adj_norm)
        mat = np.clip(mat, 0.0, 1.0)
        bmat = np.clip(bmat, 0.0, 1.0)
        write_trait_matrix(agents, mat)
        write_belief_matrix(agents, bmat)


def _compute_neighbor_latent_means(
    agents: List[Dict[str, Any]],
    social_graph: Optional[Any],
    sample_k: int = 15,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-agent neighbor mean latent vectors for richer social context.

    To avoid averaging over the entire neighborhood (which washes out signal
    in hub nodes), at most *sample_k* neighbors are sampled per agent.
    """
    if social_graph is None:
        return {}

    import random
    _rng = random.Random(seed)
    from social.network import agent_id_to_node, node_to_agent_id

    id_to_idx: Dict[str, int] = {}
    for i, a in enumerate(agents):
        p = a.get("persona")
        if p:
            id_to_idx[p.agent_id] = i

    means: Dict[str, np.ndarray] = {}
    for i, a in enumerate(agents):
        p = a.get("persona")
        if not p:
            continue
        node = agent_id_to_node(social_graph, p.agent_id)
        if node is None:
            continue
        neighbor_nodes = list(social_graph.neighbors(node))
        if len(neighbor_nodes) > sample_k:
            neighbor_nodes = _rng.sample(neighbor_nodes, sample_k)
        vecs = []
        for n_node in neighbor_nodes:
            nid = node_to_agent_id(social_graph, n_node)
            j = id_to_idx.get(nid)
            if j is not None:
                s = agents[j].get("state")
                if s and hasattr(s, "latent_state"):
                    vecs.append(s.latent_state.to_vector())
        if vecs:
            means[p.agent_id] = np.mean(vecs, axis=0)
    return means


async def run_agent_async(
    agent_id: str,
    persona: Persona,
    question: str,
    question_id: str,
    think_fn: Callable[..., Any],
    friends_using: float = 0.0,
    location_quality: float = 0.5,
) -> Dict[str, Any]:
    """Run one agent's think pipeline (async)."""
    result = await think_fn(
        persona=persona,
        question=question,
        question_id=question_id,
        friends_using=friends_using,
        location_quality=location_quality,
    )
    result["agent_id"] = agent_id
    return result


async def run_survey(
    agents: List[Dict[str, Any]],
    question: str,
    question_id: str = "",
    options: Optional[List[str]] = None,
    think_fn: Optional[Callable[..., Any]] = None,
    use_archetypes: bool = False,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run survey: each agent answers the question via think_fn.
    If use_archetypes=True, only archetype representatives call LLM;
    others get an adapted narrative via token substitution (~90% cost reduction).
    """
    settings = get_settings()
    max_concurrent = max_concurrent or settings.max_concurrent_llm_calls
    semaphore = asyncio.Semaphore(max_concurrent)

    # Shared set for batch-level opening deduplication across all agents
    used_openings: set = set()

    # Build agent-id -> state/env lookups so default_think reuses persistent state
    # across multi-question surveys instead of creating fresh state each call.
    _state_cache: Dict[str, Any] = {}
    _env_cache: Dict[str, Dict[str, Any]] = {}
    for a in agents:
        p = a.get("persona")
        if not p:
            continue
        s = a.get("state")
        if s:
            _state_cache[p.agent_id] = s
        e = a.get("environment")
        if e:
            _env_cache[p.agent_id] = e

    # Social warmup: run a few diffusion steps so latent states reflect
    # peer influence before survey decisions are made.
    import api.state as _app_state
    _sg = _app_state.social_graph
    _has_states = any(a.get("state") is not None for a in agents)
    if _sg is not None and _has_states and len(agents) >= 2:
        _social_warmup(agents, _sg, warmup_steps=3)
        _neighbor_means = _compute_neighbor_latent_means(
            agents, _sg, seed=getattr(settings, "master_seed", 42),
        )
        for a in agents:
            p = a.get("persona")
            if p:
                nm = _neighbor_means.get(p.agent_id)
                if nm is not None:
                    a.setdefault("environment", {})["neighbor_latent_mean"] = nm
                    _env_cache[p.agent_id] = a.get("environment", {})

    # --- Shared research context (one lookup per question, not per agent) ---
    _research_ctx = None
    try:
        from research.engine import ResearchEngine
        _re = ResearchEngine(
            provider=settings.research_api_provider,
            cache_path=settings.research_cache_path,
            openai_api_key=settings.openai_api_key,
        )
        _research_ctx = _re.research_question(question)
    except Exception:
        pass

    if think_fn is None:
        async def default_think(persona, question, question_id, friends_using, location_quality):
            from agents.cognitive import AgentCognitiveEngine
            from agents.state import AgentState
            from llm.prompts import reasoner_via_llm
            from memory.store import get_memory_store

            state = _state_cache.get(persona.agent_id) or AgentState.from_persona(persona)
            _state_cache[persona.agent_id] = state
            store = get_memory_store()

            def recall(agent_id, perception):
                return store.recall(agent_id, perception.raw_question, top_k=3)

            # Build simulation dynamics context for narrative generation
            sim_ctx = {}
            if hasattr(state, "current_activation"):
                sim_ctx["activation"] = state.current_activation
            if hasattr(state, "life_event_history") and state.life_event_history:
                sim_ctx["recent_life_events"] = [
                    e.get("event", "") for e in state.life_event_history[-3:]
                ]
            if hasattr(state, "current_day"):
                sim_ctx["day"] = state.current_day
            if hasattr(state, "media_exposure_history") and state.media_exposure_history:
                recent_media = state.media_exposure_history[-3:]
                headlines = []
                for entry in recent_media:
                    for f in entry.get("frames", []):
                        hl = f.get("headline") or f.get("topic", "")
                        if hl:
                            headlines.append(hl)
                if headlines:
                    sim_ctx["recent_media_headlines"] = headlines[:6]
                    sources = set()
                    for entry in recent_media:
                        for f in entry.get("frames", []):
                            src = f.get("source")
                            if src:
                                sources.add(src)
                    if sources:
                        sim_ctx["media_sources"] = list(sources)[:4]

            _local_sim_ctx = sim_ctx

            async def _reasoner_with_dedup(p, q, sa, dist, mems):
                return await reasoner_via_llm(
                    p, q, sa, dist, mems,
                    used_openings=used_openings,
                    simulation_context=_local_sim_ctx,
                    option_labels=options,
                )

            engine = AgentCognitiveEngine(
                persona=persona,
                state=state,
                memory_recall=recall,
                reasoner=_reasoner_with_dedup,
            )
            # Inject social context (neighbor latent means) from warmup pass
            _agent_env = _env_cache.get(persona.agent_id)
            if _agent_env:
                engine.set_world_environment(_agent_env)
            # Inject research context into the world environment
            if _research_ctx is not None:
                engine._world_environment["research_context"] = _research_ctx.to_prompt_text()
            engine._world_environment["simulation_context"] = sim_ctx

            result = await engine.think(question, question_id, friends_using, location_quality)
            store.add_memory(
                persona.agent_id,
                f"Q: {question} | A: {result.get('sampled_option', '')}",
                metadata={
                    "question_id": question_id,
                    "sampled_option": result.get("sampled_option", ""),
                    "topic": result.get("perception_topic", ""),
                    "domain": result.get("perception_domain", ""),
                },
            )
            return result
        think_fn = default_think

    async def bounded_think(persona, question, question_id, friends_using, location_quality):
        async with semaphore:
            return await think_fn(
                persona=persona,
                question=question,
                question_id=question_id,
                friends_using=friends_using,
                location_quality=location_quality,
            )

    personas = [a["persona"] for a in agents]
    archetype_rep: Optional[Dict[int, int]] = None
    labels: Optional[List[int]] = None
    if use_archetypes and len(personas) > settings.use_archetypes_above_agents:
        archetype_rep, labels = build_archetype_map(personas, settings.archetype_count)
        # Stamp archetype cluster ids onto personas for downstream tracing
        if labels is not None:
            for i, label in enumerate(labels):
                personas[i].meta.persona_cluster = int(label)
                rep_idx = archetype_rep.get(label) if archetype_rep else None
                personas[i].meta.archetype_id = int(rep_idx) if rep_idx is not None else None

    # Phase 1: run representatives (LLM calls)
    rep_results: Dict[int, Dict[str, Any]] = {}
    if archetype_rep is not None and labels is not None:
        rep_indices = set(archetype_rep.values())
        rep_tasks = {}
        for idx in rep_indices:
            a = agents[idx]
            persona = a["persona"]
            rep_tasks[idx] = bounded_think(
                persona=persona,
                question=question,
                question_id=question_id,
                friends_using=a.get("social_trait_fraction", 0.0),
                location_quality=a.get("location_quality", 0.5),
            )
        gathered = await asyncio.gather(*rep_tasks.values(), return_exceptions=True)
        for idx, result in zip(rep_tasks.keys(), gathered):
            if isinstance(result, Exception):
                rep_results[idx] = {
                    "agent_id": agents[idx]["persona"].agent_id,
                    "error": str(result),
                    "answer": "",
                }
            else:
                result["agent_id"] = agents[idx]["persona"].agent_id
                rep_results[idx] = result

    async def run_one(i: int) -> Dict[str, Any]:
        a = agents[i]
        persona = a["persona"]
        agent_id = persona.agent_id
        friends_using = a.get("social_trait_fraction", 0.0)
        location_quality = a.get("location_quality", 0.5)

        if archetype_rep is not None and labels is not None:
            c = labels[i]
            rep_idx = archetype_rep.get(c, 0)
            if rep_idx != i:
                from agents.cognitive import AgentCognitiveEngine
                from agents.state import AgentState
                from memory.store import get_memory_store

                state = AgentState.from_persona(persona)
                store = get_memory_store()

                def _recall(agent_id, perception, _s=store):
                    return _s.recall(agent_id, perception.raw_question, top_k=3)

                engine = AgentCognitiveEngine(
                    persona=persona, state=state, memory_recall=_recall,
                )
                perception = engine.perceive(question)
                memories = await engine.recall(perception)
                dist, sampled = engine.decide(
                    perception, memories, friends_using=friends_using,
                    location_quality=location_quality,
                )
                rep_data = rep_results.get(rep_idx, {})
                rep_narrative = rep_data.get("answer", "")
                rep_sampled = rep_data.get("sampled_option", "")

                # For open_text (empty distribution), each agent needs a fresh LLM call
                # — no shared option to adapt from.
                if not dist:
                    result = await bounded_think(
                        persona=persona, question=question,
                        question_id=question_id,
                        friends_using=friends_using,
                        location_quality=location_quality,
                    )
                    result["agent_id"] = agent_id
                    return result

                # Guard: if this agent chose a different option than the
                # archetype representative, the narrative would contradict
                # the answer -- fall back to a fresh LLM call.
                if sampled != rep_sampled:
                    result = await bounded_think(
                        persona=persona, question=question,
                        question_id=question_id,
                        friends_using=friends_using,
                        location_quality=location_quality,
                    )
                    result["agent_id"] = agent_id
                    return result

                if rep_narrative and rep_narrative != rep_sampled:
                    source_persona = agents[rep_idx]["persona"]
                    adapted = _adapt_narrative(rep_narrative, source_persona, persona, sampled)
                else:
                    adapted = sampled
                pa = persona.personal_anchors
                pmeta = persona.meta
                return {
                    "agent_id": agent_id,
                    "answer": adapted,
                    "sampled_option": sampled,
                    "distribution": dist,
                    "demographics": {
                        "age_group": persona.age,
                        "nationality": persona.nationality,
                        "income_band": persona.income,
                        "location": persona.location,
                        "occupation": persona.occupation,
                        "household_size": persona.household_size,
                        "family_children": persona.family.children,
                        "has_spouse": persona.family.spouse,
                    },
                    "lifestyle": {
                        "cuisine_preference": pa.cuisine_preference,
                        "diet": pa.diet,
                        "hobby": pa.hobby,
                        "work_schedule": pa.work_schedule,
                        "health_focus": pa.health_focus,
                        "commute_method": pa.commute_method,
                    },
                    "persona_meta": {
                        "persona_version": pmeta.persona_version,
                        "synthesis_method": pmeta.synthesis_method,
                        "generation_seed": pmeta.generation_seed,
                        "archetype_id": pmeta.archetype_id,
                        "persona_cluster": pmeta.persona_cluster,
                    },
                }
            else:
                return rep_results.get(i, await bounded_think(
                    persona=persona,
                    question=question,
                    question_id=question_id,
                    friends_using=friends_using,
                    location_quality=location_quality,
                ))

        return await bounded_think(
            persona=persona,
            question=question,
            question_id=question_id,
            friends_using=friends_using,
            location_quality=location_quality,
        )

    tasks = [run_one(i) for i in range(len(agents))]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append({"agent_id": agents[i]["persona"].agent_id, "error": str(r), "answer": ""})
        else:
            r["agent_id"] = r.get("agent_id") or agents[i]["persona"].agent_id
            out.append(r)
    return out
