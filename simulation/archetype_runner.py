"""
Archetype round runner: compute decisions for K archetypes, then expand to N agents.

This is the core 5-10x speed multiplier.  Instead of running the full cognitive
pipeline for every agent, only archetype representatives call the LLM.  Member
agents receive a perturbed copy of the archetype's distribution, re-sample their
own answer, and get an adapted narrative template.  A configurable narrative
budget controls how many non-representatives receive fresh LLM calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config.settings import get_settings
from simulation.archetypes import ArchetypeState
from simulation.orchestrator import _adapt_narrative

VARIANT_TONES = ["casual", "reflective", "humorous", "matter_of_fact", "skeptical", "practical"]


# ------------------------------------------------------------------
# Distribution perturbation
# ------------------------------------------------------------------

def _perturb_distribution(
    base: Dict[str, float],
    noise_std: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Add per-option Gaussian noise and re-normalize."""
    keys = list(base.keys())
    vals = np.array([base[k] for k in keys])
    vals = vals + rng.normal(0.0, noise_std, size=vals.shape)
    vals = np.clip(vals, 0.0, None)
    total = vals.sum()
    if total < 1e-12:
        vals = np.ones_like(vals) / len(vals)
    else:
        vals = vals / total
    return dict(zip(keys, vals.tolist()))


def _sample_from_distribution(
    dist: Dict[str, float],
    rng: np.random.Generator,
) -> str:
    """Weighted random sample from a probability distribution dict."""
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys])
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


# ------------------------------------------------------------------
# Single archetype response builder
# ------------------------------------------------------------------

def _build_response_dict(
    agent: Dict[str, Any],
    answer: str,
    sampled_option: str,
    distribution: Dict[str, float],
) -> Dict[str, Any]:
    """Build a standard response dict from an agent dict."""
    persona = agent["persona"]
    pa = persona.personal_anchors
    meta = persona.meta
    return {
        "agent_id": persona.agent_id,
        "answer": answer,
        "sampled_option": sampled_option,
        "distribution": distribution,
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
            "persona_version": meta.persona_version,
            "synthesis_method": meta.synthesis_method,
            "generation_seed": meta.generation_seed,
            "archetype_id": meta.archetype_id,
            "persona_cluster": meta.persona_cluster,
        },
    }


# ------------------------------------------------------------------
# Expand one archetype to its member agents
# ------------------------------------------------------------------

async def _expand_archetype(
    astate: ArchetypeState,
    agents: List[Dict[str, Any]],
    agent_id_to_idx: Dict[str, int],
    noise_std: float,
    narrative_budget: float,
    think_fn: Callable[..., Any],
    semaphore: asyncio.Semaphore,
    question: str,
    question_id: str,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Expand archetype result to all member agents."""
    rep_dist = astate.last_distribution
    rep_sampled = astate.last_sampled
    templates = astate.narrative_templates or [""]
    rep_persona = astate.persona_template

    # For open_text (empty distribution), each agent needs a fresh LLM call
    if not rep_dist:
        results = []
        for member_id in astate.members:
            idx = agent_id_to_idx.get(member_id)
            if idx is None:
                continue
            agent = agents[idx]
            if idx == astate.representative_idx:
                results.append(_build_response_dict(
                    agent, templates[0] if templates else "", rep_sampled, {},
                ))
                continue
            async def _llm_for_open_text(
                _agent=agent, _question=question, _qid=question_id,
            ):
                async with semaphore:
                    return await think_fn(
                        persona=_agent["persona"],
                        question=_question,
                        question_id=_qid,
                        friends_using=_agent.get("social_trait_fraction", 0.0),
                        location_quality=_agent.get("location_quality", 0.5),
                    )
            result = await _llm_for_open_text()
            result["agent_id"] = agent["persona"].agent_id
            results.append(result)
        return results

    results: List[Dict[str, Any]] = []

    # Decide which non-rep members get LLM narratives
    non_rep_ids = [
        mid for mid in astate.members
        if agent_id_to_idx.get(mid, -1) != astate.representative_idx
    ]
    n_llm = max(1, int(len(non_rep_ids) * narrative_budget)) if non_rep_ids else 0
    llm_set = set(rng.choice(
        len(non_rep_ids), size=min(n_llm, len(non_rep_ids)), replace=False,
    ).tolist()) if non_rep_ids else set()

    llm_tasks = []

    for member_pos, member_id in enumerate(astate.members):
        idx = agent_id_to_idx.get(member_id)
        if idx is None:
            continue
        agent = agents[idx]

        if idx == astate.representative_idx:
            results.append(_build_response_dict(
                agent, templates[0], rep_sampled, rep_dist,
            ))
            continue

        perturbed = _perturb_distribution(rep_dist, noise_std, rng)
        sampled = _sample_from_distribution(perturbed, rng)

        # Pick a random template from the pool for linguistic diversity
        rep_narrative = templates[int(rng.integers(0, len(templates)))]

        is_in_non_rep = member_id in [non_rep_ids[j] for j in range(len(non_rep_ids))]
        non_rep_pos = non_rep_ids.index(member_id) if is_in_non_rep else -1

        if non_rep_pos >= 0 and non_rep_pos in llm_set:
            # This agent gets a fresh LLM narrative
            async def _llm_for_member(
                _agent=agent, _question=question, _qid=question_id,
                _sampled=sampled, _perturbed=perturbed,
            ):
                async with semaphore:
                    result = await think_fn(
                        persona=_agent["persona"],
                        question=_question,
                        question_id=_qid,
                        friends_using=_agent.get("social_trait_fraction", 0.0),
                        location_quality=_agent.get("location_quality", 0.5),
                    )
                return result

            llm_tasks.append((_llm_for_member, idx, perturbed, sampled))
        else:
            # Adapt the archetype narrative via token substitution
            if sampled == rep_sampled and rep_narrative and rep_narrative != rep_sampled:
                adapted = _adapt_narrative(
                    rep_narrative, rep_persona, agent["persona"], sampled,
                )
            else:
                adapted = sampled
            results.append(_build_response_dict(
                agent, adapted, sampled, perturbed,
            ))

    # Run LLM tasks concurrently
    if llm_tasks:
        coros = [fn() for fn, _, _, _ in llm_tasks]
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        for (_, idx, perturbed, sampled), result in zip(llm_tasks, gathered):
            agent = agents[idx]
            if isinstance(result, Exception):
                results.append(_build_response_dict(
                    agent, sampled, sampled, perturbed,
                ))
            else:
                result["agent_id"] = agent["persona"].agent_id
                results.append(result)

    return results


# ------------------------------------------------------------------
# Per-agent state update after expansion
# ------------------------------------------------------------------

def _update_agent_states(
    agents: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    question_id: str,
) -> None:
    """Update each agent's individual AgentState from their specific answer."""
    agent_id_to_resp = {r["agent_id"]: r for r in responses}
    for a in agents:
        persona = a.get("persona")
        if not persona:
            continue
        resp = agent_id_to_resp.get(persona.agent_id)
        if not resp:
            continue
        state = a.get("state")
        if state is None:
            continue
        sampled = resp.get("sampled_option", "")
        if sampled and question_id:
            state.update_after_answer(question_id, sampled)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

async def run_archetype_round(
    archetype_states: Dict[int, ArchetypeState],
    labels: List[int],
    agents: List[Dict[str, Any]],
    question: str,
    question_id: str,
    options: Optional[List[str]] = None,
    think_fn: Optional[Callable[..., Any]] = None,
    max_concurrent: Optional[int] = None,
    noise_std: float = 0.05,
    narrative_budget: float = 0.20,
    seed: Optional[int] = None,
    narrative_templates_per_archetype: int = 3,
) -> List[Dict[str, Any]]:
    """Run one survey round using archetype execution + result expansion.

    Phase 1: Run full cognitive pipeline for each archetype representative.
    Phase 2: Expand results to all member agents with noise + narrative adaptation.
    Phase 3: Update per-agent state.

    Returns a flat list of response dicts (one per agent).
    """
    settings = get_settings()
    max_concurrent = max_concurrent or settings.max_concurrent_llm_calls
    semaphore = asyncio.Semaphore(max_concurrent)
    rng = np.random.default_rng(seed)

    # Build index
    agent_id_to_idx: Dict[str, int] = {}
    for i, a in enumerate(agents):
        p = a.get("persona")
        if p:
            agent_id_to_idx[p.agent_id] = i

    # Build think_fn if not supplied
    if think_fn is None:
        think_fn = _default_think_fn(agents, option_labels=options)

    # ---- Phase 1: Archetype representative decisions ----------------
    rep_tasks = {}
    for cid, astate in archetype_states.items():
        idx = astate.representative_idx
        agent = agents[idx]
        persona = agent["persona"]

        async def _run_rep(
            _persona=persona, _agent=agent, _question=question, _qid=question_id,
        ):
            async with semaphore:
                return await think_fn(
                    persona=_persona,
                    question=_question,
                    question_id=_qid,
                    friends_using=_agent.get("social_trait_fraction", 0.0),
                    location_quality=_agent.get("location_quality", 0.5),
                )

        rep_tasks[cid] = _run_rep

    rep_coros = {cid: fn() for cid, fn in rep_tasks.items()}
    gathered = await asyncio.gather(
        *rep_coros.values(), return_exceptions=True,
    )

    for cid, result in zip(rep_coros.keys(), gathered):
        astate = archetype_states[cid]
        if isinstance(result, Exception):
            astate.last_distribution = {}
            astate.last_sampled = ""
            astate.narrative_templates = []
        else:
            astate.last_distribution = result.get("distribution", {})
            astate.last_sampled = result.get("sampled_option", "")
            astate.narrative_templates = [result.get("answer", "")]

    # Phase 1b: Generate additional narrative variants per archetype,
    # each with a forced tone override for linguistic diversity.
    extra_needed = max(0, narrative_templates_per_archetype - 1)
    if extra_needed > 0:
        variant_tasks = []
        variant_cids = []
        for cid, astate in archetype_states.items():
            if not astate.narrative_templates or not astate.narrative_templates[0]:
                continue
            idx = astate.representative_idx
            agent = agents[idx]
            persona = agent["persona"]
            for vi in range(extra_needed):
                forced_tone = VARIANT_TONES[vi % len(VARIANT_TONES)]

                async def _variant(
                    _persona=persona, _agent=agent,
                    _question=question, _qid=question_id,
                    _tone=forced_tone,
                ):
                    async with semaphore:
                        return await think_fn(
                            persona=_persona,
                            question=_question,
                            question_id=_qid,
                            friends_using=_agent.get("social_trait_fraction", 0.0),
                            location_quality=_agent.get("location_quality", 0.5),
                            tone_override=_tone,
                        )
                variant_tasks.append(_variant())
                variant_cids.append(cid)

        if variant_tasks:
            variant_results = await asyncio.gather(*variant_tasks, return_exceptions=True)
            for cid_v, v_result in zip(variant_cids, variant_results):
                astate = archetype_states[cid_v]
                if isinstance(v_result, Exception):
                    astate.narrative_templates.append(astate.narrative_templates[0])
                else:
                    astate.narrative_templates.append(v_result.get("answer", astate.narrative_templates[0]))

    # ---- Phase 2: Expand to all agents ------------------------------
    expansion_tasks = []
    for cid, astate in archetype_states.items():
        expansion_tasks.append(
            _expand_archetype(
                astate=astate,
                agents=agents,
                agent_id_to_idx=agent_id_to_idx,
                noise_std=noise_std,
                narrative_budget=narrative_budget,
                think_fn=think_fn,
                semaphore=semaphore,
                question=question,
                question_id=question_id,
                rng=rng,
            )
        )

    expansion_results = await asyncio.gather(*expansion_tasks)
    all_responses: List[Dict[str, Any]] = []
    for cluster_responses in expansion_results:
        all_responses.extend(cluster_responses)

    # ---- Phase 3: Update per-agent state ----------------------------
    _update_agent_states(agents, all_responses, question_id)

    return all_responses


# ------------------------------------------------------------------
# Default think_fn factory (mirrors orchestrator.default_think)
# ------------------------------------------------------------------

def _default_think_fn(
    agents: List[Dict[str, Any]],
    option_labels: Optional[List[str]] = None,
) -> Callable[..., Any]:
    """Build a default think function that reuses persistent state."""
    _state_cache: Dict[str, Any] = {}
    _env_cache: Dict[str, Dict[str, Any]] = {}
    used_openings: set = set()

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

    async def _think(
        persona, question, question_id, friends_using=0.0, location_quality=0.5,
        tone_override=None,
    ):
        from agents.cognitive import AgentCognitiveEngine
        from agents.state import AgentState
        from llm.prompts import reasoner_via_llm
        from memory.store import get_memory_store

        state = _state_cache.get(persona.agent_id) or AgentState.from_persona(persona)
        _state_cache[persona.agent_id] = state
        store = get_memory_store()

        def recall(agent_id, perception):
            return store.recall(agent_id, perception.raw_question, top_k=3)

        _tone = tone_override

        async def _reasoner(p, q, sa, dist, mems):
            return await reasoner_via_llm(
                p, q, sa, dist, mems, used_openings=used_openings,
                tone_override=_tone,
                option_labels=option_labels,
            )

        engine = AgentCognitiveEngine(
            persona=persona, state=state,
            memory_recall=recall, reasoner=_reasoner,
        )
        agent_env = _env_cache.get(persona.agent_id)
        if agent_env:
            engine.set_world_environment(agent_env)
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

    return _think
