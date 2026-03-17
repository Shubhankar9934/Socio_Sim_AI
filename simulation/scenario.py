"""
Scenario system: define, load, run, and compare simulation scenarios.

A scenario is a JSON-serializable configuration that specifies simulation
parameters (days, seed) and a sequence of world events to inject.  Two
scenarios can be compared by running both and diffing their macro metrics.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from simulation.config import SimulationConfig
from world.events import EventScheduler, SimulationEvent


class ScenarioEvent(BaseModel):
    day: int = Field(ge=0)
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    district: Optional[str] = None


class ScenarioConfig(BaseModel):
    """Full scenario definition, JSON-loadable."""

    name: str = "unnamed"
    days: int = Field(default=30, ge=1)
    seed: Optional[int] = None
    events: List[ScenarioEvent] = Field(default_factory=list)


@dataclass
class ScenarioResult:
    """Output of a single scenario run."""

    name: str
    days: int
    seed: Optional[int]
    macro_metrics: Dict[str, float] = field(default_factory=dict)
    dimension_means: Dict[str, float] = field(default_factory=dict)
    belief_means: Dict[str, float] = field(default_factory=dict)
    population_size: int = 0
    survey_results: Optional[Dict[str, Any]] = None
    timeline: List[Dict[str, Any]] = field(default_factory=list)


def load_scenario(path: str) -> ScenarioConfig:
    """Load a scenario from a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    return ScenarioConfig.model_validate_json(text)


def _build_scheduler(scenario: ScenarioConfig) -> EventScheduler:
    scheduler = EventScheduler()
    for ev in scenario.events:
        scheduler.add(SimulationEvent(
            day=ev.day,
            type=ev.type,
            payload=ev.payload,
            district=ev.district,
        ))
    return scheduler


def _collect_belief_means(agents: List[Dict[str, Any]]) -> Dict[str, float]:
    from agents.belief_network import BELIEF_DIMENSIONS
    import numpy as np
    vecs = []
    for a in agents:
        state = a.get("state")
        if state and hasattr(state, "beliefs"):
            vecs.append(state.beliefs.to_vector())
    if not vecs:
        return {}
    arr = np.array(vecs)
    means = arr.mean(axis=0)
    return {BELIEF_DIMENSIONS[i]: round(float(means[i]), 4) for i in range(len(BELIEF_DIMENSIONS))}


def _snapshot(agents: List[Dict[str, Any]], day: int) -> Dict[str, Any]:
    from agents.vectorized import build_trait_matrix, vectorized_macro_aggregation
    mat = build_trait_matrix(agents)
    dim_means = vectorized_macro_aggregation(mat)
    belief_means = _collect_belief_means(agents)
    return {"day": day, "dimension_means": dim_means, "belief_means": belief_means}


def run_scenario(
    agents: List[Dict[str, Any]],
    scenario: ScenarioConfig,
    social_graph: Optional[Any] = None,
    enable_social: bool = True,
    enable_macro: bool = True,
    collect_timeline: bool = False,
) -> ScenarioResult:
    """Run a full scenario and return aggregate results.

    Agents are deep-copied so the original list is not mutated.
    ``enable_social`` / ``enable_macro`` can be disabled for partial
    runs used in causal attribution.
    """
    from simulation.engine import run_daily_step
    from agents.vectorized import build_trait_matrix, vectorized_macro_aggregation

    agents_copy = copy.deepcopy(agents)
    scheduler = _build_scheduler(scenario)
    config = SimulationConfig(master_seed=scenario.seed, days=scenario.days)

    timeline: List[Dict[str, Any]] = []

    if collect_timeline:
        for day in range(1, scenario.days + 1):
            run_daily_step(
                agents_copy, day=day, social_graph=social_graph,
                scheduler=scheduler, config=config,
                enable_social=enable_social, enable_macro=enable_macro,
            )
            if day % max(1, scenario.days // 10) == 0 or day == scenario.days:
                timeline.append(_snapshot(agents_copy, day))
    else:
        from simulation.engine import run_simulation
        run_simulation(
            agents_copy, days=scenario.days, social_graph=social_graph,
            scheduler=scheduler, config=config,
            enable_social=enable_social, enable_macro=enable_macro,
        )

    mat = build_trait_matrix(agents_copy)
    dim_means = vectorized_macro_aggregation(mat)
    belief_means = _collect_belief_means(agents_copy)

    return ScenarioResult(
        name=scenario.name,
        days=scenario.days,
        seed=scenario.seed,
        dimension_means=dim_means,
        belief_means=belief_means,
        population_size=len(agents_copy),
        timeline=timeline,
    )


async def run_scenario_with_survey(
    agents: List[Dict[str, Any]],
    scenario: ScenarioConfig,
    questions: List[str],
    social_graph: Optional[Any] = None,
) -> ScenarioResult:
    """Run scenario simulation, then survey the post-scenario population."""
    from simulation.orchestrator import run_survey

    result = run_scenario(agents, scenario, social_graph, collect_timeline=True)

    agents_copy = copy.deepcopy(agents)
    scheduler = _build_scheduler(scenario)
    config = SimulationConfig(master_seed=scenario.seed, days=scenario.days)
    from simulation.engine import run_simulation
    run_simulation(
        agents_copy, days=scenario.days, social_graph=social_graph,
        scheduler=scheduler, config=config,
    )

    survey_data: Dict[str, Any] = {}
    for q in questions:
        import uuid
        qid = str(uuid.uuid4())
        responses = await run_survey(agents_copy, q, question_id=qid)
        survey_data[q] = {
            "question_id": qid,
            "responses": responses,
            "n_responses": len(responses),
        }
    result.survey_results = survey_data
    return result


async def compare_scenarios_with_survey(
    agents: List[Dict[str, Any]],
    scenario_a: ScenarioConfig,
    scenario_b: ScenarioConfig,
    questions: List[str],
    social_graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run two scenarios, survey both populations, and compare."""
    import numpy as np
    from scipy.stats import chisquare

    result_a = await run_scenario_with_survey(agents, scenario_a, questions, social_graph)
    result_b = await run_scenario_with_survey(agents, scenario_b, questions, social_graph)

    dim_diff: Dict[str, float] = {}
    all_dims = set(result_a.dimension_means) | set(result_b.dimension_means)
    for dim in sorted(all_dims):
        va = result_a.dimension_means.get(dim, 0.5)
        vb = result_b.dimension_means.get(dim, 0.5)
        dim_diff[dim] = round(vb - va, 6)

    survey_comparison: Dict[str, Any] = {}
    for q in questions:
        sa = result_a.survey_results.get(q, {}).get("responses", []) if result_a.survey_results else []
        sb = result_b.survey_results.get(q, {}).get("responses", []) if result_b.survey_results else []

        def _dist(responses):
            counts: Dict[str, int] = {}
            for r in responses:
                opt = r.get("sampled_option") or r.get("answer", "")
                counts[opt] = counts.get(opt, 0) + 1
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()} if total else {}

        dist_a, dist_b = _dist(sa), _dist(sb)
        all_keys = sorted(set(list(dist_a.keys()) + list(dist_b.keys())))
        arr_a = np.array([dist_a.get(k, 0.0) for k in all_keys])
        arr_b = np.array([dist_b.get(k, 0.0) for k in all_keys])

        p_value = None
        if arr_a.sum() > 0 and arr_b.sum() > 0:
            try:
                _, p_value = chisquare(arr_b * 100, f_exp=arr_a * 100 + 1e-9)
                p_value = float(p_value)
            except Exception:
                pass

        survey_comparison[q] = {
            "distribution_a": dist_a,
            "distribution_b": dist_b,
            "p_value": p_value,
        }

    return {
        "scenario_a": {"name": result_a.name, "dimension_means": result_a.dimension_means,
                        "belief_means": result_a.belief_means},
        "scenario_b": {"name": result_b.name, "dimension_means": result_b.dimension_means,
                        "belief_means": result_b.belief_means},
        "dimension_diff_b_minus_a": dim_diff,
        "survey_comparison": survey_comparison,
        "timeline_a": result_a.timeline,
        "timeline_b": result_b.timeline,
    }


def run_scenario_with_attribution(
    agents: List[Dict[str, Any]],
    scenario: ScenarioConfig,
    social_graph: Optional[Any] = None,
) -> Dict[str, Dict[str, float]]:
    """Decompose a scenario's effects into event / social / macro contributions.

    Runs three partial simulations:
      1. Events only (no social diffusion, no macro feedback)
      2. Events + social (no macro feedback)
      3. Full scenario
    The deltas between successive runs isolate each mechanism's contribution.
    """
    from agents.vectorized import build_trait_matrix, vectorized_macro_aggregation

    baseline_mat = build_trait_matrix(agents)
    baseline_means = vectorized_macro_aggregation(baseline_mat)

    result_events_only = run_scenario(
        agents, scenario, social_graph,
        enable_social=False, enable_macro=False,
    )
    result_events_social = run_scenario(
        agents, scenario, social_graph,
        enable_social=True, enable_macro=False,
    )
    result_full = run_scenario(
        agents, scenario, social_graph,
        enable_social=True, enable_macro=True,
    )

    all_dims = sorted(
        set(baseline_means) | set(result_events_only.dimension_means)
        | set(result_events_social.dimension_means) | set(result_full.dimension_means)
    )

    events_attr: Dict[str, float] = {}
    social_attr: Dict[str, float] = {}
    macro_attr: Dict[str, float] = {}
    for dim in all_dims:
        base = baseline_means.get(dim, 0.5)
        eo = result_events_only.dimension_means.get(dim, 0.5)
        es = result_events_social.dimension_means.get(dim, 0.5)
        full = result_full.dimension_means.get(dim, 0.5)
        events_attr[dim] = round(eo - base, 6)
        social_attr[dim] = round(es - eo, 6)
        macro_attr[dim] = round(full - es, 6)

    return {
        "events": events_attr,
        "social": social_attr,
        "macro": macro_attr,
    }


def compare_scenarios(
    agents: List[Dict[str, Any]],
    scenario_a: ScenarioConfig,
    scenario_b: ScenarioConfig,
    social_graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run two scenarios side-by-side and return a comparison dict.

    Includes causal attribution for scenario B showing how much of the
    shift is due to events, social diffusion, and macro feedback.
    """
    result_a = run_scenario(agents, scenario_a, social_graph)
    result_b = run_scenario(agents, scenario_b, social_graph)

    diff: Dict[str, float] = {}
    all_dims = set(result_a.dimension_means) | set(result_b.dimension_means)
    for dim in sorted(all_dims):
        va = result_a.dimension_means.get(dim, 0.5)
        vb = result_b.dimension_means.get(dim, 0.5)
        diff[dim] = round(vb - va, 6)

    attribution_b = run_scenario_with_attribution(agents, scenario_b, social_graph)

    causal_effects = {}
    try:
        from causal.graph import build_default_causal_graph
        cg = build_default_causal_graph()
        intervention_vars = {
            dim: val for dim, val in diff.items() if abs(val) > 0.01
        }
        if intervention_vars:
            causal_effects = cg.do(intervention_vars, result_a.dimension_means)
    except Exception:
        pass

    return {
        "scenario_a": {
            "name": result_a.name,
            "dimension_means": result_a.dimension_means,
            "belief_means": result_a.belief_means,
        },
        "scenario_b": {
            "name": result_b.name,
            "dimension_means": result_b.dimension_means,
            "belief_means": result_b.belief_means,
        },
        "dimension_diff_b_minus_a": diff,
        "causal_attribution_b": attribution_b,
        "causal_counterfactual_values": causal_effects,
    }
