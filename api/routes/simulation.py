"""Simulation run, status, event injection, and scenario management."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.schemas import (
    EventInjectRequest, ScenarioCompareRequest, ScenarioCompareWithSurveyRequest,
    ScenarioRunRequest, ScenarioWithSurveyRequest, SimulateRequest,
)
from api.state import agents_store, social_graph
from simulation.engine import run_simulation
from simulation.scenario import (
    ScenarioConfig, ScenarioEvent, compare_scenarios, run_scenario,
    run_scenario_with_survey, compare_scenarios_with_survey,
)
from world.events import SimulationEvent

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("")
async def run_simulation_endpoint(body: SimulateRequest) -> Dict[str, Any]:
    """Run N days of simulation (events, social influence, state updates)."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")
    from api import state as app_state
    run_simulation(
        agents_store,
        days=body.days,
        social_graph=app_state.social_graph,
        scheduler=app_state.event_scheduler,
    )
    return {"status": "ok", "days": body.days, "n_agents": len(agents_store)}


@router.post("/events")
async def inject_event(body: EventInjectRequest) -> Dict[str, Any]:
    """Schedule a world event (price_change, policy, infrastructure, market, etc.)."""
    from api import state as app_state
    event = SimulationEvent(
        day=body.day,
        type=body.type,
        payload=body.payload,
        district=body.district,
    )
    app_state.event_scheduler.add(event)
    return {
        "status": "scheduled",
        "event_type": body.type,
        "day": body.day,
        "pending_events": len(app_state.event_scheduler._events),
    }


@router.get("/events")
def list_events() -> Dict[str, Any]:
    """List pending events and current global parameters."""
    from api import state as app_state
    return {
        "pending_events": [
            {"day": e.day, "type": e.type, "district": e.district, "payload": e.payload}
            for e in app_state.event_scheduler._events
        ],
        "global_params": app_state.event_scheduler.global_params,
    }


@router.get("/status")
def simulation_status() -> Dict[str, Any]:
    """Current simulation status."""
    from api import state as app_state
    n = len(agents_store)
    return {
        "population_size": n,
        "social_graph_loaded": app_state.social_graph is not None,
        "pending_events": len(app_state.event_scheduler._events),
    }


def _to_scenario_config(req: ScenarioRunRequest) -> ScenarioConfig:
    return ScenarioConfig(
        name=req.name,
        days=req.days,
        seed=req.seed,
        events=[
            ScenarioEvent(day=e.day, type=e.type, payload=e.payload, district=e.district)
            for e in req.events
        ],
    )


@router.post("/scenario")
async def run_scenario_endpoint(body: ScenarioRunRequest) -> Dict[str, Any]:
    """Run a named scenario (deep-copies agents, does not mutate state)."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")
    from api import state as app_state

    scenario = _to_scenario_config(body)
    result = run_scenario(agents_store, scenario, social_graph=app_state.social_graph)
    return {
        "name": result.name,
        "days": result.days,
        "seed": result.seed,
        "population_size": result.population_size,
        "dimension_means": result.dimension_means,
    }


@router.post("/scenario/compare")
async def compare_scenarios_endpoint(body: ScenarioCompareRequest) -> Dict[str, Any]:
    """Run two scenarios and return the diff of their macro metrics."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")
    from api import state as app_state

    sa = _to_scenario_config(body.scenario_a)
    sb = _to_scenario_config(body.scenario_b)
    return compare_scenarios(agents_store, sa, sb, social_graph=app_state.social_graph)


@router.post("/scenario/run-with-survey")
async def run_scenario_with_survey_endpoint(body: ScenarioWithSurveyRequest) -> Dict[str, Any]:
    """Run scenario simulation, then survey the post-scenario population."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")
    from api import state as app_state
    scenario = _to_scenario_config(body.scenario)
    result = await run_scenario_with_survey(
        agents_store, scenario, body.questions, social_graph=app_state.social_graph
    )
    return {
        "name": result.name,
        "days": result.days,
        "population_size": result.population_size,
        "dimension_means": result.dimension_means,
        "belief_means": result.belief_means,
        "survey_results": result.survey_results,
        "timeline": result.timeline,
    }


@router.post("/scenario/compare-with-survey")
async def compare_scenarios_with_survey_endpoint(
    body: ScenarioCompareWithSurveyRequest,
) -> Dict[str, Any]:
    """Run two scenarios, survey both populations, and compare."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")
    from api import state as app_state
    sa = _to_scenario_config(body.scenario_a)
    sb = _to_scenario_config(body.scenario_b)
    return await compare_scenarios_with_survey(
        agents_store, sa, sb, body.questions, social_graph=app_state.social_graph
    )


# ── Causal endpoints ──────────────────────────────────────────────

from pydantic import BaseModel, Field


class DoInterventionRequest(BaseModel):
    intervention: Dict[str, float]
    observational: Dict[str, float] = Field(default_factory=dict)


class ATERequest(BaseModel):
    treatment: str
    outcome: str
    confounders: List[str] = Field(default_factory=list)
    treatment_value: float = 1.0
    control_value: float = 0.0


@router.get("/causal/graph")
async def get_causal_graph() -> Dict[str, Any]:
    """Return the default causal graph structure."""
    from causal.graph import build_default_causal_graph
    g = build_default_causal_graph()
    return g.to_dict()


@router.post("/causal/do-intervention")
async def do_intervention(req: DoInterventionRequest) -> Dict[str, Any]:
    """Run a counterfactual do-intervention query."""
    from causal.graph import build_default_causal_graph
    g = build_default_causal_graph()
    result = g.do(req.intervention, req.observational or None)
    return {"counterfactual_values": result}


@router.post("/causal/ate")
async def estimate_ate(req: ATERequest) -> Dict[str, Any]:
    """Estimate Average Treatment Effect."""
    from causal.graph import build_default_causal_graph
    g = build_default_causal_graph()
    ate = g.estimate_ate(
        treatment=req.treatment, outcome=req.outcome,
        confounders=req.confounders,
        treatment_value=req.treatment_value,
        control_value=req.control_value,
    )
    return {"treatment": req.treatment, "outcome": req.outcome, "ate": ate}


@router.post("/causal/learn")
async def learn_causal_graph(body: Dict[str, Any]) -> Dict[str, Any]:
    """Learn causal structure from simulation timeline data."""
    from causal.learner import CausalLearner
    learner = CausalLearner()
    timeline = body.get("timeline", [])
    g = learner.learn_from_timeline(timeline)
    return g.to_dict()
