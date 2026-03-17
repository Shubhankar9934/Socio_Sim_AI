"""Population generation endpoint.

After synthesis, the route:
  - Creates AgentState for each persona (segment-primed latent state)
  - Builds a Barabasi-Albert social graph
  - Computes social_trait_fraction from the social graph so social
    influence is non-zero from the first survey
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.schemas import GeneratePopulationRequest
from api import state as app_state
from config.settings import get_settings
from population.synthesis import generate_population
from population.validator import validate_population
from social.network import build_social_network

router = APIRouter(prefix="/population", tags=["population"])


@router.post("/generate")
def generate_population_endpoint(body: GeneratePopulationRequest) -> Dict[str, Any]:
    """Generate synthetic population (Monte Carlo, Bayesian, or IPF)."""
    settings = get_settings()
    if body.n > 10000:
        raise HTTPException(status_code=400, detail="n must be <= 10000")
    personas = generate_population(
        n=body.n,
        method=body.method,
        seed=body.seed,
        id_prefix=body.id_prefix,
    )
    passed, score, per_attr = validate_population(
        personas,
        realism_threshold=settings.population_realism_threshold,
    )

    from agents.state import AgentState
    from social.influence import fraction_friends_with_trait
    from world.districts import location_quality_for_satisfaction

    agents = []
    for p in personas:
        state = AgentState.from_persona(p)
        agents.append({
            "persona": p,
            "state": state,
            "social_trait_fraction": 0.0,
            "location_quality": location_quality_for_satisfaction(p.location),
        })

    graph = build_social_network(personas, seed=body.seed)
    app_state.social_graph = graph

    trait_by_agent: Dict[str, bool] = {}
    for a in agents:
        p = a["persona"]
        if p.lifestyle.primary_service_preference >= 0.5:
            trait_by_agent[p.agent_id] = True
        else:
            trait_by_agent[p.agent_id] = False

    for a in agents:
        p = a["persona"]
        frac = fraction_friends_with_trait(graph, p.agent_id, trait_by_agent)
        a["social_trait_fraction"] = frac
        a["state"].set_social_trait_fraction(frac)

    app_state.agents_store.clear()
    app_state.agents_store.extend(agents)

    from collections import Counter
    seg_dist = dict(Counter(
        p.meta.population_segment for p in personas
    ))

    return {
        "n": len(personas),
        "method": body.method,
        "realism_passed": passed,
        "realism_score": score,
        "per_attribute": per_attr,
        "segment_distribution": seg_dist,
    }
