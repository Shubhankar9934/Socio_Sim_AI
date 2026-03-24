"""Agent listing and detail endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from api.schemas import AgentDetail, AgentSummary
from api.state import agents_store
from population.personas import Persona

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=List[AgentSummary])
def list_agents(
    location: Optional[str] = Query(None),
    nationality: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> List[AgentSummary]:
    """List agents with optional filters."""
    out = []
    for a in agents_store[offset : offset + limit]:
        p = a.get("persona")
        if not p:
            continue
        if location and getattr(p, "location", None) != location:
            continue
        if nationality and getattr(p, "nationality", None) != nationality:
            continue
        out.append(AgentSummary(
            agent_id=p.agent_id,
            age=p.age,
            nationality=p.nationality,
            income=p.income,
            location=p.location,
            occupation=p.occupation,
        ))
    return out


@router.get("/{agent_id}", response_model=AgentDetail)
def get_agent(agent_id: str, debug: bool = Query(False)) -> AgentDetail:
    """Get one agent by id with persona and state."""
    for a in agents_store:
        p = a.get("persona")
        if p and p.agent_id == agent_id:
            state = a.get("state")
            decision_profile = None
            if debug and state and hasattr(state, "latent_state") and hasattr(state, "beliefs"):
                latent = state.latent_state.to_dict()
                beliefs = state.beliefs.to_dict()
                dominant_traits = sorted(latent.items(), key=lambda kv: kv[1], reverse=True)[:3]
                decision_profile = {
                    "latent": latent,
                    "beliefs_summary": beliefs,
                    "dominant_traits": [k for k, _ in dominant_traits],
                }
            return AgentDetail(
                agent_id=agent_id,
                persona=p.model_dump(),
                state=state.to_dict() if state and hasattr(state, "to_dict") else None,
                decision_profile=decision_profile,
            )
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Agent not found")
