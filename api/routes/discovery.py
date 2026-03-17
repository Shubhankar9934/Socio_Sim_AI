"""API routes for dimension discovery and domain auto-setup."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/discovery", tags=["discovery"])


class DimensionDiscoveryRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    n_behavioral: int = Field(default=12, ge=1, le=50)
    n_belief: int = Field(default=7, ge=1, le=30)
    domain_id: Optional[str] = None
    save: bool = Field(default=False, description="Persist results to domain config")


class DimensionDiscoveryResponse(BaseModel):
    behavioral: List[dict]
    belief: List[dict]
    question_to_dimension: dict
    saved: bool = False


class DomainAutoSetupRequest(BaseModel):
    domain_name: str
    description: str = ""
    sample_questions: List[str] = Field(default_factory=list)
    city_name: str = ""
    currency: str = "USD"
    reference_data: Optional[dict] = None


class DomainAutoSetupResponse(BaseModel):
    domain_id: str
    message: str


@router.post("/domains/auto-setup", response_model=DomainAutoSetupResponse)
async def auto_setup_domain(req: DomainAutoSetupRequest):
    from discovery.domain_setup import DomainAutoSetup
    setup = DomainAutoSetup()
    domain_id = await setup.setup_domain(
        domain_name=req.domain_name,
        description=req.description,
        sample_questions=req.sample_questions,
        city_name=req.city_name,
        currency=req.currency,
        reference_data=req.reference_data,
    )
    return DomainAutoSetupResponse(
        domain_id=domain_id,
        message=f"Domain '{domain_id}' created at data/domains/{domain_id}/",
    )


@router.post("/dimensions", response_model=DimensionDiscoveryResponse)
async def discover_dimensions(req: DimensionDiscoveryRequest):
    from discovery.dimensions import (
        DimensionDiscovery,
        save_discovered_dimensions,
    )

    discoverer = DimensionDiscovery()
    result = await discoverer.discover_dimensions(
        questions=req.questions,
        n_behavioral=req.n_behavioral,
        n_belief=req.n_belief,
    )

    saved = False
    if req.save and req.domain_id:
        save_discovered_dimensions(req.domain_id, result)
        saved = True

    return DimensionDiscoveryResponse(
        behavioral=[
            {"name": d.name, "description": d.description,
             "representative_questions": d.representative_questions}
            for d in result.behavioral
        ],
        belief=[
            {"name": d.name, "description": d.description,
             "representative_questions": d.representative_questions}
            for d in result.belief
        ],
        question_to_dimension=result.question_to_dimension,
        saved=saved,
    )
