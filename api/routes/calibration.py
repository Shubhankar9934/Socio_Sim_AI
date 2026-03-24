"""API routes for calibration: auto-weights, real data upload, fit pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from api.state import agents_store
from agents.perception import detect_question_model, perceive

router = APIRouter(prefix="/calibration", tags=["calibration"])


def _interpret_weight(weight: float) -> str:
    w = abs(float(weight))
    if w >= 0.35:
        return "strong influence on decision"
    if w >= 0.20:
        return "moderate influence on decision"
    if w >= 0.08:
        return "light influence on decision"
    return "minimal influence on decision"


class AutoWeightsRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    reference_distributions: Dict[str, Dict[str, float]]
    n_iterations: int = Field(default=50, ge=5, le=500)
    seed: Optional[int] = 42


@router.post("/auto-weights")
async def auto_weights(req: AutoWeightsRequest) -> Dict[str, Any]:
    """Run factor weight optimization against reference distributions."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")

    from calibration.auto_weights import FactorWeightLearner

    learner = FactorWeightLearner(n_iterations=req.n_iterations, seed=req.seed)
    result = learner.learn_weights(
        questions=req.questions,
        reference_distributions=req.reference_distributions,
        agents=agents_store,
    )
    from config.calibrated_weights import set_calibrated_weights

    for r in result.results:
        model_key = detect_question_model(perceive(r.question)).name
        set_calibrated_weights(model_key, r.learned_weights)
    return {
        "overall_loss": result.overall_loss,
        "calibration_provenance": {
            "n_iterations": req.n_iterations,
            "seed": req.seed,
            "persisted_runtime_overrides": True,
        },
        "results": [
            {
                "question": r.question,
                "learned_weights": r.learned_weights,
                "interpretation": {k: _interpret_weight(v) for k, v in r.learned_weights.items()},
                "best_loss": r.best_loss,
                "converged": r.converged,
            }
            for r in result.results
        ],
    }


class FitRequest(BaseModel):
    question: str
    reference_distribution: Dict[str, float]
    demographics_cols: Optional[List[str]] = None
    n_iterations: int = Field(default=50, ge=5)


@router.post("/fit")
async def fit_calibration(req: FitRequest) -> Dict[str, Any]:
    """Run calibration pipeline for a single question."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")

    from calibration.auto_weights import FactorWeightLearner

    learner = FactorWeightLearner(n_iterations=req.n_iterations)
    wr = learner.learn_weights_for_question(
        question=req.question,
        reference_distribution=req.reference_distribution,
        agents=agents_store,
    )
    from config.calibrated_weights import set_calibrated_weights
    model_key = detect_question_model(perceive(wr.question)).name
    set_calibrated_weights(model_key, wr.learned_weights)
    return {
        "question": wr.question,
        "learned_weights": wr.learned_weights,
        "interpretation": {k: _interpret_weight(v) for k, v in wr.learned_weights.items()},
        "best_loss": wr.best_loss,
        "converged": wr.converged,
        "n_iterations": wr.n_iterations,
        "calibration_provenance": {
            "seed": learner.seed,
            "persisted_runtime_overrides": True,
        },
    }


class UploadDataRequest(BaseModel):
    question: str
    responses: List[str]
    demographics: Optional[List[Dict[str, str]]] = None


@router.post("/upload-data")
async def upload_real_data(req: UploadDataRequest) -> Dict[str, Any]:
    """Upload real survey data and compute reference distribution."""
    from calibration.data_loader import RealSurveyData

    data = RealSurveyData.from_raw(req.question, req.responses, req.demographics)
    return {
        "question": data.question,
        "n_responses": data.n_responses,
        "reference_distribution": data.to_reference_distribution(),
    }
