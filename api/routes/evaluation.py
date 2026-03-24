"""Evaluation run and report."""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.schemas import DashboardMetrics, EvaluateRequest, EvaluationReportResponse
from api.state import agents_store, response_histories, survey_results
from evaluation.report import export_evaluation_report, run_evaluation

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("/{survey_id}", response_model=EvaluationReportResponse)
async def evaluate_survey(survey_id: str, body: EvaluateRequest) -> EvaluationReportResponse:
    """Run evaluation framework on a survey's responses."""
    if survey_id not in survey_results:
        raise HTTPException(status_code=404, detail="Survey not found")
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded.")
    data = survey_results[survey_id]
    responses = data.get("responses", [])
    personas = [a["persona"] for a in agents_store if a.get("persona")]
    report = await run_evaluation(
        personas=personas,
        survey_responses=responses,
        response_histories=response_histories if response_histories else None,
        realism_threshold=body.realism_threshold,
        drift_threshold=body.drift_threshold,
        run_judge=body.run_judge,
        judge_sample=body.judge_sample,
        run_similarity=body.run_similarity,
        similarity_threshold=body.similarity_threshold,
        reference_distribution=body.reference_distribution,
        question_model_key=body.question_model_key,
    )

    first_persona = personas[0] if personas else None
    system_info = {
        "population_size": len(personas),
        "question": data.get("question", ""),
        "question_id": data.get("question_id", ""),
        "n_responses": len(responses),
    }
    if first_persona:
        system_info["synthesis_method"] = first_persona.meta.synthesis_method
        system_info["generation_seed"] = first_persona.meta.generation_seed
    export_evaluation_report(
        report,
        output_path=f"evaluation_report_{survey_id}.json",
        system_info=system_info,
    )

    dashboard_data = report.get("dashboard", {})
    dashboard = DashboardMetrics(**dashboard_data) if dashboard_data else None

    return EvaluationReportResponse(
        population_realism=report["population_realism"],
        drift=report["drift"],
        consistency_score=report.get("consistency_score", 1.0),
        consistency_valid=report.get("consistency_valid", False),
        distribution_validation=report.get("distribution_validation"),
        narrative_similarity=report.get("narrative_similarity"),
        llm_judge=report.get("llm_judge"),
        dashboard=dashboard,
        quantitative_metrics=report.get("quantitative_metrics"),
        summary=report["summary"],
    )


@router.get("/{evaluation_id}/report")
def get_evaluation_report(evaluation_id: str) -> Dict[str, Any]:
    """Load exported evaluation report JSON for a survey id."""
    report_path = Path(f"evaluation_report_{evaluation_id}.json")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found")
    return json.loads(report_path.read_text(encoding="utf-8"))
