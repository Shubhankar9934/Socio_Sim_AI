"""Analytics by survey and segment."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from analytics.aggregator import aggregate_with_personas, verbatim_examples_by_segment
from api.state import agents_store, survey_results
from analytics.insights import delivery_frequency_insight, generate_insights

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{survey_id}")
def get_analytics(
    survey_id: str,
    segment_by: str = Query("location", description="location | income | nationality | age"),
    answer_key: str = Query("sampled_option_canonical", description="sampled_option_canonical | sampled_option | answer"),
) -> Dict[str, Any]:
    """Segmented analytics for a survey."""
    if survey_id not in survey_results:
        raise HTTPException(status_code=404, detail="Survey not found")
    data = survey_results[survey_id]
    responses = data.get("responses", [])
    personas = [a.get("persona") for a in agents_store if a.get("persona")]
    if not personas:
        aggregated = {}
        insights = []
    else:
        aggregated = aggregate_with_personas(
            responses,
            personas,
            segment_by=segment_by,
            answer_key=answer_key,
        )
        insights = generate_insights(aggregated, segment_name=segment_by)
        try:
            insights.append(delivery_frequency_insight(aggregated))
        except Exception:
            pass
        verbatim_examples = verbatim_examples_by_segment(responses, personas, segment_by=segment_by)
    return {
        "survey_id": survey_id,
        "segment_by": segment_by,
        "answer_key": answer_key,
        "aggregated": aggregated,
        "verbatim_examples": verbatim_examples if personas else {},
        "insights": insights,
    }

