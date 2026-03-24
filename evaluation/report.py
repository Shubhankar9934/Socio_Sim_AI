"""
Full evaluation report: realism, consistency, drift, similarity, distribution, judge scores.
Produces a dashboard metrics dict for quick quality assessment.
Supports file export for reproducible benchmarking.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from evaluation.consistency import consistency_report_from_responses
from evaluation.distribution_validation import validate_survey_distribution
from evaluation.drift import drift_report
from evaluation.realism import compute_realism_report
from population.personas import Persona

QUALITY_TARGETS = {
    "duplicate_narrative_rate": {"target": "<0.05", "threshold": 0.05, "direction": "below"},
    "persona_realism_score": {"target": ">0.90", "threshold": 0.90, "direction": "above"},
    "distribution_similarity": {"target": ">0.85", "threshold": 0.85, "direction": "above"},
    "consistency_score": {"target": ">0.90", "threshold": 0.90, "direction": "above"},
    "drift_rate": {"target": "<0.10", "threshold": 0.10, "direction": "below"},
    "mean_judge_score": {"target": ">3.50", "threshold": 3.50, "direction": "above"},
}


async def run_evaluation(
    personas: List[Persona],
    survey_responses: List[Dict[str, Any]],
    response_histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    realism_threshold: float = 0.85,
    drift_threshold: float = 0.3,
    run_judge: bool = False,
    judge_sample: Optional[int] = 20,
    run_similarity: bool = True,
    similarity_threshold: float = 0.9,
    reference_distribution: Optional[Dict[str, float]] = None,
    question_model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    survey_responses: list of {agent_id, answer, ...} for the survey.
    response_histories: optional agent_id -> list of past responses for drift.
    If run_judge=True, calls LLM judge on a sample (costly).
    If run_similarity=True, computes narrative duplicate rate.
    """
    from evaluation.judge import judge_responses_batch

    report: Dict[str, Any] = {}

    # --- Population realism ---
    realism = compute_realism_report(personas, threshold=realism_threshold)
    report["population_realism"] = realism

    # --- Drift ---
    response_histories = response_histories or {p.agent_id: [] for p in personas}
    for r in survey_responses:
        aid = r.get("agent_id")
        if aid and aid in response_histories:
            response_histories[aid].append(r)
    drift = drift_report(personas, response_histories, threshold=drift_threshold)
    report["drift"] = drift

    # --- Cross-question consistency ---
    agent_ids = [p.agent_id for p in personas]
    unique_questions = set()
    for aid, hist in response_histories.items():
        for entry in hist:
            qid = entry.get("question_id", "")
            if qid:
                unique_questions.add(qid)

    if len(unique_questions) >= 2:
        response_sets: List[Dict[str, Dict[str, Any]]] = []
        for qid in unique_questions:
            qid_responses: Dict[str, Dict[str, Any]] = {}
            for aid, hist in response_histories.items():
                for entry in hist:
                    if entry.get("question_id") == qid:
                        qid_responses[aid] = {"answer": entry.get("sampled_option", entry.get("answer", ""))}
                        break
            if qid_responses:
                response_sets.append(qid_responses)
        consistency_report = consistency_report_from_responses(response_sets, agent_ids)
        report["consistency_score"] = consistency_report["consistency_score"]
        report["consistency_valid"] = consistency_report["consistency_valid"]
    else:
        report["consistency_score"] = 1.0
        report["consistency_valid"] = False

    # --- Distribution validation ---
    dist_validation = validate_survey_distribution(
        survey_responses,
        reference=reference_distribution,
        question_model_key=question_model_key or (
            survey_responses[0].get("question_model_key") if survey_responses else None
        ),
    )
    report["distribution_validation"] = dist_validation

    # --- Narrative similarity ---
    similarity_result: Dict[str, Any] = {}
    if run_similarity and survey_responses:
        narratives = [
            r.get("answer", "") for r in survey_responses
            if r.get("answer") and r.get("answer") != r.get("sampled_option")
        ]
        if len(narratives) >= 2:
            from evaluation.similarity import compute_narrative_similarity
            similarity_result = compute_narrative_similarity(
                narratives, threshold=similarity_threshold,
            )
    report["narrative_similarity"] = similarity_result

    # --- LLM judge (optional) ---
    mean_judge_score = 0.0
    if run_judge and survey_responses:
        persona_by_id = {p.agent_id: p for p in personas}
        judged = [
            (persona_by_id[r["agent_id"]], "Survey question", r.get("answer", ""))
            for r in survey_responses if r.get("agent_id") in persona_by_id
        ]
        if judged:
            personas_j, questions_j, responses_j = zip(*judged)
            judge_result = await judge_responses_batch(
                list(personas_j), list(questions_j), list(responses_j),
                sample_size=judge_sample,
            )
            report["llm_judge"] = judge_result
            scores = judge_result if isinstance(judge_result, list) else []
            if scores:
                all_scores = []
                for s in scores:
                    if isinstance(s, dict):
                        for v in s.values():
                            if isinstance(v, (int, float)):
                                all_scores.append(v)
                if all_scores:
                    mean_judge_score = sum(all_scores) / len(all_scores)

    # --- Dashboard (quick quality metrics) ---
    dashboard_values = {
        "duplicate_narrative_rate": similarity_result.get("duplicate_rate", 0.0),
        "persona_realism_score": realism.get("population_realism_score", 0.0),
        "distribution_similarity": dist_validation.get("js_similarity", 0.0),
        "consistency_score": report["consistency_score"],
        "drift_rate": drift.get("rate", 0.0),
        "mean_judge_score": round(mean_judge_score, 2),
    }
    report["dashboard"] = dashboard_values

    # --- Quantitative validation with pass/fail ---
    quantitative_metrics: Dict[str, Any] = {}
    for metric_name, value in dashboard_values.items():
        spec = QUALITY_TARGETS.get(metric_name)
        if spec:
            if spec["direction"] == "below":
                passed = value < spec["threshold"]
            else:
                passed = value > spec["threshold"]
            quantitative_metrics[metric_name] = {
                "value": round(value, 4),
                "target": spec["target"],
                "passed": passed,
            }
    report["quantitative_metrics"] = quantitative_metrics

    report["summary"] = {
        "realism_passed": realism.get("passed", False),
        "realism_score": realism.get("population_realism_score", 0),
        "drift_count": drift.get("count", 0),
        "drift_rate": drift.get("rate", 0),
        "distribution_passed": dist_validation.get("passed", False),
        "duplicate_narrative_rate": similarity_result.get("duplicate_rate", 0.0),
        "all_targets_passed": all(
            m.get("passed", False) for m in quantitative_metrics.values()
        ),
    }
    return report


def export_evaluation_report(
    report: Dict[str, Any],
    output_path: str = "evaluation_report.json",
    system_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a standalone evaluation report JSON file for reproducible benchmarking.

    system_info can include population_size, model_version, question_model,
    synthesis_method, generation_seed, etc.
    """
    from population.personas import PERSONA_SCHEMA_VERSION

    export = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "model_version": f"sociosim-{PERSONA_SCHEMA_VERSION}",
            **(system_info or {}),
        },
        "quantitative_metrics": report.get("quantitative_metrics", {}),
        "dashboard": report.get("dashboard", {}),
        "distribution_validation": report.get("distribution_validation", {}),
        "narrative_similarity": {
            k: v for k, v in report.get("narrative_similarity", {}).items()
            if k != "flagged_pairs"
        },
        "population_realism": report.get("population_realism", {}),
        "summary": report.get("summary", {}),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, default=str)
    return os.path.abspath(output_path)
