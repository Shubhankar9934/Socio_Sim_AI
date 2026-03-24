"""Survey run and results endpoints.

Single-question surveys use ``POST /survey``.
Multi-question sequential surveys use ``POST /survey/multi`` which runs
rounds with inter-round social influence and belief updates.

Uses ``think_fn=None`` so the orchestrator's ``default_think`` handles:
  - persistent state cache (cross-question structured memory)
  - world environment injection (neighbor latent means from social warmup)
  - narrative style profiles (from persona)
  - memory store add/recall
  - opening deduplication
"""

import logging
import uuid

from llm.client import get_llm_client

logger = logging.getLogger(__name__)
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import (
    AgentDemographics,
    AgentLifestyle,
    MultiSurveyProgress,
    MultiSurveyRequest,
    RoundResultItem,
    SurveyRequest,
    SurveyResponseItem,
    SurveyResult,
    SurveySessionResult,
)
from api.state import agents_store, response_histories, survey_results, survey_sessions
from simulation.orchestrator import run_survey
from agents.intent_router import strip_survey_options_if_qualitative
from agents.perception import perceive, perceive_with_llm, detect_question_model
from config.option_space import canonicalize_options, validate_option_compatibility_hybrid

router = APIRouter(prefix="/survey", tags=["survey"])


@router.post("", response_model=SurveyResult, response_model_exclude_none=True)
async def run_survey_endpoint(body: SurveyRequest) -> SurveyResult:
    """Run a survey question across the population."""
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded. Generate population first.")
    question = body.question
    question_id = body.question_id or str(uuid.uuid4())
    use_archetypes = body.use_archetypes
    routed_options = strip_survey_options_if_qualitative(question, body.options)
    canonical_options = routed_options
    if routed_options:
        p = await perceive_with_llm(
            question, options=routed_options, question_id=question_id,
        )
        qm = detect_question_model(p)
        canonical_options = canonicalize_options(qm.name, routed_options)
        compatible, normalized, warnings = await validate_option_compatibility_hybrid(
            qm.name,
            routed_options,
            qm.scale,
        )
        if not compatible:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "incompatible_option_space",
                    "question_model_key": qm.name,
                    "expected_scale": qm.scale,
                    "provided_options_normalized": normalized,
                    "warnings": warnings,
                },
            )
        canonical_options = normalized

    client = get_llm_client()
    client.reset_survey_stats()

    api_survey_run_id = str(uuid.uuid4())
    # think_fn=None triggers the orchestrator's default_think which
    # reuses persistent AgentState (structured memory, latent state),
    # injects social neighbor means, and applies narrative style profiles.
    responses = await run_survey(
        agents_store,
        question=question,
        question_id=question_id,
        options=canonical_options,
        think_fn=None,
        use_archetypes=use_archetypes,
        current_events=body.current_events,
        diagnostics_enabled=body.diagnostics,
        survey_run_id=api_survey_run_id,
    )
    items = [
        SurveyResponseItem(
            agent_id=r.get("agent_id", ""),
            answer=r.get("answer", r.get("error", "")),
            interaction_mode=r.get("interaction_mode"),
            turn_understanding=r.get("turn_understanding"),
            sampled_option=r.get("sampled_option"),
            sampled_option_canonical=r.get("sampled_option_canonical"),
            distribution=r.get("distribution"),
            question_model_key=r.get("question_model_key"),
            option_space_key=r.get("option_space_key"),
            decision_trace=r.get("decision_trace"),
            narrative_alignment_status=r.get("narrative_alignment_status"),
            run_metadata=r.get("run_metadata"),
            response_diagnostics=r.get("response_diagnostics"),
            fallback_flags=r.get("fallback_flags"),
            demographics=AgentDemographics(**r["demographics"]) if r.get("demographics") else None,
            lifestyle=AgentLifestyle(**r["lifestyle"]) if r.get("lifestyle") else None,
            error=r.get("error"),
        )
        for r in responses
    ]
    survey_id = str(uuid.uuid4())
    survey_results[survey_id] = {
        "survey_id": survey_id,
        "question": question,
        "question_id": question_id,
        "responses": responses,
        "items": items,
    }
    for r in responses:
        aid = r.get("agent_id")
        if aid:
            if aid not in response_histories:
                response_histories[aid] = []
            response_histories[aid].append({
                "question_id": question_id,
                "survey_id": survey_id,
                "answer": r.get("answer", ""),
                "sampled_option": r.get("sampled_option", ""),
                "sampled_option_canonical": r.get("sampled_option_canonical", r.get("sampled_option", "")),
                "question_model_key": r.get("question_model_key", ""),
            })

    n_calls = client.session_call_count
    n_prompt = client.session_prompt_tokens
    n_completion = client.session_completion_tokens
    logger.info(
        "[Survey] question_id=%s | LLM calls=%d | prompt_tokens=%d | completion_tokens=%d | n_agents=%d",
        question_id, n_calls, n_prompt, n_completion, len(items),
    )
    print(
        f"[Survey] question_id={question_id} | LLM calls={n_calls} | "
        f"prompt_tokens={n_prompt} | completion_tokens={n_completion} | n_agents={len(items)}"
    )

    return SurveyResult(
        survey_id=survey_id,
        question=question,
        responses=items,
        n_total=len(items),
    )


@router.get("/{survey_id}/results", response_model=SurveyResult, response_model_exclude_none=True)
def get_survey_results(survey_id: str) -> SurveyResult:
    """Get stored survey results by id."""
    if survey_id not in survey_results:
        raise HTTPException(status_code=404, detail="Survey not found")
    data = survey_results[survey_id]
    return SurveyResult(
        survey_id=data["survey_id"],
        question=data["question"],
        responses=data["items"],
        n_total=len(data["items"]),
    )


# ---------------------------------------------------------------------------
# Multi-question survey endpoints
# ---------------------------------------------------------------------------

def _responses_to_items(responses: List[Dict[str, Any]]) -> List[SurveyResponseItem]:
    return [
        SurveyResponseItem(
            agent_id=r.get("agent_id", ""),
            answer=r.get("answer", r.get("error", "")),
            interaction_mode=r.get("interaction_mode"),
            turn_understanding=r.get("turn_understanding"),
            sampled_option=r.get("sampled_option"),
            sampled_option_canonical=r.get("sampled_option_canonical"),
            distribution=r.get("distribution"),
            question_model_key=r.get("question_model_key"),
            option_space_key=r.get("option_space_key"),
            decision_trace=r.get("decision_trace"),
            narrative_alignment_status=r.get("narrative_alignment_status"),
            run_metadata=r.get("run_metadata"),
            response_diagnostics=r.get("response_diagnostics"),
            fallback_flags=r.get("fallback_flags"),
            demographics=(
                AgentDemographics(**r["demographics"]) if r.get("demographics") else None
            ),
            lifestyle=(
                AgentLifestyle(**r["lifestyle"]) if r.get("lifestyle") else None
            ),
            error=r.get("error"),
        )
        for r in responses
    ]


async def _run_multi_survey_task(session_id: str, body: MultiSurveyRequest) -> None:
    """Background task that drives the SurveyEngine and streams progress via WebSocket."""
    from api.state import social_graph
    from api.websocket import ws_manager
    from simulation.survey_engine import SurveyEngine, SurveyEngineConfig
    from storage.writer import JSONLWriter

    config = SurveyEngineConfig(
        use_archetypes=body.use_archetypes,
        diagnostics_enabled=body.diagnostics,
        social_influence_between_rounds=body.social_influence_between_rounds,
        summarize_every=body.summarize_every,
    )
    engine = SurveyEngine(
        agents=agents_store,
        social_graph=social_graph,
        config=config,
    )
    writer = JSONLWriter()
    ws_channel = f"survey:{session_id}"

    async def _on_progress(
        round_idx: int,
        total_rounds: int,
        sid: str,
        question: str,
        responses: List[Dict[str, Any]],
    ) -> None:
        await ws_manager.broadcast(ws_channel, {
            "event": "round_complete",
            "session_id": sid,
            "round_idx": round_idx,
            "total_rounds": total_rounds,
            "question": question,
            "n_responses": len(responses),
        })
        for r in responses:
            writer.write_response(sid, round_idx, r)

    engine.on_progress(_on_progress)

    questions = []
    for q in body.questions:
        qid = q.question_id or str(uuid.uuid4())
        options = None
        routed = strip_survey_options_if_qualitative(q.question, q.options)
        if routed:
            p = await perceive_with_llm(
                q.question, options=routed, question_id=qid,
            )
            qm = detect_question_model(p)
            compatible, normalized, warnings = await validate_option_compatibility_hybrid(
                qm.name,
                routed,
                qm.scale,
            )
            if not compatible:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "incompatible_option_space",
                        "question": q.question,
                        "question_model_key": qm.name,
                        "expected_scale": qm.scale,
                        "provided_options_normalized": normalized,
                        "warnings": warnings,
                    },
                )
            options = normalized
        questions.append({
            "question": q.question,
            "question_id": qid,
            "options": options,
        })

    session = survey_sessions[session_id]
    try:
        result = await engine.run(questions)
        session["status"] = "completed"
        session["result"] = result

        # Persist per-round results for GET endpoints
        for rr in result.rounds:
            items = _responses_to_items(rr.responses)
            round_survey_id = f"{session_id}_r{rr.round_idx}"
            survey_results[round_survey_id] = {
                "survey_id": round_survey_id,
                "question": rr.question,
                "question_id": rr.question_id,
                "responses": rr.responses,
                "items": items,
            }
            for r in rr.responses:
                aid = r.get("agent_id")
                if aid:
                    response_histories.setdefault(aid, []).append({
                        "question_id": rr.question_id,
                        "survey_id": round_survey_id,
                        "answer": r.get("answer", ""),
                        "sampled_option": r.get("sampled_option", ""),
                        "sampled_option_canonical": r.get("sampled_option_canonical", r.get("sampled_option", "")),
                        "question_model_key": r.get("question_model_key", ""),
                    })

        await ws_manager.broadcast(ws_channel, {
            "event": "session_complete",
            "session_id": session_id,
            "total_responses": result.total_responses,
            "elapsed_seconds": result.elapsed_seconds,
        })
    except Exception as exc:
        session["status"] = "failed"
        session["error"] = str(exc)
        await ws_manager.broadcast(ws_channel, {
            "event": "session_failed",
            "session_id": session_id,
            "error": str(exc),
        })
    finally:
        writer.flush()


@router.post("/multi", response_model=MultiSurveyProgress)
async def run_multi_survey(body: MultiSurveyRequest, background: BackgroundTasks) -> MultiSurveyProgress:
    """Start a multi-question survey session.

    Returns the ``session_id`` immediately.  Progress streams via
    ``/ws/survey/{session_id}``; final results via
    ``GET /survey/session/{session_id}/results``.
    """
    if not agents_store:
        raise HTTPException(status_code=400, detail="No population loaded. Generate population first.")
    if not body.questions:
        raise HTTPException(status_code=400, detail="At least one question is required.")

    session_id = str(uuid.uuid4())
    survey_sessions[session_id] = {
        "session_id": session_id,
        "total_rounds": len(body.questions),
        "current_round": 0,
        "status": "running",
        "completed_questions": [],
        "result": None,
        "error": None,
    }

    background.add_task(_run_multi_survey_task, session_id, body)

    return MultiSurveyProgress(
        session_id=session_id,
        current_round=0,
        total_rounds=len(body.questions),
        status="running",
    )


@router.get("/session/{session_id}/progress", response_model=MultiSurveyProgress)
def get_multi_survey_progress(session_id: str) -> MultiSurveyProgress:
    """Poll progress for a running multi-question survey session."""
    session = survey_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    result = session.get("result")
    completed = []
    current_round = 0
    if result is not None:
        completed = [rr.question_id for rr in result.rounds]
        current_round = len(result.rounds)
    return MultiSurveyProgress(
        session_id=session_id,
        current_round=current_round,
        total_rounds=session["total_rounds"],
        status=session["status"],
        completed_questions=completed,
    )


@router.get("/session/{session_id}/results", response_model=SurveySessionResult, response_model_exclude_none=True)
def get_multi_survey_results(session_id: str) -> SurveySessionResult:
    """Retrieve full results for a completed multi-question survey session."""
    session = survey_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] == "running":
        raise HTTPException(status_code=409, detail="Session still running")
    if session["status"] == "failed":
        raise HTTPException(status_code=500, detail=session.get("error", "Unknown error"))

    result = session["result"]
    rounds = [
        RoundResultItem(
            round_idx=rr.round_idx,
            question=rr.question,
            question_id=rr.question_id,
            responses=_responses_to_items(rr.responses),
            n_total=len(rr.responses),
            elapsed_seconds=rr.elapsed_seconds,
        )
        for rr in result.rounds
    ]
    return SurveySessionResult(
        session_id=result.session_id,
        questions=result.questions,
        rounds=rounds,
        total_responses=result.total_responses,
        elapsed_seconds=result.elapsed_seconds,
        status="completed",
    )


@router.get("/session/{session_id}/round/{round_idx}", response_model=SurveyResult, response_model_exclude_none=True)
def get_multi_survey_round(session_id: str, round_idx: int) -> SurveyResult:
    """Retrieve results for a specific round of a multi-question survey session."""
    round_survey_id = f"{session_id}_r{round_idx}"
    if round_survey_id not in survey_results:
        raise HTTPException(status_code=404, detail="Round not found")
    data = survey_results[round_survey_id]
    return SurveyResult(
        survey_id=round_survey_id,
        question=data["question"],
        responses=data["items"],
        n_total=len(data["items"]),
    )
