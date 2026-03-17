"""
Response aggregation by segment: location, income, nationality, age.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


def aggregate_responses(
    responses: List[Dict[str, Any]],
    segment_by: str = "location",
    answer_key: str = "answer",
    agent_id_key: str = "agent_id",
) -> Dict[str, Dict[str, float]]:
    """
    responses: list of {agent_id, answer, ...} with optional persona/segment fields.
    segment_by: "location" | "income" | "nationality" | "age".
    Returns segment -> {answer_value: proportion}.
    """
    if not responses:
        return {}
    df = pd.DataFrame(responses)
    if segment_by not in df.columns:
        # Try to get from nested persona
        segments = []
        for r in responses:
            p = r.get("persona") or {}
            segments.append(p.get(segment_by, "unknown"))
        df[segment_by] = segments
    df = df.dropna(subset=[segment_by, answer_key])
    if df.empty:
        return {}
    out = {}
    for seg, group in df.groupby(segment_by):
        counts = group[answer_key].value_counts(normalize=True)
        out[str(seg)] = counts.to_dict()
    return out


def aggregate_with_personas(
    responses: List[Dict[str, Any]],
    personas: List[Any],
    segment_by: str = "location",
    answer_key: str = "answer",
) -> Dict[str, Dict[str, float]]:
    """Join responses with personas to get segment; then aggregate."""
    persona_by_id = {p.agent_id: p for p in personas}
    for r in responses:
        aid = r.get("agent_id")
        if aid and aid not in r and persona_by_id.get(aid):
            p = persona_by_id[aid]
            r["persona"] = p
            r["location"] = getattr(p, "location", None)
            r["income"] = getattr(p, "income", None)
            r["nationality"] = getattr(p, "nationality", None)
            r["age"] = getattr(p, "age", None)
    return aggregate_responses(responses, segment_by=segment_by, answer_key=answer_key)


def frequency_distribution_by_segment(
    responses: List[Dict[str, Any]],
    personas: List[Any],
    segment_by: str = "location",
) -> Dict[str, Dict[str, float]]:
    """Convenience: aggregate_responses with answer_key='answer' and personas."""
    return aggregate_with_personas(responses, personas, segment_by=segment_by)
