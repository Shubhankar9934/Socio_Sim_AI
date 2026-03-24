"""
Cross-question consistency: logically related answers should align.
"""

from typing import Any, Dict, List, Tuple

from config.option_space import canonicalize_option


def _to_level(answer: str) -> int:
    a = (answer or "").strip().lower()
    mapping = {
        "never": 0,
        "rarely": 1,
        "1-2 per week": 2,
        "sometimes": 2,
        "3-4 per week": 3,
        "often": 3,
        "daily": 4,
        "very often": 4,
        "multiple per day": 5,
    }
    return mapping.get(a, -1)


def check_frequency_consistency(
    q1_answers: Dict[str, str],
    q2_answers: Dict[str, str],
    agent_ids: List[str],
) -> Tuple[float, List[str]]:
    """
    q1 and q2 are agent_id -> answer for two related questions (e.g. "how often delivery" vs "last week").
    Returns (consistency_rate 0-1, list of inconsistent agent_ids).
    """
    inconsistent = []
    for aid in agent_ids:
        a1 = q1_answers.get(aid, "")
        a2 = q2_answers.get(aid, "")
        if not a1 or not a2:
            continue
        c1 = canonicalize_option("food_delivery_frequency", a1)
        c2 = canonicalize_option("food_delivery_frequency", a2)
        l1 = _to_level(c1)
        l2 = _to_level(c2)
        if l1 >= 0 and l2 >= 0 and abs(l1 - l2) >= 3:
            inconsistent.append(aid)
    checked = sum(1 for aid in agent_ids if q1_answers.get(aid) and q2_answers.get(aid))
    rate = 1.0 - (len(inconsistent) / checked) if checked else 1.0
    return rate, inconsistent


def consistency_score_from_responses(
    response_sets: List[Dict[str, Dict[str, Any]]],
    agent_ids: List[str],
) -> float:
    """
    response_sets: list of {agent_id: {answer: ...}} for each question.
    Returns average pairwise consistency (simplified).
    """
    if len(response_sets) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(response_sets)):
        for j in range(i + 1, len(response_sets)):
            r1 = {aid: r.get("answer", "") for aid, r in response_sets[i].items()}
            r2 = {aid: r.get("answer", "") for aid, r in response_sets[j].items()}
            rate, _ = check_frequency_consistency(r1, r2, agent_ids)
            total += rate
            count += 1
    return total / count if count else 1.0


def consistency_report_from_responses(
    response_sets: List[Dict[str, Dict[str, Any]]],
    agent_ids: List[str],
) -> Dict[str, Any]:
    """Return score + validity metadata."""
    if len(response_sets) < 2:
        return {
            "consistency_score": 1.0,
            "consistency_valid": False,
            "checked_pairs": 0,
            "reason": "insufficient_question_pairs",
        }
    score = consistency_score_from_responses(response_sets, agent_ids)
    return {
        "consistency_score": float(score),
        "consistency_valid": True,
        "checked_pairs": len(response_sets),
    }
