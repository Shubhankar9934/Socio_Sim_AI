"""
Cross-question consistency: logically related answers should align.
"""

from typing import Any, Dict, List, Tuple


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
        # Simple heuristic: if q1 says "rarely" and q2 says "5 times", inconsistent
        a1_low = a1.lower() in ("rarely", "never", "0")
        a2_high = any(x in str(a2) for x in ["3", "4", "5", "6", "7", "week", "times"])
        if a1_low and a2_high:
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
