"""
Tiered memory management for agents.

Three tiers:
  - Short-term: ``structured_memory`` + ``last_answers`` (per-question, capped)
  - Medium-term: ``medium_term_memory`` (compressed summaries, ~100 items)
  - Long-term: ``long_term_preferences`` (distilled preference float signals)

Compression functions move information down the tiers so agents retain
key behavioural signals across long multi-session surveys without
unbounded memory growth.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.state import AgentState


MAX_MEDIUM_TERM = 100
MAX_SHORT_TERM_ANSWERS = 10
MAX_STRUCTURED_MEMORY = 20

_ANSWER_LEVEL_MAP = {
    "rarely": 0, "never": 0,
    "1-2 per week": 1, "1-2 per month": 1,
    "3-4 per week": 2, "2-3 per week": 2,
    "daily": 3, "often": 3,
    "multiple per day": 4, "very often": 4, "always": 4,
    "very unlikely": 0, "unlikely": 1,
    "neutral": 2, "likely": 3, "very likely": 4,
    "strongly disagree": 0, "disagree": 1,
    "agree": 3, "strongly agree": 4,
    "very dissatisfied": 0, "dissatisfied": 1,
    "satisfied": 3, "very satisfied": 4,
    "strongly oppose": 0, "oppose": 1,
    "support": 3, "strongly support": 4,
}

_RELATED_SEMANTIC_KEYS = {
    "delivery_frequency": ["cooking_frequency", "dining_out_frequency"],
    "cooking_frequency": ["delivery_frequency"],
    "transport_usage": ["parking_satisfaction", "metro_usage"],
    "parking_satisfaction": ["transport_usage"],
    "housing_satisfaction": ["cost_of_living_satisfaction"],
    "cost_of_living_satisfaction": ["housing_satisfaction"],
}


def _answer_to_level(answer: str) -> Optional[int]:
    """Map an answer string to a numeric level 0-4."""
    if not answer:
        return None
    normalized = answer.strip().lower()
    if normalized in _ANSWER_LEVEL_MAP:
        return _ANSWER_LEVEL_MAP[normalized]
    if normalized.isdigit():
        return min(4, max(0, int(normalized) - 1))
    return None


def detect_contradiction(
    new_answer: str,
    semantic_key: str,
    state: "AgentState",
    contradiction_threshold: int = 2,
) -> Optional[Dict[str, Any]]:
    """Check if new_answer contradicts any stored answer for related keys.

    Returns a dict describing the contradiction, or None if consistent.
    """
    new_level = _answer_to_level(new_answer)
    if new_level is None:
        return None

    same_entry = state.structured_memory.get(semantic_key)
    if same_entry:
        old_level = _answer_to_level(str(same_entry.get("answer", "")))
        if old_level is not None and abs(new_level - old_level) >= contradiction_threshold:
            return {
                "type": "direct",
                "key": semantic_key,
                "old_answer": same_entry.get("answer"),
                "new_answer": new_answer,
                "gap": abs(new_level - old_level),
            }

    related_keys = _RELATED_SEMANTIC_KEYS.get(semantic_key, [])
    for rel_key in related_keys:
        rel_entry = state.structured_memory.get(rel_key)
        if not rel_entry:
            continue
        old_level = _answer_to_level(str(rel_entry.get("answer", "")))
        if old_level is None:
            continue
        gap = abs(new_level - old_level)
        if gap >= contradiction_threshold + 1:
            return {
                "type": "related",
                "key": rel_key,
                "old_answer": rel_entry.get("answer"),
                "new_answer": new_answer,
                "gap": gap,
            }
    return None


def check_question_repetition(
    question: str,
    state: "AgentState",
) -> Optional[str]:
    """Check if this question was already asked and return the prior answer.

    Uses normalized question hash stored in state.question_history.
    Returns the previous answer string if repeated, None otherwise.
    """
    from agents.intent_router import question_hash
    qhash = question_hash(question)
    return state.question_history.get(qhash)


def record_question(
    question: str,
    answer: str,
    state: "AgentState",
) -> None:
    """Record a question+answer pair for future repetition detection."""
    from agents.intent_router import question_hash
    qhash = question_hash(question)
    state.question_history[qhash] = answer


def compress_short_to_medium(state: "AgentState") -> None:
    """Move oldest short-term memories into compressed medium-term summaries.

    Called between survey rounds when short-term memory is near capacity.
    Produces human-readable summary strings and evicts the originals.
    """
    if len(state.last_answers) <= MAX_SHORT_TERM_ANSWERS // 2:
        return

    items = list(state.last_answers.items())
    to_compress = items[:len(items) - MAX_SHORT_TERM_ANSWERS // 2]

    for qid, answer in to_compress:
        summary = f"{qid}: {answer}"
        state.medium_term_memory.append(summary)
        del state.last_answers[qid]

    if len(state.medium_term_memory) > MAX_MEDIUM_TERM:
        state.medium_term_memory = state.medium_term_memory[-MAX_MEDIUM_TERM:]


def distill_to_long_term(state: "AgentState") -> None:
    """Distill medium-term memories into long-term preference signals.

    Scans medium-term summaries for recurring semantic keys (e.g.
    satisfaction, frequency) and produces running-average preference
    scores in ``long_term_preferences``.
    """
    if not state.structured_memory:
        return

    for sem_key, entry in state.structured_memory.items():
        score = entry.get("answer_score")
        if score is None:
            continue
        score = float(score)
        existing = state.long_term_preferences.get(sem_key)
        if existing is not None:
            state.long_term_preferences[sem_key] = 0.7 * existing + 0.3 * score
        else:
            state.long_term_preferences[sem_key] = score


def compress_all(state: "AgentState") -> None:
    """Run the full compression pipeline: short -> medium -> long."""
    compress_short_to_medium(state)
    distill_to_long_term(state)


def get_relevant_memories(
    state: "AgentState",
    topic: str = "",
    max_items: int = 8,
    decay_constant: float = 5.0,
) -> List[str]:
    """Retrieve the most relevant memories across all tiers.

    Uses recency-weighted scoring: memories from recent turns are
    weighted higher via exponential decay.  Topic-matching memories
    get a relevance boost.

    Returns a flat list of strings suitable for injection into the
    recall step of the cognitive pipeline.
    """
    scored: List[Tuple[float, str]] = []
    turn_count = getattr(state, "turn_count", 0) or 0
    total_structured = len(state.structured_memory)

    for idx, (sem_key, entry) in enumerate(state.structured_memory.items()):
        turns_ago = max(1, total_structured - idx)
        recency = math.exp(-turns_ago / max(1.0, decay_constant))
        topic_boost = 1.5 if (topic and topic.lower() in sem_key.lower()) else 1.0
        score = recency * topic_boost
        scored.append((score, f"[recent] {sem_key}: {entry.get('answer', '')}"))

    if state.medium_term_memory:
        total_medium = len(state.medium_term_memory)
        for idx, m in enumerate(state.medium_term_memory):
            turns_ago = total_structured + (total_medium - idx)
            recency = math.exp(-turns_ago / max(1.0, decay_constant))
            topic_boost = 1.3 if (topic and topic.lower() in m.lower()) else 1.0
            score = recency * topic_boost
            scored.append((score, f"[past] {m}"))

    if state.long_term_preferences:
        for pref_key, pref_val in state.long_term_preferences.items():
            stance = "positive" if pref_val > 0.6 else ("negative" if pref_val < 0.4 else "neutral")
            topic_boost = 1.2 if (topic and topic.lower() in pref_key.lower()) else 0.8
            scored.append((0.3 * topic_boost, f"[preference] {pref_key}: generally {stance}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:max_items]]
