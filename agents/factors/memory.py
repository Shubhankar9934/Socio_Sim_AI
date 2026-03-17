"""
Memory factor: combines keyword sentiment from recalled memories with
structured cross-question memory influence.

Structured memory (stored answers from prior questions) provides targeted
biases via the rules in ``agents.memory_rules``.  Keyword sentiment from
the recall store provides a secondary nudge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext

_POSITIVE_KEYWORDS = frozenset({
    "great", "excellent", "love", "good", "happy", "satisfied",
    "convenient", "fast", "easy", "improved", "recommend", "enjoy",
})
_NEGATIVE_KEYWORDS = frozenset({
    "bad", "terrible", "hate", "slow", "expensive", "poor",
    "frustrated", "broken", "worst", "complaint", "disappointing", "delay",
})

_MEMORY_WEIGHT = 0.08


def _keyword_sentiment_score(memories: list) -> float:
    """Scan recalled memory strings for sentiment keywords -> 0-1 score."""
    if not memories:
        return 0.5
    nudge = 0.0
    for mem in memories:
        words = set(mem.lower().split())
        nudge += len(words & _POSITIVE_KEYWORDS) * _MEMORY_WEIGHT
        nudge -= len(words & _NEGATIVE_KEYWORDS) * _MEMORY_WEIGHT
    return max(0.0, min(1.0, 0.5 + nudge))


def _structured_memory_score(ctx: "DecisionContext") -> float:
    """Compute a 0-1 bias score from structured cross-question memory.

    If the agent has answered related questions before, the rules in
    memory_rules.py provide directional influence.  Returns 0.5 (neutral)
    if no rules fire.
    """
    env = ctx.environment or {}
    structured = env.get("structured_memory")
    if not structured:
        return 0.5

    qm_key = ""
    perception = ctx.perception
    if perception and hasattr(perception, "question_model_key"):
        qm_key = perception.question_model_key or ""

    if not qm_key:
        return 0.5

    from agents.memory_rules import CROSS_QUESTION_RULES
    nudge = 0.0
    rule_count = 0
    for rule in CROSS_QUESTION_RULES:
        if qm_key not in rule.target_model_keys:
            continue
        mem_entry = structured.get(rule.source_key)
        if mem_entry is None:
            continue
        try:
            if not rule.condition(mem_entry):
                continue
        except Exception:
            continue
        avg_bias = sum(rule.bias.values()) / max(len(rule.bias), 1)
        nudge += (avg_bias - 1.0) * 0.3
        rule_count += 1

    if rule_count == 0:
        return 0.5
    return max(0.0, min(1.0, 0.5 + nudge / rule_count))


def memory_factor(ctx: "DecisionContext") -> float:
    """Combined memory factor: structured memory + keyword sentiment.

    Structured memory (cross-question consistency) takes priority;
    keyword sentiment is a secondary signal.
    """
    struct_score = _structured_memory_score(ctx)
    kw_score = _keyword_sentiment_score(ctx.memories)

    if abs(struct_score - 0.5) > 0.01:
        return 0.65 * struct_score + 0.35 * kw_score
    return kw_score
