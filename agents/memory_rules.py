"""
Cross-question memory influence rules.

When an agent answers one question, the answer is stored under a semantic key.
Later questions consult this memory and apply multiplicative biases to the
probability distribution so that responses are behaviorally consistent.

Example: answering "rarely" to delivery_frequency biases future
delivery_satisfaction toward neutral/dissatisfied.

The rules are declarative: each ``MemoryRule`` specifies a source semantic key,
the target question-model keys it influences, a condition on the stored answer,
and a dict of option -> multiplicative bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class MemoryRule:
    """One cross-question influence rule.

    Attributes
    ----------
    source_key : semantic key in structured_memory (e.g. "delivery_frequency")
    target_model_keys : question-model names this rule affects
    condition : predicate on the stored memory dict; if True the bias is applied
    bias : option_label -> multiplicative weight (>1 = boost, <1 = suppress)
    """
    source_key: str
    target_model_keys: List[str]
    condition: Callable[[Dict[str, Any]], bool]
    bias: Dict[str, float]


def _load_question_to_semantic_key() -> Dict[str, str]:
    """Load question-to-semantic-key mapping from domain config."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.question_to_semantic_key:
            return dict(cfg.question_to_semantic_key)
    except Exception:
        pass
    return {
        "generic_frequency": "generic_frequency",
        "generic_likert": "generic_likert",
        "generic_duration": "tenure",
    }


QUESTION_TO_SEMANTIC_KEY: Dict[str, str] = _load_question_to_semantic_key()


CROSS_QUESTION_RULES: List[MemoryRule] = [
    # Rarely orders delivery => likely not very satisfied with delivery apps
    MemoryRule(
        source_key="delivery_frequency",
        target_model_keys=["generic_likert", "generic_frequency"],
        condition=lambda m: m.get("answer") in ("rarely",),
        bias={
            "5": 0.55, "4": 0.70, "3": 1.25, "2": 1.15, "1": 1.10,
            "very often": 0.50, "often": 0.65, "sometimes": 1.20,
            "rarely": 1.20, "never": 1.10,
        },
    ),
    # Daily+ delivery => probably satisfied, tech-comfortable
    MemoryRule(
        source_key="delivery_frequency",
        target_model_keys=["tech_adoption_likelihood", "generic_likert"],
        condition=lambda m: m.get("answer") in ("daily", "multiple per day"),
        bias={
            "very likely": 1.25, "likely": 1.15, "neutral": 0.90,
            "unlikely": 0.70, "very unlikely": 0.55,
            "5": 1.25, "4": 1.15, "3": 0.95, "2": 0.75, "1": 0.60,
        },
    ),
    # Moderate delivery (1-2/week) => slightly positive on convenience topics
    MemoryRule(
        source_key="delivery_frequency",
        target_model_keys=["generic_likert"],
        condition=lambda m: m.get("answer") in ("1-2 per week",),
        bias={"5": 0.90, "4": 1.10, "3": 1.15, "2": 0.95, "1": 0.85},
    ),
    # Heavy delivery => less shopping in person
    MemoryRule(
        source_key="delivery_frequency",
        target_model_keys=["shopping_frequency"],
        condition=lambda m: m.get("answer") in ("daily", "multiple per day"),
        bias={
            "daily": 0.75, "2-3 per week": 0.85, "weekly": 1.10,
            "1-2 per month": 1.15, "rarely": 1.10,
        },
    ),
    # Bad parking => more delivery-friendly, supports transport policy
    MemoryRule(
        source_key="parking_satisfaction",
        target_model_keys=["food_delivery_frequency", "policy_support"],
        condition=lambda m: m.get("answer") in ("1", "2"),
        bias={
            "daily": 1.15, "3-4 per week": 1.10, "1-2 per week": 1.05,
            "rarely": 0.85, "multiple per day": 1.10,
            "strongly support": 1.20, "support": 1.15, "neutral": 1.00,
            "oppose": 0.80, "strongly oppose": 0.70,
        },
    ),
    # Good parking => less reliant on delivery
    MemoryRule(
        source_key="parking_satisfaction",
        target_model_keys=["food_delivery_frequency"],
        condition=lambda m: m.get("answer") in ("4", "5"),
        bias={
            "daily": 0.85, "3-4 per week": 0.90, "1-2 per week": 1.05,
            "rarely": 1.15, "multiple per day": 0.75,
        },
    ),
    # Happy with housing => generally more positive on satisfaction questions
    MemoryRule(
        source_key="housing_satisfaction",
        target_model_keys=["parking_satisfaction", "transport_satisfaction"],
        condition=lambda m: m.get("answer") in ("4", "5"),
        bias={"5": 1.15, "4": 1.10, "3": 1.00, "2": 0.85, "1": 0.75},
    ),
    # Unhappy with housing => generally more negative
    MemoryRule(
        source_key="housing_satisfaction",
        target_model_keys=["parking_satisfaction", "transport_satisfaction"],
        condition=lambda m: m.get("answer") in ("1", "2"),
        bias={"5": 0.70, "4": 0.80, "3": 1.05, "2": 1.15, "1": 1.20},
    ),
    # Tech adopter => more likely to order delivery, shop online
    MemoryRule(
        source_key="tech_adoption",
        target_model_keys=["food_delivery_frequency", "shopping_frequency"],
        condition=lambda m: m.get("answer") in ("very likely", "likely"),
        bias={
            "daily": 1.15, "3-4 per week": 1.10, "multiple per day": 1.10,
            "rarely": 0.80,
            "2-3 per week": 1.10, "weekly": 1.05,
        },
    ),
]


def apply_memory_rules(
    distribution: Dict[str, float],
    question_model_key: str,
    structured_memory: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Apply cross-question memory biases to a probability distribution.

    Scans all rules whose target_model_keys include the current question
    and whose source_key exists in structured_memory.  Matching rules
    multiplicatively adjust option probabilities.
    """
    if not structured_memory:
        return distribution

    adjusted = dict(distribution)
    applied_any = False

    for rule in CROSS_QUESTION_RULES:
        if question_model_key not in rule.target_model_keys:
            continue
        mem_entry = structured_memory.get(rule.source_key)
        if mem_entry is None:
            continue
        try:
            if not rule.condition(mem_entry):
                continue
        except Exception:
            continue

        for option, mult in rule.bias.items():
            if option in adjusted:
                adjusted[option] *= mult
                applied_any = True

    if not applied_any:
        return distribution

    total = sum(adjusted.values())
    if total <= 0:
        return distribution
    return {k: v / total for k, v in adjusted.items()}
