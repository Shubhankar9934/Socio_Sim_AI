"""Context relevance policy: which persona slices and contract fields reach the LLM.

Tiered filtering prevents over-injection (e.g. cuisine on pure housing/rent questions)
while keeping transport/food topics fully grounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.intent_router import LIFESTYLE_KEYWORDS, normalize_turn


@dataclass(frozen=True)
class ContextRelevancePolicy:
    """Flags for prompt construction in ``build_agent_prompt``."""

    include_core_demographics: bool = True
    include_family: bool = True
    include_mobility: bool = True
    include_biography: bool = True
    include_lifestyle_anchors: bool = False
    include_archetype_cultural: bool = False
    include_behavior_floats: bool = True
    include_beliefs: bool = True
    include_personality_summary: bool = True
    include_tradeoff: bool = True
    include_dominant_factor: bool = True
    include_runner_up: bool = True
    include_narrative_guidance: bool = True


def _question_has_lifestyle_keywords(normalized: str) -> bool:
    return any(kw in normalized for kw in LIFESTYLE_KEYWORDS)


def resolve_context_relevance(
    *,
    question: str,
    topic: str = "general",
    domain: str = "general",
    interaction_mode: str = "survey",
    allow_lifestyle_anchors_rule: bool = False,
) -> ContextRelevancePolicy:
    """Derive tiered relevance from topic/domain and anchor rule outcome."""
    normalized = normalize_turn(question)
    t = (topic or "general").strip().lower()
    d = (domain or "general").strip().lower()
    mode = (interaction_mode or "survey").strip().lower()

    lifestyle = bool(allow_lifestyle_anchors_rule)
    if t in {"housing", "cost_of_living"} and not _question_has_lifestyle_keywords(normalized):
        lifestyle = False

    if mode in {"conversation", "qualitative_interview"}:
        lifestyle = lifestyle or _question_has_lifestyle_keywords(normalized) or t in {
            "transport",
            "food_delivery",
            "general",
            "satisfaction",
        }

    biography = True
    if t in {"housing", "cost_of_living"} and not _question_has_lifestyle_keywords(normalized):
        biography = False

    beliefs = True
    personality = True
    if d == "technology" and t == "general" and not lifestyle:
        beliefs = False
        personality = False

    dominant = True
    runner_up = True
    tradeoff = True
    narrative = True
    if mode == "conversation":
        dominant = runner_up = tradeoff = narrative = False

    return ContextRelevancePolicy(
        include_core_demographics=True,
        include_family=True,
        include_mobility=True,
        include_biography=biography,
        include_lifestyle_anchors=lifestyle,
        include_archetype_cultural=lifestyle,
        include_behavior_floats=True,
        include_beliefs=beliefs,
        include_personality_summary=personality,
        include_tradeoff=tradeoff,
        include_dominant_factor=dominant,
        include_runner_up=runner_up,
        include_narrative_guidance=narrative,
    )


def build_relevance_from_turn_understanding(
    question: str,
    turn_understanding: Optional[Dict[str, Any]],
    *,
    interaction_mode: Optional[str] = None,
) -> ContextRelevancePolicy:
    """Convenience: read topic/domain/anchor flag from hybrid understanding dict."""
    tu = turn_understanding or {}
    topic = str(tu.get("topic") or "general")
    domain = str(tu.get("domain") or "general")
    mode = interaction_mode or str(tu.get("interaction_mode") or "survey")
    allow = bool(tu.get("persona_anchor_allowed", False))
    return resolve_context_relevance(
        question=question,
        topic=topic,
        domain=domain,
        interaction_mode=mode,
        allow_lifestyle_anchors_rule=allow,
    )


def filter_memories_for_topic(memories: List[str], topic: str, max_items: int = 5) -> List[str]:
    """Light keyword filter so weakly related memories drop off for narrow topics."""
    if not memories or not topic:
        return memories[:max_items]
    t = topic.lower()
    topic_kw = {
        "housing": ("rent", "lease", "flat", "apartment", "home", "landlord", "room", "housing", "move", "jvc", "marina"),
        "cost_of_living": ("price", "cost", "afford", "budget", "expensive", "bill", "salary", "money", "inflation"),
        "food_delivery": ("food", "deliver", "order", "meal", "restaurant", "eat", "cook"),
        "transport": ("commute", "metro", "car", "drive", "traffic", "parking", "bus", "transport"),
    }
    kws = topic_kw.get(t, ())
    if not kws:
        return memories[:max_items]
    scored: List[tuple[int, str]] = []
    for i, m in enumerate(memories):
        low = m.lower()
        score = sum(1 for k in kws if k in low)
        scored.append((score, i, m))
    scored.sort(key=lambda x: (-x[0], x[1]))
    high = [m for s, _, m in scored if s > 0]
    low = [m for s, _, m in scored if s == 0]
    return (high + low)[:max_items]


def sample_response_shape(
    question_model_key: str,
    interaction_mode: str,
    rng,
    *,
    base_weights: Optional[Dict[str, float]] = None,
) -> str:
    """Explicit short / medium / long mix keyed off question class (diversity policy).

    Returns one of: ``micro``, ``short``, ``medium``, ``long`` (compatible with
    ``build_style_instruction`` / length handling in prompts).
    """
    import random as _random

    r = rng or _random
    mode = (interaction_mode or "survey").strip().lower()
    key = (question_model_key or "").strip().lower()

    if mode in {"conversation", "qualitative_interview"}:
        weights = {"micro": 0.15, "short": 0.35, "medium": 0.40, "long": 0.10}
    elif "open_text" in key or key == "generic_open_text":
        weights = {"micro": 0.05, "short": 0.25, "medium": 0.45, "long": 0.25}
    elif "likert" in key or "nps" in key or "satisfaction" in key:
        weights = {"micro": 0.10, "short": 0.40, "medium": 0.40, "long": 0.10}
    elif "frequency" in key or "likelihood" in key:
        weights = {"micro": 0.15, "short": 0.45, "medium": 0.35, "long": 0.05}
    else:
        weights = {"micro": 0.08, "short": 0.37, "medium": 0.45, "long": 0.10}

    if base_weights:
        merged = dict(weights)
        for k, v in base_weights.items():
            if k in merged:
                merged[k] = float(v)
        s = sum(merged.values())
        if s > 0:
            weights = {k: v / s for k, v in merged.items()}

    choices = list(weights.keys())
    probs = [weights[c] for c in choices]
    return str(r.choices(choices, weights=probs, k=1)[0])
