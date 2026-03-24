"""Voice diversity: crutch rate, first-sentence spread, opening pools, tradeoff copy."""

import random
import re

import pytest

from agents.narrative import (
    NarrativeStyleProfile,
    _opening_pool_for_profile,
    derive_narrative_style_profile,
    format_voice_instruction_line,
)
from agents.response_contract import _compute_tradeoff_guidance
from population.personas import NarrativeStyleFields, PersonalityVector


def test_derive_profile_includes_voice_axes():
    rng = random.Random(42)
    pvec = PersonalityVector(
        openness_to_experience=0.8,
        conscientiousness=0.35,
        agreeableness=0.5,
    )
    prof = derive_narrative_style_profile(
        "25-34", "25-50k", "professional", "Western", rng, personality=pvec,
    )
    assert prof.voice_register in ("analytical", "conversational", "blunt", "rambling")
    assert prof.rhetorical_habit in ("direct", "narrative", "list_pros_cons", "emotional_lead")
    assert isinstance(prof.avoid_phrases, tuple)


def test_opening_pool_analytical_excludes_many_crutch_openings():
    blunt = NarrativeStyleProfile(
        verbosity="short",
        preferred_tone="blunt",
        preferred_style="routine",
        slang_level=0.2,
        grammar_quality=0.8,
        voice_register="analytical",
        rhetorical_habit="direct",
        avoid_phrases=(),
    )
    pool = _opening_pool_for_profile(blunt)
    full = _opening_pool_for_profile(None)
    assert len(pool) < len(full)
    low = " ".join(pool).lower()
    assert "honestly" not in low or sum("honestly" in o.lower() for o in pool) < sum(
        "honestly" in o.lower() for o in full
    )


def test_tradeoff_guidance_varies_by_rhetorical_habit():
    dist = {"A": 0.41, "B": 0.40, "C": 0.19}
    d = _compute_tradeoff_guidance("A", "B", dist, rhetorical_habit="direct")
    n = _compute_tradeoff_guidance("A", "B", dist, rhetorical_habit="narrative")
    assert d and n
    assert d != n


def test_format_voice_instruction_non_empty():
    p = NarrativeStyleProfile(
        verbosity="medium",
        preferred_tone="casual",
        preferred_style="routine",
        slang_level=0.4,
        grammar_quality=0.6,
        voice_register="blunt",
        rhetorical_habit="list_pros_cons",
        avoid_phrases=("honestly", "I mean"),
    )
    line = format_voice_instruction_line(p)
    assert "Voice:" in line
    assert "blunt" in line.lower() or "direct" in line.lower()


_CRUTCH_RE = re.compile(
    r"\b(honestly|i mean|kinda torn|you know|kind of|i guess|tbh)\b",
    re.IGNORECASE,
)


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    for sep in ".!?":
        if sep in t[:220]:
            return t.split(sep)[0].strip()
    return t[:120]


def test_crutch_rate_and_first_sentence_diversity_metrics():
    """Synthetic batch: metric helpers (no LLM)."""
    samples = [
        "Honestly I would pick daily for that.",
        "I mean, probably rarely if I'm cooking.",
        "Three or four times a week fits me.",
        "Kinda torn but I'd say sometimes.",
        "You know, it depends on the week.",
    ]
    crutches = sum(1 for s in samples if _CRUTCH_RE.search(s))
    assert crutches >= 3
    firsts = [_first_sentence(s) for s in samples]
    unique = len(set(firsts))
    assert unique >= 3

    # NarrativeStyleFields defaults include voice_register
    ns = NarrativeStyleFields()
    assert ns.voice_register == "conversational"
    assert ns.rhetorical_habit == "direct"
    assert ns.avoid_phrases == []
