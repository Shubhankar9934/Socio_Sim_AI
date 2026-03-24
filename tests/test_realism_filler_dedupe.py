"""Realism injectors must not double-stamp fillers (basically/basically, like like, etc.)."""

import random

from agents.narrative import NarrativeStyleProfile
from agents.realism import (
    _remainder_prefix_matches_marker,
    inject_thinking_markers,
    maybe_add_hedging,
    maybe_add_redundancy,
)


def _analytical_profile() -> NarrativeStyleProfile:
    return NarrativeStyleProfile(
        verbosity="medium",
        preferred_tone="matter_of_fact",
        preferred_style="routine",
        slang_level=0.2,
        grammar_quality=0.80,
        voice_register="analytical",
        rhetorical_habit="direct",
        avoid_phrases=(),
    )


def _casual_profile() -> NarrativeStyleProfile:
    return NarrativeStyleProfile(
        verbosity="medium",
        preferred_tone="casual",
        preferred_style="routine",
        slang_level=0.45,
        grammar_quality=0.50,
        voice_register="conversational",
        rhetorical_habit="direct",
        avoid_phrases=(),
    )


def test_remainder_prefix_detects_duplicate_marker():
    assert _remainder_prefix_matches_marker("basically", " basically I would start")
    assert not _remainder_prefix_matches_marker("well", " basically I would start")
    assert _remainder_prefix_matches_marker("I mean", " I mean what I said")


def test_inject_thinking_analytical_never_emits_double_basically():
    text = "Yeah so basically I would start cutting extras first."
    prof = _analytical_profile()
    for seed in range(400):
        out = inject_thinking_markers(
            text, prof, rng=random.Random(seed), marker_probability=1.0,
        )
        assert ", basically, basically" not in out.lower()
        assert out.lower().count("basically") <= text.lower().count("basically") + 1


def test_maybe_add_hedging_skips_when_basically_already_near_start():
    text = "Yeah so basically I cut back on extras this month."
    prof = _casual_profile()
    for seed in range(200):
        out = maybe_add_hedging(
            text,
            rng=random.Random(seed),
            hedge_probability=1.0,
            profile=prof,
        )
        assert not out.lower().startswith("so basically yeah")


def test_redundancy_echo_avoids_like_like():
    text = "With tolls up the car is going to feel like a luxury now."
    bad = ", like like"
    for seed in range(2500):
        out = maybe_add_redundancy(
            text,
            rng=random.Random(seed),
            redundancy_probability=1.0,
        )
        assert bad not in out.lower()
