"""Guards on maybe_fragmentize: no splits on in-word hyphens; completeness."""

import random

from agents.narrative import NarrativeStyleProfile
from agents.realism import maybe_fragmentize, trim_weak_terminal_suffix


def _profile_fragment_always():
    return NarrativeStyleProfile(
        verbosity="medium",
        preferred_tone="casual",
        preferred_style="routine",
        slang_level=0.35,
        grammar_quality=0.55,
        voice_register="conversational",
        rhetorical_habit="direct",
        avoid_phrases=(),
    )


def test_fragmentize_does_not_cut_on_hyphen_inside_word():
    text = (
        "I will start bulk-buying staples at cheaper stores this month because "
        "prices keep climbing and I need to stretch the budget further."
    )
    rng = random.Random(0)
    prof = _profile_fragment_always()
    for _ in range(40):
        out = maybe_fragmentize(text, prof, rng=rng, fragment_probability=1.0)
        assert "bulk-buying" in out or out == text
        assert not out.endswith("bulk.")
        assert not out.endswith("start bulk.")


def test_fragmentize_rejects_too_short_clause():
    text = "Sometimes I wonder about prices, but sometimes I wonder."
    rng = random.Random(42)
    prof = _profile_fragment_always()
    for _ in range(30):
        out = maybe_fragmentize(text, prof, rng=rng, fragment_probability=1.0)
        if out != text:
            assert len(out.split()) >= 5
            assert out.endswith((".", "!", "?"))


def test_fragmentize_preserves_short_text():
    short = "Fine, I guess."
    rng = random.Random(1)
    assert maybe_fragmentize(short, _profile_fragment_always(), rng=rng, fragment_probability=1.0) == short


def test_trim_weak_terminal_suffix_drops_last_sentence():
    t = (
        "We switched to cheaper stores and buy in bulk when it makes sense. "
        "Still stressful but it helps a bit with the total bill, though sometimes I wonder."
    )
    out = trim_weak_terminal_suffix(t)
    assert "wonder" not in out.lower()
    assert len(out.split()) >= 8


def test_trim_weak_terminal_noop_when_too_short():
    t = "Maybe prices will ease, I guess."
    assert trim_weak_terminal_suffix(t) == t
