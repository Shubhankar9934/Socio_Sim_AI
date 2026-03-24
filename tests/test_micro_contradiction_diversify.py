"""Micro-contradiction suffix uses per-agent text hash so batches do not clone."""

from agents.realism import maybe_add_micro_contradiction


def _fake_persona(young: bool):
    class NS:
        rhetorical_habit = "direct"

    class PA:
        narrative_style = NS()

    class P:
        agent_id = "DXB_0001" if young else "DXB_0099"
        age = "28" if young else "62"
        personal_anchors = PA()

    return P()


def test_different_agents_get_different_micro_suffix_same_text():
    text = (
        "Prices are rough so I'm buying more rice and lentils and cutting snacks. "
        "Still manageable if I plan the week."
    )
    young = _fake_persona(young=True)
    outs = [
        maybe_add_micro_contradiction(
            text,
            confidence_band="medium",
            contradiction_probability=1.0,
            agent_id=f"DXB_{i:04d}",
            persona=young,
        )
        for i in range(1, 22)
    ]
    # Per-agent hash → varied qualifiers; batch should not collapse to one string.
    assert len(set(outs)) >= 10


def test_senior_direct_persona_skips_micro_contradiction():
    text = (
        "We buy less meat and more vegetables. The bill still goes up every month "
        "but we cope by planning meals."
    )
    out = maybe_add_micro_contradiction(
        text,
        confidence_band="medium",
        contradiction_probability=1.0,
        agent_id="DXB_0002",
        persona=_fake_persona(young=False),
    )
    assert out == text
