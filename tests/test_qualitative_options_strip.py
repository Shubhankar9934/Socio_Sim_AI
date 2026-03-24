"""Survey options routing: explicit labels from caller take precedence over question shape."""

from agents.intent_router import strip_survey_options_if_qualitative


def test_explicit_options_kept_for_qualitative_wording():
    q = "How are you adjusting to higher grocery prices?"
    opts = ["Absorbing the increase", "Cutting non-essentials"]
    assert strip_survey_options_if_qualitative(q, opts) == opts


def test_keep_options_for_explicit_scale():
    q = "On a scale of 1 to 5, how satisfied are you with public transport?"
    opts = ["1", "2", "3", "4", "5"]
    assert strip_survey_options_if_qualitative(q, opts) == opts


def test_empty_options_unchanged():
    assert strip_survey_options_if_qualitative("Why do you say that?", None) is None
    assert strip_survey_options_if_qualitative("Why?", []) == []


def test_all_blank_options_cleared_for_qualitative():
    q = "What matters most to you when choosing where to live?"
    assert strip_survey_options_if_qualitative(q, ["", "  "]) is None


def test_all_blank_options_kept_when_not_qualitative():
    q = "On a scale of 1 to 5, rate convenience."
    assert strip_survey_options_if_qualitative(q, ["", "  "]) == ["", "  "]
