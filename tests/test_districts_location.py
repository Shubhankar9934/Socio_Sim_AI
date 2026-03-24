"""Location factor must not collapse to 0.5 for major Dubai districts."""

from world.districts import (
    canonicalize_district_name,
    location_quality_for_satisfaction,
    resolve_location_quality,
)


def test_marina_and_deira_are_not_identical_neutral():
    m = location_quality_for_satisfaction("Dubai Marina")
    d = location_quality_for_satisfaction("Deira")
    assert abs(m - 0.5) > 0.02
    assert abs(d - 0.5) > 0.02
    assert m != d


def test_low_parking_metro_districts_were_formula_collapsed_to_half():
    # Regression: old formula gave exactly 0.5 for low parking + metro=True
    assert location_quality_for_satisfaction("Business Bay") != 0.5
    assert location_quality_for_satisfaction("JLT") != 0.5


def test_canonicalize_aliases():
    assert canonicalize_district_name("  dubai marina ") == "Dubai Marina"
    assert canonicalize_district_name("JLT") == "JLT"
    assert canonicalize_district_name("jumeirah village circle") == "JVC"


def test_resolve_uses_explicit_override():
    assert resolve_location_quality("Dubai Marina", 0.91) == 0.91
    assert resolve_location_quality("Dubai Marina", None) == location_quality_for_satisfaction("Dubai Marina")
