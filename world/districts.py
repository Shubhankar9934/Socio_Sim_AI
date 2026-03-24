"""
District properties: population_density, metro_access, restaurant_density, etc.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from config.demographics import get_demographics


@dataclass
class DistrictProperties:
    """Properties of a district that affect agent behavior."""

    name: str
    population_density: str  # "low" | "medium" | "high"
    average_income: int  # AED monthly, approximate
    metro_access: bool
    restaurant_density: str  # "low" | "medium" | "high"
    parking_availability: str  # "low" | "medium" | "high"


# Default properties per district (can be overridden by events)
DEFAULT_DISTRICT_PROPERTIES: Dict[str, DistrictProperties] = {}

def _build_default_districts() -> Dict[str, DistrictProperties]:
    metro = get_demographics().metro_access_by_location
    # Heuristics: premium areas higher income, metro areas higher density
    data = {
        "Dubai Marina": DistrictProperties("Dubai Marina", "high", 35000, True, "high", "low"),
        "Jumeirah": DistrictProperties("Jumeirah", "medium", 40000, False, "high", "medium"),
        "Deira": DistrictProperties("Deira", "high", 18000, True, "medium", "low"),
        "Business Bay": DistrictProperties("Business Bay", "high", 32000, True, "high", "low"),
        "Al Barsha": DistrictProperties("Al Barsha", "medium", 28000, True, "medium", "medium"),
        "JLT": DistrictProperties("JLT", "high", 30000, True, "high", "low"),
        "Downtown": DistrictProperties("Downtown", "high", 38000, True, "high", "low"),
        "Al Karama": DistrictProperties("Al Karama", "high", 15000, True, "medium", "low"),
        "JVC": DistrictProperties("JVC", "medium", 22000, False, "medium", "high"),
        "Others": DistrictProperties("Others", "medium", 25000, False, "medium", "medium"),
    }
    # Sync metro from dubai_data
    for name, prop in data.items():
        prop.metro_access = metro.get(name, False)
    return data


# Populate on import
DEFAULT_DISTRICT_PROPERTIES = _build_default_districts()

# Aliases and minor variants seen in personas / data (→ canonical registry key)
_DISTRICT_ALIASES: Dict[str, str] = {
    "dubai marina": "Dubai Marina",
    "marina": "Dubai Marina",
    "business bay": "Business Bay",
    "businessbay": "Business Bay",
    "jlt": "JLT",
    "jumeirah lakes towers": "JLT",
    "jumeirah": "Jumeirah",
    "jumeirah village circle": "JVC",
    "jvc": "JVC",
    "deira": "Deira",
    "downtown": "Downtown",
    "downtown dubai": "Downtown",
    "al barsha": "Al Barsha",
    "barsha": "Al Barsha",
    "al karama": "Al Karama",
    "karama": "Al Karama",
    "others": "Others",
    "other": "Others",
}

# Explicit LPFG location factor signal per district (spread away from 0.5).
# Interpretation: mobility / environment / rent-pressure blend for relocation & satisfaction
# graphs — MUST differ across Dubai Marina vs Deira vs JVC so importance is non-zero.
DISTRICT_LOCATION_FACTOR_SCORE: Dict[str, float] = {
    "Dubai Marina": 0.38,
    "Jumeirah": 0.44,
    "Business Bay": 0.40,
    "JLT": 0.39,
    "Downtown": 0.36,
    "Al Barsha": 0.46,
    "JVC": 0.58,
    "Deira": 0.63,
    "Al Karama": 0.61,
    "Others": 0.52,
}


def canonicalize_district_name(name: str) -> str:
    """Map free-text location to a known district key when possible."""
    raw = (name or "").strip()
    if not raw:
        return "Others"
    if raw in DEFAULT_DISTRICT_PROPERTIES:
        return raw
    key = raw.lower()
    if key in _DISTRICT_ALIASES:
        return _DISTRICT_ALIASES[key]
    for canon in DEFAULT_DISTRICT_PROPERTIES:
        if canon.lower() == key:
            return canon
    return raw


def get_district(name: str) -> DistrictProperties:
    """Get district properties by name."""
    canon = canonicalize_district_name(name)
    if canon in DEFAULT_DISTRICT_PROPERTIES:
        return DEFAULT_DISTRICT_PROPERTIES[canon]
    return DEFAULT_DISTRICT_PROPERTIES.get(
        name,
        DistrictProperties(name, "medium", 25000, False, "medium", "medium"),
    )


def location_quality_for_satisfaction(district_name: str) -> float:
    """Scalar in [0,1] for LPFG location_factor — differentiated per district.

    The old formula ``parking*0.5 + 0.5*(0.5+metro)`` collapsed to **exactly 0.5**
    for every low-parking + metro district (e.g. Dubai Marina, Business Bay),
    yielding zero importance (|raw-0.5|) and starving the decision graph.
    """
    canon = canonicalize_district_name(district_name)
    if canon in DISTRICT_LOCATION_FACTOR_SCORE:
        return float(DISTRICT_LOCATION_FACTOR_SCORE[canon])

    d = get_district(canon)
    # Non-collapsing blend: parking and metro push in different directions so
    # few rows land exactly on 0.5.
    parking = {"low": 0.28, "medium": 0.52, "high": 0.78}.get(d.parking_availability, 0.50)
    metro_adj = 0.10 if d.metro_access else -0.06
    density_adj = {"low": -0.04, "medium": 0.0, "high": 0.05}.get(d.population_density, 0.0)
    score = 0.62 * parking + 0.28 * (0.45 + metro_adj) + density_adj
    return float(max(0.0, min(1.0, score)))


def resolve_location_quality(
    district_name: str,
    explicit: Optional[float] = None,
) -> float:
    """Use an explicit survey-time override when set; otherwise derive from district."""
    if explicit is not None:
        return max(0.0, min(1.0, float(explicit)))
    return float(location_quality_for_satisfaction(district_name or ""))
