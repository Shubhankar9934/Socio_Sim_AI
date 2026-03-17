"""
District properties: population_density, metro_access, restaurant_density, etc.
"""

from dataclasses import dataclass
from typing import Dict

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


def get_district(name: str) -> DistrictProperties:
    """Get district properties by name."""
    return DEFAULT_DISTRICT_PROPERTIES.get(
        name,
        DistrictProperties(name, "medium", 25000, False, "medium", "medium"),
    )


def location_quality_for_satisfaction(district_name: str) -> float:
    """Scalar 0-1 for use in satisfaction models (parking, transport)."""
    d = get_district(district_name)
    parking_score = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(d.parking_availability, 0.5)
    metro_score = 0.2 if d.metro_access else 0.0
    return min(1.0, parking_score * 0.5 + 0.5 * (0.5 + metro_score))
