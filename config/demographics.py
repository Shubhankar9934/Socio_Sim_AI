"""
Generic demographics loader: reads from ``data/domains/{domain_id}/demographics.json``.

Replaces the hardcoded ``config/dubai_data`` module with a config-driven
approach that works for any city/domain.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DOMAINS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "domains"


@dataclass
class DemographicData:
    """All demographic distributions + conditional tables for one domain."""

    age: Dict[str, float] = field(default_factory=dict)
    nationality: Dict[str, float] = field(default_factory=dict)
    income: Dict[str, float] = field(default_factory=dict)
    location: Dict[str, float] = field(default_factory=dict)
    household_size: Dict[str, float] = field(default_factory=dict)
    occupation: Dict[str, float] = field(default_factory=dict)

    income_given_nationality: Dict[str, Dict[str, float]] = field(default_factory=dict)
    location_given_income: Dict[str, Dict[str, float]] = field(default_factory=dict)
    car_given_location: Dict[str, float] = field(default_factory=dict)
    metro_access_by_location: Dict[str, bool] = field(default_factory=dict)
    occupation_given_nationality: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_all_marginals(self) -> Dict[str, Dict[str, float]]:
        return {
            "age": self.age,
            "nationality": self.nationality,
            "income": self.income,
            "location": self.location,
            "household_size": self.household_size,
            "occupation": self.occupation,
        }

    def get_nationality_keys(self) -> List[str]:
        return list(self.nationality.keys())

    def get_age_keys(self) -> List[str]:
        return list(self.age.keys())

    def get_income_keys(self) -> List[str]:
        return list(self.income.keys())

    def get_location_keys(self) -> List[str]:
        return list(self.location.keys())


_cached_demo: Optional[DemographicData] = None
_cached_demo_id: Optional[str] = None


def get_demographics(domain_id: Optional[str] = None) -> DemographicData:
    """Load demographic data for a domain, with caching.

    If *domain_id* is None, reads from settings.
    """
    global _cached_demo, _cached_demo_id
    if domain_id is None:
        try:
            from config.settings import get_settings
            domain_id = get_settings().domain_id
        except Exception:
            domain_id = "dubai"
    if _cached_demo is not None and _cached_demo_id == domain_id:
        return _cached_demo

    path = _DOMAINS_DIR / domain_id / "demographics.json"
    if not path.exists():
        logger.warning("Demographics file %s not found, returning empty", path)
        _cached_demo = DemographicData()
        _cached_demo_id = domain_id
        return _cached_demo

    raw = json.loads(path.read_text(encoding="utf-8"))

    demo = DemographicData(
        age=raw.get("age", {}),
        nationality=raw.get("nationality", {}),
        income=raw.get("income", {}),
        location=raw.get("location", {}),
        household_size=raw.get("household_size", {}),
        occupation=raw.get("occupation", {}),
        income_given_nationality=raw.get("income_given_nationality", {}),
        location_given_income=raw.get("location_given_income", {}),
        car_given_location={k: float(v) for k, v in raw.get("car_given_location", {}).items()},
        metro_access_by_location={k: bool(v) for k, v in raw.get("metro_access_by_location", {}).items()},
        occupation_given_nationality=raw.get("occupation_given_nationality", {}),
    )

    _cached_demo = demo
    _cached_demo_id = domain_id
    return demo


def reset_demographics_cache() -> None:
    global _cached_demo, _cached_demo_id
    _cached_demo = None
    _cached_demo_id = None
