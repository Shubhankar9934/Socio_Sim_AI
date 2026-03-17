"""
Domain configuration: city/region-specific data loaded from JSON files.

All domain-specific content (demographics, cultural priors, prompts,
topic keywords, strategic actors, etc.) lives under
``data/domains/{domain_id}/`` and is loaded once via ``get_domain_config()``.

To add a new city: create ``data/domains/{city_id}/domain.json`` and
``data/domains/{city_id}/demographics.json`` following the Dubai template.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DOMAINS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "domains"


@dataclass
class DomainConfig:
    """All domain-specific configuration for one city/region."""

    city_id: str = "dubai"
    city_name: str = "Dubai"
    currency: str = "AED"
    id_prefix: str = "DXB"
    districts: List[str] = field(default_factory=list)
    nationalities: List[str] = field(default_factory=list)
    premium_areas: List[str] = field(default_factory=list)

    # Topic detection keywords for perception layer
    topic_keywords: Dict[str, List[str]] = field(default_factory=dict)
    domain_keywords: Dict[str, List[str]] = field(default_factory=dict)
    topic_to_model_key: Dict[str, Dict[str, str]] = field(default_factory=dict)
    location_terms: List[str] = field(default_factory=list)

    # Services and price levels for world feedback
    services: Dict[str, float] = field(default_factory=dict)
    price_levels: Dict[str, float] = field(default_factory=dict)

    # Cultural priors for realism layer
    cultural_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    family_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    income_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    implausible_combos: List[Dict[str, Any]] = field(default_factory=list)
    answer_habit_scores: Dict[str, float] = field(default_factory=dict)
    vague_answers: Dict[str, List[str]] = field(default_factory=dict)

    # LLM prompt templates
    system_prompts: List[str] = field(default_factory=list)
    archetype_hints: Dict[str, str] = field(default_factory=dict)
    cultural_hints: Dict[str, str] = field(default_factory=dict)
    frequency_interpretation: Dict[str, str] = field(default_factory=dict)
    lifestyle_keywords: List[str] = field(default_factory=list)

    # Strategic media actors
    strategic_actors: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-question memory rules
    memory_rules: List[Dict[str, Any]] = field(default_factory=list)
    question_to_semantic_key: Dict[str, str] = field(default_factory=dict)

    # Reference distributions for evaluation
    reference_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Belief mappings
    question_belief_map: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Population segments (loaded separately if present)
    segments: Dict[str, Any] = field(default_factory=dict)

    # Cuisine / diet / cultural data for synthesis
    cuisine_by_nationality: Dict[str, List[str]] = field(default_factory=dict)
    diet_pool: Dict[str, List[Any]] = field(default_factory=dict)
    cultural_family_multiplier: Dict[str, float] = field(default_factory=dict)
    late_dinner_nationalities: List[str] = field(default_factory=list)


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_domain_config(domain_id: str = "dubai") -> DomainConfig:
    """Load a full DomainConfig from ``data/domains/{domain_id}/``."""
    base = _DOMAINS_DIR / domain_id
    if not base.exists():
        logger.warning("Domain directory %s not found, using defaults", base)
        return DomainConfig(city_id=domain_id)

    domain_data = _load_json(base / "domain.json")
    demo_data = _load_json(base / "demographics.json")
    ref_data = _load_json(base / "reference_distributions.json")
    segments_data = _load_json(base / "segments.json")

    districts = list(demo_data.get("location", {}).keys())
    nationalities = list(demo_data.get("nationality", {}).keys())

    cfg = DomainConfig(
        city_id=domain_data.get("city_id", domain_id),
        city_name=domain_data.get("city_name", domain_id.title()),
        currency=domain_data.get("currency", "USD"),
        id_prefix=domain_data.get("id_prefix", domain_id[:3].upper()),
        districts=districts,
        nationalities=nationalities,
        premium_areas=domain_data.get("premium_areas", []),
        topic_keywords=domain_data.get("topic_keywords", {}),
        domain_keywords=domain_data.get("domain_keywords", {}),
        topic_to_model_key=domain_data.get("topic_to_model_key", {}),
        location_terms=domain_data.get("location_terms", []),
        services=domain_data.get("services", {}),
        price_levels=domain_data.get("price_levels", {}),
        cultural_priors=domain_data.get("cultural_priors", {}),
        family_modifiers=domain_data.get("family_modifiers", {}),
        income_modifiers=domain_data.get("income_modifiers", {}),
        implausible_combos=domain_data.get("implausible_combos", []),
        answer_habit_scores=domain_data.get("answer_habit_scores", {}),
        vague_answers=domain_data.get("vague_answers", {}),
        system_prompts=domain_data.get("system_prompts", []),
        archetype_hints=domain_data.get("archetype_hints", {}),
        cultural_hints=domain_data.get("cultural_hints", {}),
        frequency_interpretation=domain_data.get("frequency_interpretation", {}),
        lifestyle_keywords=domain_data.get("lifestyle_keywords", []),
        strategic_actors=domain_data.get("strategic_actors", []),
        memory_rules=domain_data.get("memory_rules", []),
        question_to_semantic_key=domain_data.get("question_to_semantic_key", {}),
        reference_distributions=ref_data if ref_data else domain_data.get("reference_distributions", {}),
        question_belief_map=domain_data.get("question_belief_map", {}),
        segments=segments_data,
        cuisine_by_nationality=domain_data.get("cuisine_by_nationality", {}),
        diet_pool=domain_data.get("diet_pool", {}),
        cultural_family_multiplier=domain_data.get("cultural_family_multiplier", {}),
        late_dinner_nationalities=domain_data.get("late_dinner_nationalities", []),
    )
    return cfg


_cached_config: Optional[DomainConfig] = None
_cached_domain_id: Optional[str] = None


def get_domain_config(domain_id: Optional[str] = None) -> DomainConfig:
    """Get (cached) domain configuration.

    If *domain_id* is None, reads from ``config.settings.get_settings().domain_id``.
    """
    global _cached_config, _cached_domain_id
    if domain_id is None:
        try:
            from config.settings import get_settings
            domain_id = get_settings().domain_id
        except Exception:
            domain_id = "dubai"
    if _cached_config is not None and _cached_domain_id == domain_id:
        return _cached_config
    _cached_config = load_domain_config(domain_id)
    _cached_domain_id = domain_id
    return _cached_config


def reset_domain_cache() -> None:
    """Clear cached config (useful for tests)."""
    global _cached_config, _cached_domain_id
    _cached_config = None
    _cached_domain_id = None
