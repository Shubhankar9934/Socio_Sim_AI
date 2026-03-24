"""Persona model for synthetic agents."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

PERSONA_SCHEMA_VERSION = "v3.0"


class PersonaMeta(BaseModel):
    """Provenance and reproducibility metadata for a synthetic persona."""

    persona_version: str = Field(default=PERSONA_SCHEMA_VERSION)
    synthesis_method: str = Field(default="bayesian", description="monte_carlo | bayesian | ipf")
    generation_seed: Optional[int] = Field(default=None)
    archetype_id: Optional[int] = Field(default=None, description="Cluster label from archetype compression")
    persona_cluster: Optional[int] = Field(default=None, description="KMeans cluster id if archetypes were built")
    population_segment: Optional[str] = Field(default=None, description="Behavioral population segment for multimodal clustering")


class FamilyStructure(BaseModel):
    """Household structure."""

    spouse: bool = False
    children: int = Field(default=0, ge=0, le=8)


class MobilityProfile(BaseModel):
    """Transport/mobility preferences."""

    car: bool = True
    metro_usage: str = Field(default="rare", description="rare | occasional | frequent")


class LifestyleCoefficients(BaseModel):
    """Behavioral coefficients 0.0-1.0 for decision models."""

    luxury_preference: float = Field(default=0.5, ge=0.0, le=1.0)
    tech_adoption: float = Field(default=0.5, ge=0.0, le=1.0)
    dining_out: float = Field(default=0.5, ge=0.0, le=1.0)
    convenience_preference: float = Field(default=0.5, ge=0.0, le=1.0)
    price_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    primary_service_preference: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def food_delivery_preference(self) -> float:
        """Backward-compat alias."""
        return self.primary_service_preference

    @food_delivery_preference.setter
    def food_delivery_preference(self, v: float) -> None:
        self.primary_service_preference = v


class NarrativeStyleFields(BaseModel):
    """Persistent writing-style identity stored on the persona."""

    verbosity: str = Field(default="medium", description="micro | short | medium | long")
    preferred_tone: str = Field(default="casual")
    preferred_style: str = Field(default="routine")
    slang_level: float = Field(default=0.3, ge=0.0, le=1.0)
    grammar_quality: float = Field(default=0.6, ge=0.0, le=1.0)
    voice_register: str = Field(
        default="conversational",
        description="analytical | conversational | blunt | rambling",
    )
    rhetorical_habit: str = Field(
        default="direct",
        description="direct | narrative | list_pros_cons | emotional_lead",
    )
    avoid_phrases: List[str] = Field(
        default_factory=list,
        description="Discourse markers this agent should avoid in survey answers",
    )


class PersonalAnchors(BaseModel):
    """Unique personal details that anchor LLM narrative diversity."""

    cuisine_preference: str = Field(default="Mixed")
    diet: str = Field(default="no restriction")
    hobby: str = Field(default="reading")
    secondary_hobby: str = Field(default="")
    work_schedule: str = Field(default="9-to-5")
    typical_dinner_time: str = Field(default="8 PM")
    commute_method: str = Field(default="car")
    health_focus: str = Field(default="moderate")
    weekend_habit: str = Field(default="relaxing at home")
    spending_pattern: str = Field(default="balanced")
    social_media_preference: str = Field(default="")
    pet: str = Field(default="none")
    music_preference: str = Field(default="")
    reading_preference: str = Field(default="")
    archetype: str = Field(default="default", description="Behavioral archetype label")
    narrative_style: NarrativeStyleFields = Field(default_factory=NarrativeStyleFields)


class PersonalityVector(BaseModel):
    """Stored personality traits (0-1) seeded per-agent at generation time.

    Unlike the derived ``PersonalityTraits`` in ``agents/personality.py``,
    this vector persists on the persona and includes additional Big-Five
    inspired dimensions so that every agent has a unique, stable personality.
    """

    risk_aversion: float = Field(default=0.5, ge=0.0, le=1.0)
    openness_to_experience: float = Field(default=0.5, ge=0.0, le=1.0)
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0)
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_stability: float = Field(default=0.5, ge=0.0, le=1.0)
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0)
    impulsivity: float = Field(default=0.5, ge=0.0, le=1.0)
    optimism: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_list(self) -> List[float]:
        return [
            self.risk_aversion, self.openness_to_experience,
            self.conscientiousness, self.agreeableness,
            self.emotional_stability, self.extraversion,
            self.impulsivity, self.optimism,
        ]


class LifePathEntry(BaseModel):
    """One milestone in a retroactively generated life history."""

    age_at_event: int
    event: str
    detail: str = ""


class LifePath(BaseModel):
    """Pre-simulated life history for biographical depth."""

    biography: str = Field(default="", description="1-2 sentence compressed biography")
    milestones: List[LifePathEntry] = Field(default_factory=list)
    career_trajectory: List[str] = Field(default_factory=list, description="Sequence of job titles / career stages")
    education_level: str = Field(default="secondary")


class Persona(BaseModel):
    """Full synthetic persona for one agent."""

    agent_id: str = Field(..., description="Unique agent identifier e.g. DXB_001")
    age: str = Field(..., description="Age group e.g. 25-34")
    nationality: str = Field(...)
    income: str = Field(..., description="Income band e.g. 10-25k")
    location: str = Field(..., description="District/area")
    occupation: str = Field(default="professional")
    household_size: str = Field(default="2")
    family: FamilyStructure = Field(default_factory=FamilyStructure)
    mobility: MobilityProfile = Field(default_factory=MobilityProfile)
    lifestyle: LifestyleCoefficients = Field(default_factory=LifestyleCoefficients)
    personal_anchors: PersonalAnchors = Field(default_factory=PersonalAnchors)
    personality: PersonalityVector = Field(default_factory=PersonalityVector)
    life_path: Optional[LifePath] = Field(default=None)
    meta: PersonaMeta = Field(default_factory=PersonaMeta)
    media_subscriptions: List[str] = Field(default_factory=list, description="Subscribed media sources")

    def to_dict(self) -> Dict[str, Any]:
        """For JSON storage and LLM context."""
        return self.model_dump()

    def to_compressed_summary(self) -> str:
        """Natural language summary for LLM prompts, including personal anchors."""
        try:
            from config.domain import get_domain_config
            currency = get_domain_config().currency
        except Exception:
            currency = "USD"
        pa = self.personal_anchors
        parts = [
            f"Age group {self.age}, {self.nationality}",
            f"living in {self.location}.",
            f"Income band: {currency} {self.income}/month.",
        ]
        if self.family.spouse:
            parts.append(f"Has spouse and {self.family.children} children.")
        parts.append(
            f"Occupation: {self.occupation}. "
            f"Car: {'yes' if self.mobility.car else 'no'}, metro usage: {self.mobility.metro_usage}."
        )
        anchor_parts = [
            f"Cuisine preference: {pa.cuisine_preference}. Diet: {pa.diet}.",
            f"Hobby: {pa.hobby}.",
        ]
        if pa.secondary_hobby:
            anchor_parts.append(f"Also enjoys {pa.secondary_hobby}.")
        anchor_parts.append(
            f"Work schedule: {pa.work_schedule}. "
            f"Dinner time: {pa.typical_dinner_time}. Commute: {pa.commute_method}. "
            f"Health focus: {pa.health_focus}."
        )
        if pa.weekend_habit and pa.weekend_habit != "relaxing at home":
            anchor_parts.append(f"Weekends: {pa.weekend_habit}.")
        if pa.pet and pa.pet != "none":
            anchor_parts.append(f"Pet: {pa.pet}.")
        parts.extend(anchor_parts)
        if self.life_path and self.life_path.biography:
            parts.append(f"Background: {self.life_path.biography}")
        return " ".join(parts)
