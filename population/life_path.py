"""
Retroactive life-path generator.

Given a persona's current demographics, deterministically generates a
plausible sequence of life milestones from age 18 to the persona's
current age -- education, career stages, relationships, relocations.

This provides biographical depth so that older agents feel lived-in
and LLM prompts can reference life experience.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from core.rng import agent_seed_from_id
from population.personas import LifePath, LifePathEntry, Persona


# ---------------------------------------------------------------------------
# Education level inference
# ---------------------------------------------------------------------------

_EDUCATION_BY_OCCUPATION: Dict[str, List[tuple[str, float]]] = {
    "professional": [("university", 0.65), ("postgraduate", 0.25), ("secondary", 0.10)],
    "managerial": [("university", 0.50), ("postgraduate", 0.40), ("secondary", 0.10)],
    "technical": [("university", 0.55), ("vocational", 0.30), ("secondary", 0.15)],
    "service": [("secondary", 0.50), ("vocational", 0.30), ("university", 0.20)],
    "other": [("secondary", 0.45), ("university", 0.35), ("vocational", 0.20)],
    "student": [("secondary", 0.60), ("university", 0.40)],
    "retired": [("secondary", 0.35), ("university", 0.40), ("postgraduate", 0.15), ("vocational", 0.10)],
}

# ---------------------------------------------------------------------------
# Career trajectory templates
# ---------------------------------------------------------------------------

_CAREER_PATHS: Dict[str, List[List[str]]] = {
    "professional": [
        ["intern", "junior analyst", "analyst", "senior analyst", "manager"],
        ["trainee", "associate", "specialist", "lead", "director"],
        ["graduate", "junior developer", "developer", "senior developer", "tech lead"],
    ],
    "managerial": [
        ["team lead", "department head", "senior manager", "director", "VP"],
        ["supervisor", "manager", "senior manager", "general manager"],
    ],
    "technical": [
        ["apprentice", "technician", "senior technician", "specialist", "lead engineer"],
        ["junior engineer", "engineer", "senior engineer", "principal engineer"],
    ],
    "service": [
        ["part-time worker", "full-time staff", "shift supervisor", "assistant manager"],
        ["crew member", "team member", "senior staff", "floor manager"],
    ],
    "other": [
        ["entry-level", "mid-level", "experienced"],
    ],
}

# ---------------------------------------------------------------------------
# Life milestone catalog (age-gated, probabilistic)
# ---------------------------------------------------------------------------

_RELATIONSHIP_MILESTONES = [
    (22, 0.15, "Started first serious relationship"),
    (25, 0.30, "Got engaged"),
    (27, 0.40, "Got married"),
    (29, 0.25, "Had first child"),
    (32, 0.20, "Had second child"),
    (38, 0.08, "Had third child"),
    (35, 0.06, "Went through a divorce"),
    (40, 0.04, "Remarried"),
]

_CAREER_MILESTONES = [
    (20, 0.60, "Got first real job"),
    (24, 0.30, "Changed careers"),
    (28, 0.25, "Got a significant promotion"),
    (33, 0.15, "Started a side business"),
    (38, 0.10, "Experienced a layoff"),
    (42, 0.20, "Reached senior position"),
    (50, 0.10, "Started mentoring younger colleagues"),
]

_LIFESTYLE_MILESTONES = [
    (19, 0.50, "Moved away from family for the first time"),
    (23, 0.30, "Relocated to a new city"),
    (30, 0.20, "Bought first home"),
    (26, 0.15, "Traveled internationally for the first time"),
    (35, 0.10, "Adopted a pet"),
    (40, 0.08, "Took up a new hobby"),
    (45, 0.10, "Started focusing more on health"),
]


def _parse_age_midpoint(age: str) -> int:
    """Return the midpoint of an age-group string."""
    cleaned = age.replace("+", "")
    parts = cleaned.split("-")
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) // 2
    return int(parts[0]) + 5  # e.g. "55+" -> 60


def _sample_weighted(pool: List[tuple[str, float]], rng: random.Random) -> str:
    items = [t[0] for t in pool]
    weights = [t[1] for t in pool]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for item, w in zip(items, weights):
        cumulative += w
        if r <= cumulative:
            return item
    return items[-1]


def generate_life_path(
    persona: Persona,
    seed: int | None = None,
) -> LifePath:
    """Generate a retroactive life history for a persona.

    Uses the persona's agent_id as the seed source so the same persona
    always gets the same life path (deterministic).
    """
    agent_hash = agent_seed_from_id(persona.agent_id, base_seed=seed)
    rng = random.Random(agent_hash)
    current_age = _parse_age_midpoint(persona.age)

    # Education
    edu_pool = _EDUCATION_BY_OCCUPATION.get(persona.occupation, _EDUCATION_BY_OCCUPATION["other"])
    education = _sample_weighted(edu_pool, rng)

    # Career trajectory
    career_templates = _CAREER_PATHS.get(persona.occupation, _CAREER_PATHS["other"])
    career_template = rng.choice(career_templates)
    career_years = max(0, current_age - (22 if education in ("university", "postgraduate") else 18))
    steps = max(1, min(len(career_template), career_years // 5 + 1))
    career_trajectory = career_template[:steps]

    # Milestones
    milestones: List[LifePathEntry] = []

    all_milestones = _RELATIONSHIP_MILESTONES + _CAREER_MILESTONES + _LIFESTYLE_MILESTONES
    for min_age, prob, description in all_milestones:
        if min_age > current_age:
            continue
        if rng.random() < prob:
            actual_age = min_age + rng.randint(0, min(3, current_age - min_age))
            milestones.append(LifePathEntry(
                age_at_event=actual_age,
                event=description,
            ))

    # Coherence adjustments: if persona has spouse/children, ensure milestones reflect that
    has_marriage = any("married" in m.event.lower() for m in milestones)
    if persona.family.spouse and not has_marriage and current_age >= 25:
        marriage_age = rng.randint(24, min(current_age, 40))
        milestones.append(LifePathEntry(age_at_event=marriage_age, event="Got married"))

    has_child = any("child" in m.event.lower() for m in milestones)
    if persona.family.children > 0 and not has_child and current_age >= 25:
        child_age = rng.randint(25, min(current_age, 42))
        milestones.append(LifePathEntry(
            age_at_event=child_age,
            event=f"Had first child",
        ))

    milestones.sort(key=lambda m: m.age_at_event)

    # Biography
    bio_parts = []
    if education in ("university", "postgraduate"):
        bio_parts.append(f"Completed {education} education")
    if len(career_trajectory) > 1:
        bio_parts.append(f"worked up from {career_trajectory[0]} to {career_trajectory[-1]}")
    elif career_trajectory:
        bio_parts.append(f"currently working as {career_trajectory[-1]}")
    if persona.family.spouse:
        bio_parts.append("married")
    if persona.family.children > 0:
        bio_parts.append(f"with {persona.family.children} {'child' if persona.family.children == 1 else 'children'}")
    key_events = [m.event.lower() for m in milestones if "reloc" in m.event.lower() or "first" in m.event.lower()]
    if key_events:
        bio_parts.append(key_events[0])

    biography = ". ".join(bio_parts).capitalize()
    if biography and not biography.endswith("."):
        biography += "."

    return LifePath(
        biography=biography,
        milestones=milestones,
        career_trajectory=career_trajectory,
        education_level=education,
    )


def stamp_life_paths(
    personas: List[Persona],
    seed: int | None = None,
) -> List[Persona]:
    """Attach retroactive life paths to all personas in a batch."""
    for p in personas:
        p.life_path = generate_life_path(p, seed=seed)
    return personas
