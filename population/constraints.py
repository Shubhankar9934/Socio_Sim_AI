"""
Declarative constraint engine for demographic plausibility.

Validates that synthetically generated personas don't contain impossible
or implausible attribute combinations (e.g. 18-year-old CEO with 4 children).
Repairs violations by adjusting the offending field to the nearest plausible value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.rng import agent_rng_pack
from population.personas import Persona


@dataclass(frozen=True)
class Constraint:
    """One declarative plausibility rule.

    ``check`` returns True when the persona is valid.
    ``repair`` mutates the persona to fix the violation.
    """

    name: str
    check: Callable[[Persona], bool]
    repair: Callable[[Persona], None]


def _parse_age_min(age: str) -> int:
    return int(age.replace("+", "").split("-")[0])


def _parse_age_max(age: str) -> int:
    if "+" in age:
        return 99
    parts = age.split("-")
    return int(parts[1]) if len(parts) > 1 else int(parts[0])


# ---------------------------------------------------------------------------
# Constraint catalog
# ---------------------------------------------------------------------------

_LOW_INCOME = ("<10k", "10-25k")
_STUDENT_OCCUPATIONS = ("other", "service")

CONSTRAINTS: List[Constraint] = [
    Constraint(
        name="young_adult_max_children",
        check=lambda p: not (p.age == "18-24" and p.family.children > 1),
        repair=lambda p: setattr(p.family, "children", min(p.family.children, 1)),
    ),
    Constraint(
        name="young_adult_no_spouse_if_single_household",
        check=lambda p: not (p.age == "18-24" and p.household_size == "1" and p.family.spouse),
        repair=lambda p: setattr(p.family, "spouse", False),
    ),
    Constraint(
        name="student_income_cap",
        check=lambda p: not (
            p.occupation in ("student",)
            and p.income not in _LOW_INCOME
        ),
        repair=lambda p: setattr(p, "income", "10-25k"),
    ),
    Constraint(
        name="children_require_min_age",
        check=lambda p: not (
            _parse_age_min(p.age) < 20 and p.family.children > 0
        ),
        repair=lambda p: setattr(p.family, "children", 0),
    ),
    Constraint(
        name="high_children_require_spouse",
        check=lambda p: not (p.family.children >= 3 and not p.family.spouse),
        repair=lambda p: setattr(p.family, "spouse", True),
    ),
    Constraint(
        name="retiree_work_schedule",
        check=lambda p: not (
            p.age == "55+"
            and p.occupation == "retired"
            and p.personal_anchors.work_schedule in ("shift work", "night shift", "long hours")
        ),
        repair=lambda p: setattr(p.personal_anchors, "work_schedule", "flexible hours"),
    ),
    Constraint(
        name="managerial_requires_experience",
        check=lambda p: not (p.age == "18-24" and p.occupation == "managerial"),
        repair=lambda p: setattr(p, "occupation", "professional"),
    ),
    Constraint(
        name="household_family_coherence",
        check=lambda p: not (
            p.household_size == "1"
            and (p.family.spouse or p.family.children > 0)
        ),
        repair=lambda p: (
            setattr(p.family, "spouse", False),
            setattr(p.family, "children", 0),
        )[-1] if True else None,
    ),
]


def validate(persona: Persona) -> List[str]:
    """Return list of violated constraint names."""
    return [c.name for c in CONSTRAINTS if not c.check(persona)]


def repair(persona: Persona) -> List[str]:
    """Fix all constraint violations in-place; return names of repaired constraints."""
    repaired = []
    for _ in range(3):
        violations = validate(persona)
        if not violations:
            break
        for c in CONSTRAINTS:
            if c.name in violations:
                c.repair(persona)
                repaired.append(c.name)
    return repaired


def validate_and_repair_all(
    personas: List[Persona],
    seed: int | None = None,
) -> List[Persona]:
    """Validate and repair all personas in a population batch."""
    for p in personas:
        repair(p)
    return personas
