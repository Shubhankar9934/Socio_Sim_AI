"""
Population synthesis: Monte Carlo, IPF, and Bayesian conditional sampling.
Generates synthetic personas matching demographic distributions loaded from
the domain config layer.

Includes age-aware family generation and age-conditional household sizing
to prevent implausible combinations (e.g. 18-year-old with 5 children).
"""

import random
from typing import Dict, List, Literal

import numpy as np

from config.demographics import get_demographics
from population.personas import (
    FamilyStructure,
    LifestyleCoefficients,
    MobilityProfile,
    Persona,
    PersonalAnchors,
    PersonaMeta,
)


def _weighted_choice(dist: Dict[str, float], rng: random.Random | None = None) -> str:
    """Sample one key from a distribution (values = probabilities)."""
    items = list(dist.keys())
    weights = list(dist.values())
    total = sum(weights)
    if total <= 0:
        r = rng or random.Random()
        return r.choice(items)
    probs = [w / total for w in weights]
    return np.random.choice(items, p=probs)


_CATEGORICAL_NOISE_STD = 0.05


def _noisy_weighted_choice(
    dist: Dict[str, float],
    rng: random.Random | None = None,
) -> str:
    """Sample with small Gaussian noise injected into the distribution.

    Prevents synthetic clustering by jittering probabilities slightly
    before sampling, so that repeated draws from the same conditional
    distribution produce more varied outcomes.
    """
    items = list(dist.keys())
    weights = np.array(list(dist.values()), dtype=np.float64)
    r = rng or random.Random()
    noise = np.array([r.gauss(0, _CATEGORICAL_NOISE_STD) for _ in weights])
    weights = np.clip(weights + noise, 0.01, None)
    weights = weights / weights.sum()
    return str(np.random.choice(items, p=weights))


def _sample_income_given_nationality(nationality: str, rng: random.Random | None = None) -> str:
    demo = get_demographics()
    dist = demo.income_given_nationality.get(nationality, demo.income)
    return _noisy_weighted_choice(dist, rng)


def _sample_location_given_income(income: str, rng: random.Random | None = None) -> str:
    demo = get_demographics()
    dist = demo.location_given_income.get(income, demo.location)
    return _noisy_weighted_choice(dist, rng)


def _sample_occupation_given_nationality(nationality: str, rng: random.Random | None = None) -> str:
    demo = get_demographics()
    dist = demo.occupation_given_nationality.get(nationality, demo.occupation)
    return _noisy_weighted_choice(dist, rng)


# ---------------------------------------------------------------------------
# Lifestyle coefficient noise — tripled from +/-0.05 to +/-0.15
# ---------------------------------------------------------------------------

_LIFESTYLE_NOISE_RANGE = 0.30  # uniform noise width (centered → +/- 0.15)


def _lifestyle_from_demographics(
    income: str,
    location: str,
    nationality: str,
    rng: random.Random,
) -> LifestyleCoefficients:
    """Derive lifestyle coefficients from demographics with significant noise."""
    try:
        from config.domain import get_domain_config
        premium_areas = tuple(get_domain_config().premium_areas)
    except Exception:
        premium_areas = ()
    income_high = income in ("25-50k", "50k+")
    is_premium = location in premium_areas
    western = nationality == "Western" or nationality == "Emirati"

    luxury = 0.5 + (0.2 if income_high else 0) + (0.1 if is_premium else 0) + (0.1 if western else 0)
    tech = 0.5 + (0.2 if income_high else 0) + (0.1 if western else 0)
    dining = 0.5 + (0.15 if income_high else 0) + (0.1 if is_premium else 0)
    convenience = 0.5 + (0.2 if income_high else 0)
    price_sens = 0.5 - (0.2 if income_high else 0) + (0.1 if not western else 0)
    service = 0.5 + (0.2 if income_high else 0) + (0.1 if is_premium else 0)

    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x + (rng.random() - 0.5) * _LIFESTYLE_NOISE_RANGE))

    return LifestyleCoefficients(
        luxury_preference=clamp(luxury),
        tech_adoption=clamp(tech),
        dining_out=clamp(dining),
        convenience_preference=clamp(convenience),
        price_sensitivity=clamp(price_sens),
        primary_service_preference=clamp(service),
    )


def _mobility_from_location(location: str, rng: random.Random) -> MobilityProfile:
    """Car and metro usage from demographics config."""
    demo = get_demographics()
    car_prob = demo.car_given_location.get(location, 0.65)
    car = rng.random() < car_prob
    has_metro = demo.metro_access_by_location.get(location, False)
    metro_usage = "frequent" if not car and has_metro else (
        "occasional" if rng.random() < 0.4 else "rare"
    )
    return MobilityProfile(car=car, metro_usage=metro_usage)


# ---------------------------------------------------------------------------
# Personal anchors: demographic-conditional pools for narrative diversity
# ---------------------------------------------------------------------------

def _load_cuisine_and_diet() -> tuple:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        cuisine = cfg.cuisine_by_nationality if cfg.cuisine_by_nationality else {}
        diet_raw = cfg.diet_pool if cfg.diet_pool else {}
        diet = {k: [(item[0], item[1]) for item in v] for k, v in diet_raw.items()}
        return cuisine, diet
    except Exception:
        return {}, {}


_cuisine_cfg, _diet_cfg = _load_cuisine_and_diet()

_CUISINE_BY_NATIONALITY: Dict[str, List[str]] = _cuisine_cfg if _cuisine_cfg else {
    "Other": ["Mixed", "International", "Continental"],
}

_DIET_POOL: Dict[str, List[tuple[str, float]]] = _diet_cfg if _diet_cfg else {
    "Other": [("no restriction", 0.7), ("vegetarian", 0.15), ("low-carb", 0.15)],
}

# Weighted hobby pool: mundane hobbies dominate (like real survey data)
_HOBBY_POOL_WEIGHTED: List[tuple[str, float]] = [
    # Mundane/common hobbies — high weight (real people are boring)
    ("watching TV", 0.10), ("Netflix", 0.06), ("nothing really", 0.04),
    ("sleeping", 0.03), ("hanging out with friends", 0.04),
    ("scrolling social media", 0.05), ("shopping", 0.04),
    ("walking", 0.04), ("watching cricket", 0.03), ("watching football", 0.03),
    ("cooking", 0.05), ("reading", 0.04), ("gym", 0.05),
    ("gaming", 0.04), ("music", 0.03),
    # Moderate hobbies
    ("cycling", 0.03), ("swimming", 0.03), ("running", 0.03),
    ("yoga", 0.02), ("photography", 0.02), ("cricket", 0.02),
    ("football", 0.02), ("travel", 0.02),
    # Niche hobbies — low weight
    ("tennis", 0.01), ("art", 0.01), ("gardening", 0.01),
    ("fishing", 0.01), ("chess", 0.01), ("dancing", 0.01),
    ("hiking", 0.01), ("padel", 0.005), ("martial arts", 0.005),
    ("calligraphy", 0.003), ("surfing", 0.003), ("horse riding", 0.002),
]
# Legacy flat pool for backward compat
_HOBBY_POOL = [h for h, _ in _HOBBY_POOL_WEIGHTED]

_WORK_SCHEDULE_BY_OCCUPATION: Dict[str, List[tuple[str, float]]] = {
    "professional": [("9-to-5", 0.5), ("9-to-6", 0.25), ("flexible hours", 0.15), ("remote", 0.1)],
    "service": [("shift work", 0.5), ("split shift", 0.2), ("early morning", 0.15), ("night shift", 0.15)],
    "technical": [("9-to-5", 0.35), ("flexible hours", 0.3), ("remote", 0.2), ("9-to-6", 0.15)],
    "managerial": [("9-to-6", 0.4), ("long hours", 0.3), ("flexible hours", 0.2), ("9-to-5", 0.1)],
    "other": [("9-to-5", 0.4), ("shift work", 0.25), ("flexible hours", 0.2), ("part-time", 0.15)],
}

_DINNER_TIMES = ["7 PM", "7:30 PM", "8 PM", "8:30 PM", "9 PM", "9:30 PM", "10 PM", "10:30 PM"]

_HEALTH_FOCUS_POOL = [
    ("moderate", 0.30), ("relaxed", 0.20), ("active", 0.18),
    ("don't think about it", 0.10), ("very health-conscious", 0.10),
    ("trying to be healthier", 0.07), ("fitness-focused", 0.05),
]


def _sample_weighted_tuples(pool: List[tuple[str, float]], rng: random.Random) -> str:
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


def _personal_anchors_from_demographics(
    nationality: str,
    occupation: str,
    income: str,
    location: str,
    mobility: MobilityProfile,
    rng: random.Random,
) -> PersonalAnchors:
    cuisine_pool = _CUISINE_BY_NATIONALITY.get(nationality, _CUISINE_BY_NATIONALITY["Other"])
    cuisine = rng.choice(cuisine_pool)

    diet_pool = _DIET_POOL.get(nationality, _DIET_POOL["Other"])
    diet = _sample_weighted_tuples(diet_pool, rng)

    hobby = _sample_weighted_tuples(_HOBBY_POOL_WEIGHTED, rng)

    sched_pool = _WORK_SCHEDULE_BY_OCCUPATION.get(occupation, _WORK_SCHEDULE_BY_OCCUPATION["other"])
    work_schedule = _sample_weighted_tuples(sched_pool, rng)

    try:
        from config.domain import get_domain_config
        _late_nats = set(get_domain_config().late_dinner_nationalities)
    except Exception:
        _late_nats = set()
    if nationality in _late_nats:
        dinner_time = rng.choice(_DINNER_TIMES[3:])  # 8:30 PM onwards
    else:
        dinner_time = rng.choice(_DINNER_TIMES)

    if mobility.metro_usage == "frequent":
        commute = rng.choice(["metro", "metro", "metro + walk", "metro + bus"])
    elif mobility.car:
        commute = _sample_weighted_tuples([
            ("car", 0.65), ("car + metro", 0.10), ("ride-hailing", 0.10),
            ("carpool", 0.05), ("company transport", 0.05), ("cycling", 0.05),
        ], rng)
    else:
        commute = _sample_weighted_tuples([
            ("bus", 0.30), ("taxi", 0.20), ("walk", 0.15),
            ("ride-hailing", 0.15), ("company transport", 0.10), ("cycling", 0.10),
        ], rng)

    health_focus = _sample_weighted_tuples(_HEALTH_FOCUS_POOL, rng)

    return PersonalAnchors(
        cuisine_preference=cuisine,
        diet=diet,
        hobby=hobby,
        work_schedule=work_schedule,
        typical_dinner_time=dinner_time,
        commute_method=commute,
        health_focus=health_focus,
    )


# ---------------------------------------------------------------------------
# Age-aware family generation with cultural multiplier
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Conditional Probability Table: P(family | age, household_size)
# Replaces heuristic guards with a principled generative model.
# Each entry: (spouse_prob, max_children)
# ---------------------------------------------------------------------------

_FAMILY_CPT: Dict[tuple, tuple] = {
    # 18-24: young adults; large households = roommates/shared housing
    ("18-24", "1"):   (0.00, 0),
    ("18-24", "2"):   (0.10, 0),
    ("18-24", "3-4"): (0.05, 0),  # shared flat, not family
    ("18-24", "5+"):  (0.00, 0),  # shared accommodation
    # 25-34: early career, starting families
    ("25-34", "1"):   (0.00, 0),
    ("25-34", "2"):   (0.65, 1),
    ("25-34", "3-4"): (0.80, 2),
    ("25-34", "5+"):  (0.85, 3),
    # 35-44: established families
    ("35-44", "1"):   (0.00, 0),
    ("35-44", "2"):   (0.80, 1),
    ("35-44", "3-4"): (0.90, 3),
    ("35-44", "5+"):  (0.95, 4),
    # 45-54: mature families
    ("45-54", "1"):   (0.00, 0),
    ("45-54", "2"):   (0.85, 1),
    ("45-54", "3-4"): (0.90, 3),
    ("45-54", "5+"):  (0.95, 5),
    # 55+: empty-nesters or large extended families
    ("55+", "1"):     (0.00, 0),
    ("55+", "2"):     (0.90, 0),
    ("55+", "3-4"):   (0.90, 2),
    ("55+", "5+"):    (0.90, 4),
}

def _load_cultural_family_multiplier() -> Dict[str, float]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.cultural_family_multiplier:
            return cfg.cultural_family_multiplier
    except Exception:
        pass
    return {"Other": 1.0}


CULTURAL_FAMILY_MULTIPLIER: Dict[str, float] = _load_cultural_family_multiplier()


def _parse_age_min(age: str) -> int:
    """Extract the lower bound from an age-group string like '18-24' or '55+'."""
    cleaned = age.replace("+", "")
    return int(cleaned.split("-")[0])


def _family_from_household(
    household_size: str,
    age: str,
    nationality: str,
    rng: random.Random,
) -> FamilyStructure:
    """Generate family structure from conditional probability table.

    Directly samples spouse and children from P(family | age, household_size)
    with a cultural multiplier on max_children. No post-hoc reinterpretation.
    """
    cfg = _FAMILY_CPT.get((age, household_size))
    if cfg is None:
        return FamilyStructure(spouse=False, children=0)

    spouse_prob, max_children = cfg
    cultural_mult = CULTURAL_FAMILY_MULTIPLIER.get(nationality, 1.0)
    max_children = max(0, int(max_children * cultural_mult))

    has_spouse = rng.random() < spouse_prob
    children = rng.randint(0, max_children) if max_children > 0 else 0

    return FamilyStructure(spouse=has_spouse, children=children)


# ---------------------------------------------------------------------------
# Age-conditional household size distribution (Fix 3B)
# ---------------------------------------------------------------------------

HOUSEHOLD_GIVEN_AGE: Dict[str, Dict[str, float]] = {
    "18-24": {"1": 0.55, "2": 0.30, "3-4": 0.12, "5+": 0.03},
    "25-34": {"1": 0.20, "2": 0.30, "3-4": 0.30, "5+": 0.20},
    "35-44": {"1": 0.10, "2": 0.20, "3-4": 0.35, "5+": 0.35},
    "45-54": {"1": 0.10, "2": 0.20, "3-4": 0.30, "5+": 0.40},
    "55+":   {"1": 0.15, "2": 0.30, "3-4": 0.30, "5+": 0.25},
}


def _sample_household_size(age: str) -> str:
    """Sample household size conditionally on age group."""
    demo = get_demographics()
    dist = HOUSEHOLD_GIVEN_AGE.get(age, demo.household_size)
    return _weighted_choice(dist)


# ---------------------------------------------------------------------------
# Monte Carlo: independent weighted sampling from marginals
# ---------------------------------------------------------------------------


def generate_monte_carlo(
    n: int,
    seed: int | None = None,
    id_prefix: str | None = None,
) -> List[Persona]:
    """Generate n personas via Monte Carlo (marginals only)."""
    demo = get_demographics()
    if id_prefix is None:
        try:
            from config.domain import get_domain_config
            id_prefix = get_domain_config().id_prefix
        except Exception:
            id_prefix = "AGT"
    rng = random.Random(seed)
    personas: List[Persona] = []
    for i in range(n):
        agent_id = f"{id_prefix}_{i:04d}"
        age = _weighted_choice(demo.age)
        nationality = _weighted_choice(demo.nationality)
        income = _weighted_choice(demo.income)
        location = _weighted_choice(demo.location)
        household_size = _sample_household_size(age)
        occupation = _weighted_choice(demo.occupation)

        family = _family_from_household(household_size, age, nationality, rng)
        mobility = _mobility_from_location(location, rng)
        lifestyle = _lifestyle_from_demographics(income, location, nationality, rng)
        anchors = _personal_anchors_from_demographics(
            nationality, occupation, income, location, mobility, rng,
        )

        p = Persona(
            agent_id=agent_id,
            age=age,
            nationality=nationality,
            income=income,
            location=location,
            occupation=occupation,
            household_size=household_size,
            family=family,
            mobility=mobility,
            lifestyle=lifestyle,
            personal_anchors=anchors,
            meta=PersonaMeta(synthesis_method="monte_carlo", generation_seed=seed + i if seed is not None else None),
        )
        personas.append(p)
    return personas


# ---------------------------------------------------------------------------
# Bayesian: conditional chain nationality -> income -> location -> ...
# ---------------------------------------------------------------------------


def generate_bayesian(
    n: int,
    seed: int | None = None,
    id_prefix: str | None = None,
) -> List[Persona]:
    """Generate n personas via Bayesian conditional sampling."""
    demo = get_demographics()
    if id_prefix is None:
        try:
            from config.domain import get_domain_config
            id_prefix = get_domain_config().id_prefix
        except Exception:
            id_prefix = "AGT"
    rng = random.Random(seed)
    personas = []
    for i in range(n):
        agent_id = f"{id_prefix}_{i:04d}"
        age = _weighted_choice(demo.age)
        nationality = _weighted_choice(demo.nationality)
        income = _sample_income_given_nationality(nationality, rng)
        location = _sample_location_given_income(income, rng)
        household_size = _sample_household_size(age)
        occupation = _sample_occupation_given_nationality(nationality, rng)

        family = _family_from_household(household_size, age, nationality, rng)
        mobility = _mobility_from_location(location, rng)
        lifestyle = _lifestyle_from_demographics(income, location, nationality, rng)
        anchors = _personal_anchors_from_demographics(
            nationality, occupation, income, location, mobility, rng,
        )

        p = Persona(
            agent_id=agent_id,
            age=age,
            nationality=nationality,
            income=income,
            location=location,
            occupation=occupation,
            household_size=household_size,
            family=family,
            mobility=mobility,
            lifestyle=lifestyle,
            personal_anchors=anchors,
            meta=PersonaMeta(synthesis_method="bayesian", generation_seed=seed + i if seed is not None else None),
        )
        personas.append(p)
    return personas


# ---------------------------------------------------------------------------
# IPF: Iterative Proportional Fitting for joint (age x nationality) then fill
# ---------------------------------------------------------------------------

# We implement a simplified IPF: fit age x nationality matrix to marginals,
# then sample from that joint and fill rest via Bayesian.


def _ipf_2d(
    matrix: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Iterative proportional fitting for 2D matrix."""
    m = matrix.copy()
    for _ in range(max_iter):
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        m = m * (row_target.reshape(-1, 1) / row_sums)
        col_sums = m.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        m = m * (col_target.reshape(1, -1) / col_sums)
        if np.allclose(m.sum(axis=1), row_target) and np.allclose(m.sum(axis=0), col_target):
            break
    return m


def generate_ipf(
    n: int,
    seed: int | None = None,
    id_prefix: str | None = None,
) -> List[Persona]:
    """
    Generate n personas using IPF for age x nationality joint distribution,
    then Bayesian for income/location/occupation.
    """
    demo = get_demographics()
    if id_prefix is None:
        try:
            from config.domain import get_domain_config
            id_prefix = get_domain_config().id_prefix
        except Exception:
            id_prefix = "AGT"
    rng = random.Random(seed)
    age_keys = demo.get_age_keys()
    nat_keys = demo.get_nationality_keys()
    row_target = np.array([demo.age[k] for k in age_keys])
    col_target = np.array([demo.nationality[k] for k in nat_keys])
    # Uniform start
    matrix = np.ones((len(age_keys), len(nat_keys))) / (len(age_keys) * len(nat_keys))
    matrix = _ipf_2d(matrix, row_target, col_target)
    # Normalize to probabilities
    matrix = matrix / matrix.sum()

    personas = []
    for i in range(n):
        agent_id = f"{id_prefix}_{i:04d}"
        # Sample (age, nationality) from joint
        flat = matrix.flatten()
        idx = np.random.choice(len(flat), p=flat)
        ai, ni = np.unravel_index(idx, matrix.shape)
        age = age_keys[ai]
        nationality = nat_keys[ni]

        income = _sample_income_given_nationality(nationality, rng)
        location = _sample_location_given_income(income, rng)
        household_size = _sample_household_size(age)
        occupation = _sample_occupation_given_nationality(nationality, rng)

        family = _family_from_household(household_size, age, nationality, rng)
        mobility = _mobility_from_location(location, rng)
        lifestyle = _lifestyle_from_demographics(income, location, nationality, rng)
        anchors = _personal_anchors_from_demographics(
            nationality, occupation, income, location, mobility, rng,
        )

        p = Persona(
            agent_id=agent_id,
            age=age,
            nationality=nationality,
            income=income,
            location=location,
            occupation=occupation,
            household_size=household_size,
            family=family,
            mobility=mobility,
            lifestyle=lifestyle,
            personal_anchors=anchors,
            meta=PersonaMeta(synthesis_method="ipf", generation_seed=seed + i if seed is not None else None),
        )
        personas.append(p)
    return personas


# ---------------------------------------------------------------------------
# Behavioral archetype assignment (rule-based)
# ---------------------------------------------------------------------------

_ARCHETYPE_RULES = [
    ("busy_professional", lambda p: p.occupation in ("professional", "managerial") and p.lifestyle.convenience_preference > 0.55),
    ("family_cook", lambda p: p.family.children >= 1 and p.household_size in ("3-4", "5+") and p.lifestyle.primary_service_preference < 0.5),
    ("health_focused", lambda p: p.personal_anchors.health_focus in ("active", "very health-conscious", "fitness-focused")),
    ("budget_conscious", lambda p: p.income in ("<10k", "10-25k") and p.lifestyle.price_sensitivity > 0.5),
    ("student", lambda p: p.age == "18-24" and p.occupation in ("other", "service") and p.income == "<10k"),
    ("convenience_seeker", lambda p: p.lifestyle.convenience_preference > 0.65 and p.lifestyle.primary_service_preference > 0.55),
    ("social_foodie", lambda p: p.lifestyle.dining_out > 0.60),
    ("young_explorer", lambda p: p.age in ("18-24", "25-34") and p.lifestyle.tech_adoption > 0.55),
    ("traditionalist", lambda p: p.personal_anchors.diet in ("halal", "vegetarian") and p.lifestyle.primary_service_preference < 0.50),
]


def _assign_archetype(persona: Persona) -> str:
    """Return the first matching archetype label, or 'default'."""
    for name, rule in _ARCHETYPE_RULES:
        try:
            if rule(persona):
                return name
        except Exception:
            continue
    return "default"


def _stamp_archetypes(personas: List[Persona]) -> List[Persona]:
    """Assign behavioral archetype to each persona in-place."""
    for p in personas:
        p.personal_anchors.archetype = _assign_archetype(p)
    return personas


def _stamp_segments(personas: List[Persona], seed: int | None = None) -> List[Persona]:
    """Assign population segments for multimodal behavioral clustering."""
    from population.segments import assign_segment

    rng = np.random.default_rng(seed)
    for p in personas:
        p.meta.population_segment = assign_segment(
            p.age, p.income, p.location, rng,
        )
    return personas


def _stamp_narrative_styles(personas: List[Persona], seed: int | None = None) -> List[Persona]:
    """Assign persistent narrative style profiles from demographics."""
    from agents.narrative import derive_narrative_style_profile

    rng = random.Random(seed)
    for p in personas:
        profile = derive_narrative_style_profile(
            p.age, p.income, p.occupation, p.nationality, rng,
        )
        p.personal_anchors.narrative_style.verbosity = profile.verbosity
        p.personal_anchors.narrative_style.preferred_tone = profile.preferred_tone
        p.personal_anchors.narrative_style.preferred_style = profile.preferred_style
        p.personal_anchors.narrative_style.slang_level = profile.slang_level
        p.personal_anchors.narrative_style.grammar_quality = profile.grammar_quality
    return personas


def generate_population(
    n: int,
    method: Literal["monte_carlo", "bayesian", "ipf"] = "bayesian",
    seed: int | None = None,
    id_prefix: str | None = None,
) -> List[Persona]:
    """Unified entry point for population synthesis."""
    if method == "monte_carlo":
        personas = generate_monte_carlo(n, seed=seed, id_prefix=id_prefix)
    elif method == "bayesian":
        personas = generate_bayesian(n, seed=seed, id_prefix=id_prefix)
    elif method == "ipf":
        personas = generate_ipf(n, seed=seed, id_prefix=id_prefix)
    else:
        raise ValueError(f"Unknown method: {method}")
    personas = _stamp_archetypes(personas)
    personas = _stamp_segments(personas, seed=seed)
    personas = _stamp_narrative_styles(personas, seed=seed)
    personas = _stamp_media_subscriptions(personas, seed=seed)
    return personas


def _stamp_media_subscriptions(personas: List[Persona], seed: int | None = None) -> List[Persona]:
    """Assign media diet based on belief-aligned homophilic selection."""
    from agents.belief_network import init_beliefs_from_persona
    from agents.personality import personality_from_persona
    from media.sources import assign_media_diet

    rng = np.random.default_rng(seed)
    for p in personas:
        traits = personality_from_persona(p)
        beliefs = init_beliefs_from_persona(p, traits)
        p.media_subscriptions = assign_media_diet(beliefs.to_vector(), rng=rng)
    return personas
