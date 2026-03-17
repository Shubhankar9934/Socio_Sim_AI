"""
Real Dubai demographic distributions and conditional probability tables.
Used for synthetic population generation (IPF, Bayesian, Monte Carlo).
Sources: Dubai Statistics Center, UAE census, and urban research estimates.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Marginal distributions (percentages, sum to 100)
# ---------------------------------------------------------------------------

AGE_DISTRIBUTION: Dict[str, float] = {
    "18-24": 0.20,
    "25-34": 0.35,
    "35-44": 0.20,
    "45-54": 0.15,
    "55+": 0.10,
}

NATIONALITY_DISTRIBUTION: Dict[str, float] = {
    "Emirati": 0.12,
    "Indian": 0.35,
    "Pakistani": 0.15,
    "Filipino": 0.10,
    "Western": 0.10,
    "Egyptian": 0.05,
    "Other": 0.13,
}

INCOME_DISTRIBUTION: Dict[str, float] = {
    "<10k": 0.30,
    "10-25k": 0.40,
    "25-50k": 0.20,
    "50k+": 0.10,
}

LOCATION_DISTRIBUTION: Dict[str, float] = {
    "Dubai Marina": 0.10,
    "Jumeirah": 0.08,
    "Deira": 0.15,
    "Business Bay": 0.12,
    "Al Barsha": 0.10,
    "JLT": 0.08,
    "Downtown": 0.07,
    "Al Karama": 0.08,
    "JVC": 0.07,
    "Others": 0.15,
}

HOUSEHOLD_SIZE_DISTRIBUTION: Dict[str, float] = {
    "1": 0.25,
    "2": 0.30,
    "3-4": 0.28,
    "5+": 0.17,
}

OCCUPATION_DISTRIBUTION: Dict[str, float] = {
    "professional": 0.35,
    "service": 0.25,
    "technical": 0.20,
    "managerial": 0.12,
    "other": 0.08,
}

# ---------------------------------------------------------------------------
# Conditional: P(income | nationality)
# ---------------------------------------------------------------------------

INCOME_GIVEN_NATIONALITY: Dict[str, Dict[str, float]] = {
    "Emirati": {"<10k": 0.05, "10-25k": 0.25, "25-50k": 0.35, "50k+": 0.35},
    "Indian": {"<10k": 0.25, "10-25k": 0.45, "25-50k": 0.25, "50k+": 0.05},
    "Pakistani": {"<10k": 0.35, "10-25k": 0.45, "25-50k": 0.15, "50k+": 0.05},
    "Filipino": {"<10k": 0.40, "10-25k": 0.45, "25-50k": 0.12, "50k+": 0.03},
    "Western": {"<10k": 0.10, "10-25k": 0.25, "25-50k": 0.35, "50k+": 0.30},
    "Egyptian": {"<10k": 0.30, "10-25k": 0.45, "25-50k": 0.20, "50k+": 0.05},
    "Other": {"<10k": 0.28, "10-25k": 0.42, "25-50k": 0.22, "50k+": 0.08},
}

# ---------------------------------------------------------------------------
# Conditional: P(location | income) — higher income -> premium areas
# ---------------------------------------------------------------------------

LOCATION_GIVEN_INCOME: Dict[str, Dict[str, float]] = {
    "<10k": {
        "Dubai Marina": 0.04, "Jumeirah": 0.03, "Deira": 0.22, "Business Bay": 0.06,
        "Al Barsha": 0.12, "JLT": 0.06, "Downtown": 0.02, "Al Karama": 0.15,
        "JVC": 0.10, "Others": 0.20,
    },
    "10-25k": {
        "Dubai Marina": 0.08, "Jumeirah": 0.06, "Deira": 0.14, "Business Bay": 0.12,
        "Al Barsha": 0.11, "JLT": 0.10, "Downtown": 0.06, "Al Karama": 0.10,
        "JVC": 0.09, "Others": 0.14,
    },
    "25-50k": {
        "Dubai Marina": 0.14, "Jumeirah": 0.12, "Deira": 0.08, "Business Bay": 0.16,
        "Al Barsha": 0.10, "JLT": 0.12, "Downtown": 0.10, "Al Karama": 0.05,
        "JVC": 0.06, "Others": 0.07,
    },
    "50k+": {
        "Dubai Marina": 0.18, "Jumeirah": 0.18, "Deira": 0.04, "Business Bay": 0.18,
        "Al Barsha": 0.08, "JLT": 0.10, "Downtown": 0.14, "Al Karama": 0.02,
        "JVC": 0.04, "Others": 0.08,
    },
}

# ---------------------------------------------------------------------------
# Conditional: P(car_ownership | location) — metro areas -> less car
# ---------------------------------------------------------------------------

CAR_GIVEN_LOCATION: Dict[str, float] = {
    "Dubai Marina": 0.65,
    "Jumeirah": 0.75,
    "Deira": 0.55,
    "Business Bay": 0.60,
    "Al Barsha": 0.70,
    "JLT": 0.62,
    "Downtown": 0.58,
    "Al Karama": 0.52,
    "JVC": 0.72,
    "Others": 0.68,
}

# Metro access by district (for world model)
METRO_ACCESS_BY_LOCATION: Dict[str, bool] = {
    "Dubai Marina": True,
    "Jumeirah": False,
    "Deira": True,
    "Business Bay": True,
    "Al Barsha": True,
    "JLT": True,
    "Downtown": True,
    "Al Karama": True,
    "JVC": False,
    "Others": False,
}

# ---------------------------------------------------------------------------
# Occupation given nationality (simplified)
# ---------------------------------------------------------------------------

OCCUPATION_GIVEN_NATIONALITY: Dict[str, Dict[str, float]] = {
    "Emirati": {"professional": 0.25, "service": 0.15, "technical": 0.15, "managerial": 0.35, "other": 0.10},
    "Indian": {"professional": 0.40, "service": 0.22, "technical": 0.25, "managerial": 0.08, "other": 0.05},
    "Pakistani": {"professional": 0.30, "service": 0.30, "technical": 0.25, "managerial": 0.08, "other": 0.07},
    "Filipino": {"professional": 0.20, "service": 0.50, "technical": 0.15, "managerial": 0.05, "other": 0.10},
    "Western": {"professional": 0.35, "service": 0.15, "technical": 0.20, "managerial": 0.25, "other": 0.05},
    "Egyptian": {"professional": 0.35, "service": 0.30, "technical": 0.20, "managerial": 0.10, "other": 0.05},
    "Other": {"professional": 0.32, "service": 0.28, "technical": 0.22, "managerial": 0.10, "other": 0.08},
}


def get_all_marginals() -> Dict[str, Dict[str, float]]:
    """Return all marginal distributions for validation/synthesis."""
    return {
        "age": AGE_DISTRIBUTION,
        "nationality": NATIONALITY_DISTRIBUTION,
        "income": INCOME_DISTRIBUTION,
        "location": LOCATION_DISTRIBUTION,
        "household_size": HOUSEHOLD_SIZE_DISTRIBUTION,
        "occupation": OCCUPATION_DISTRIBUTION,
    }


def get_nationality_keys() -> List[str]:
    return list(NATIONALITY_DISTRIBUTION.keys())


def get_age_keys() -> List[str]:
    return list(AGE_DISTRIBUTION.keys())


def get_income_keys() -> List[str]:
    return list(INCOME_DISTRIBUTION.keys())


def get_location_keys() -> List[str]:
    return list(LOCATION_DISTRIBUTION.keys())
