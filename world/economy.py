"""
Economic simulation: income -> budget allocation.
"""

from dataclasses import dataclass
from typing import Dict

INCOME_BAND_TO_AMOUNT: Dict[str, int] = {
    "<10k": 7_500,
    "10-25k": 17_500,
    "25-50k": 37_500,
    "50k+": 75_000,
}

INCOME_BAND_TO_AED = INCOME_BAND_TO_AMOUNT


@dataclass
class BudgetAllocation:
    """Monthly budget by category (AED)."""

    food: float
    transport: float
    rent: float
    entertainment: float
    other: float


def income_to_amount(income_band: str) -> int:
    """Convert income band to approximate monthly amount in local currency."""
    return INCOME_BAND_TO_AMOUNT.get(income_band, 20_000)


income_to_aed = income_to_amount


def monthly_budget(
    income_band: str,
    rent_share: float = 0.40,
    food_share: float = 0.20,
    transport_share: float = 0.10,
    entertainment_share: float = 0.10,
) -> BudgetAllocation:
    """
    Allocate monthly income to categories. Rest goes to other.
    Shares can be modified by nationality/location in extended model.
    """
    aed = income_to_amount(income_band)
    return BudgetAllocation(
        food=aed * food_share,
        transport=aed * transport_share,
        rent=aed * rent_share,
        entertainment=aed * entertainment_share,
        other=aed * (1.0 - rent_share - food_share - transport_share - entertainment_share),
    )


def service_budget_share(
    income_band: str,
    luxury_preference: float = 0.5,
    household_size: int = 2,
) -> float:
    """Fraction of food/service budget that might go to the primary service (0-1).

    Larger households increase service propensity (convenience effect)
    but with diminishing returns past 4 members.
    """
    base = 0.3 + luxury_preference * 0.3
    if income_band == "50k+":
        base += 0.1
    elif income_band == "25-50k":
        base += 0.05
    if household_size >= 5:
        base *= 1.15
    elif household_size >= 3:
        base *= 1.08
    return min(0.8, base)


food_delivery_budget_share = service_budget_share
