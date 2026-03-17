"""
Income factor: economically-grounded scoring via budget allocation model.

Uses the world economy module to compute what fraction of an agent's
budget is plausible for the primary service, combined with their luxury
preference.  Falls back to a static lookup if economy module fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext

_INCOME_SCORE_FALLBACK = {
    "<10k": 0.20,
    "10-25k": 0.45,
    "25-50k": 0.70,
    "50k+": 0.90,
}


def income_factor(ctx: "DecisionContext") -> float:
    try:
        from world.economy import service_budget_share
        luxury = ctx.persona.lifestyle.luxury_preference
        hs_raw = ctx.persona.household_size
        household_size = int(hs_raw.split("-")[0]) if "-" in str(hs_raw) else int(hs_raw)
        share = service_budget_share(
            ctx.persona.income,
            luxury_preference=luxury,
            household_size=household_size,
        )
        return max(0.0, min(1.0, share))
    except Exception:
        return _INCOME_SCORE_FALLBACK.get(ctx.persona.income, 0.45)
