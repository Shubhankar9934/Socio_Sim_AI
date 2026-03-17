"""
Location / environment factor: district-aware scoring.

Uses the pre-computed *location_quality* value (0-1) from
world.districts.location_quality_for_satisfaction, which encodes
parking availability, metro access, and area density.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.factor_graph import DecisionContext


def location_factor(ctx: "DecisionContext") -> float:
    return max(0.0, min(1.0, ctx.location_quality))
