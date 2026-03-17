"""
Closed-loop world feedback engine.

Aggregate agent behaviour feeds back into mutable ``WorldState``, which
in turn generates ``SimulationEvent``s injected into the scheduler.  This
converts the simulation from a one-way agent behavioural model into a
full policy / market simulator where macro outcomes influence agent
perception on the next time step.

The feedback loop:

    WorldState -> Agent Perception -> Decisions -> Macro Metrics
         ^                                              |
         |---------- apply_demand_feedback -------------|
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from world.events import SimulationEvent

logger = logging.getLogger(__name__)

def _default_services() -> Dict[str, float]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.services:
            return dict(cfg.services)
    except Exception:
        pass
    return {"primary_service": 1.0}


def _default_prices() -> Dict[str, float]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.price_levels:
            return dict(cfg.price_levels)
    except Exception:
        pass
    return {"service_fee": 1.0}


_EVENT_PRIORITY_WEIGHT: Dict[str, float] = {
    "price_change": 2.0,
    "market": 1.5,
    "infrastructure": 1.0,
}


@dataclass
class WorldState:
    """Mutable environment state that evolves from agent behaviour."""

    service_availability: Dict[str, float] = field(default_factory=lambda: _default_services())
    price_levels: Dict[str, float] = field(default_factory=lambda: _default_prices())
    infrastructure_stress: Dict[str, float] = field(default_factory=dict)
    market_competition: float = 0.5

    def apply_demand_feedback(
        self,
        macro: Any,
        current_day: int,
        max_events_per_step: int = 5,
    ) -> List[SimulationEvent]:
        """Generate world events from aggregate demand signals.

        Returns at most *max_events_per_step* events to be scheduled
        for *current_day + 1*.  When more candidates are generated
        than the cap allows, the highest-impact events are kept.
        """
        events: List[SimulationEvent] = []
        next_day = current_day + 1

        adoption = getattr(macro, "adoption_rate", 0.0)
        demand_pressure = getattr(macro, "demand_pressure", 0.0)
        utilization = getattr(macro, "service_utilization", 0.0)

        # High adoption -> service fee rises (capacity strain)
        if adoption > 0.7:
            fee_bump = 1.0 + 0.1 * (adoption - 0.7)
            magnitude = fee_bump - 1.0
            fee_key = next(iter(self.price_levels), "service_fee")
            self.price_levels[fee_key] = fee_bump
            events.append(SimulationEvent(
                day=next_day,
                type="price_change",
                payload={
                    fee_key: fee_bump,
                    "_impact": magnitude * _EVENT_PRIORITY_WEIGHT.get("price_change", 1.0),
                },
            ))

        # Low adoption -> service providers exit (state mutation only)
        if adoption < 0.2:
            for svc in self.service_availability:
                self.service_availability[svc] = max(
                    0.1, self.service_availability[svc] * 0.95,
                )

        # High demand pressure -> infrastructure stress
        if demand_pressure > 0.6:
            stress_delta = 0.05 * (demand_pressure - 0.6)
            for district in list(self.infrastructure_stress):
                self.infrastructure_stress[district] = min(
                    1.0, self.infrastructure_stress[district] + stress_delta,
                )
            events.append(SimulationEvent(
                day=next_day,
                type="infrastructure",
                payload={
                    "infra_type": "congestion",
                    "dimension_impacts": {"time_pressure": 0.01},
                    "_impact": stress_delta * _EVENT_PRIORITY_WEIGHT.get("infrastructure", 1.0),
                },
            ))

        # High utilization -> competition attracts new entrants
        if utilization > 0.8:
            self.market_competition = min(1.0, self.market_competition + 0.02)
            magnitude = utilization - 0.8
            events.append(SimulationEvent(
                day=next_day,
                type="market",
                payload={
                    "name": "new_competitor_entry",
                    "effect": {"competition": self.market_competition},
                    "dimension_impacts": {"novelty_seeking": 0.01},
                    "_impact": magnitude * _EVENT_PRIORITY_WEIGHT.get("market", 1.0),
                },
            ))

        # Low utilization -> consolidation (state mutation only)
        if utilization < 0.3:
            self.market_competition = max(0.0, self.market_competition - 0.01)

        if len(events) > max_events_per_step:
            logger.warning(
                "World feedback generated %d events, capping to %d",
                len(events), max_events_per_step,
            )
            events.sort(
                key=lambda e: e.payload.get("_impact", 0.0), reverse=True,
            )
            events = events[:max_events_per_step]

        for e in events:
            e.payload.pop("_impact", None)

        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_availability": dict(self.service_availability),
            "price_levels": dict(self.price_levels),
            "infrastructure_stress": dict(self.infrastructure_stress),
            "market_competition": self.market_competition,
        }
