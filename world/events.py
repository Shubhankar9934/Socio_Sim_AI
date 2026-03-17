"""
Event system: events change district properties, global parameters,
and agent behavioral dimensions.

Supported event types:
  - new_metro_station : enables metro access for a district
  - new_service       : adds service availability flag per district
  - price_change      : global price multipliers (delivery, fuel, etc.)
  - policy            : regulatory changes (congestion pricing, subsidies)
  - infrastructure    : road closures, developments, metro expansion
  - market            : competitor launches, discount campaigns
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from world.districts import DEFAULT_DISTRICT_PROPERTIES, DistrictProperties


@dataclass
class SimulationEvent:
    """One event to be applied at a given day."""

    day: int
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    district: Optional[str] = None

    def apply(
        self,
        district_properties: Dict[str, DistrictProperties],
        global_params: Dict[str, Any],
    ) -> None:
        """Apply this event to the world state (mutates in place)."""
        handler = _EVENT_HANDLERS.get(self.type)
        if handler is not None:
            handler(self, district_properties, global_params)


def _accumulate_belief_impacts(event: SimulationEvent, gp: Dict[str, Any]) -> None:
    """Merge belief_impacts from event payload into global params."""
    for dim, shift in event.payload.get("belief_impacts", {}).items():
        bi = gp.setdefault("event_belief_impacts", {})
        bi[dim] = bi.get(dim, 0.0) + float(shift)


def _apply_new_metro_station(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    if event.district and event.district in dp:
        d = dp[event.district]
        dp[event.district] = DistrictProperties(
            d.name, d.population_density, d.average_income,
            True, d.restaurant_density, d.parking_availability,
        )
    _accumulate_belief_impacts(event, gp)


def _apply_new_service(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    service_name = event.payload.get("service_name", "unknown")
    districts = [event.district] if event.district else list(dp.keys())
    services = gp.setdefault("available_services", {})
    for d in districts:
        services.setdefault(d, set()).add(service_name)
    # Auto-generate dimension impacts for new tech/digital services
    impacts = dict(event.payload.get("dimension_impacts", {}))
    if not impacts:
        impacts = {"technology_openness": 0.02, "novelty_seeking": 0.01}
    for dim, shift in impacts.items():
        di = gp.setdefault("event_dimension_impacts", {})
        di[dim] = di.get(dim, 0.0) + float(shift)
    _accumulate_belief_impacts(event, gp)


def _apply_price_change(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    multipliers = gp.setdefault("price_multipliers", {})
    explicit_impacts = event.payload.get("dimension_impacts", {})
    for key, value in event.payload.items():
        if key in ("dimension_impacts", "belief_impacts"):
            continue
        multipliers[key] = float(value)
    # Auto-generate: price increases raise price_sensitivity
    impacts = dict(explicit_impacts)
    if not impacts:
        avg_mult = sum(
            float(v) for k, v in event.payload.items()
            if k not in ("dimension_impacts", "belief_impacts")
        ) / max(1, len(multipliers))
        if avg_mult > 1.0:
            impacts = {"price_sensitivity": 0.03}
        elif avg_mult < 1.0:
            impacts = {"price_sensitivity": -0.02}
    for dim, shift in impacts.items():
        di = gp.setdefault("event_dimension_impacts", {})
        di[dim] = di.get(dim, 0.0) + float(shift)
    _accumulate_belief_impacts(event, gp)


def _apply_policy(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    policies = gp.setdefault("active_policies", [])
    policies.append({
        "name": event.payload.get("policy_name", "unnamed"),
        "effect": event.payload.get("effect", {}),
        "day": event.day,
        "district": event.district,
    })
    for dim, shift in event.payload.get("dimension_impacts", {}).items():
        impacts = gp.setdefault("event_dimension_impacts", {})
        impacts[dim] = impacts.get(dim, 0.0) + float(shift)
    _accumulate_belief_impacts(event, gp)


def _apply_infrastructure(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    infra_type = event.payload.get("infra_type", "general")
    if infra_type == "metro_expansion" and event.district and event.district in dp:
        d = dp[event.district]
        dp[event.district] = DistrictProperties(
            d.name, d.population_density, d.average_income,
            True, d.restaurant_density, d.parking_availability,
        )
    elif infra_type == "road_closure" and event.district and event.district in dp:
        d = dp[event.district]
        dp[event.district] = DistrictProperties(
            d.name, d.population_density, d.average_income,
            d.metro_access, d.restaurant_density, "low",
        )
    gp.setdefault("infrastructure_events", []).append({
        "type": infra_type,
        "district": event.district,
        "day": event.day,
        "payload": event.payload,
    })
    # Auto-generate dimension impacts based on infrastructure type
    explicit_impacts = event.payload.get("dimension_impacts", {})
    impacts = dict(explicit_impacts)
    if not impacts:
        if infra_type in ("metro_expansion", "new_road", "bike_lane"):
            impacts = {"convenience_seeking": 0.02}
        elif infra_type == "road_closure":
            impacts = {"convenience_seeking": -0.02, "time_pressure": 0.02}
    for dim, shift in impacts.items():
        di = gp.setdefault("event_dimension_impacts", {})
        di[dim] = di.get(dim, 0.0) + float(shift)
    _accumulate_belief_impacts(event, gp)


def _apply_market(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    market_events = gp.setdefault("market_events", [])
    market_events.append({
        "name": event.payload.get("name", "market_event"),
        "effect": event.payload.get("effect", {}),
        "day": event.day,
    })
    for dim, shift in event.payload.get("dimension_impacts", {}).items():
        impacts = gp.setdefault("event_dimension_impacts", {})
        impacts[dim] = impacts.get(dim, 0.0) + float(shift)
    _accumulate_belief_impacts(event, gp)


def _apply_cultural_shift(
    event: SimulationEvent,
    dp: Dict[str, DistrictProperties],
    gp: Dict[str, Any],
) -> None:
    """Shift cultural norms for a district (or all districts if no district specified)."""
    from world.culture import apply_cultural_shift

    shifts = event.payload.get("cultural_shifts", {})
    districts = [event.district] if event.district else list(dp.keys())
    for d in districts:
        apply_cultural_shift(d, shifts)

    updates = gp.setdefault("cultural_field_updates", [])
    updates.append({
        "districts": districts,
        "shifts": shifts,
        "day": event.day,
    })

    for dim, shift in event.payload.get("dimension_impacts", {}).items():
        impacts = gp.setdefault("event_dimension_impacts", {})
        impacts[dim] = impacts.get(dim, 0.0) + float(shift)
    _accumulate_belief_impacts(event, gp)


def _apply_media_campaign(
    event: SimulationEvent, dp: Dict, gp: Dict[str, Any]
) -> None:
    """Inject a media campaign: payload should have narrative, source, belief_impacts."""
    campaigns = gp.setdefault("active_media_campaigns", [])
    campaigns.append({
        "narrative": event.payload.get("narrative", ""),
        "source": event.payload.get("source", "government"),
        "intensity": event.payload.get("intensity", 0.5),
        "day": event.day,
    })
    _accumulate_belief_impacts(event, gp)
    for dim, shift in event.payload.get("dimension_impacts", {}).items():
        impacts = gp.setdefault("event_dimension_impacts", {})
        impacts[dim] = impacts.get(dim, 0.0) + float(shift)


def _apply_subsidy(
    event: SimulationEvent, dp: Dict, gp: Dict[str, Any]
) -> None:
    """Subsidy: reduce price level for a service."""
    amount = event.payload.get("amount", 0.1)
    target = event.payload.get("target_service", "")
    price_mult = gp.setdefault("price_multipliers", {})
    current = price_mult.get(target, 1.0)
    price_mult[target] = max(0.1, current - amount)
    impacts = gp.setdefault("event_dimension_impacts", {})
    impacts["price_sensitivity"] = impacts.get("price_sensitivity", 0.0) - amount * 0.3


def _apply_tax(
    event: SimulationEvent, dp: Dict, gp: Dict[str, Any]
) -> None:
    """Tax: increase price level for a service."""
    amount = event.payload.get("amount", 0.1)
    target = event.payload.get("target_service", "")
    price_mult = gp.setdefault("price_multipliers", {})
    current = price_mult.get(target, 1.0)
    price_mult[target] = current + amount
    impacts = gp.setdefault("event_dimension_impacts", {})
    impacts["price_sensitivity"] = impacts.get("price_sensitivity", 0.0) + amount * 0.3


def _apply_information_campaign(
    event: SimulationEvent, dp: Dict, gp: Dict[str, Any]
) -> None:
    """Shift belief dimensions directly (e.g., public health campaign)."""
    _accumulate_belief_impacts(event, gp)
    for dim, shift in event.payload.get("dimension_impacts", {}).items():
        impacts = gp.setdefault("event_dimension_impacts", {})
        impacts[dim] = impacts.get(dim, 0.0) + float(shift)


_EVENT_HANDLERS = {
    "new_metro_station": _apply_new_metro_station,
    "new_service": _apply_new_service,
    "price_change": _apply_price_change,
    "policy": _apply_policy,
    "infrastructure": _apply_infrastructure,
    "market": _apply_market,
    "cultural_shift": _apply_cultural_shift,
    "media_campaign": _apply_media_campaign,
    "subsidy": _apply_subsidy,
    "tax": _apply_tax,
    "information_campaign": _apply_information_campaign,
}


class EventScheduler:
    """Queue events by day; process in order.  Maintains global parameters."""

    def __init__(self) -> None:
        self._events: List[SimulationEvent] = []
        self._district_properties: Dict[str, DistrictProperties] = dict(DEFAULT_DISTRICT_PROPERTIES)
        self.global_params: Dict[str, Any] = {}

    def add(self, event: SimulationEvent) -> None:
        self._events.append(event)
        self._events.sort(key=lambda e: e.day)

    def process_until(self, day: int) -> None:
        """Process all events with event.day <= day.

        Per-day ephemeral impacts (dimension and belief shifts) are reset each
        call so they only reflect the events processed *this* day, preventing
        unbounded cumulative drift across simulation steps.
        """
        self.global_params.pop("event_dimension_impacts", None)
        self.global_params.pop("event_belief_impacts", None)

        to_remove = []
        for e in self._events:
            if e.day <= day:
                e.apply(self._district_properties, self.global_params)
                to_remove.append(e)
        for e in to_remove:
            self._events.remove(e)

    def district_properties(self) -> Dict[str, DistrictProperties]:
        return self._district_properties

    def get_environment(self) -> Dict[str, Any]:
        """Aggregate environment dict for factor graph consumption."""
        env: Dict[str, Any] = {}
        env["price_multipliers"] = self.global_params.get("price_multipliers", {})
        env["active_policies"] = self.global_params.get("active_policies", [])
        env["available_services"] = {
            k: list(v) if isinstance(v, set) else v
            for k, v in self.global_params.get("available_services", {}).items()
        }
        env["event_dimension_impacts"] = self.global_params.get("event_dimension_impacts", {})
        env["event_belief_impacts"] = self.global_params.get("event_belief_impacts", {})
        env["market_events"] = self.global_params.get("market_events", [])
        env["infrastructure_events"] = self.global_params.get("infrastructure_events", [])
        env["cultural_field_updates"] = self.global_params.get("cultural_field_updates", [])
        return env
