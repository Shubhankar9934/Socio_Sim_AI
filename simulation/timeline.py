"""
Event scheduler: queue events by day, process in order.
"""

from typing import List

from world.events import EventScheduler, SimulationEvent


def run_timeline(scheduler: EventScheduler, up_to_day: int) -> None:
    """Process all events with day <= up_to_day."""
    scheduler.process_until(up_to_day)


def schedule_metro_opening(scheduler: EventScheduler, day: int, district: str) -> None:
    """Schedule a new metro station opening."""
    scheduler.add(SimulationEvent(day=day, type="new_metro_station", district=district))


def schedule_new_service(scheduler: EventScheduler, day: int, service: str, area: str) -> None:
    """Schedule a new service (e.g. delivery app) in an area."""
    scheduler.add(SimulationEvent(
        day=day, type="new_service",
        payload={"service": service, "area": area},
        district=area,
    ))
