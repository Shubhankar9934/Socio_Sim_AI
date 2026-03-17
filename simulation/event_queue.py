"""
Priority-queue event-driven scheduler for large-scale agent simulations.

Unlike the day-based ``EventScheduler`` in ``world.events`` (which processes
events linearly by day), this scheduler uses a min-heap so that agents only
run when triggered.  Survey questions, social influence passes, belief
updates, and world events all become ``SimEvent`` instances on a single
unified timeline.

Backward compatible: the existing ``EventScheduler`` is unaffected.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(order=True)
class SimEvent:
    """One event on the simulation timeline.

    Ordering is by ``time`` only; all other fields are excluded from
    comparison so that events at the same time are stable-sorted by
    insertion order.
    """

    time: float
    agent_id: str = field(compare=False)
    event_type: str = field(compare=False)
    payload: Any = field(compare=False, default=None)

    def __repr__(self) -> str:
        return (
            f"SimEvent(t={self.time}, agent={self.agent_id}, "
            f"type={self.event_type})"
        )


EventHandler = Callable[["SimEvent", Dict[str, Any]], Optional[Any]]


class EventDrivenScheduler:
    """Min-heap event queue with pluggable handler dispatch.

    Usage::

        scheduler = EventDrivenScheduler()
        scheduler.register("survey_question", handle_survey)
        scheduler.schedule(SimEvent(time=1.0, agent_id="DXB_001",
                                    event_type="survey_question",
                                    payload={"question": "..."}))
        processed = await scheduler.process_all_async(agents)
    """

    def __init__(self) -> None:
        self._queue: List[SimEvent] = []
        self._current_time: float = 0.0
        self._handlers: Dict[str, EventHandler] = {}
        self._processed_count: int = 0

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._handlers[event_type] = handler

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(self, event: SimEvent) -> None:
        heapq.heappush(self._queue, event)

    def schedule_batch(self, events: List[SimEvent]) -> None:
        for e in events:
            heapq.heappush(self._queue, e)

    # ------------------------------------------------------------------
    # Synchronous processing
    # ------------------------------------------------------------------

    def process_next(self, agents: Dict[str, Any]) -> Optional[SimEvent]:
        """Pop and process the next event.  Returns the event or None."""
        if not self._queue:
            return None
        event = heapq.heappop(self._queue)
        self._current_time = event.time
        handler = self._handlers.get(event.event_type)
        if handler is not None:
            handler(event, agents)
        self._processed_count += 1
        return event

    def process_until(self, time: float, agents: Dict[str, Any]) -> int:
        """Process all events with ``event.time <= time``."""
        count = 0
        while self._queue and self._queue[0].time <= time:
            self.process_next(agents)
            count += 1
        return count

    def process_all(self, agents: Dict[str, Any]) -> int:
        """Drain the entire queue synchronously."""
        count = 0
        while self._queue:
            self.process_next(agents)
            count += 1
        return count

    # ------------------------------------------------------------------
    # Async processing (for LLM-backed handlers)
    # ------------------------------------------------------------------

    async def process_next_async(
        self,
        agents: Dict[str, Any],
    ) -> Optional[SimEvent]:
        if not self._queue:
            return None
        event = heapq.heappop(self._queue)
        self._current_time = event.time
        handler = self._handlers.get(event.event_type)
        if handler is not None:
            result = handler(event, agents)
            if hasattr(result, "__await__"):
                await result
        self._processed_count += 1
        return event

    async def process_until_async(
        self,
        time: float,
        agents: Dict[str, Any],
    ) -> int:
        """Async version of ``process_until`` -- awaits coroutine handlers."""
        count = 0
        while self._queue and self._queue[0].time <= time:
            await self.process_next_async(agents)
            count += 1
        return count

    async def process_all_async(self, agents: Dict[str, Any]) -> int:
        count = 0
        while self._queue:
            await self.process_next_async(agents)
            count += 1
        return count

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        return len(self._queue)

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def processed_count(self) -> int:
        return self._processed_count

    def peek(self) -> Optional[SimEvent]:
        """Return the next event without removing it."""
        return self._queue[0] if self._queue else None

    def clear(self) -> None:
        self._queue.clear()
        self._current_time = 0.0
        self._processed_count = 0
