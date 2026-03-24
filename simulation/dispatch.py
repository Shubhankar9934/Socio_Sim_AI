"""
Stateless event dispatcher that decouples event routing from agent logic.

Instead of embedding event-type switch logic inside ``AgentCognitiveEngine``,
handlers are registered on this singleton dispatcher and looked up at runtime.
This keeps the cognitive engine focused on perception/decision/reasoning while
event routing remains a separate, composable concern.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.cognitive import AgentCognitiveEngine
    from simulation.event_queue import SimEvent


AsyncHandler = Callable[["SimEvent", "AgentCognitiveEngine"], Coroutine[Any, Any, Optional[Dict[str, Any]]]]


class EventDispatcher:
    """Global event-type -> handler registry with async dispatch."""

    _handlers: Dict[str, AsyncHandler] = {}

    @classmethod
    def register(cls, event_type: str, handler: AsyncHandler) -> None:
        cls._handlers[event_type] = handler

    @classmethod
    async def dispatch(
        cls,
        event: "SimEvent",
        engine: "AgentCognitiveEngine",
    ) -> Optional[Dict[str, Any]]:
        handler = cls._handlers.get(event.event_type)
        if handler is not None:
            return await handler(event, engine)
        return None

    @classmethod
    def registered_types(cls) -> list[str]:
        return list(cls._handlers)


# ------------------------------------------------------------------
# Built-in handlers
# ------------------------------------------------------------------

async def _handle_survey_question(
    event: "SimEvent", engine: "AgentCognitiveEngine",
) -> Optional[Dict[str, Any]]:
    """Survey events should set ``option_labels`` when the turn has discrete choices (any scale type)."""
    payload = event.payload or {}
    ol = payload.get("option_labels")
    if ol is not None and not isinstance(ol, list):
        ol = None
    return await engine.think(
        payload.get("question", ""),
        payload.get("question_id", ""),
        option_labels=ol,
    )


async def _handle_social_update(
    event: "SimEvent", engine: "AgentCognitiveEngine",
) -> None:
    engine.state.beliefs.apply_social_diffusion(event.payload)
    return None


async def _handle_belief_update(
    event: "SimEvent", engine: "AgentCognitiveEngine",
) -> None:
    engine.state.beliefs.update_from_answer(event.payload or {}, 0.5)
    return None


async def _handle_memory_decay(
    event: "SimEvent", engine: "AgentCognitiveEngine",
) -> None:
    engine.state.summarize_memory()
    return None


# Register built-in event types
EventDispatcher.register("survey_question", _handle_survey_question)
EventDispatcher.register("social_update", _handle_social_update)
EventDispatcher.register("belief_update", _handle_belief_update)
EventDispatcher.register("memory_decay", _handle_memory_decay)
