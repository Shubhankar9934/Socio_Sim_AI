"""Shared app state: agents, survey results, social graph, response history, event scheduler."""

from typing import Any, Dict, List

from simulation.event_queue import EventDrivenScheduler
from world.events import EventScheduler

agents_store: List[Dict[str, Any]] = []
survey_results: Dict[str, Dict[str, Any]] = {}
social_graph: Any = None
event_scheduler: EventScheduler = EventScheduler()

# Priority-queue event-driven scheduler for large-scale simulations
event_driven_scheduler: EventDrivenScheduler = EventDrivenScheduler()

# agent_id -> list of {question_id, answer, sampled_option, ...} across all surveys
response_histories: Dict[str, List[Dict[str, Any]]] = {}

# session_id -> session metadata dict for multi-question survey sessions
survey_sessions: Dict[str, Dict[str, Any]] = {}
