"""
WebSocket endpoints for real-time survey and simulation progress streaming.

Clients connect to a channel and receive JSON messages as events occur:
  - ``/ws/survey/{session_id}`` -- per-round progress during multi-question surveys
  - ``/ws/simulation``          -- simulation step updates + client commands
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.websocket import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/survey/{session_id}")
async def survey_progress_ws(websocket: WebSocket, session_id: str) -> None:
    """Stream round-by-round survey progress for a session.

    The server pushes messages of the form::

        {
            "event": "round_complete",
            "round_idx": 0,
            "total_rounds": 20,
            "question": "...",
            "n_responses": 500,
            "elapsed_seconds": 3.2
        }
    """
    channel = f"survey:{session_id}"
    await ws_manager.connect(channel, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(channel, websocket)


@router.websocket("/ws/simulation")
async def simulation_ws(websocket: WebSocket) -> None:
    """Bidirectional simulation channel.

    Server pushes step updates; client can send commands::

        {"action": "inject_event", "day": 5, "type": "price_change", ...}
        {"action": "pause"}
        {"action": "resume"}
    """
    channel = "simulation"
    await ws_manager.connect(channel, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send_personal(
                    websocket, {"error": "invalid JSON"},
                )
                continue

            action = data.get("action")
            if action == "inject_event":
                from api.state import event_scheduler
                from world.events import SimulationEvent

                event = SimulationEvent(
                    day=int(data.get("day", 0)),
                    type=str(data.get("type", "market")),
                    payload=data.get("payload", {}),
                    district=data.get("district"),
                )
                event_scheduler.add(event)
                await ws_manager.send_personal(
                    websocket,
                    {"ack": "event_scheduled", "type": event.type, "day": event.day},
                )
            elif action == "status":
                from api.state import agents_store

                await ws_manager.send_personal(
                    websocket,
                    {"population_size": len(agents_store)},
                )
            else:
                await ws_manager.send_personal(
                    websocket,
                    {"error": f"unknown action: {action}"},
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(channel, websocket)
