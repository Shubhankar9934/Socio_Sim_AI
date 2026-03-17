"""
WebSocket connection manager for real-time simulation and survey progress.

Channels are arbitrary string keys (e.g. ``"survey:<session_id>"``,
``"simulation"``).  Multiple clients can subscribe to the same channel
and receive broadcast messages.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections grouped by channel."""

    def __init__(self) -> None:
        self._active: Dict[str, List[WebSocket]] = {}

    async def connect(self, channel: str, ws: WebSocket) -> None:
        await ws.accept()
        self._active.setdefault(channel, []).append(ws)
        logger.info("WS connected: channel=%s", channel)

    def disconnect(self, channel: str, ws: WebSocket) -> None:
        conns = self._active.get(channel)
        if conns is not None:
            self._active[channel] = [c for c in conns if c is not ws]
            if not self._active[channel]:
                del self._active[channel]
        logger.info("WS disconnected: channel=%s", channel)

    async def broadcast(self, channel: str, data: Dict[str, Any]) -> None:
        """Send a JSON message to all clients on *channel*."""
        conns = self._active.get(channel)
        if not conns:
            return
        msg = json.dumps(data, default=str)
        stale: List[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(channel, ws)

    async def send_personal(self, ws: WebSocket, data: Dict[str, Any]) -> None:
        """Send a JSON message to a single client."""
        try:
            await ws.send_text(json.dumps(data, default=str))
        except Exception:
            pass

    @property
    def channels(self) -> List[str]:
        return list(self._active.keys())

    def subscriber_count(self, channel: str) -> int:
        return len(self._active.get(channel, []))


ws_manager = ConnectionManager()
