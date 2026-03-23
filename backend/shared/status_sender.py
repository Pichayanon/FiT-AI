"""
WebSocket status and info message sender.

Provides throttled sending of status updates, phase updates, and
informational messages over WebSocket connections to iOS clients.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Protocol

from fastapi import WebSocket


class HasStatusFields(Protocol):
    """Protocol for stream state objects that support status throttling."""

    status_tick: int
    last_status: str
    session_id: str


class HasPhaseFields(Protocol):
    """Protocol for stream state objects that support phase throttling."""

    phase_tick: int
    last_phase: str
    session_id: str


class StatusSender:
    """Send throttled status, phase, and info messages over WebSocket.

    Supports:
        - send_info: Unthrottled informational messages
        - send_status: Throttled status phase updates (e.g., NO_POSE, BUFFERING)
        - send_phase: Throttled exercise phase updates (e.g., eccentric, concentric)
    """

    def __init__(
        self,
        every_n_frames: int,
        phase_every_n: int = 2,
    ) -> None:
        """Initialize the status sender.

        Args:
            every_n_frames: Send status updates every N frames (throttle rate).
            phase_every_n: Send phase updates every N frames (throttle rate).
        """
        self.every_n_frames = max(1, int(every_n_frames))
        self.phase_every_n = max(1, int(phase_every_n))

    async def send_info(
        self,
        websocket: WebSocket,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an informational message to the client (not throttled).

        Args:
            websocket: Active WebSocket connection.
            message: Human-readable message string.
            extra: Optional additional key-value pairs to include in the payload.
        """
        payload: Dict[str, Any] = {"type": "info", "message": message}
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload))

    async def send_status(
        self,
        websocket: WebSocket,
        state: Any,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Send a status update to the client with optional throttling.

        Status messages are throttled to reduce WebSocket traffic. A status
        is sent if: force=True, the phase changed, or the throttle interval
        has elapsed.

        Args:
            websocket: Active WebSocket connection.
            state: Stream state object with status_tick, last_status, session_id.
            phase: Current status phase string (e.g., "NO_POSE", "BUFFERING").
            extra: Optional additional key-value pairs to include in the payload.
            force: If True, bypass throttling and send immediately.
        """
        state.status_tick += 1

        if not force:
            if (state.status_tick % self.every_n_frames != 0) and (
                phase == state.last_status
            ):
                return

        payload: Dict[str, Any] = {
            "type": "status",
            "state": phase,
            "session_id": state.session_id,
        }
        if extra:
            payload.update(extra)

        state.last_status = phase
        await websocket.send_text(json.dumps(payload))

    async def send_phase(
        self,
        websocket: WebSocket,
        state: Any,
        phase: str,
        force: bool = False,
    ) -> None:
        """Send a phase update to the client with optional throttling.

        Used by exercises that track eccentric/concentric phases (squat, lunges).

        Args:
            websocket: Active WebSocket connection.
            state: Stream state object with phase_tick, last_phase, session_id.
            phase: Current exercise phase string (e.g., "eccentric", "concentric").
            force: If True, bypass throttling and send immediately.
        """
        state.phase_tick += 1
        if not force and (state.phase_tick % self.phase_every_n != 0) and (
            phase == state.last_phase
        ):
            return

        payload: Dict[str, Any] = {
            "type": "phase",
            "phase": phase,
            "session_id": state.session_id,
        }

        state.last_phase = phase
        await websocket.send_text(json.dumps(payload))
