"""
base_session.py — Abstract base class for all exercise WebSocket streaming sessions.

Eliminates the duplicated main receive loop and start/stop handling that previously
existed in every exercise module (squat, lunges, plank, wall_sit).

Subclasses must:
    - Set self.ws, self.status, self.debug, and self.st in __init__
    - self.st must have: started: bool, session_id: str
    - Override _create_state() to return a fresh exercise StreamState
    - Override _handle_frame() with exercise-specific frame logic
    - Optionally override _initial_status, _on_start(), _on_stop(), _stop_extra()
    - Optionally override _on_connected() for welcome messages
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from shared.json_utils import parse_json
from shared.status_sender import StatusSender


class BaseWebSocketSession:
    """Abstract base for all exercise WebSocket sessions.

    Provides the common receive loop and start/stop handling so each exercise
    module only needs to implement exercise-specific logic.

    Attributes set by subclasses in __init__:
        ws: Active WebSocket connection.
        status: StatusSender instance for throttled messaging.
        debug: Debug logging flag.
        st: Session state object — must have .started (bool) and .session_id (str).
    """

    ws: WebSocket
    status: StatusSender
    debug: bool
    st: Any  # Any dataclass/object with .started and .session_id

    # ---------------------------------------------------------------
    # State factory (override in subclasses)
    # ---------------------------------------------------------------

    def _create_state(self) -> Any:
        """Return a brand-new exercise StreamState for a fresh session.

        The base class will set .started = True and .session_id automatically
        after calling this method — subclasses should NOT set those fields here.

        Example:
            def _create_state(self) -> StreamState:
                return StreamState()
        """
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Status phase (override for exercises using NO_POSE phases)
    # ---------------------------------------------------------------

    @property
    def _initial_status(self) -> str:
        """Phase string sent on session start and stop.

        Override to return a different string (e.g. "NO_POSE") for exercises
        that use named pose phases instead of "waiting".
        """
        return "waiting"

    # ---------------------------------------------------------------
    # Optional hooks (override as needed)
    # ---------------------------------------------------------------

    async def _on_connected(self) -> None:
        """Called once after ws.accept().

        Override to send dimension warnings, welcome messages, or any
        exercise-specific initialization message to the client.
        """

    async def _on_start(self) -> None:
        """Called after state is reset, before start messages are sent.

        Override to perform exercise-specific initialization such as resetting
        frame counters (self.frame_i = 0) or smoothers.
        """

    async def _on_stop(self) -> None:
        """Called after stop messages are sent.

        Override to perform exercise-specific cleanup, e.g.:
            self._reset_gate_and_buffers(reset_watchdog=True)
        """

    def _stop_extra(self) -> Dict[str, Any]:
        """Return extra fields merged into the stop info payload.

        Override to include exercise-specific summary data, e.g.:
            return {"reps": {"total": ..., "correct": ..., ...}}
        """
        return {}

    # ---------------------------------------------------------------
    # Common start / stop (do not override)
    # ---------------------------------------------------------------

    async def _handle_start(self) -> None:
        """Handle {"type": "start"} message. Reset state for a new session."""
        if getattr(self, "st", None) is not None and getattr(self.st, "started", False):
            await self._on_stop()
        self.st = self._create_state()
        self.st.started = True
        self.st.session_id = str(int(time.time() * 1000))
        print(f"[SESSION] START session_id={self.st.session_id}")
        await self._on_start()
        await self.status.send_info(
            self.ws, "Start streaming", {"session_id": self.st.session_id}
        )
        await self.status.send_status(
            self.ws, self.st, self._initial_status,
            {"reason": "session_started"}, force=True,
        )

    async def _handle_stop(self) -> None:
        """Handle {"type": "stop"} message. Finalize and clean up session."""
        print(f"[SESSION] STOP session_id={self.st.session_id}")
        self.st.started = False
        await self.status.send_info(
            self.ws, "Stop streaming",
            {"session_id": self.st.session_id, **self._stop_extra()},
        )
        await self.status.send_status(
            self.ws, self.st, self._initial_status,
            {"reason": "session_stopped"}, force=True,
        )
        await self._on_stop()

    # ---------------------------------------------------------------
    # Frame handler (override in subclasses)
    # ---------------------------------------------------------------

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        """Handle {"type": "frame", "jpeg_b64": "..."} message."""
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Common main loop (do not override)
    # ---------------------------------------------------------------

    async def run(self) -> None:
        """Common WebSocket receive loop shared by all exercise sessions.

        Flow:
            1. Accept the WebSocket connection.
            2. Call _on_connected() for exercise-specific welcome messages.
            3. Loop: receive JSON messages and dispatch to handlers.
            4. Handle WebSocketDisconnect and unexpected exceptions gracefully.

        Do not override this method — put exercise logic in _handle_frame,
        _create_state, _on_start, _on_stop, _stop_extra, and _on_connected.
        """
        await self.ws.accept()
        await self._on_connected()

        try:
            while True:
                msg = await self.ws.receive_text()
                data = parse_json(msg)
                if data is None:
                    await self.status.send_info(self.ws, "Invalid JSON")
                    continue

                mtype = data.get("type")
                if mtype == "start":
                    await self._handle_start()
                elif mtype == "stop":
                    await self._handle_stop()
                elif mtype == "frame" and self.st.started:
                    await self._handle_frame(data)

        except WebSocketDisconnect:
            print(f"[WS] disconnect session_id={self.st.session_id}")
            try:
                await self._on_stop()
            except Exception:  # pylint: disable=broad-except
                pass

        except Exception as e:  # pylint: disable=broad-except
            print(f"[WS] error: {e}")
            print(traceback.format_exc())
            try:
                await self._on_stop()
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                await self.status.send_info(self.ws, f"Server error: {e}")
            except Exception:  # pylint: disable=broad-except
                pass
