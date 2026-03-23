"""
base_session.py — Abstract base class for all exercise WebSocket streaming sessions.

Eliminates the duplicated main receive loop and start/stop handling that previously
existed in every exercise module (squat, lunges, plank, wall_sit).

Subclasses must:
    - Set self.websocket, self.status_sender, self.debug, and self.state in __init__
    - self.state must have: started: bool, session_id: str
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
        websocket: Active WebSocket connection.
        status_sender: StatusSender instance for throttled messaging.
        debug: Debug logging flag.
        state: Session state object — must have .started (bool) and .session_id (str).
    """

    websocket: WebSocket
    status_sender: StatusSender
    debug: bool
    state: Any  # Any dataclass/object with .started and .session_id

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
        """Called once after websocket.accept().

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
        if getattr(self, "state", None) is not None and getattr(self.state, "started", False):
            await self._on_stop()
        self.state = self._create_state()
        self.state.started = True
        self.state.session_id = str(int(time.time() * 1000))
        print(f"[SESSION] START session_id={self.state.session_id}")
        await self._on_start()
        await self.status_sender.send_info(
            self.websocket,
            "Start streaming",
            {"session_id": self.state.session_id},
        )
        await self.status_sender.send_status(
            self.websocket,
            self.state,
            self._initial_status,
            {"reason": "session_started"}, force=True,
        )

    async def _handle_stop(self) -> None:
        """Handle {"type": "stop"} message. Finalize and clean up session."""
        print(f"[SESSION] STOP session_id={self.state.session_id}")
        self.state.started = False
        await self.status_sender.send_info(
            self.websocket,
            "Stop streaming",
            {"session_id": self.state.session_id, **self._stop_extra()},
        )
        await self.status_sender.send_status(
            self.websocket,
            self.state,
            self._initial_status,
            {"reason": "session_stopped"}, force=True,
        )
        await self._on_stop()

    # ---------------------------------------------------------------
    # Shared watchdog helpers
    # (used by all exercise sessions that have darkness/no-pose detection)
    # ---------------------------------------------------------------

    async def _update_dark_watchdog(
        self,
        too_dark: bool,
        brightness_mean: float,
    ) -> None:
        """Send a light-adjust warning once darkness persists beyond the threshold.

        Subclasses must set self.dark_adjust_seconds and self.dark_brightness_th in __init__.
        self.state must have dark_since (Optional[float]) and dark_alerted (bool) fields.
        """
        now = time.time()

        if not too_dark:
            self.state.dark_since = None
            self.state.dark_alerted = False
            return

        if self.state.dark_since is None:
            self.state.dark_since = now
            self.state.dark_alerted = False
            return

        if self.state.dark_alerted:
            return

        if now - self.state.dark_since >= self.dark_adjust_seconds:
            await self.status_sender.send_info(
                self.websocket,
                "Please adjust your lights.",
                {
                    "brightness_mean": round(brightness_mean, 1),
                    "brightness_th": self.dark_brightness_th,
                },
            )
            self.state.dark_alerted = True

    async def _handle_no_pose(
        self,
        too_dark: bool,
        brightness_mean: float,
        gate_debug: Dict[str, Any],
    ) -> None:
        """Handle frames where pose detection or side-view gate fails.

        Manages the NO_POSE watchdog timer, sends a camera-adjust message
        after no_pose_adjust_seconds of consecutive no-pose frames, then
        resets gate/buffers and sends a NO_POSE status.

        Subclasses must set no_pose_adjust_seconds, dark_brightness_th,
        phase_no_pose, ready_streak_n, window_frames in __init__.
        self.state must have no_pose_since, no_pose_alerted, dark_since,
        dark_alerted, and the gate/buffer fields cleared by _reset_gate_and_buffers.
        """
        now = time.time()

        if self.state.no_pose_since is None:
            self.state.no_pose_since = now
            self.state.no_pose_alerted = False

        if (
            (not self.state.no_pose_alerted)
            and (now - self.state.no_pose_since >= self.no_pose_adjust_seconds)
            and (not too_dark)
        ):
            await self.status_sender.send_info(
                self.websocket,
                "Adjust your camera to see your full body",
                {
                    "brightness_mean": round(brightness_mean, 1),
                    "brightness_th": self.dark_brightness_th,
                },
            )
            self.state.no_pose_alerted = True

        self._reset_gate_and_buffers(reset_watchdog=False)

        await self.status_sender.send_status(
            self.websocket,
            self.state,
            self.phase_no_pose,
            {
                "chosen_side": None,
                "ready_streak": 0,
                "needed_streak": self.ready_streak_n,
                "window_fill": 0,
                "window_size": self.window_frames,
                "too_dark": too_dark,
                "brightness_mean": round(brightness_mean, 1),
                "brightness_th": self.dark_brightness_th,
                "debug": gate_debug if self.debug else None,
            },
        )

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
        await self.websocket.accept()
        await self._on_connected()

        try:
            while True:
                message_text = await self.websocket.receive_text()
                data = parse_json(message_text)
                if data is None:
                    await self.status_sender.send_info(self.websocket, "Invalid JSON")
                    continue

                message_type = data.get("type")
                if message_type == "start":
                    await self._handle_start()
                elif message_type == "stop":
                    await self._handle_stop()
                elif message_type == "frame" and self.state.started:
                    await self._handle_frame(data)

        except WebSocketDisconnect:
            print(f"[WS] disconnect session_id={self.state.session_id}")
            try:
                await self._on_stop()
            except Exception:  # pylint: disable=broad-except
                pass

        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WS] error: {exc}")
            print(traceback.format_exc())
            try:
                await self._on_stop()
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                await self.status_sender.send_info(
                    self.websocket,
                    f"Server error: {exc}",
                )
            except Exception:  # pylint: disable=broad-except
                pass
