from __future__ import annotations

import time
import traceback
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from shared.json_utils import parse_json
from shared.status_sender import StatusSender


class BaseWebSocketSession:
    websocket: WebSocket
    status_sender: StatusSender
    debug: bool
    state: Any

    def _create_state(self) -> Any:
        raise NotImplementedError

    @property
    def _initial_status(self) -> str:
        return "waiting"

    async def _on_connected(self) -> None:
        pass

    async def _on_start(self) -> None:
        pass

    async def _on_stop(self) -> None:
        pass

    def _stop_extra(self) -> Dict[str, Any]:
        return {}

    async def _handle_start(self) -> None:
        if getattr(self, "state", None) is not None and getattr(
            self.state, "started", False
        ):
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
            {"reason": "session_started"},
            force=True,
        )

    async def _handle_stop(self) -> None:
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
            {"reason": "session_stopped"},
            force=True,
        )
        await self._on_stop()

    async def _update_dark_watchdog(
        self,
        too_dark: bool,
        brightness_mean: float,
    ) -> None:
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

    async def _handle_frame(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def run(self) -> None:
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
            except Exception:
                pass

        except Exception as exc:
            print(f"[WS] error: {exc}")
            print(traceback.format_exc())
            try:
                await self._on_stop()
            except Exception:
                pass
            try:
                await self.status_sender.send_info(
                    self.websocket,
                    f"Server error: {exc}",
                )
            except Exception:
                pass
