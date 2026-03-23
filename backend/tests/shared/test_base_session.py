from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from shared.base_session import BaseWebSocketSession


class FakeStatusSender:
    def __init__(self) -> None:
        self.info_calls: list[dict[str, Any]] = []
        self.status_calls: list[dict[str, Any]] = []

    async def send_info(self, websocket, message: str, extra: dict[str, Any] | None = None) -> None:
        self.info_calls.append({"message": message, "extra": extra})

    async def send_status(
        self,
        websocket,
        state,
        phase: str,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        state.last_status = phase
        self.status_calls.append({"phase": phase, "extra": extra, "force": force})


class FakeWebSocket:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = list(messages or [])
        self.accepted = False
        self.sent_texts: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if not self.messages:
            raise WebSocketDisconnect(1000)
        next_message = self.messages.pop(0)
        if isinstance(next_message, Exception):
            raise next_message
        return next_message

    async def send_text(self, payload: str) -> None:
        self.sent_texts.append(payload)


@dataclass
class DummyBaseState:
    started: bool = False
    session_id: str = ""
    dark_since: float | None = None
    dark_alerted: bool = False
    no_pose_since: float | None = None
    no_pose_alerted: bool = False
    last_status: str = ""


class DummyBaseSession(BaseWebSocketSession):
    def __init__(self, websocket: FakeWebSocket, status_sender: FakeStatusSender) -> None:
        self.websocket = websocket
        self.status_sender = status_sender
        self.debug = True
        self.dark_adjust_seconds = 2.0
        self.dark_brightness_th = 15.0
        self.no_pose_adjust_seconds = 0.0
        self.phase_no_pose = "waiting"
        self.ready_streak_n = 3
        self.window_frames = 5
        self.state = self._create_state()
        self.connected_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.handle_frame_calls: list[dict[str, Any]] = []
        self.reset_calls: list[bool] = []
        self.raise_on_frame = False

    def _create_state(self) -> DummyBaseState:
        return DummyBaseState()

    async def _on_connected(self) -> None:
        self.connected_calls += 1

    async def _on_start(self) -> None:
        self.start_calls += 1

    async def _on_stop(self) -> None:
        self.stop_calls += 1

    def _stop_extra(self) -> dict[str, Any]:
        return {"extra": "value"}

    def _reset_gate_and_buffers(self, reset_watchdog: bool) -> None:
        self.reset_calls.append(reset_watchdog)

    async def _handle_frame(self, data: dict[str, Any]) -> None:
        self.handle_frame_calls.append(data)
        if self.raise_on_frame:
            raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_base_session_start_stop_and_dark_watchdog(monkeypatch) -> None:
    websocket = FakeWebSocket()
    sender = FakeStatusSender()
    session = DummyBaseSession(websocket, sender)

    time_values = iter([1.001, 2.002, 10.0, 12.0, 14.1, 15.0])
    monkeypatch.setattr("shared.base_session.time.time", lambda: next(time_values))

    await session._handle_start()
    first_session_id = session.state.session_id
    await session._handle_start()

    assert first_session_id == "1000"
    assert session.state.session_id == "2001"
    assert session.start_calls == 2
    assert session.stop_calls == 1
    assert sender.info_calls[0]["message"] == "Start streaming"
    assert sender.status_calls[0]["phase"] == "waiting"

    await session._update_dark_watchdog(False, 1.0)
    assert session.state.dark_since is None
    await session._update_dark_watchdog(True, 5.0)
    assert session.state.dark_since == 12.0
    await session._update_dark_watchdog(True, 5.0)
    assert session.state.dark_alerted is True
    await session._update_dark_watchdog(True, 5.0)

    light_messages = [call for call in sender.info_calls if call["message"] == "Please adjust your lights."]
    assert len(light_messages) == 1

    await session._handle_stop()
    assert session.state.started is False
    assert session.stop_calls == 2
    assert sender.info_calls[-1]["message"] == "Stop streaming"
    assert sender.info_calls[-1]["extra"]["extra"] == "value"


@pytest.mark.asyncio
async def test_base_session_handle_no_pose_resets_buffers_and_sends_guidance(monkeypatch) -> None:
    websocket = FakeWebSocket()
    sender = FakeStatusSender()
    session = DummyBaseSession(websocket, sender)

    monkeypatch.setattr("shared.base_session.time.time", lambda: 50.0)

    await session._handle_no_pose(
        too_dark=False,
        brightness_mean=7.77,
        gate_debug={"side": "unknown"},
    )

    assert session.state.no_pose_since == 50.0
    assert session.state.no_pose_alerted is True
    assert session.reset_calls == [False]
    assert sender.info_calls[0]["message"] == "Adjust your camera to see your full body"
    assert sender.status_calls[0]["phase"] == "waiting"
    assert sender.status_calls[0]["extra"]["brightness_mean"] == 7.8
    assert sender.status_calls[0]["extra"]["debug"] == {"side": "unknown"}


@pytest.mark.asyncio
async def test_base_session_run_processes_messages_and_disconnects() -> None:
    websocket = FakeWebSocket(
        [
            "{broken",
            '{"type":"start"}',
            '{"type":"frame","jpeg_b64":"abc"}',
            '{"type":"stop"}',
        ]
    )
    sender = FakeStatusSender()
    session = DummyBaseSession(websocket, sender)

    await session.run()

    assert websocket.accepted is True
    assert session.connected_calls == 1
    assert session.start_calls == 1
    assert session.stop_calls == 2
    assert session.handle_frame_calls == [{"type": "frame", "jpeg_b64": "abc"}]
    assert any(call["message"] == "Invalid JSON" for call in sender.info_calls)


@pytest.mark.asyncio
async def test_base_session_run_reports_server_errors() -> None:
    websocket = FakeWebSocket(
        [
            '{"type":"start"}',
            '{"type":"frame","jpeg_b64":"abc"}',
        ]
    )
    sender = FakeStatusSender()
    session = DummyBaseSession(websocket, sender)
    session.raise_on_frame = True

    await session.run()

    assert session.stop_calls == 1
    assert any(call["message"] == "Server error: boom" for call in sender.info_calls)
