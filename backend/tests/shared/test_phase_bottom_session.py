from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from shared.phase_bottom_session import PhaseBottomSessionMixin


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
        self.status_calls.append({"phase": phase, "extra": extra, "force": force})


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


class FakeModelService:
    def __init__(self) -> None:
        self.bottom_loaded = True
        self.bottom_T = 3
        self.calls: list[np.ndarray] = []

    def predict_bottom(self, window_features: np.ndarray) -> tuple[str, float, np.ndarray]:
        self.calls.append(window_features)
        return "good_depth", 0.876, np.array([0.124, 0.876], dtype=np.float32)


@dataclass
class DummyPhaseState:
    started: bool = False
    session_id: str = "phase-1"
    ready: bool = False
    ready_streak: int = 0
    last_gate_debug: dict[str, Any] = field(default_factory=dict)
    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_frame: int = -(10**9)
    last_sent_bottom_event_frame: int = -(10**9)
    history: deque = field(default_factory=deque)
    phase_features: deque = field(default_factory=deque)
    prev_phase: str = ""
    last_bottom_event_frame: int = -(10**9)
    pending_bottom_event: dict[str, Any] | None = None


class DummyPhaseBottomSession(PhaseBottomSessionMixin):
    def __init__(self) -> None:
        self.state = DummyPhaseState()
        self.ready_streak_n = 2
        self.dark_brightness_th = 15.0
        self.debug = True
        self.pre_frames = 1
        self.post_frames = 1
        self.bottom_feature_dim = 2
        self.goal_good_reps = 5
        self.frame_index = 0
        self.websocket = FakeWebSocket()
        self.status_sender = FakeStatusSender()
        self.model_service = FakeModelService()
        self.full_reset_calls = 0
        self.ready_transition_calls = 0

    def _bottom_feature_from_history_record(self, record: Any) -> np.ndarray:
        return np.asarray(record[1], dtype=np.float32)

    def _after_full_buffer_reset(self) -> None:
        self.full_reset_calls += 1

    def _after_ready_transition(self) -> None:
        self.ready_transition_calls += 1


def test_phase_bottom_reset_helpers_and_rep_counter() -> None:
    session = DummyPhaseBottomSession()
    session.state.ready = True
    session.state.ready_streak = 5
    session.state.last_gate_debug = {"ok": True}
    session.state.history.extend([(1, np.array([1.0, 2.0]))])
    session.state.phase_features.extend([np.array([3.0])])
    session.state.prev_phase = "eccentric"
    session.state.pending_bottom_event = {"event_frame": 9}

    session._reset_phase_bottom_state()

    assert session.state.ready is False
    assert session.state.ready_streak == 0
    assert session.state.last_gate_debug == {}
    assert list(session.state.history) == []
    assert list(session.state.phase_features) == []
    assert session.state.prev_phase == ""
    assert session.state.pending_bottom_event is None
    assert session.full_reset_calls == 1

    session._activate_ready_phase_bottom_state()

    assert session.state.ready is True
    assert session.ready_transition_calls == 1

    session._increment_rep_counter(10, "good_depth")
    session._increment_rep_counter(10, "bad_depth")
    session._increment_rep_counter(11, "bad_depth")

    assert session.state.total_reps == 2
    assert session.state.good_reps == 1
    assert session.state.bad_reps == 1


@pytest.mark.asyncio
async def test_phase_bottom_waiting_status_and_ready_streak_flow() -> None:
    session = DummyPhaseBottomSession()

    await session._send_waiting_status(
        too_dark=False,
        brightness_mean=7.77,
        gate_debug={"gate": "ok"},
        reason="camera_adjust",
    )

    assert session.status_sender.status_calls[0]["phase"] == "waiting"
    assert session.status_sender.status_calls[0]["extra"]["brightness_mean"] == 7.8
    assert session.status_sender.status_calls[0]["extra"]["gate"] == {"gate": "ok"}
    assert session.status_sender.status_calls[0]["extra"]["reason"] == "camera_adjust"

    ready = await session._advance_ready_streak(
        gate_debug={"pose": "almost"},
        ok_message="View OK",
    )
    assert ready is False
    assert session.status_sender.status_calls[1]["phase"] == "warming_up"

    ready = await session._advance_ready_streak(
        gate_debug={"pose": "good"},
        ok_message="View OK",
    )
    assert ready is True
    assert session.state.ready is True
    assert session.status_sender.info_calls[0]["message"] == "View OK"
    assert session.status_sender.status_calls[2]["phase"] == "ready"
    assert session.status_sender.status_calls[2]["force"] is True


@pytest.mark.asyncio
async def test_phase_bottom_predict_and_send_bottom_handles_pending_empty_and_result() -> None:
    session = DummyPhaseBottomSession()

    session.state.history.extend(
        [
            (0, np.array([0.0, 1.0], dtype=np.float32)),
            (1, np.array([1.0, 2.0], dtype=np.float32)),
        ]
    )
    await session._predict_and_send_bottom(1, "eccentric")
    assert session.state.pending_bottom_event == {
        "event_frame": 1,
        "start_frame": 0,
        "end_frame": 2,
    }

    session.state.history.clear()
    await session._predict_and_send_bottom(1, "eccentric", force=True)
    assert session.state.pending_bottom_event is None

    session.state.history.extend(
        [
            (0, np.array([0.0, 1.0], dtype=np.float32)),
            (1, np.array([1.0, 2.0], dtype=np.float32)),
            (2, np.array([2.0, 3.0], dtype=np.float32)),
        ]
    )
    await session._predict_and_send_bottom(1, "concentric")

    assert session.status_sender.status_calls[-1]["phase"] == "predicting"
    assert session.model_service.calls[0].shape == (3, 2)
    assert session.websocket.messages[0]["type"] == "result"
    assert session.websocket.messages[0]["prediction"] == "good_depth"
    assert session.websocket.messages[0]["reps"]["total"] == 1
    assert session.websocket.messages[0]["reps"]["correct"] == 1

    await session._predict_and_send_bottom(1, "concentric", force=True)
    assert len(session.websocket.messages) == 1


@pytest.mark.asyncio
async def test_phase_bottom_resolve_pending_prediction_uses_window_completion() -> None:
    session = DummyPhaseBottomSession()
    calls: list[tuple[int, str, bool]] = []

    async def fake_predict(event_frame: int, phase: str, force: bool = False) -> None:
        calls.append((event_frame, phase, force))

    session._predict_and_send_bottom = fake_predict
    session.state.pending_bottom_event = {
        "event_frame": 5,
        "start_frame": 4,
        "end_frame": 6,
    }
    session.state.history.extend(
        [
            (4, np.array([4.0, 5.0], dtype=np.float32)),
            (5, np.array([5.0, 6.0], dtype=np.float32)),
        ]
    )

    await session._resolve_pending_bottom_prediction("eccentric")
    assert calls == []

    session.frame_index = 6
    await session._resolve_pending_bottom_prediction("eccentric")
    assert calls == [(5, "eccentric", True)]
