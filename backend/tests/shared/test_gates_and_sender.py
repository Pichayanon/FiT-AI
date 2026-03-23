from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from shared.front_view_gate_dynamic import FrontViewGateDynamic
from shared.side_gate import SideGate
from shared.side_view_gate_dynamic import SideViewGateDynamic
from shared.status_sender import StatusSender
from tests.helpers.landmarks import landmarks_to_list, make_landmark_frame, set_landmark
from tests.helpers.mp_pose import make_fake_mp_pose


def build_visibility_landmarks() -> list:
    frame = make_landmark_frame(default_visibility=0.1)
    for index in [11, 23, 25, 27, 31]:
        set_landmark(frame, index, x=0.0, y=0.0, visibility=0.9)
    for index in [12, 24, 26, 28, 32]:
        set_landmark(frame, index, x=1.0, y=0.0, visibility=0.4)
    return landmarks_to_list(frame)


def build_front_gate_landmarks(*, visibility: float = 0.9, gap: float = 0.5) -> list:
    frame = make_landmark_frame(default_visibility=visibility)
    set_landmark(frame, 11, x=0.0, y=0.0, visibility=visibility)
    set_landmark(frame, 12, x=gap, y=0.0, visibility=visibility)
    set_landmark(frame, 23, x=0.0, y=1.0, visibility=visibility)
    set_landmark(frame, 24, x=gap, y=1.0, visibility=visibility)
    set_landmark(frame, 25, x=0.0, y=2.0, visibility=visibility)
    set_landmark(frame, 26, x=gap, y=2.0, visibility=visibility)
    set_landmark(frame, 27, x=0.0, y=3.0, visibility=visibility)
    set_landmark(frame, 28, x=gap, y=3.0, visibility=visibility)
    return landmarks_to_list(frame)


def build_side_view_frame(*, shoulder_vis=(0.9, 0.2), leg_vis=(0.9, 0.9, 0.9, 0.3, 0.3, 0.3)) -> np.ndarray:
    frame = make_landmark_frame(default_visibility=0.0)
    indices = [23, 24, 25, 26, 27, 28]
    for index, visibility in zip(indices, leg_vis):
        set_landmark(frame, index, x=float(index), y=0.0, visibility=visibility)
    set_landmark(frame, 11, x=0.0, y=0.0, visibility=shoulder_vis[0])
    set_landmark(frame, 12, x=1.0, y=0.0, visibility=shoulder_vis[1])
    return frame


def test_side_gate_scores_visibility_and_chooses_best_side() -> None:
    gate = SideGate(make_fake_mp_pose(), side_mode="auto", vis_th=0.5)
    landmarks = build_visibility_landmarks()

    left_ok, left_avg, left_vis = gate.score_side_visibility(landmarks, "left")
    chosen_side, debug_info = gate.choose_best_side(landmarks)

    assert left_ok is True
    assert left_avg == pytest.approx(0.9)
    assert left_vis["L_SHO"] == pytest.approx(0.9)
    assert chosen_side == "left"
    assert debug_info["right_ok"] is False


def test_side_gate_covers_forced_left_right_only_both_visible_and_neither_visible() -> None:
    left_gate = SideGate(make_fake_mp_pose(), side_mode="left", vis_th=0.5)
    right_gate = SideGate(make_fake_mp_pose(), side_mode="auto", vis_th=0.5)

    left_visible = build_visibility_landmarks()
    chosen_left, _ = left_gate.choose_best_side(left_visible)
    assert chosen_left == "left"

    right_only = build_visibility_landmarks()
    for index in [11, 23, 25, 27, 31]:
        right_only[index].visibility = 0.2
    for index in [12, 24, 26, 28, 32]:
        right_only[index].visibility = 0.9
    chosen_right, right_debug = right_gate.choose_best_side(right_only)
    assert chosen_right == "right"
    assert right_debug["left_ok"] is False
    assert right_debug["right_ok"] is True

    both_visible = build_visibility_landmarks()
    for index in [12, 24, 26, 28, 32]:
        both_visible[index].visibility = 0.95
    chosen_best, both_debug = right_gate.choose_best_side(both_visible)
    assert chosen_best == "right"
    assert both_debug["left_ok"] is True
    assert both_debug["right_ok"] is True

    neither_visible = build_visibility_landmarks()
    for landmark in neither_visible:
        landmark.visibility = 0.1
    chosen_none, none_debug = right_gate.choose_best_side(neither_visible)
    assert chosen_none is None
    assert none_debug["left_ok"] is False
    assert none_debug["right_ok"] is False


def test_side_gate_honors_forced_side_modes() -> None:
    gate = SideGate(make_fake_mp_pose(), side_mode="right", vis_th=0.5)
    landmarks = build_visibility_landmarks()

    chosen_side, debug_info = gate.choose_best_side(landmarks)

    assert chosen_side is None
    assert debug_info["mode"] == "right"


def test_front_view_gate_dynamic_reports_visibility_and_gap_failures() -> None:
    gate = FrontViewGateDynamic(make_fake_mp_pose(), vis_th=0.5, min_sho_gap=0.4, min_hip_gap=0.4)

    ok_landmarks = build_front_gate_landmarks(visibility=0.9, gap=0.6)
    bad_landmarks = build_front_gate_landmarks(visibility=0.3, gap=0.1)

    ok, ok_debug = gate.evaluate(ok_landmarks)
    bad, bad_debug = gate.evaluate(bad_landmarks)

    assert ok is True
    assert ok_debug["vis_ok"] is True
    assert ok_debug["gap_ok"] is True
    assert bad is False
    assert bad_debug["vis_ok"] is False
    assert bad_debug["gap_ok"] is False


def test_side_view_gate_dynamic_handles_numpy_single_side_profile_case() -> None:
    gate = SideViewGateDynamic(make_fake_mp_pose(), vis_th=0.5)
    frame = build_side_view_frame(
        shoulder_vis=(0.9, 0.1),
        leg_vis=(0.9, 0.1, 0.9, 0.1, 0.9, 0.1),
    )

    ok, debug_info = gate.evaluate(frame)

    assert ok is False
    assert debug_info["sho_ok"] is True
    assert debug_info["left_chain_ok"] is True
    assert debug_info["single_side_profile_ok"] is True
    assert debug_info["reason"] == "Single side profile visible"


def test_side_view_gate_dynamic_handles_landmark_list_and_upper_body_failure() -> None:
    gate = SideViewGateDynamic(make_fake_mp_pose(), vis_th=0.5)
    frame = build_side_view_frame(
        shoulder_vis=(0.1, 0.2),
        leg_vis=(0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
    )

    ok, debug_info = gate.evaluate(landmarks_to_list(frame))

    assert ok is False
    assert debug_info["legs_ok"] is True
    assert debug_info["sho_ok"] is False
    assert debug_info["reason"] == "Upper body not visible"


def test_side_view_gate_dynamic_tracks_left_chain_failures() -> None:
    gate = SideViewGateDynamic(make_fake_mp_pose(), vis_th=0.5)
    frame = build_side_view_frame(
        shoulder_vis=(0.9, 0.9),
        leg_vis=(0.1, 0.9, 0.1, 0.9, 0.1, 0.9),
    )

    ok, debug_info = gate.evaluate(frame)

    assert ok is False
    assert debug_info["left_chain_ok"] is False
    assert debug_info["right_chain_ok"] is True
    assert debug_info["reason"] == "Single side profile visible"


class DummyWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_text(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


@dataclass
class DummyStatusState:
    session_id: str = "session-1"
    status_tick: int = 0
    phase_tick: int = 0
    last_status: str = "ready"
    last_phase: str = "unknown"


@pytest.mark.asyncio
async def test_status_sender_sends_info_status_and_phase_with_throttling() -> None:
    websocket = DummyWebSocket()
    sender = StatusSender(every_n_frames=3, phase_every_n=2)
    state = DummyStatusState()

    await sender.send_info(websocket, "hello", {"x": 1})
    await sender.send_status(websocket, state, "ready")
    await sender.send_status(websocket, state, "waiting")
    await sender.send_phase(websocket, state, "eccentric")
    await sender.send_phase(websocket, state, "concentric")

    assert websocket.messages[0] == {"type": "info", "message": "hello", "x": 1}
    assert websocket.messages[1]["type"] == "status"
    assert websocket.messages[1]["state"] == "waiting"
    assert websocket.messages[2] == {
        "type": "phase",
        "phase": "eccentric",
        "session_id": "session-1",
    }
    assert websocket.messages[3] == {
        "type": "phase",
        "phase": "concentric",
        "session_id": "session-1",
    }


@pytest.mark.asyncio
async def test_status_sender_force_bypasses_throttling() -> None:
    websocket = DummyWebSocket()
    sender = StatusSender(every_n_frames=10, phase_every_n=10)
    state = DummyStatusState()

    await sender.send_status(websocket, state, "ready", force=True)
    await sender.send_phase(websocket, state, "eccentric", force=True)

    assert [message["type"] for message in websocket.messages] == ["status", "phase"]


@pytest.mark.asyncio
async def test_status_sender_adds_extra_fields_and_skips_duplicate_phase_messages() -> None:
    websocket = DummyWebSocket()
    sender = StatusSender(every_n_frames=10, phase_every_n=10)
    state = DummyStatusState(last_status="ready", last_phase="eccentric")

    await sender.send_status(websocket, state, "ready", extra={"debug": True}, force=True)
    await sender.send_phase(websocket, state, "eccentric")

    assert websocket.messages == [
        {
            "type": "status",
            "state": "ready",
            "session_id": "session-1",
            "debug": True,
        }
    ]
