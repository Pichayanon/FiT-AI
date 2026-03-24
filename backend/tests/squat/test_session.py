from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from squat.session import SquatModelService, SquatWebSocketSession, StreamState
from tests.helpers.landmarks import landmarks_to_list, make_landmark_frame
from tests.helpers.session_fakes import (
    FakePose,
    RecordingStatusSender,
    RecordingWebSocket,
    make_pose_result,
)


class SequenceFrontGate:
    def __init__(self, results):
        self.results = list(results)

    def evaluate(self, landmarks):
        if self.results:
            return self.results.pop(0)
        return False, {"reason": "missing"}


class DummySquatModelService:
    def __init__(
        self,
        *,
        bottom_loaded: bool = True,
        bottom_in_dim: int = 41,
        bottom_T: int | None = 3,
        stand_loaded: bool = True,
        stand_in_dim: int = 16,
        stand_T: int | None = 1,
        phase_loaded: bool = False,
        phase_window: int | None = 1,
        phase_label: str = "unknown",
        stand_prediction: str = "good_stand",
    ) -> None:
        self.bottom_loaded = bottom_loaded
        self.bottom_in_dim = bottom_in_dim
        self.bottom_T = bottom_T
        self.stand_loaded = stand_loaded
        self.stand_in_dim = stand_in_dim
        self.stand_T = stand_T
        self.phase_loaded = phase_loaded
        self.phase_window = phase_window
        self.phase_label = phase_label
        self.phase_inputs: list[np.ndarray] = []
        self.stand_inputs: list[np.ndarray] = []
        self.stand_prediction = stand_prediction

    def predict_phase(self, feature_window: np.ndarray, decision_mode: str = "last_logits") -> str:
        self.phase_inputs.append(feature_window)
        return self.phase_label

    def predict_stand(self, feature_window: np.ndarray):
        self.stand_inputs.append(feature_window)
        return self.stand_prediction, 0.812, np.array([0.188, 0.812], dtype=np.float32)


def make_landmarks() -> list:
    return landmarks_to_list(make_landmark_frame())


def build_session(monkeypatch, *, pose_results, gate_results, model_service, pre_frames=1, post_frames=1):
    fake_pose = FakePose(pose_results)
    import mediapipe as mp
    monkeypatch.setattr(mp.solutions.pose, "Pose", lambda **kwargs: fake_pose)
    monkeypatch.setattr(
        "squat.session.FrameDecoder.decode_jpeg_base64",
        lambda payload: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "squat.session.FrameQuality.is_too_dark",
        lambda frame, threshold: (False, 22.0),
    )
    monkeypatch.setattr("squat.session.cv2.cvtColor", lambda frame, code: frame)

    websocket = RecordingWebSocket()
    status_sender = RecordingStatusSender()
    session = SquatWebSocketSession(
        websocket=websocket,
        model_service=model_service,
        gate=SequenceFrontGate(gate_results),
        status_sender=status_sender,
        ready_streak_n=2,
        debug=False,
        bottom_feature_dim=41,
        stand_feature_dim=16,
        stand_ok_labels={"good_stand"},
        pre_frames=pre_frames,
        post_frames=post_frames,
        min_gap=0,
        phase_decision_mode="last_logits",
        stand_knee_angle_deg_th=160.0,
        stand_knee_delta_max_deg=5.0,
        stand_min_streak=1,
        stand_pred_cooldown=0,
        dark_adjust_seconds=3.0,
        dark_brightness_th=55.0,
        goal_good_reps=5,
        mp_min_det_conf=0.8,
        mp_min_track_conf=0.8,
    )
    session.pose = fake_pose
    return session, status_sender, websocket


def test_squat_model_service_forwards_paths(monkeypatch) -> None:
    captured = {}

    def fake_init(self, bottom_path, *, stand_path=None, phase_path=None) -> None:
        captured["bottom_path"] = bottom_path
        captured["stand_path"] = stand_path
        captured["phase_path"] = phase_path

    monkeypatch.setattr("squat.session.PhaseAwareTCNModelService.__init__", fake_init)

    SquatModelService("bottom.pt", "stand.pt", "phase.pt")

    assert captured == {
        "bottom_path": "bottom.pt",
        "stand_path": "stand.pt",
        "phase_path": "phase.pt",
    }


@pytest.mark.asyncio
async def test_squat_session_hooks_and_connected_payload(monkeypatch) -> None:
    model_service = DummySquatModelService(bottom_in_dim=99, stand_in_dim=77)
    session, status_sender, _ = build_session(
        monkeypatch,
        pose_results=[],
        gate_results=[],
        model_service=model_service,
    )

    await session._on_connected()

    assert status_sender.info_calls[0]["message"].startswith("WARNING: Bottom model")
    assert status_sender.info_calls[1]["message"].startswith("WARNING: Stand model")
    assert status_sender.info_calls[2]["message"] == "WebSocket connected"

    state = session._create_state()
    assert isinstance(state, StreamState)
    assert state.history.maxlen == 242

    session.frame_index = 11
    await session._on_start()
    assert session.frame_index == 0

    session.state.total_reps = 4
    session.state.good_reps = 3
    session.state.bad_reps = 1
    session.state.stand_ok = True
    stop_extra = session._stop_extra()
    assert stop_extra["reps"]["goal_correct"] == 5
    assert stop_extra["stand_ok"] is True

    session.state.last_sent_stand_label = "old"
    session.state.stand_streak = 5
    session.state.previous_knee_angle = 123.0
    session._reset_buffers()
    assert session.state.last_sent_stand_label == ""
    assert session.state.stand_streak == 0
    assert session.state.previous_knee_angle is None

    session.state.previous_knee_angle = 90.0
    session._after_ready_transition()
    assert session.state.previous_knee_angle is None

    record = (5, np.array([1.0], dtype=np.float32), np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(session._bottom_feature_from_history_record(record), record[2])


@pytest.mark.asyncio
async def test_squat_handle_frame_decode_no_pose_and_gate_fail(monkeypatch) -> None:
    session, status_sender, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(None), make_pose_result(make_landmarks())],
        gate_results=[(False, {"gate": "fail"})],
        model_service=DummySquatModelService(),
    )

    monkeypatch.setattr("squat.session.FrameDecoder.decode_jpeg_base64", lambda payload: None)
    await session._handle_frame({"jpeg_b64": "bad"})
    assert status_sender.info_calls[0]["message"] == "Decode failed"

    monkeypatch.setattr(
        "squat.session.FrameDecoder.decode_jpeg_base64",
        lambda payload: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    await session._handle_frame({"jpeg_b64": "frame"})
    assert status_sender.status_calls[0]["phase"] == "waiting"
    assert session.frame_index == 1

    await session._handle_frame({"jpeg_b64": "frame"})
    assert status_sender.status_calls[1]["phase"] == "waiting"
    assert status_sender.status_calls[1]["extra"]["reason"] == "front_gate_not_ok"
    assert session.frame_index == 2


@pytest.mark.asyncio
async def test_squat_handle_frame_ready_gate_warmup(monkeypatch) -> None:
    session, _, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(make_landmarks())],
        gate_results=[(True, {"front_ok": True})],
        model_service=DummySquatModelService(),
    )

    calls = []

    async def fake_advance_ready_streak(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(session, "_advance_ready_streak", fake_advance_ready_streak)

    await session._handle_frame({"jpeg_b64": "frame"})

    assert calls[0]["ok_message"] == "Front View OK"
    assert session.frame_index == 1


@pytest.mark.asyncio
async def test_squat_handle_frame_triggers_bottom_event(monkeypatch) -> None:
    model_service = DummySquatModelService(
        stand_loaded=False,
        phase_loaded=True,
        phase_window=1,
        phase_label="concentric",
    )
    session, status_sender, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(make_landmarks())],
        gate_results=[(True, {"front_ok": True})],
        model_service=model_service,
    )
    session.state.prev_phase = "eccentric"

    async def fake_advance_ready_streak(**kwargs):
        session.state.ready = True
        return True

    predicted = []
    resolved = []

    async def fake_predict_bottom(event_frame, phase):
        predicted.append((event_frame, phase))

    async def fake_resolve(phase):
        resolved.append(phase)

    monkeypatch.setattr(session, "_advance_ready_streak", fake_advance_ready_streak)
    monkeypatch.setattr(session, "_predict_and_send_bottom", fake_predict_bottom)
    monkeypatch.setattr(session, "_resolve_pending_bottom_prediction", fake_resolve)
    monkeypatch.setattr("squat.session.angle_from_points", lambda *args: 170.0)
    monkeypatch.setattr(
        "squat.session.extract_stand_features",
        lambda landmarks: np.array([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "squat.session.extract_bottom_features",
        lambda landmarks: np.array([2.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "squat.session.extract_phase_features",
        lambda landmarks: np.array([3.0], dtype=np.float32),
    )

    await session._handle_frame({"jpeg_b64": "frame"})

    assert status_sender.phase_calls[0]["phase"] == "concentric"
    assert predicted == [(0, "concentric")]
    assert resolved == ["concentric"]
    assert len(session.state.history) == 1
    assert len(session.state.phase_features) == 1
    assert session.state.last_bottom_event_frame == 0
    assert session.frame_index == 1


@pytest.mark.asyncio
async def test_squat_handle_frame_runs_stand_prediction(monkeypatch) -> None:
    model_service = DummySquatModelService(
        bottom_loaded=False,
        phase_loaded=False,
        stand_loaded=True,
        stand_T=1,
        stand_prediction="good_stand",
    )
    session, status_sender, websocket = build_session(
        monkeypatch,
        pose_results=[make_pose_result(make_landmarks())],
        gate_results=[(True, {"front_ok": True})],
        model_service=model_service,
        pre_frames=0,
        post_frames=0,
    )

    async def fake_advance_ready_streak(**kwargs):
        session.state.ready = True
        return True

    async def fake_resolve(phase):
        return None

    monkeypatch.setattr(session, "_advance_ready_streak", fake_advance_ready_streak)
    monkeypatch.setattr(session, "_resolve_pending_bottom_prediction", fake_resolve)
    monkeypatch.setattr("squat.session.angle_from_points", lambda *args: 170.0)
    monkeypatch.setattr(
        "squat.session.extract_stand_features",
        lambda landmarks: np.array([1.0, 2.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "squat.session.extract_bottom_features",
        lambda landmarks: np.array([3.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "squat.session.extract_phase_features",
        lambda landmarks: np.array([4.0], dtype=np.float32),
    )

    await session._handle_frame({"jpeg_b64": "frame"})

    assert status_sender.phase_calls[0]["phase"] == "unknown"
    assert model_service.stand_inputs[0].shape == (1, 2)
    assert session.state.stand_ok is True
    assert websocket.sent_messages[0]["mode"] == "stand"
    assert websocket.sent_messages[0]["prediction"] == "good_stand"
    assert session.frame_index == 1
