from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lunges.features import LandmarkSmoother
from lunges.session import LungeModelService, LungeWebSocketSession, StreamState
from tests.helpers.landmarks import landmarks_to_list, make_landmark_frame
from tests.helpers.session_fakes import (
    FakePose,
    RecordingStatusSender,
    RecordingWebSocket,
    make_pose_result,
)


class SequenceGate:
    def __init__(self, results):
        self.results = list(results)

    def evaluate(self, landmark_array):
        if self.results:
            return self.results.pop(0)
        return False, {"reason": "missing"}


class DummyLungeModelService:
    def __init__(
        self,
        *,
        bottom_loaded: bool = True,
        bottom_in_dim: int = 42,
        bottom_T: int | None = 3,
        phase_loaded: bool = False,
        phase_window: int | None = 1,
        phase_label: str = "unknown",
    ) -> None:
        self.bottom_loaded = bottom_loaded
        self.bottom_in_dim = bottom_in_dim
        self.bottom_T = bottom_T
        self.phase_loaded = phase_loaded
        self.phase_window = phase_window
        self.phase_label = phase_label
        self.phase_inputs: list[np.ndarray] = []

    def predict_phase(self, feature_window: np.ndarray) -> str:
        self.phase_inputs.append(feature_window)
        return self.phase_label


def make_landmarks() -> list:
    return landmarks_to_list(make_landmark_frame())


def build_session(monkeypatch, *, pose_results, gate_results, model_service):
    fake_pose = FakePose(pose_results)
    monkeypatch.setattr("lunges.session.mp.solutions.pose.Pose", lambda **kwargs: fake_pose)
    monkeypatch.setattr(
        "lunges.session.FrameDecoder.decode_jpeg_base64",
        lambda payload: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "lunges.session.FrameQuality.is_too_dark",
        lambda frame, threshold: (False, 21.5),
    )
    monkeypatch.setattr("lunges.session.cv2.cvtColor", lambda frame, code: frame)

    websocket = RecordingWebSocket()
    status_sender = RecordingStatusSender()
    session = LungeWebSocketSession(
        websocket=websocket,
        model_service=model_service,
        gate=SequenceGate(gate_results),
        status_sender=status_sender,
        ready_streak_n=2,
        debug=False,
        bottom_feature_dim=42,
        pre_frames=1,
        post_frames=1,
        min_gap=0,
        gate_knee_angle=130.0,
        dark_adjust_seconds=3.0,
        dark_brightness_th=55.0,
        goal_good_reps=5,
        mp_min_det_conf=0.5,
        mp_min_track_conf=0.5,
    )
    session.pose = fake_pose
    return session, status_sender, websocket


def test_lunge_model_service_forwards_phase_path(monkeypatch) -> None:
    captured = {}

    def fake_init(self, bottom_path, *, stand_path=None, phase_path=None) -> None:
        captured["bottom_path"] = bottom_path
        captured["stand_path"] = stand_path
        captured["phase_path"] = phase_path

    monkeypatch.setattr("lunges.session.PhaseAwareTCNModelService.__init__", fake_init)

    LungeModelService("bottom.pt", phase_path="phase.pt")

    assert captured == {
        "bottom_path": "bottom.pt",
        "stand_path": None,
        "phase_path": "phase.pt",
    }


@pytest.mark.asyncio
async def test_lunge_session_hooks_and_connected_payload(monkeypatch) -> None:
    model_service = DummyLungeModelService(bottom_in_dim=99)
    session, status_sender, _ = build_session(
        monkeypatch,
        pose_results=[],
        gate_results=[],
        model_service=model_service,
    )

    await session._on_connected()

    assert status_sender.info_calls[0]["message"].startswith("WARNING: Bottom model")
    assert status_sender.info_calls[1]["message"] == "WebSocket connected"

    state = session._create_state()
    assert isinstance(state, StreamState)
    assert state.history.maxlen == 242

    old_smoother = session.smoother
    session.frame_index = 9
    await session._on_start()
    assert session.frame_index == 0
    assert isinstance(session.smoother, LandmarkSmoother)
    assert session.smoother is not old_smoother

    session.state.total_reps = 3
    session.state.good_reps = 2
    session.state.bad_reps = 1
    assert session._stop_extra()["reps"]["goal_correct"] == 5

    session.state.previous_phase_values = (1.0, 2.0, 3.0)
    session._after_full_buffer_reset()
    assert session.state.previous_phase_values is None

    session.state.previous_phase_values = (4.0, 5.0, 6.0)
    session._after_ready_transition()
    assert session.state.previous_phase_values is None

    assert session._is_good_rep_label("good_form") is True
    assert session._is_good_rep_label("correct") is True
    assert session._is_good_rep_label("bad_form") is False

    record = (5, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(session._bottom_feature_from_history_record(record), record[1])


@pytest.mark.asyncio
async def test_lunge_handle_frame_decode_no_pose_and_gate_fail(monkeypatch) -> None:
    session, status_sender, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(None), make_pose_result(make_landmarks())],
        gate_results=[(False, {"reason": "bad_side"})],
        model_service=DummyLungeModelService(),
    )

    monkeypatch.setattr("lunges.session.FrameDecoder.decode_jpeg_base64", lambda payload: None)
    await session._handle_frame({"jpeg_b64": "bad"})
    assert status_sender.info_calls[0]["message"] == "Decode failed"

    monkeypatch.setattr(
        "lunges.session.FrameDecoder.decode_jpeg_base64",
        lambda payload: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    await session._handle_frame({"jpeg_b64": "frame"})
    assert status_sender.status_calls[0]["phase"] == "waiting"
    assert session.frame_index == 1

    await session._handle_frame({"jpeg_b64": "frame"})
    assert status_sender.status_calls[1]["phase"] == "waiting"
    assert status_sender.status_calls[1]["extra"]["reason"] == "bad_side"
    assert session.frame_index == 2


@pytest.mark.asyncio
async def test_lunge_handle_frame_ready_gate_warmup(monkeypatch) -> None:
    session, _, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(make_landmarks())],
        gate_results=[(True, {"vis_ok": True})],
        model_service=DummyLungeModelService(),
    )

    calls = []

    async def fake_advance_ready_streak(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(session, "_advance_ready_streak", fake_advance_ready_streak)

    await session._handle_frame({"jpeg_b64": "frame"})

    assert calls[0]["ok_message"] == "Side View OK"
    assert session.frame_index == 1


@pytest.mark.asyncio
async def test_lunge_handle_frame_predicts_bottom_event(monkeypatch) -> None:
    model_service = DummyLungeModelService(
        phase_loaded=True,
        phase_window=1,
        phase_label="concentric",
    )
    session, status_sender, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(make_landmarks())],
        gate_results=[(True, {"vis_ok": True})],
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

    bottom_features = np.zeros(42, dtype=np.float32)
    bottom_features[30] = 0.5
    bottom_features[31] = 0.5

    monkeypatch.setattr(session, "_advance_ready_streak", fake_advance_ready_streak)
    monkeypatch.setattr(session, "_predict_and_send_bottom", fake_predict_bottom)
    monkeypatch.setattr(session, "_resolve_pending_bottom_prediction", fake_resolve)
    monkeypatch.setattr("lunges.session.extract_bottom_features", lambda landmarks: bottom_features)
    monkeypatch.setattr(
        "lunges.session.extract_phase_features_from_lm",
        lambda landmarks, previous: (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            (1.0, 2.0, 3.0),
        ),
    )

    await session._handle_frame({"jpeg_b64": "frame"})

    assert status_sender.phase_calls[0]["phase"] == "concentric"
    assert predicted == [(0, "concentric")]
    assert resolved == ["concentric"]
    assert len(session.state.history) == 1
    assert len(session.state.phase_features) == 1
    assert session.state.last_bottom_event_frame == 0
    assert session.frame_index == 1
