from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from shared.side_window_session import SideWindowSession
from tests.helpers.landmarks import landmarks_to_list, make_landmark_frame


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


class FakePose:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)

    def process(self, image: np.ndarray) -> Any:
        if self.results:
            return self.results.pop(0)
        return SimpleNamespace(pose_landmarks=None)


class FakeGate:
    def __init__(self, results: list[tuple[str | None, dict[str, Any]]]) -> None:
        self.results = list(results)
        self.calls = 0

    def choose_best_side(self, landmarks: Any) -> tuple[str | None, dict[str, Any]]:
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return None, {}


class FakeFeatureExtractor:
    def __init__(self, features: list[Any]) -> None:
        self.features = list(features)
        self.aggregate_calls: list[list[Any]] = []

    def extract_features(self, pose_result: Any, side: str) -> Any:
        if self.features:
            return self.features.pop(0)
        return None

    def aggregate_window(self, frame_features: list[Any]) -> np.ndarray:
        self.aggregate_calls.append(list(frame_features))
        return np.array([9.0, 8.0], dtype=np.float32)


class FakeLabelMapper:
    def label_of(self, label_id: int) -> str:
        return {0: "bad", 1: "good"}.get(label_id, str(label_id))


class FakeModelService:
    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded
        self.calls: list[np.ndarray] = []

    def predict(self, aggregated_features: np.ndarray) -> tuple[int, float]:
        self.calls.append(aggregated_features)
        return 1, 0.876


@dataclass
class DummySideState:
    started: bool = False
    session_id: str = "side-1"
    frame_features: list[Any] = field(default_factory=list)
    ready: bool = False
    ready_streak: int = 0
    chosen_side: str | None = None
    last_sent_label: str = ""
    last_sent_confidence: float | None = None
    last_prediction_label: str = ""
    last_prediction_confidence: float | None = None
    label_streak: int = 0
    label_streak_label: str = ""
    frame_count: int = 0
    no_pose_since: float | None = None
    no_pose_alerted: bool = False
    dark_since: float | None = None
    dark_alerted: bool = False


class DummySideWindowSession(SideWindowSession):
    def __init__(self, **kwargs) -> None:
        self.pre_ready_payload_override: dict[str, Any] | None = None
        self.pre_buffer_payload_override: dict[str, Any] | None = None
        self.reset_specific_calls = 0
        self.missing_features_calls = 0
        super().__init__(**kwargs)

    def _create_state(self) -> DummySideState:
        return DummySideState()

    def _reset_gate_specific_fields(self) -> None:
        self.reset_specific_calls += 1

    def _on_missing_features(self) -> None:
        self.missing_features_calls += 1

    def _pre_ready_pose_payload(self, landmarks: Any, chosen_side: str) -> dict[str, Any] | None:
        return self.pre_ready_payload_override

    def _pre_buffer_feature_payload(self, frame_features: Any) -> dict[str, Any] | None:
        return self.pre_buffer_payload_override


def make_pose_result(landmarks: list[Any] | None) -> Any:
    if landmarks is None:
        return SimpleNamespace(pose_landmarks=None)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))


def make_landmarks() -> list[Any]:
    return landmarks_to_list(make_landmark_frame())


def build_session(
    monkeypatch,
    *,
    pose_results: list[Any],
    gate_results: list[tuple[str | None, dict[str, Any]]],
    extracted_features: list[Any],
    model_loaded: bool = True,
) -> tuple[DummySideWindowSession, FakeStatusSender, FakeFeatureExtractor, FakeModelService, FakeWebSocket, FakeGate]:
    fake_pose = FakePose(pose_results)
    import mediapipe as mp
    monkeypatch.setattr(
        mp.solutions.pose, "Pose",
        lambda **kwargs: fake_pose,
    )
    monkeypatch.setattr(
        "shared.side_window_session.FrameDecoder.decode_jpeg_base64",
        lambda payload: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "shared.side_window_session.FrameQuality.is_too_dark",
        lambda frame, threshold: (False, 21.5),
    )
    monkeypatch.setattr("shared.side_window_session.cv2.cvtColor", lambda frame, code: frame)

    websocket = FakeWebSocket()
    status_sender = FakeStatusSender()
    gate = FakeGate(gate_results)
    feature_extractor = FakeFeatureExtractor(extracted_features)
    model_service = FakeModelService(loaded=model_loaded)

    session = DummySideWindowSession(
        websocket=websocket,
        model_service=model_service,
        gate=gate,
        feature_extractor=feature_extractor,
        labels=FakeLabelMapper(),
        status_sender=status_sender,
        window_frames=2,
        ready_streak_n=2,
        debug=True,
        side_mode="auto",
        vis_th=0.5,
        mp_min_det_conf=0.5,
        mp_min_track_conf=0.5,
        no_pose_adjust_seconds=0.0,
        dark_adjust_seconds=1.0,
        dark_brightness_th=10.0,
        phase_no_pose="no_pose",
        phase_have_pose="have_pose",
        phase_buffering="buffering",
        phase_inferencing="inferencing",
        status_send_every_n_frames=1,
    )
    session.pose = fake_pose
    return session, status_sender, feature_extractor, model_service, websocket, gate


@pytest.mark.asyncio
async def test_side_window_on_connected_and_reset_buffers(monkeypatch) -> None:
    session, sender, _, _, _, _ = build_session(
        monkeypatch,
        pose_results=[],
        gate_results=[],
        extracted_features=[],
        model_loaded=False,
    )

    await session._on_connected()
    session.state.frame_features = [(1.0, 2.0)]
    session.state.ready = True
    session.state.ready_streak = 2
    session.state.chosen_side = "left"
    session.state.last_prediction_label = "good"
    session.state.last_prediction_confidence = 0.9
    session.state.no_pose_since = 1.0
    session.state.dark_since = 2.0
    session._reset_gate_and_buffers(reset_watchdog=True)

    assert sender.info_calls[0]["message"] == "WebSocket connected"
    assert sender.info_calls[1]["message"] == "Model not loaded (check MODEL_PATH)"
    assert session.state.frame_features == []
    assert session.state.ready is False
    assert session.state.ready_streak == 0
    assert session.state.chosen_side is None
    assert session.state.no_pose_since is None
    assert session.state.dark_since is None
    assert session.reset_specific_calls == 1


@pytest.mark.asyncio
async def test_side_window_handle_frame_decode_failure_and_no_pose(monkeypatch) -> None:
    session, sender, _, _, _, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(None)],
        gate_results=[],
        extracted_features=[],
    )
    session.state.started = True

    monkeypatch.setattr(
        "shared.side_window_session.FrameDecoder.decode_jpeg_base64",
        lambda payload: None,
    )
    await session._handle_frame({"jpeg_b64": "bad"})
    assert sender.info_calls[0]["message"] == "Decode failed"

    monkeypatch.setattr(
        "shared.side_window_session.FrameDecoder.decode_jpeg_base64",
        lambda payload: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    await session._handle_frame({"jpeg_b64": "ok"})

    assert session.state.frame_count == 0
    assert sender.info_calls[1]["message"] == "Adjust your camera to see your full body"
    assert sender.status_calls[0]["phase"] == "no_pose"
    assert sender.status_calls[0]["extra"]["window_size"] == 2


@pytest.mark.asyncio
async def test_side_window_pre_ready_gate_and_ready_transition(monkeypatch) -> None:
    landmarks = make_landmarks()
    session, sender, _, model_service, _, gate = build_session(
        monkeypatch,
        pose_results=[
            make_pose_result(landmarks),
            make_pose_result(landmarks),
            make_pose_result(landmarks),
        ],
        gate_results=[
            ("left", {"gate": "pre"}),
            ("left", {"gate": "warm"}),
            ("left", {"gate": "ready"}),
        ],
        extracted_features=[],
        model_loaded=False,
    )
    session.state.started = True
    session.pre_ready_payload_override = {"blocked": True}

    await session._handle_frame({"jpeg_b64": "frame"})
    assert sender.status_calls[0]["phase"] == "have_pose"
    assert sender.status_calls[0]["extra"] == {"blocked": True}
    assert gate.calls == 1

    session.pre_ready_payload_override = None
    await session._handle_frame({"jpeg_b64": "frame"})
    assert sender.status_calls[1]["phase"] == "have_pose"
    assert sender.status_calls[1]["extra"]["ready_streak"] == 1

    await session._handle_frame({"jpeg_b64": "frame"})
    assert session.state.ready is True
    assert any(call["message"] == "Side View OK" for call in sender.info_calls)
    assert sender.status_calls[2]["phase"] == "buffering"
    assert sender.status_calls[2]["force"] is True
    assert sender.status_calls[3]["phase"] == "buffering"
    assert sender.status_calls[3]["extra"]["window_fill"] == 0
    assert model_service.calls == []


@pytest.mark.asyncio
async def test_side_window_feature_buffering_missing_features_and_prebuffer(monkeypatch) -> None:
    landmarks = make_landmarks()
    session, sender, _, _, _, gate = build_session(
        monkeypatch,
        pose_results=[
            make_pose_result(landmarks),
            make_pose_result(landmarks),
            make_pose_result(landmarks),
        ],
        gate_results=[],
        extracted_features=[None, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
    )
    session.state.started = True
    session.state.ready = True
    session.state.chosen_side = "left"

    await session._handle_frame({"jpeg_b64": "frame"})
    assert session.missing_features_calls == 1
    assert sender.status_calls[0]["phase"] == "buffering"
    assert sender.status_calls[0]["extra"]["window_fill"] == 0
    assert gate.calls == 0

    session.pre_buffer_payload_override = {"standing": True}
    await session._handle_frame({"jpeg_b64": "frame"})
    assert sender.status_calls[1]["phase"] == "have_pose"
    assert sender.status_calls[1]["extra"] == {"standing": True}

    session.pre_buffer_payload_override = None
    await session._handle_frame({"jpeg_b64": "frame"})
    assert sender.status_calls[2]["phase"] == "buffering"
    assert sender.status_calls[2]["extra"]["window_fill"] == 1


@pytest.mark.asyncio
async def test_side_window_inference_sends_result_and_trims_window(monkeypatch) -> None:
    landmarks = make_landmarks()
    session, sender, extractor, model_service, websocket, _ = build_session(
        monkeypatch,
        pose_results=[make_pose_result(landmarks)],
        gate_results=[],
        extracted_features=[(7.0, 8.0, 9.0)],
    )
    session.state.started = True
    session.state.ready = True
    session.state.chosen_side = "left"
    session.state.frame_features = [(0.0, 0.0, 0.0)] * 62
    # Pre-fill label streak so prediction is emitted on this frame
    session.state.label_streak_label = "good"
    session.state.label_streak = session.label_streak_n - 1

    await session._handle_frame({"jpeg_b64": "frame"})

    assert sender.status_calls[0]["phase"] == "inferencing"
    assert extractor.aggregate_calls
    assert model_service.calls[0].tolist() == [9.0, 8.0]
    assert session.state.last_prediction_label == "good"
    assert session.state.last_prediction_confidence == 0.876
    assert websocket.messages[0]["type"] == "result"
    assert websocket.messages[0]["prediction"] == "good"
    assert websocket.messages[0]["confidence"] == 0.876
    assert len(session.state.frame_features) == 62
