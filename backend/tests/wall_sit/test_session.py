from __future__ import annotations

from types import SimpleNamespace

from wall_sit.features import StreamState
from wall_sit.session import WallSitWebSocketSession
from tests.helpers.session_fakes import FakePose, RecordingStatusSender, RecordingWebSocket


def build_session(monkeypatch) -> WallSitWebSocketSession:
    fake_pose = FakePose()
    import mediapipe as mp
    monkeypatch.setattr(
        mp.solutions.pose, "Pose",
        lambda **kwargs: fake_pose,
    )

    return WallSitWebSocketSession(
        websocket=RecordingWebSocket(),
        model_service=SimpleNamespace(loaded=True),
        gate=SimpleNamespace(),
        feature_extractor=SimpleNamespace(),
        labels=SimpleNamespace(),
        status_sender=RecordingStatusSender(),
        window_frames=15,
        ready_streak_n=3,
        debug=False,
        side_mode="auto",
        vis_th=0.8,
        mp_min_det_conf=0.8,
        mp_min_track_conf=0.8,
        no_pose_adjust_seconds=5.0,
        dark_adjust_seconds=5.0,
        dark_brightness_th=55.0,
        stand_knee_angle_deg_th=0.5,
        stand_streak_n=2,
        phase_no_pose="NO_POSE",
        phase_have_pose="HAVE_POSE",
        phase_buffering="BUFFERING",
        phase_inferencing="INFERENCING",
        status_send_every_n_frames=3,
    )


def test_wall_sit_session_constructor_and_reset_hooks(monkeypatch) -> None:
    session = build_session(monkeypatch)

    assert isinstance(session.state, StreamState)
    assert isinstance(session._create_state(), StreamState)

    session.state.stand_streak = 4
    session._reset_gate_specific_fields()
    assert session.state.stand_streak == 0

    session.state.stand_streak = 3
    session._on_missing_features()
    assert session.state.stand_streak == 0


def test_wall_sit_pre_buffer_feature_payload_tracks_standing_streak(monkeypatch) -> None:
    session = build_session(monkeypatch)
    session.state.session_id = "wall-1"
    session.state.chosen_side = "left"
    session.state.ready_streak = 2

    assert session._pre_buffer_feature_payload((0.0, 0.2, 0.0, 0.0)) is None
    assert session.state.stand_streak == 0

    assert session._pre_buffer_feature_payload((0.0, 0.6, 0.0, 0.0)) is None
    assert session.state.stand_streak == 1

    session.state.frame_features = [(1.0, 2.0, 3.0, 0.0)]
    payload = session._pre_buffer_feature_payload((0.0, 0.7, 0.0, 0.0))

    assert payload == {
        "chosen_side": "left",
        "window_fill": 0,
        "window_size": 15,
        "ready_streak": 2,
        "needed_streak": 3,
        "standing": True,
        "knee_angle": 0.7,
        "knee_th": 0.5,
    }
    assert session.state.frame_features == []

    session.state.frame_features = [(1.0, 2.0, 3.0, 0.0)]
    next_payload = session._pre_buffer_feature_payload((0.0, 0.8, 0.0, 0.0))
    assert next_payload is not None
    assert session.state.frame_features == []
