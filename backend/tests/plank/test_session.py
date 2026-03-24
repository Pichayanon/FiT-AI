from __future__ import annotations

from types import SimpleNamespace

from plank.features import StreamState
from plank.session import PlankWebSocketSession
from tests.helpers.landmarks import landmarks_to_list, make_landmark_frame
from tests.helpers.session_fakes import FakePose, RecordingStatusSender, RecordingWebSocket


def build_session(monkeypatch) -> PlankWebSocketSession:
    fake_pose = FakePose()
    import mediapipe as mp
    monkeypatch.setattr(
        mp.solutions.pose, "Pose",
        lambda **kwargs: fake_pose,
    )

    return PlankWebSocketSession(
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
        dark_adjust_seconds=3.0,
        dark_brightness_th=55.0,
        plank_ready_max_body_axis_angle_deg=35.0,
        plank_ready_max_torso_angle_deg=45.0,
        plank_ready_max_leg_angle_deg=45.0,
        phase_no_pose="NO_POSE",
        phase_have_pose="HAVE_POSE",
        phase_buffering="BUFFERING",
        phase_inferencing="INFERENCING",
        status_send_every_n_frames=3,
    )


def test_plank_session_constructor_and_create_state(monkeypatch) -> None:
    session = build_session(monkeypatch)

    assert session.plank_ready_max_body_axis_angle_deg == 35.0
    assert session.plank_ready_max_torso_angle_deg == 45.0
    assert session.plank_ready_max_leg_angle_deg == 45.0
    assert isinstance(session.state, StreamState)
    assert isinstance(session._create_state(), StreamState)


def test_plank_session_pre_ready_pose_payload(monkeypatch) -> None:
    session = build_session(monkeypatch)
    landmarks = landmarks_to_list(make_landmark_frame())

    monkeypatch.setattr(
        "plank.session.is_plank_ready_pose",
        lambda landmarks, chosen_side, **kwargs: (True, {"ok": True}),
    )
    assert session._pre_ready_pose_payload(landmarks, "left") is None

    monkeypatch.setattr(
        "plank.session.is_plank_ready_pose",
        lambda landmarks, chosen_side, **kwargs: (
            False,
            {"body_axis_angle_deg": 70.0, "plank_pose_ok": False},
        ),
    )
    payload = session._pre_ready_pose_payload(landmarks, "right")

    assert payload == {
        "chosen_side": "right",
        "window_fill": 0,
        "window_size": 15,
        "ready_streak": 0,
        "needed_streak": 3,
        "body_axis_angle_deg": 70.0,
        "plank_pose_ok": False,
    }
