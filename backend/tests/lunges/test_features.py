from __future__ import annotations

import numpy as np
import pytest

from lunges.features import (
    BOTTOM_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    LandmarkSmoother,
    extract_bottom_features,
    extract_phase_features,
    normalize_facing_direction,
    select_front_back_joint_indices,
)
from tests.helpers.landmarks import make_landmark_frame, set_landmark


def build_lunge_frame(*, facing_right: bool = True) -> np.ndarray:
    frame = make_landmark_frame()
    set_landmark(frame, 7, x=0.5, y=0.0)
    set_landmark(frame, 8, x=-0.5, y=0.0)
    set_landmark(frame, 11, x=0.5, y=0.5)
    set_landmark(frame, 12, x=-0.5, y=0.5)
    set_landmark(frame, 23, x=0.5, y=1.5)
    set_landmark(frame, 24, x=-0.5, y=1.5)
    set_landmark(frame, 25, x=1.0, y=2.5)
    set_landmark(frame, 26, x=-1.0, y=2.7)
    set_landmark(frame, 27, x=1.0, y=3.5)
    set_landmark(frame, 28, x=-1.0, y=3.5)

    if facing_right:
        set_landmark(frame, 29, x=0.8, y=3.5)
        set_landmark(frame, 31, x=1.2, y=3.5)
        set_landmark(frame, 30, x=-1.2, y=3.5)
        set_landmark(frame, 32, x=-0.8, y=3.5)
        return frame

    mirrored = frame.copy()
    mirrored[:, 0] *= -1.0
    set_landmark(mirrored, 29, x=-0.8, y=3.5)
    set_landmark(mirrored, 31, x=-1.2, y=3.5)
    set_landmark(mirrored, 30, x=1.2, y=3.5)
    set_landmark(mirrored, 32, x=0.8, y=3.5)
    return mirrored


def build_lunge_phase_frame() -> np.ndarray:
    frame = make_landmark_frame()
    set_landmark(frame, 11, x=-1.0, y=1.0)
    set_landmark(frame, 12, x=1.0, y=1.0)
    set_landmark(frame, 23, x=-1.0, y=2.0)
    set_landmark(frame, 24, x=1.0, y=2.0)
    set_landmark(frame, 25, x=-1.0, y=3.0)
    set_landmark(frame, 26, x=1.0, y=3.0)
    return frame


def test_normalize_facing_direction_flips_x_when_subject_faces_left() -> None:
    frame = build_lunge_frame(facing_right=False)

    normalized = normalize_facing_direction(frame[:, :3])

    assert normalized[31, 0] > normalized[29, 0]
    assert normalized[32, 0] > normalized[30, 0]


def test_select_front_back_joint_indices_chooses_front_leg_by_ankle_position() -> None:
    frame = build_lunge_frame()
    indices = select_front_back_joint_indices(frame[:, :3])

    assert indices["front_hip"] == 23
    assert indices["back_hip"] == 24
    assert indices["front_ankle"] == 27
    assert indices["back_ankle"] == 28


def test_select_front_back_joint_indices_supports_right_leg_front() -> None:
    frame = build_lunge_frame().copy()
    frame[27, 0] = -2.0
    frame[28, 0] = 2.0

    indices = select_front_back_joint_indices(frame[:, :3])

    assert indices["front_hip"] == 24
    assert indices["back_hip"] == 23
    assert indices["front_ankle"] == 28
    assert indices["back_ankle"] == 27


def test_extract_bottom_features_returns_expected_key_values() -> None:
    features = extract_bottom_features(build_lunge_frame())

    assert features.shape == (BOTTOM_FEATURE_DIM,)
    np.testing.assert_allclose(
        features[:6],
        np.array([0.5, -1.5, 0.0, -0.5, -1.5, 0.0], dtype=np.float32),
        atol=1e-5,
    )
    assert features[34] == pytest.approx(0.0, abs=1e-3)
    assert features[35] == pytest.approx(2.0)
    assert features[36] == pytest.approx(0.0)
    assert features[37] == pytest.approx(0.0)
    assert features[38] == pytest.approx(1.0)
    assert features[39] == pytest.approx(0.8)
    assert features[40] == pytest.approx(1.0, abs=1e-3)
    assert features[41] == pytest.approx(2.0)


def test_extract_bottom_features_accepts_sequences() -> None:
    frame = build_lunge_frame()
    sequence = np.stack([frame, frame], axis=0)

    features = extract_bottom_features(sequence)

    assert features.shape == (2, BOTTOM_FEATURE_DIM)
    np.testing.assert_allclose(features[0], features[1])


def test_extract_phase_features_returns_heights_and_velocities() -> None:
    features, current_heights = extract_phase_features(
        build_lunge_phase_frame(),
        previous_heights=(0.25, -0.75, 0.5),
    )

    assert features.shape == (PHASE_FEATURE_DIM,)
    np.testing.assert_allclose(
        features,
        np.array([0.0, -1.0, 1.0, -0.25, -0.25, 0.5], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        current_heights,
        np.array([0.0, -1.0, 1.0], dtype=np.float32),
        atol=1e-5,
    )


def test_extract_phase_features_uses_zero_velocity_without_previous_heights() -> None:
    features, current_heights = extract_phase_features(
        build_lunge_phase_frame(),
        previous_heights=None,
    )

    np.testing.assert_allclose(
        features,
        np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        current_heights,
        np.array([0.0, -1.0, 1.0], dtype=np.float32),
        atol=1e-5,
    )


def test_landmark_smoother_updates_and_resets() -> None:
    smoother = LandmarkSmoother(alpha=0.5)
    first = np.zeros((33, 4), dtype=np.float32)
    second = np.ones((33, 4), dtype=np.float32)

    np.testing.assert_allclose(smoother.update(first), first)
    np.testing.assert_allclose(smoother.update(second), np.full((33, 4), 0.5, dtype=np.float32))
    smoother.reset()
    assert smoother.previous_landmarks is None
