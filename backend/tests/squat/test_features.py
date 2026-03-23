from __future__ import annotations

import numpy as np
import pytest

from squat.features import (
    BOTTOM_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    STAND_FEATURE_DIM,
    extract_bottom_features,
    extract_phase_features,
    extract_stand_features,
)
from tests.helpers.landmarks import make_landmark_frame, set_landmark


def build_squat_frame() -> np.ndarray:
    frame = make_landmark_frame()
    set_landmark(frame, 7, x=-1.0, y=-1.0)
    set_landmark(frame, 8, x=1.0, y=-1.0)
    set_landmark(frame, 11, x=-1.0, y=0.0)
    set_landmark(frame, 12, x=1.0, y=0.0)
    set_landmark(frame, 23, x=-1.0, y=2.0)
    set_landmark(frame, 24, x=1.0, y=2.0)
    set_landmark(frame, 25, x=-1.0, y=4.0)
    set_landmark(frame, 26, x=1.0, y=4.0)
    set_landmark(frame, 27, x=-1.0, y=6.0)
    set_landmark(frame, 28, x=1.0, y=6.0)
    return frame


def test_extract_phase_features_returns_expected_values_for_upright_pose() -> None:
    features = extract_phase_features(build_squat_frame())

    assert features.shape == (PHASE_FEATURE_DIM,)
    np.testing.assert_allclose(
        features,
        np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32),
        atol=1e-3,
    )


def test_extract_phase_features_accepts_sequences() -> None:
    frame = build_squat_frame()
    sequence = np.stack([frame, frame], axis=0)

    features = extract_phase_features(sequence)

    assert features.shape == (2, PHASE_FEATURE_DIM)
    np.testing.assert_allclose(features[0], features[1])


def test_extract_stand_features_returns_expected_ratios_and_offsets() -> None:
    features = extract_stand_features(build_squat_frame())

    assert features.shape == (STAND_FEATURE_DIM,)
    np.testing.assert_allclose(
        features,
        np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                -0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
        atol=1e-3,
    )


def test_extract_stand_features_accepts_sequences() -> None:
    frame = build_squat_frame()
    sequence = np.stack([frame, frame], axis=0)

    features = extract_stand_features(sequence)

    assert features.shape == (2, STAND_FEATURE_DIM)
    np.testing.assert_allclose(features[0], features[1])


def test_extract_bottom_features_returns_expected_normalized_pose() -> None:
    features = extract_bottom_features(build_squat_frame())

    assert features.shape == (BOTTOM_FEATURE_DIM,)
    np.testing.assert_allclose(
        features[:30],
        np.array(
            [
                -0.5,
                -1.5,
                0.0,
                0.5,
                -1.5,
                0.0,
                -0.5,
                -1.0,
                0.0,
                0.5,
                -1.0,
                0.0,
                -0.5,
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
                -0.5,
                1.0,
                0.0,
                0.5,
                1.0,
                0.0,
                -0.5,
                2.0,
                0.0,
                0.5,
                2.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(features[30:], np.ones(11, dtype=np.float32), atol=1e-3)


def test_extract_bottom_features_accepts_sequences() -> None:
    frame = build_squat_frame()
    sequence = np.stack([frame, frame, frame], axis=0)

    features = extract_bottom_features(sequence)

    assert features.shape == (3, BOTTOM_FEATURE_DIM)
    np.testing.assert_allclose(features[0], features[2])
