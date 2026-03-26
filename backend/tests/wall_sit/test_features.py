from __future__ import annotations

import numpy as np

from tests.helpers.landmarks import (
    landmarks_to_list,
    make_landmark_frame,
    make_pose_result,
    set_landmark,
)
from wall_sit.features import (
    StreamState,
    WallSitFeatureExtractor,
    aggregate_window,
    aggregate_window_features,
    extract_frame_features,
    side_indices,
)


def build_wall_sit_landmarks() -> list:
    frame = make_landmark_frame()
    set_landmark(frame, 11, x=-1.0, y=0.0)
    set_landmark(frame, 12, x=1.0, y=0.0)
    set_landmark(frame, 23, x=-1.0, y=1.0)
    set_landmark(frame, 24, x=1.0, y=1.0)
    set_landmark(frame, 25, x=-2.0, y=1.0)
    set_landmark(frame, 26, x=2.0, y=1.0)
    set_landmark(frame, 27, x=-2.0, y=2.0)
    set_landmark(frame, 28, x=2.0, y=2.0)
    return landmarks_to_list(frame)


def test_side_indices_return_expected_joint_order() -> None:
    assert side_indices("right") == (12, 24, 26, 28)
    assert side_indices("left") == (11, 23, 25, 27)


def test_extract_frame_features_returns_expected_wall_sit_values() -> None:
    features = extract_frame_features(build_wall_sit_landmarks(), "right")

    # (foot_wall_norm, knee_angle, torso_alignment, hip_knee_height_ratio)
    assert len(features) == 4
    np.testing.assert_allclose(
        features[:3],
        np.array([0.5, 90.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    # hip and knee at same y=1.0, so height ratio ≈ 0.0
    np.testing.assert_allclose(features[3], 0.0, atol=1e-5)


def test_aggregate_window_features_returns_expected_statistics() -> None:
    frames = [
        (0.5, 90.0, 0.0, 0.1),
        (1.0, 45.0, 0.5, 0.8),
    ]
    aggregated = aggregate_window_features(frames)

    # 7 aggregated features now (added hip_knee_height weighted mean + std)
    assert aggregated.shape == (7,)
    # Weighted mean of foot_wall should be > simple mean (0.75)
    assert aggregated[0] > 0.75
    # std is unweighted, stays the same
    np.testing.assert_allclose(aggregated[1], 0.25, atol=1e-5)
    # Weighted mean of knee_angle should be < simple mean (67.5)
    assert aggregated[2] < 67.5
    # min is unweighted, stays the same
    np.testing.assert_allclose(aggregated[3], 45.0, atol=1e-5)
    # Weighted mean of hip_knee_height should be > simple mean (0.45)
    assert aggregated[5] > 0.45
    # std of hip_knee_height
    np.testing.assert_allclose(aggregated[6], np.std([0.1, 0.8]), atol=1e-5)


def test_static_aggregate_window_matches_module_function() -> None:
    frame_values = [(0.5, 90.0, 0.0, 0.1), (1.0, 45.0, 0.5, 0.8)]

    np.testing.assert_allclose(
        WallSitFeatureExtractor.aggregate_window(frame_values),
        aggregate_window(frame_values),
    )


def test_wall_sit_feature_extractor_handles_missing_and_present_pose_results() -> None:
    extractor = WallSitFeatureExtractor(mp_pose=object())

    assert extractor.extract_features(make_pose_result(None), "right") is None
    result = extractor.extract_features(make_pose_result(build_wall_sit_landmarks()), "right")
    assert len(result) == 4
    np.testing.assert_allclose(
        result[:3],
        np.array([0.5, 90.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(result[3], 0.0, atol=1e-5)


def test_stream_state_starts_with_empty_features_and_defaults() -> None:
    state = StreamState()

    assert state.started is False
    assert state.frame_features == []
    assert state.ready is False
    assert state.chosen_side is None
