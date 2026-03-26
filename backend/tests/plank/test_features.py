from __future__ import annotations

import numpy as np
import pytest

from plank.features import (
    PlankFeatureExtractor,
    StreamState,
    aggregate_window,
    aggregate_window_features,
    choose_visible_side,
    extract_frame_features,
    extract_posture_metrics,
    horizontal_angle,
    is_plank_ready_pose,
    side_indices,
)
from tests.helpers.landmarks import (
    landmarks_to_list,
    make_landmark_frame,
    make_pose_result,
    set_landmark,
)


def build_horizontal_plank_frame() -> list:
    frame = make_landmark_frame(default_visibility=0.5)
    set_landmark(frame, 11, x=0.0, y=1.0)
    set_landmark(frame, 12, x=0.0, y=0.0)
    set_landmark(frame, 23, x=1.0, y=1.0, visibility=0.2)
    set_landmark(frame, 24, x=1.0, y=0.0, visibility=0.9)
    set_landmark(frame, 27, x=2.0, y=1.0)
    set_landmark(frame, 28, x=2.0, y=0.0)
    return landmarks_to_list(frame)


def build_upright_plank_frame() -> list:
    frame = make_landmark_frame()
    set_landmark(frame, 11, x=0.0, y=0.0)
    set_landmark(frame, 12, x=1.0, y=0.0)
    set_landmark(frame, 23, x=0.0, y=1.0, visibility=0.1)
    set_landmark(frame, 24, x=1.0, y=1.0, visibility=0.9)
    set_landmark(frame, 27, x=0.0, y=2.0)
    set_landmark(frame, 28, x=1.0, y=2.0)
    return landmarks_to_list(frame)


def test_choose_visible_side_prefers_more_visible_hip() -> None:
    assert choose_visible_side(build_horizontal_plank_frame()) == "right"


def test_side_indices_returns_expected_left_and_right_landmarks() -> None:
    assert side_indices("right") == (12, 24, 28)
    assert side_indices("left") == (11, 23, 27)


def test_horizontal_angle_folds_obtuse_angles_toward_horizontal() -> None:
    assert horizontal_angle(np.array([0.0, 0.0]), np.array([-1.0, 1.0])) == pytest.approx(45.0)


def test_extract_posture_metrics_reports_horizontal_pose() -> None:
    metrics = extract_posture_metrics(build_horizontal_plank_frame(), "right")

    assert metrics["torso_angle_deg"] == pytest.approx(0.0)
    assert metrics["leg_angle_deg"] == pytest.approx(0.0)
    assert metrics["body_axis_angle_deg"] == pytest.approx(0.0)


def test_is_plank_ready_pose_accepts_horizontal_pose_and_rejects_upright_pose() -> None:
    ready, ready_metrics = is_plank_ready_pose(build_horizontal_plank_frame(), "right")
    not_ready, upright_metrics = is_plank_ready_pose(build_upright_plank_frame(), "right")

    assert ready is True
    assert ready_metrics["plank_pose_ok"] is True
    assert not_ready is False
    assert upright_metrics["plank_pose_ok"] is False


def test_extract_frame_features_returns_expected_line_geometry() -> None:
    features = extract_frame_features(build_horizontal_plank_frame(), "right")

    np.testing.assert_allclose(
        features,
        np.array([0.0, 0.0, 180.0], dtype=np.float32),
        atol=0.1,
    )


def test_aggregate_window_features_returns_mean_and_std() -> None:
    frames = [
        (0.0, 0.0, 180.0),
        (2.0, 1.0, 120.0),
    ]
    aggregated = aggregate_window_features(frames)

    # With exponential weighting, recent frames are weighted more heavily
    assert aggregated.shape == (6,)
    # Weighted mean of hip_signed_dist should be > simple mean (1.0)
    # because the larger value (2.0) is more recent
    assert aggregated[0] > 1.0
    # std is unweighted, stays the same
    np.testing.assert_allclose(aggregated[1], 1.0, atol=1e-5)
    # Weighted mean of body_angle should be < simple mean (150.0)
    # because the smaller value (120.0) is more recent
    assert aggregated[4] < 150.0
    # std is unweighted, stays the same
    np.testing.assert_allclose(aggregated[5], 30.0, atol=1e-5)


def test_static_aggregate_window_matches_module_function() -> None:
    frame_values = [(0.0, 0.0, 180.0), (2.0, 1.0, 120.0)]

    np.testing.assert_allclose(
        PlankFeatureExtractor.aggregate_window(frame_values),
        aggregate_window(frame_values),
    )


def test_plank_feature_extractor_handles_missing_and_present_pose_results() -> None:
    extractor = PlankFeatureExtractor(mp_pose=object())

    assert extractor.extract_features(make_pose_result(None), "right") is None
    np.testing.assert_allclose(
        extractor.extract_features(make_pose_result(build_horizontal_plank_frame()), "right"),
        np.array([0.0, 0.0, 180.0], dtype=np.float32),
        atol=0.1,
    )


def test_stream_state_starts_with_empty_window_and_no_side() -> None:
    state = StreamState()

    assert state.started is False
    assert state.frame_features == []
    assert state.chosen_side is None
