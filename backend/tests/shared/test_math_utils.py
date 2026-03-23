from __future__ import annotations

import re

import numpy as np
import pytest

from shared.math_utils import (
    angle_from_points,
    angle_to_direction,
    as_xy_frame,
    as_xyz_frame,
    as_xyz_sequence,
    landmarks_to_xyz,
    pick_scale,
    point_distance,
    position_normalize,
    safe_norm,
)
from tests.helpers.landmarks import landmarks_to_list, make_landmark_frame, set_landmark


def test_angle_from_points_returns_right_angle() -> None:
    angle = angle_from_points(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    assert angle == pytest.approx(90.0)


def test_safe_norm_returns_epsilon_for_zero_vector() -> None:
    assert safe_norm(np.zeros(3, dtype=np.float32)) == pytest.approx(1e-6)


def test_angle_to_direction_handles_parallel_and_opposite_vectors() -> None:
    direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert angle_to_direction(direction, direction) < 0.2
    assert angle_to_direction(-direction, direction) > 179.8


def test_pick_scale_returns_first_valid_candidate() -> None:
    assert pick_scale(1e-6, 0.25, 2.0, fallback=9.0) == pytest.approx(0.25)


def test_pick_scale_returns_fallback_when_all_candidates_are_small() -> None:
    assert pick_scale(0.0, 1e-8, fallback=3.0) == pytest.approx(3.0)


def test_position_normalize_supports_scalar_and_array_inputs() -> None:
    assert position_normalize(3.0, center=1.0, scale=2.0) == pytest.approx(1.0)
    normalized = position_normalize(
        np.array([2.0, 4.0], dtype=np.float32),
        center=np.array([1.0, 2.0], dtype=np.float32),
        scale=2.0,
    )
    np.testing.assert_allclose(
        normalized,
        np.array([0.5, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_point_distance_returns_euclidean_distance() -> None:
    distance = point_distance(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([3.0, 4.0, 0.0], dtype=np.float32),
    )
    assert distance == pytest.approx(5.0)


def test_landmarks_to_xyz_and_frame_parsers_accept_mediapipe_like_objects() -> None:
    frame = make_landmark_frame()
    set_landmark(frame, 0, x=1.0, y=2.0, z=3.0)
    landmark_list = landmarks_to_list(frame)

    xyz = landmarks_to_xyz(landmark_list)
    np.testing.assert_allclose(xyz[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(as_xyz_frame(landmark_list)[0], xyz[0])
    np.testing.assert_allclose(as_xy_frame(landmark_list)[0], xyz[0, :2])


def test_numpy_frame_parsers_slice_xyz_and_xy() -> None:
    frame = make_landmark_frame()
    set_landmark(frame, 5, x=3.0, y=4.0, z=5.0, visibility=0.7)

    xyz = as_xyz_frame(frame)
    xy = as_xy_frame(frame)

    np.testing.assert_allclose(xyz[5], np.array([3.0, 4.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(xy[5], np.array([3.0, 4.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("value", "expected_message"),
    [
        (np.zeros((32, 4), dtype=np.float32), "Expected landmark frame with shape (33, 3/4)."),
        (np.zeros((33, 1), dtype=np.float32), "Expected landmark frame with shape (33, 2/3/4)."),
        (np.zeros((3, 32, 4), dtype=np.float32), "Expected landmark sequence with shape (T, 33, 3/4)."),
    ],
)
def test_shape_validators_raise_value_error(
    value: np.ndarray,
    expected_message: str,
) -> None:
    if value.ndim == 2 and value.shape[1] == 1:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            as_xy_frame(value)
        return
    if value.ndim == 2:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            as_xyz_frame(value)
        return
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        as_xyz_sequence(value)
