from __future__ import annotations

from typing import Any

import numpy as np


def angle_from_points(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    point_a_array = np.array(point_a, dtype=np.float32)
    point_b_array = np.array(point_b, dtype=np.float32)
    point_c_array = np.array(point_c, dtype=np.float32)
    vector_ba = point_a_array - point_b_array
    vector_bc = point_c_array - point_b_array
    denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc) + 1e-6
    cosine_angle = np.clip(np.dot(vector_ba, vector_bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def safe_norm(vector: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.sqrt(np.sum(vector * vector)) + eps)


def angle_to_direction(vector: np.ndarray, direction: np.ndarray) -> float:
    vector_array = np.asarray(vector, dtype=np.float32)
    direction_array = np.asarray(direction, dtype=np.float32)
    denominator = safe_norm(vector_array) * safe_norm(direction_array) + 1e-6
    cosine_angle = np.clip(
        np.dot(vector_array, direction_array) / denominator,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine_angle)))


def pick_scale(
    *candidates: float,
    min_scale: float = 1e-4,
    fallback: float = 1.0,
) -> float:
    for candidate in candidates:
        candidate_value = float(candidate)
        if candidate_value > min_scale:
            return candidate_value
    return float(fallback)


def position_normalize(
    value: float | np.ndarray,
    center: float | np.ndarray = 0.5,
    scale: float | np.ndarray = 0.5,
    eps: float = 1e-6,
) -> float | np.ndarray:
    value_arr = np.asarray(value, dtype=np.float32)
    center_arr = np.asarray(center, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    normalized = (value_arr - center_arr) / (scale_arr + eps)

    if normalized.ndim == 0:
        return float(normalized)
    return normalized.astype(np.float32)


def point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((point_a - point_b) ** 2)))


def landmarks_to_xyz(landmarks: list) -> np.ndarray:
    xyz_coordinates = np.zeros((33, 3), dtype=np.float32)
    for index in range(33):
        xyz_coordinates[index, 0] = landmarks[index].x
        xyz_coordinates[index, 1] = landmarks[index].y
        xyz_coordinates[index, 2] = landmarks[index].z
    return xyz_coordinates


def as_xyz_frame(landmarks: Any) -> np.ndarray:
    if isinstance(landmarks, np.ndarray):
        landmark_array = np.asarray(landmarks, dtype=np.float32)
        if (
            landmark_array.ndim != 2
            or landmark_array.shape[0] != 33
            or landmark_array.shape[1] < 3
        ):
            raise ValueError("Expected landmark frame with shape (33, 3/4).")
        return landmark_array[:, :3].astype(np.float32)
    return landmarks_to_xyz(landmarks)


def as_xy_frame(landmarks: Any) -> np.ndarray:
    if isinstance(landmarks, np.ndarray):
        landmark_array = np.asarray(landmarks, dtype=np.float32)
        if (
            landmark_array.ndim != 2
            or landmark_array.shape[0] != 33
            or landmark_array.shape[1] < 2
        ):
            raise ValueError("Expected landmark frame with shape (33, 2/3/4).")
        return landmark_array[:, :2].astype(np.float32)
    return landmarks_to_xyz(landmarks)[:, :2]


def as_xyz_sequence(landmark_sequence: np.ndarray) -> np.ndarray:
    landmark_array = np.asarray(landmark_sequence, dtype=np.float32)
    if (
        landmark_array.ndim != 3
        or landmark_array.shape[1] != 33
        or landmark_array.shape[2] < 3
    ):
        raise ValueError("Expected landmark sequence with shape (T, 33, 3/4).")
    return landmark_array[..., :3].astype(np.float32)


angle_3pts = angle_from_points
dist = point_distance
get_xyz = landmarks_to_xyz
