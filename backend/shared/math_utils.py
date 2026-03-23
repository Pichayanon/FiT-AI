"""
Shared math utility functions for pose analysis.

Provides angle computation, position normalization, distance calculation, and other
geometric utilities used across all exercise feature extractors.
"""

from __future__ import annotations

import numpy as np


def angle_from_points(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    """Compute the angle ABC (in degrees) from 2D or 3D points.

    Args:
        point_a: First point (vertex of the angle is at point_b).
        point_b: Vertex point.
        point_c: Third point.

    Returns:
        Angle in degrees (0 to 180).
    """
    point_a_array = np.array(point_a, dtype=np.float32)
    point_b_array = np.array(point_b, dtype=np.float32)
    point_c_array = np.array(point_c, dtype=np.float32)
    vector_ba = point_a_array - point_b_array
    vector_bc = point_c_array - point_b_array
    denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc) + 1e-6
    cosine_angle = np.clip(np.dot(vector_ba, vector_bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def safe_norm(vector: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the L2 norm of a vector with a small epsilon for stability.

    Args:
        vector: Input vector.
        eps: Small constant added to prevent division by zero.

    Returns:
        L2 norm plus eps.
    """
    return float(np.sqrt(np.sum(vector * vector)) + eps)


def position_normalize(
    value: float | np.ndarray,
    center: float | np.ndarray = 0.5,
    scale: float | np.ndarray = 0.5,
    eps: float = 1e-6,
) -> float | np.ndarray:
    """Normalize a single x/y value or array of positions.

    Uses the shared `(value - center) / scale` convention that appears across
    the pose feature extractors. Defaults map MediaPipe-style image coordinates
    from `[0, 1]` into roughly `[-1, 1]`.

    Args:
        value: Position value(s) to normalize.
        center: Reference center to subtract before scaling.
        scale: Reference scale used to normalize the offset.
        eps: Small constant added to prevent division by zero.

    Returns:
        Normalized value(s) with the same scalar/array shape.
    """
    value_arr = np.asarray(value, dtype=np.float32)
    center_arr = np.asarray(center, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    normalized = (value_arr - center_arr) / (scale_arr + eps)

    if normalized.ndim == 0:
        return float(normalized)
    return normalized.astype(np.float32)


def point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point_a: First point.
        point_b: Second point.

    Returns:
        Euclidean distance.
    """
    return float(np.sqrt(np.sum((point_a - point_b) ** 2)))


def landmarks_to_xyz(landmarks: list) -> np.ndarray:
    """Extract (33, 3) xyz coordinates from MediaPipe landmarks.

    Args:
        landmarks: MediaPipe landmark list (33 landmarks).

    Returns:
        Array of shape (33, 3) with x, y, z coordinates.
    """
    xyz_coordinates = np.zeros((33, 3), dtype=np.float32)
    for index in range(33):
        xyz_coordinates[index, 0] = landmarks[index].x
        xyz_coordinates[index, 1] = landmarks[index].y
        xyz_coordinates[index, 2] = landmarks[index].z
    return xyz_coordinates


# Backward-compatible aliases while the rest of the codebase is being renamed.
angle_3pts = angle_from_points
dist = point_distance
get_xyz = landmarks_to_xyz
