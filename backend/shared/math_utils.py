"""
Shared math utility functions for pose analysis.

Provides angle computation, distance calculation, and other
geometric utilities used across all exercise feature extractors.
"""

from __future__ import annotations

import numpy as np


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle ABC (in degrees) from 2D or 3D points.

    Args:
        a: First point (vertex of the angle is at b).
        b: Vertex point.
        c: Third point.

    Returns:
        Angle in degrees (0 to 180).
    """
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    c_arr = np.array(c, dtype=np.float32)
    ba = a_arr - b_arr
    bc = c_arr - b_arr
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the L2 norm of a vector with a small epsilon for stability.

    Args:
        v: Input vector.
        eps: Small constant added to prevent division by zero.

    Returns:
        L2 norm plus eps.
    """
    return float(np.sqrt(np.sum(v * v)) + eps)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        a: First point.
        b: Second point.

    Returns:
        Euclidean distance.
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def get_xyz(lm: list) -> np.ndarray:
    """Extract (33, 3) xyz coordinates from MediaPipe landmarks.

    Args:
        lm: MediaPipe landmark list (33 landmarks).

    Returns:
        Array of shape (33, 3) with x, y, z coordinates.
    """
    xyz = np.zeros((33, 3), dtype=np.float32)
    for i in range(33):
        xyz[i, 0] = lm[i].x
        xyz[i, 1] = lm[i].y
        xyz[i, 2] = lm[i].z
    return xyz
