"""Shared plank feature extraction helpers.

Used by both training and streaming so plank feature computation stays
consistent across offline and realtime paths.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from shared.math_utils import (
    angle_from_points,
    landmarks_to_xyz,
    point_distance,
    position_normalize,
)


LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28


def _side_indices(side: str) -> Tuple[int, int, int]:
    """Return shoulder/hip/ankle landmark indices for the chosen side."""
    if side == "right":
        return RIGHT_SHOULDER_INDEX, RIGHT_HIP_INDEX, RIGHT_ANKLE_INDEX
    return LEFT_SHOULDER_INDEX, LEFT_HIP_INDEX, LEFT_ANKLE_INDEX


def _angle_from_horizontal_deg(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """Return the segment angle relative to horizontal in the range [0, 90]."""
    delta = np.asarray(b_xy, dtype=np.float32) - np.asarray(a_xy, dtype=np.float32)
    angle_deg = abs(float(np.degrees(np.arctan2(delta[1], delta[0]))))
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg
    return angle_deg


def choose_visible_side(landmarks: list) -> str:
    """Select the body side with the more visible hip landmark."""
    right_hip_visibility = landmarks[RIGHT_HIP_INDEX].visibility
    left_hip_visibility = landmarks[LEFT_HIP_INDEX].visibility
    return "right" if right_hip_visibility > left_hip_visibility else "left"


def extract_posture_metrics(landmarks: list, side: str) -> Dict[str, float]:
    """Extract orientation metrics used to gate plank-ready posture.

    The model features are intentionally translation/scale normalized, which makes
    them poor at separating an upright body from a horizontal plank. These metrics
    preserve image-space orientation so realtime inference only runs once the body
    is sufficiently horizontal.
    """
    landmark_xyz = landmarks_to_xyz(landmarks)
    shoulder_index, hip_index, ankle_index = _side_indices(side)

    selected_shoulder = landmark_xyz[shoulder_index]
    selected_hip = landmark_xyz[hip_index]
    selected_ankle = landmark_xyz[ankle_index]

    torso_angle = _angle_from_horizontal_deg(
        selected_shoulder[:2],
        selected_hip[:2],
    )
    leg_angle = _angle_from_horizontal_deg(
        selected_hip[:2],
        selected_ankle[:2],
    )
    body_axis_angle = _angle_from_horizontal_deg(
        selected_shoulder[:2],
        selected_ankle[:2],
    )
    torso_dx = abs(float(selected_shoulder[0] - selected_hip[0]))
    torso_dy = abs(float(selected_shoulder[1] - selected_hip[1]))
    leg_dx = abs(float(selected_ankle[0] - selected_hip[0]))
    leg_dy = abs(float(selected_ankle[1] - selected_hip[1]))
    body_x_span = float(
        max(selected_shoulder[0], selected_hip[0], selected_ankle[0])
        - min(selected_shoulder[0], selected_hip[0], selected_ankle[0])
    )
    body_y_span = float(
        max(selected_shoulder[1], selected_hip[1], selected_ankle[1])
        - min(selected_shoulder[1], selected_hip[1], selected_ankle[1])
    )

    return {
        "torso_angle_deg": float(torso_angle),
        "leg_angle_deg": float(leg_angle),
        "body_axis_angle_deg": float(body_axis_angle),
        "torso_horizontal_ratio": torso_dx / (torso_dy + 1e-6),
        "leg_horizontal_ratio": leg_dx / (leg_dy + 1e-6),
        "body_horizontal_ratio": body_x_span / (body_y_span + 1e-6),
        "body_x_span": body_x_span,
        "body_y_span": body_y_span,
    }


def is_plank_ready_pose(
    landmarks: list,
    side: str,
    *,
    max_body_axis_angle_deg: float = 35.0,
    max_torso_angle_deg: float = 45.0,
    max_leg_angle_deg: float = 45.0,
    min_body_horizontal_ratio: float = 1.15,
    min_torso_horizontal_ratio: float = 0.75,
    min_leg_horizontal_ratio: float = 0.75,
) -> Tuple[bool, Dict[str, float]]:
    """Check whether the body is horizontal enough to start plank inference."""
    metrics = extract_posture_metrics(landmarks, side)
    is_ready = (
        metrics["body_axis_angle_deg"] <= max_body_axis_angle_deg
        and metrics["torso_angle_deg"] <= max_torso_angle_deg
        and metrics["leg_angle_deg"] <= max_leg_angle_deg
        and metrics["body_horizontal_ratio"] >= min_body_horizontal_ratio
        and metrics["torso_horizontal_ratio"] >= min_torso_horizontal_ratio
        and metrics["leg_horizontal_ratio"] >= min_leg_horizontal_ratio
    )
    metrics.update(
        {
            "plank_pose_ok": is_ready,
            "max_body_axis_angle_deg": float(max_body_axis_angle_deg),
            "max_torso_angle_deg": float(max_torso_angle_deg),
            "max_leg_angle_deg": float(max_leg_angle_deg),
            "min_body_horizontal_ratio": float(min_body_horizontal_ratio),
            "min_torso_horizontal_ratio": float(min_torso_horizontal_ratio),
            "min_leg_horizontal_ratio": float(min_leg_horizontal_ratio),
        }
    )
    return is_ready, metrics


def extract_frame_features(
    landmarks: list,
    side: str,
) -> Tuple[float, float, float]:
    """Extract normalized plank features from one landmark frame.

    The selected shoulder, hip, and ankle are first normalized in xyz space
    relative to the chosen-side hip. The resulting features are:
        - hip_signed_dist: signed hip distance from shoulder-ankle line
        - hip_height_norm: hip vertical offset from shoulder/ankle midpoint
        - body_angle: shoulder-hip-ankle angle in degrees
    """
    landmark_xyz = landmarks_to_xyz(landmarks)

    shoulder_index, hip_index, ankle_index = _side_indices(side)

    selected_shoulder = landmark_xyz[shoulder_index]
    selected_hip = landmark_xyz[hip_index]
    selected_ankle = landmark_xyz[ankle_index]

    torso_length = point_distance(selected_shoulder, selected_hip)
    body_length = point_distance(selected_shoulder, selected_ankle)
    hip_width = point_distance(
        landmark_xyz[LEFT_HIP_INDEX],
        landmark_xyz[RIGHT_HIP_INDEX],
    )
    shoulder_width = point_distance(
        landmark_xyz[LEFT_SHOULDER_INDEX],
        landmark_xyz[RIGHT_SHOULDER_INDEX],
    )
    reference_scale = torso_length
    if reference_scale <= 1e-4:
        reference_scale = body_length
    if reference_scale <= 1e-4:
        reference_scale = hip_width
    if reference_scale <= 1e-4:
        reference_scale = shoulder_width
    if reference_scale <= 1e-4:
        reference_scale = 1e-6

    normalized_shoulder = position_normalize(
        selected_shoulder,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_hip = position_normalize(
        selected_hip,
        center=selected_hip,
        scale=reference_scale,
    )
    normalized_ankle = position_normalize(
        selected_ankle,
        center=selected_hip,
        scale=reference_scale,
    )

    normalized_shoulder_xy = normalized_shoulder[:2]
    normalized_hip_xy = normalized_hip[:2]
    normalized_ankle_xy = normalized_ankle[:2]

    shoulder_to_ankle_vector = normalized_ankle_xy - normalized_shoulder_xy
    shoulder_to_hip_vector = normalized_hip_xy - normalized_shoulder_xy
    hip_signed_dist = np.cross(
        shoulder_to_ankle_vector,
        shoulder_to_hip_vector,
    ) / (np.linalg.norm(shoulder_to_ankle_vector) + 1e-6)

    body_angle = angle_from_points(
        normalized_shoulder,
        normalized_hip,
        normalized_ankle,
    )

    normalized_body_length = (
        np.linalg.norm(normalized_ankle_xy - normalized_shoulder_xy) + 1e-6
    )
    hip_height_norm = (
        normalized_hip_xy[1]
        - 0.5 * (normalized_shoulder_xy[1] + normalized_ankle_xy[1])
    ) / normalized_body_length

    return float(hip_signed_dist), float(hip_height_norm), float(body_angle)


def aggregate_window_features(
    frame_values: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Aggregate per-frame tuples into the model's 6-D plank feature vector.

    Feature order:
        [0] mean hip signed distance
        [1] std hip signed distance
        [2] mean hip height offset
        [3] std hip height offset
        [4] mean body angle
        [5] std body angle
    """
    hip_signed_distances = [frame_value[0] for frame_value in frame_values]
    hip_height_offsets = [frame_value[1] for frame_value in frame_values]
    body_angles = [frame_value[2] for frame_value in frame_values]

    return np.array(
        [
            np.mean(hip_signed_distances),
            np.std(hip_signed_distances),
            np.mean(hip_height_offsets),
            np.std(hip_height_offsets),
            np.mean(body_angles),
            np.std(body_angles),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------
# Feature extractor (streaming adapter)
# ---------------------------------------------------------------

from typing import Any, Optional


class PlankFeatureExtractor:
    """Adapter that wraps plank feature helpers for use in the streaming session."""

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    def extract_features(
        self,
        pose_result: Any,
        side: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract a single-frame feature tuple for the given side."""
        if not pose_result.pose_landmarks:
            return None
        return extract_frame_features(pose_result.pose_landmarks.landmark, side)

    @staticmethod
    def aggregate_window(
        frame_features: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Aggregate a window of per-frame tuples into a 6-D feature vector."""
        return aggregate_window_features(frame_features)


# Backward-compatible alias
FeatureExtractor = PlankFeatureExtractor


# ---------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class StreamState:
    """Session-level state for a plank streaming WebSocket connection."""

    started: bool = False
    # Per-frame features: (signed_dist, hip_height_norm, body_angle)
    frame_features: List[Tuple[float, float, float]] = field(default_factory=list)

    # Gate
    ready: bool = False
    ready_streak: int = 0
    chosen_side: Optional[str] = None

    # Session metadata
    session_id: str = ""

    # Status throttle
    last_status: str = ""
    status_tick: int = 0

    # Last prediction (for logs / debugging)
    last_prediction_label: str = ""
    last_prediction_confidence: Optional[float] = None

    # Last sent (for optional deduplication)
    last_sent_label: str = ""
    last_sent_confidence: Optional[float] = None

    # Frame counter
    frame_count: int = 0

    # NO_POSE watchdog
    no_pose_since: Optional[float] = None
    no_pose_alerted: bool = False

    # DARK watchdog
    dark_since: Optional[float] = None
    dark_alerted: bool = False


# Backward-compatible alias for older imports.
aggregate_window = aggregate_window_features
