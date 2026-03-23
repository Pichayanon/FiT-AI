from __future__ import annotations

from types import SimpleNamespace

import numpy as np

LANDMARK_COUNT = 33


def make_landmark_frame(default_visibility: float = 1.0) -> np.ndarray:
    frame = np.zeros((LANDMARK_COUNT, 4), dtype=np.float32)
    frame[:, 3] = default_visibility
    return frame


def set_landmark(
    frame: np.ndarray,
    index: int,
    *,
    x: float,
    y: float,
    z: float = 0.0,
    visibility: float | None = None,
) -> np.ndarray:
    if visibility is None:
        visibility = float(frame[index, 3]) if frame.shape[1] > 3 else 1.0
    frame[index] = np.array([x, y, z, visibility], dtype=np.float32)
    return frame


def landmarks_to_list(frame: np.ndarray) -> list[SimpleNamespace]:
    landmark_list: list[SimpleNamespace] = []
    for landmark in frame:
        visibility = float(landmark[3]) if landmark.shape[0] > 3 else 1.0
        landmark_list.append(
            SimpleNamespace(
                x=float(landmark[0]),
                y=float(landmark[1]),
                z=float(landmark[2]),
                visibility=visibility,
            )
        )
    return landmark_list


def make_pose_result(landmarks: list[SimpleNamespace] | None) -> SimpleNamespace:
    if landmarks is None:
        return SimpleNamespace(pose_landmarks=None)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))
