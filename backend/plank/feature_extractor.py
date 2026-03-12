from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from shared.math_utils import angle_3pts


class FeatureExtractor:
    """Extract per-frame and window-level features for the plank model.

    Per-frame feature:
        (signed_dist, hip_height_norm, body_angle)
    Window-level aggregate feature (6-D):
        mean/std of signed_dist, hip_height_norm, body_angle.
    """

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose



    def extract_features(self, res: Any, side: str) -> Optional[Tuple[float, float, float]]:
        """Extract a single-frame feature tuple for the given side."""
        if not res.pose_landmarks:
            return None

        landmarks = res.pose_landmarks.landmark

        if side == "right":
            hip = self.mp_pose.PoseLandmark.RIGHT_HIP
            ankle = self.mp_pose.PoseLandmark.RIGHT_ANKLE
            shoulder = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        else:
            hip = self.mp_pose.PoseLandmark.LEFT_HIP
            ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE
            shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER

        shoulder_xy = np.array([landmarks[shoulder].x, landmarks[shoulder].y])
        hip_xy = np.array([landmarks[hip].x, landmarks[hip].y])
        ankle_xy = np.array([landmarks[ankle].x, landmarks[ankle].y])

        # 1) Signed distance of hip from shoulder-ankle line
        line_vec = ankle_xy - shoulder_xy
        hip_vec = hip_xy - shoulder_xy
        signed_dist = np.cross(line_vec, hip_vec) / (np.linalg.norm(line_vec) + 1e-6)

        # 2) Body angle (shoulder-hip-ankle)
        body_angle = angle_3pts(shoulder_xy, hip_xy, ankle_xy)

        # 3) Normalized hip height
        body_length = np.linalg.norm(ankle_xy - shoulder_xy) + 1e-6
        hip_height_norm = (
            landmarks[hip].y - 0.5 * (landmarks[shoulder].y + landmarks[ankle].y)
        ) / body_length

        return float(signed_dist), float(hip_height_norm), float(body_angle)

    @staticmethod
    def aggregate_window(values: List[Tuple[float, float, float]]) -> np.ndarray:
        """Aggregate a window of per-frame tuples into a 6-D feature vector."""
        signed_dists = [v[0] for v in values]
        hip_height_norms = [v[1] for v in values]
        body_angles = [v[2] for v in values]

        return np.array(
            [
                np.mean(signed_dists),
                np.std(signed_dists),
                np.mean(hip_height_norms),
                np.std(hip_height_norms),
                np.mean(body_angles),
                np.std(body_angles),
            ],
            dtype=np.float32,
        )

