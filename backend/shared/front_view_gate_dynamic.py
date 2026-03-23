"""
Front-view visibility gate for exercises requiring frontal camera angle.

Used by squat to verify that the user is positioned facing the camera
with sufficient landmark visibility and left-right body separation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class FrontViewGateDynamic:
    """Front-view gate: visibility check plus left-right body separation.

    Ensures that both sides of the body are visible and that
    shoulders and hips show sufficient horizontal separation
    (indicating a front-facing view).
    """

    def __init__(
        self,
        mp_pose: Any,
        vis_th: float,
        min_sho_gap: float,
        min_hip_gap: float,
    ) -> None:
        """Initialize the front view gate.

        Args:
            mp_pose: MediaPipe pose solutions module (mp.solutions.pose).
            vis_th: Minimum visibility threshold for required landmarks.
            min_sho_gap: Minimum horizontal gap between left/right shoulders.
            min_hip_gap: Minimum horizontal gap between left/right hips.
        """
        self.mp_pose = mp_pose
        self.vis_th = vis_th
        self.min_sho_gap = min_sho_gap
        self.min_hip_gap = min_hip_gap

        # Build landmark lists from the passed mp_pose module
        self.FRONT_LM: List[int] = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

        self.FRONT_LM_LABELS: Dict[int, str] = {
            mp_pose.PoseLandmark.LEFT_SHOULDER: "L_SHO",
            mp_pose.PoseLandmark.RIGHT_SHOULDER: "R_SHO",
            mp_pose.PoseLandmark.LEFT_HIP: "L_HIP",
            mp_pose.PoseLandmark.RIGHT_HIP: "R_HIP",
            mp_pose.PoseLandmark.LEFT_KNEE: "L_KNE",
            mp_pose.PoseLandmark.RIGHT_KNEE: "R_KNE",
            mp_pose.PoseLandmark.LEFT_ANKLE: "L_ANK",
            mp_pose.PoseLandmark.RIGHT_ANKLE: "R_ANK",
        }

    def evaluate(self, landmarks: List[Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if the front-view gate passes.

        Args:
            landmarks: MediaPipe landmark list.

        Returns:
            Tuple of (gate_passes, debug_info_dict).
        """
        visibility_by_landmark: Dict[str, float] = {}
        visibility_ok = True
        for idx in self.FRONT_LM:
            visibility = float(landmarks[idx].visibility)
            visibility_by_landmark[self.FRONT_LM_LABELS.get(idx, str(idx))] = round(
                visibility,
                3,
            )
            if visibility < self.vis_th:
                visibility_ok = False

        left_shoulder_x = float(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        right_shoulder_x = float(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        left_hip_x = float(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x)
        right_hip_x = float(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x)
        shoulder_gap = abs(left_shoulder_x - right_shoulder_x)
        hip_gap = abs(left_hip_x - right_hip_x)
        gap_ok = (shoulder_gap >= self.min_sho_gap) and (hip_gap >= self.min_hip_gap)

        debug_info = {
            "vis_ok": visibility_ok,
            "gap_ok": gap_ok,
            "sho_gap": round(shoulder_gap, 3),
            "hip_gap": round(hip_gap, 3),
            "vis_th": float(self.vis_th),
            "vis": visibility_by_landmark,
        }
        return (visibility_ok and gap_ok), debug_info
