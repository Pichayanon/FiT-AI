"""
Side-view visibility gate for exercises requiring lateral camera angle.

Used by plank and wall_sit to verify that the user is positioned in a
side view with sufficient landmark visibility before processing begins.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class SideGate:
    """Side-view visibility gate and side selector.

    Chooses the best side (left/right) based on landmark visibility and
    ensures that required landmarks are above a given visibility threshold.
    """

    def __init__(self, mp_pose: Any, side_mode: str, vis_th: float) -> None:
        """Initialize the side gate.

        Args:
            mp_pose: MediaPipe pose solutions module (mp.solutions.pose).
            side_mode: Side selection mode — "auto", "left", or "right".
            vis_th: Minimum visibility threshold for required landmarks.
        """
        self.mp_pose = mp_pose
        self.side_mode = side_mode
        self.vis_th = vis_th

        self.SIDE_LM: Dict[str, List[int]] = {
            "left": [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            ],
            "right": [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            ],
        }

        self.REQ_LM_LABELS: Dict[int, str] = {
            mp_pose.PoseLandmark.LEFT_SHOULDER: "L_SHO",
            mp_pose.PoseLandmark.LEFT_HIP: "L_HIP",
            mp_pose.PoseLandmark.LEFT_KNEE: "L_KNEE",
            mp_pose.PoseLandmark.LEFT_ANKLE: "L_ANK",
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX: "L_FOOT",
            mp_pose.PoseLandmark.RIGHT_SHOULDER: "R_SHO",
            mp_pose.PoseLandmark.RIGHT_HIP: "R_HIP",
            mp_pose.PoseLandmark.RIGHT_KNEE: "R_KNE",
            mp_pose.PoseLandmark.RIGHT_ANKLE: "R_ANK",
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: "R_FOOT",
        }

    def score_side_visibility(
        self,
        landmarks: List[Any],
        side: str,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Compute visibility score for the given side.

        Args:
            landmarks: MediaPipe landmark list.
            side: "left" or "right".

        Returns:
            Tuple of (all_visible, average_visibility, visibility_map).
        """
        visibility_by_landmark: Dict[str, float] = {}
        all_visible = True
        visibility_sum = 0.0

        for idx in self.SIDE_LM[side]:
            visibility = float(landmarks[idx].visibility)
            visibility_by_landmark[self.REQ_LM_LABELS.get(idx, str(idx))] = visibility
            visibility_sum += visibility
            if visibility < self.vis_th:
                all_visible = False

        average_visibility = visibility_sum / max(1, len(self.SIDE_LM[side]))
        return all_visible, average_visibility, visibility_by_landmark

    def choose_best_side(
        self, landmarks: List[Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Choose the best visible side for processing.

        In "auto" mode, selects the side with better visibility.
        In "left"/"right" mode, only accepts the specified side.

        Args:
            landmarks: MediaPipe landmark list.

        Returns:
            Tuple of (side_name_or_None, debug_info_dict).
        """
        left_visible, left_average, left_visibility = self.score_side_visibility(
            landmarks,
            "left",
        )
        right_visible, right_average, right_visibility = self.score_side_visibility(
            landmarks,
            "right",
        )

        debug_info: Dict[str, Any] = {
            "left_ok": left_visible,
            "left_avg": round(left_average, 3),
            "left_vis": left_visibility,
            "right_ok": right_visible,
            "right_avg": round(right_average, 3),
            "right_vis": right_visibility,
            "mode": self.side_mode,
            "vis_th": self.vis_th,
        }

        if self.side_mode == "left":
            return ("left" if left_visible else None), debug_info
        if self.side_mode == "right":
            return ("right" if right_visible else None), debug_info

        # auto mode
        if left_visible and not right_visible:
            return "left", debug_info
        if right_visible and not left_visible:
            return "right", debug_info
        if left_visible and right_visible:
            return (
                "left" if left_average >= right_average else "right"
            ), debug_info

        return None, debug_info
