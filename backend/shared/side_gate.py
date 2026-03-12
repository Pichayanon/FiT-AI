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

    def side_score(
        self, landmarks: List[Any], side: str
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Compute visibility score for the given side.

        Args:
            landmarks: MediaPipe landmark list.
            side: "left" or "right".

        Returns:
            Tuple of (all_visible, average_visibility, visibility_map).
        """
        vis_map: Dict[str, float] = {}
        ok = True
        vis_sum = 0.0

        for idx in self.SIDE_LM[side]:
            v = float(landmarks[idx].visibility)
            vis_map[self.REQ_LM_LABELS.get(idx, str(idx))] = v
            vis_sum += v
            if v < self.vis_th:
                ok = False

        avg = vis_sum / max(1, len(self.SIDE_LM[side]))
        return ok, avg, vis_map

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
        left_ok, left_avg, left_map = self.side_score(landmarks, "left")
        right_ok, right_avg, right_map = self.side_score(landmarks, "right")

        debug: Dict[str, Any] = {
            "left_ok": left_ok,
            "left_avg": round(left_avg, 3),
            "left_vis": left_map,
            "right_ok": right_ok,
            "right_avg": round(right_avg, 3),
            "right_vis": right_map,
            "mode": self.side_mode,
            "vis_th": self.vis_th,
        }

        if self.side_mode == "left":
            return ("left" if left_ok else None), debug
        if self.side_mode == "right":
            return ("right" if right_ok else None), debug

        # auto mode
        if left_ok and not right_ok:
            return "left", debug
        if right_ok and not left_ok:
            return "right", debug
        if left_ok and right_ok:
            return ("left" if left_avg >= right_avg else "right"), debug

        return None, debug
