"""
Side-view visibility gate for dynamic exercises requiring lateral camera angle.

Used by lunges to verify that the user's full body (legs + upper body)
is sufficiently visible before processing begins. Unlike the existing
SideGate (used by plank/wall_sit for left/right side selection), this
gate simply checks overall visibility for dynamic side-view exercises.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SideViewGateDynamic:
    """Side-view visibility gate for dynamic exercises (e.g., lunges).

    Checks that:
        - All leg landmarks (hips, knees, ankles) are visible
        - At least one shoulder is visible (upper body)

    Returns (ok, debug) tuple matching FrontViewGateDynamic interface.
    """

    def __init__(
        self,
        mp_pose: Any,
        vis_th: float = 0.65,
    ) -> None:
        """Initialize the side-view dynamic gate.

        Args:
            mp_pose: MediaPipe pose solutions module (mp.solutions.pose).
            vis_th: Minimum visibility threshold for required landmarks.
        """
        self.mp_pose = mp_pose
        self.vis_th = vis_th

        # Required: all leg landmarks must be visible
        self.LEG_LM: List[int] = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

        # Required: at least one shoulder must be visible
        self.SHOULDER_LM: List[int] = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]
        self.SIDE_LEG_CHAINS: Dict[str, List[int]] = {
            "left": [
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
            ],
            "right": [
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
            ],
        }

        self.LM_LABELS: Dict[int, str] = {
            mp_pose.PoseLandmark.LEFT_HIP: "L_HIP",
            mp_pose.PoseLandmark.RIGHT_HIP: "R_HIP",
            mp_pose.PoseLandmark.LEFT_KNEE: "L_KNE",
            mp_pose.PoseLandmark.RIGHT_KNEE: "R_KNE",
            mp_pose.PoseLandmark.LEFT_ANKLE: "L_ANK",
            mp_pose.PoseLandmark.RIGHT_ANKLE: "R_ANK",
            mp_pose.PoseLandmark.LEFT_SHOULDER: "L_SHO",
            mp_pose.PoseLandmark.RIGHT_SHOULDER: "R_SHO",
        }

    def evaluate(self, landmarks: Any) -> Tuple[bool, Dict[str, Any]]:
        """Check if the side-view visibility gate passes.

        Accepts either MediaPipe landmark list or numpy array (N, 4)
        where column 3 is visibility.

        Args:
            landmarks: MediaPipe landmark list or numpy array of shape (33, 4).

        Returns:
            Tuple of (gate_passes, debug_info_dict).
        """
        import numpy as np

        # Support both MediaPipe landmarks and numpy arrays
        is_numpy_array = isinstance(landmarks, np.ndarray)

        visibility_by_landmark: Dict[str, float] = {}
        fail_reason = ""

        # Check legs
        legs_ok = True
        for idx in self.LEG_LM:
            visibility = (
                float(landmarks[idx, 3])
                if is_numpy_array
                else float(landmarks[idx].visibility)
            )
            visibility_by_landmark[self.LM_LABELS.get(idx, str(idx))] = round(
                visibility,
                3,
            )
            if visibility < self.vis_th:
                legs_ok = False
                if not fail_reason:
                    fail_reason = "Legs/Feet not visible"

        # Check shoulders (at least one)
        sho_ok = False
        for idx in self.SHOULDER_LM:
            visibility = (
                float(landmarks[idx, 3])
                if is_numpy_array
                else float(landmarks[idx].visibility)
            )
            visibility_by_landmark[self.LM_LABELS.get(idx, str(idx))] = round(
                visibility,
                3,
            )
            if visibility >= self.vis_th:
                sho_ok = True

        left_chain_ok = True
        right_chain_ok = True
        for idx in self.SIDE_LEG_CHAINS["left"]:
            visibility = (
                float(landmarks[idx, 3])
                if is_numpy_array
                else float(landmarks[idx].visibility)
            )
            if visibility < self.vis_th:
                left_chain_ok = False
        for idx in self.SIDE_LEG_CHAINS["right"]:
            visibility = (
                float(landmarks[idx, 3])
                if is_numpy_array
                else float(landmarks[idx].visibility)
            )
            if visibility < self.vis_th:
                right_chain_ok = False

        single_side_profile_ok = sho_ok and (left_chain_ok or right_chain_ok)

        if not sho_ok and not fail_reason:
            fail_reason = "Upper body not visible"
        elif not legs_ok and single_side_profile_ok:
            fail_reason = "Single side profile visible"

        gate_ok = legs_ok and sho_ok

        debug_info: Dict[str, Any] = {
            "vis_ok": gate_ok,
            "legs_ok": legs_ok,
            "sho_ok": sho_ok,
            "left_chain_ok": left_chain_ok,
            "right_chain_ok": right_chain_ok,
            "single_side_profile_ok": single_side_profile_ok,
            "vis_th": float(self.vis_th),
            "vis": visibility_by_landmark,
        }
        if fail_reason:
            debug_info["reason"] = fail_reason

        return gate_ok, debug_info
