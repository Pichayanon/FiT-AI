from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SideViewGateDynamic:
    def __init__(
        self,
        mp_pose: Any,
        vis_th: float = 0.65,
    ) -> None:
        self.mp_pose = mp_pose
        self.vis_th = vis_th

        self.LEG_LM: List[int] = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

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
        import numpy as np

        is_numpy_array = isinstance(landmarks, np.ndarray)

        visibility_by_landmark: Dict[str, float] = {}
        fail_reason = ""

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
