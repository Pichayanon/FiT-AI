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

    def check(self, lm: List[Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if the front-view gate passes.

        Args:
            lm: MediaPipe landmark list.

        Returns:
            Tuple of (gate_passes, debug_info_dict).
        """
        vis_map: Dict[str, float] = {}
        ok_vis = True
        for idx in self.FRONT_LM:
            v = float(lm[idx].visibility)
            vis_map[self.FRONT_LM_LABELS.get(idx, str(idx))] = round(v, 3)
            if v < self.vis_th:
                ok_vis = False

        lsho_x = float(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        rsho_x = float(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        lhip_x = float(lm[self.mp_pose.PoseLandmark.LEFT_HIP].x)
        rhip_x = float(lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x)
        sho_gap = abs(lsho_x - rsho_x)
        hip_gap = abs(lhip_x - rhip_x)
        ok_gap = (sho_gap >= self.min_sho_gap) and (hip_gap >= self.min_hip_gap)

        dbg = {
            "vis_ok": ok_vis,
            "gap_ok": ok_gap,
            "sho_gap": round(sho_gap, 3),
            "hip_gap": round(hip_gap, 3),
            "vis_th": float(self.vis_th),
            "vis": vis_map,
        }
        return (ok_vis and ok_gap), dbg
