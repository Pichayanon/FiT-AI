"""Plank WebSocket session handler."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import WebSocket

from shared.label_mapper import LabelMapper
from shared.side_gate import SideGate
from shared.side_window_session import SideWindowSession
from shared.sklearn_model_service import SklearnModelService
from shared.status_sender import StatusSender

from plank.features import PlankFeatureExtractor, StreamState, is_plank_ready_pose


class PlankWebSocketSession(SideWindowSession):
    """Handle a single plank WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_service: SklearnModelService,
        gate: SideGate,
        feature_extractor: PlankFeatureExtractor,
        labels: LabelMapper,
        status_sender: StatusSender,
        window_frames: int,
        ready_streak_n: int,
        debug: bool,
        side_mode: str,
        vis_th: float,
        mp_min_det_conf: float,
        mp_min_track_conf: float,
        no_pose_adjust_seconds: float,
        dark_adjust_seconds: float,
        dark_brightness_th: float,
        plank_ready_max_body_axis_angle_deg: float,
        plank_ready_max_torso_angle_deg: float,
        plank_ready_max_leg_angle_deg: float,
        phase_no_pose: str,
        phase_have_pose: str,
        phase_buffering: str,
        phase_inferencing: str,
        status_send_every_n_frames: int,
    ) -> None:
        self.plank_ready_max_body_axis_angle_deg = (
            plank_ready_max_body_axis_angle_deg
        )
        self.plank_ready_max_torso_angle_deg = plank_ready_max_torso_angle_deg
        self.plank_ready_max_leg_angle_deg = plank_ready_max_leg_angle_deg
        super().__init__(
            websocket=websocket,
            model_service=model_service,
            gate=gate,
            feature_extractor=feature_extractor,
            labels=labels,
            status_sender=status_sender,
            window_frames=window_frames,
            ready_streak_n=ready_streak_n,
            debug=debug,
            side_mode=side_mode,
            vis_th=vis_th,
            mp_min_det_conf=mp_min_det_conf,
            mp_min_track_conf=mp_min_track_conf,
            no_pose_adjust_seconds=no_pose_adjust_seconds,
            dark_adjust_seconds=dark_adjust_seconds,
            dark_brightness_th=dark_brightness_th,
            phase_no_pose=phase_no_pose,
            phase_have_pose=phase_have_pose,
            phase_buffering=phase_buffering,
            phase_inferencing=phase_inferencing,
            status_send_every_n_frames=status_send_every_n_frames,
        )

    def _create_state(self) -> StreamState:
        """Return a fresh StreamState; base will set started=True and session_id."""
        return StreamState()

    def _pre_ready_pose_payload(
        self,
        landmarks: Any,
        chosen_side: str,
    ) -> Optional[Dict[str, Any]]:
        """Keep buffering disabled until the user is in a plank-like pose."""
        is_plank_pose_ready, posture_debug = is_plank_ready_pose(
            landmarks,
            chosen_side,
            max_body_axis_angle_deg=self.plank_ready_max_body_axis_angle_deg,
            max_torso_angle_deg=self.plank_ready_max_torso_angle_deg,
            max_leg_angle_deg=self.plank_ready_max_leg_angle_deg,
        )
        if is_plank_pose_ready:
            return None
        return self._window_payload(
            chosen_side,
            0,
            {
                "ready_streak": 0,
                "needed_streak": self.ready_streak_n,
                **posture_debug,
            },
        )
