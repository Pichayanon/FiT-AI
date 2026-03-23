"""Wall sit WebSocket session handler."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from fastapi import WebSocket

from shared.label_mapper import LabelMapper
from shared.side_gate import SideGate
from shared.side_window_session import SideWindowSession
from shared.sklearn_model_service import SklearnModelService
from shared.status_sender import StatusSender

from wall_sit.features import WallSitFeatureExtractor, StreamState


class WallSitWebSocketSession(SideWindowSession):
    """Handle a single wall sit WebSocket streaming session."""

    def __init__(
        self,
        websocket: WebSocket,
        model_service: SklearnModelService,
        gate: SideGate,
        feature_extractor: WallSitFeatureExtractor,
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
        stand_knee_angle_deg_th: float,
        stand_streak_n: int,
        phase_no_pose: str,
        phase_have_pose: str,
        phase_buffering: str,
        phase_inferencing: str,
        status_send_every_n_frames: int,
    ) -> None:
        self.stand_knee_angle_deg_th = stand_knee_angle_deg_th
        self.stand_streak_n = stand_streak_n
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
        """Return a fresh StreamState for the current session."""
        return StreamState()

    def _reset_gate_specific_fields(self) -> None:
        """Reset wall-sit specific gating state."""
        self.state.stand_streak = 0

    def _on_missing_features(self) -> None:
        """Reset stand streak when frame features are unavailable."""
        self.state.stand_streak = 0

    def _pre_buffer_feature_payload(
        self,
        frame_features: Tuple[float, float, float],
    ) -> Optional[Dict[str, object]]:
        """Keep inference disabled while the user is still standing upright."""
        knee_angle = float(frame_features[1])
        if knee_angle >= self.stand_knee_angle_deg_th:
            self.state.stand_streak += 1
        else:
            self.state.stand_streak = 0

        if self.state.stand_streak < self.stand_streak_n:
            return None

        if self.state.stand_streak == self.stand_streak_n:
            print(
                f"[WALLSIT] STANDING gate: session_id={self.state.session_id} "
                f"side={self.state.chosen_side} knee_angle={knee_angle:.1f} "
                f"th={self.stand_knee_angle_deg_th}"
            )
        self.state.frame_features.clear()
        return self._window_payload(
            self.state.chosen_side,
            0,
            {
                "ready_streak": self.state.ready_streak,
                "needed_streak": self.ready_streak_n,
                "standing": True,
                "knee_angle": round(knee_angle, 1),
                "knee_th": self.stand_knee_angle_deg_th,
            },
        )
