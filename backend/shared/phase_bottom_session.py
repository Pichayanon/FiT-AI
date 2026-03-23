from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

_EVENT_SENTINEL = -(10**9)


class PhaseBottomSessionMixin:
    def _is_good_rep_label(self, pred_label: str) -> bool:
        return pred_label.startswith("good")

    def _bottom_feature_from_history_record(self, record: Any) -> np.ndarray:
        raise NotImplementedError

    def _after_full_buffer_reset(self) -> None:
        pass

    def _after_ready_transition(self) -> None:
        pass

    def _reset_common_bottom_buffers(self) -> None:
        self.state.history.clear()
        self.state.phase_features.clear()
        self.state.prev_phase = ""
        self.state.last_bottom_event_frame = _EVENT_SENTINEL
        self.state.pending_bottom_event = None
        self.state.last_sent_bottom_event_frame = _EVENT_SENTINEL

    def _reset_phase_bottom_state(self) -> None:
        self.state.ready = False
        self.state.ready_streak = 0
        self.state.last_gate_debug = {}
        self._reset_common_bottom_buffers()
        self._after_full_buffer_reset()

    def _activate_ready_phase_bottom_state(self) -> None:
        self.state.ready = True
        self._reset_common_bottom_buffers()
        self._after_ready_transition()

    def _increment_rep_counter(self, event_frame: int, pred_label: str) -> None:
        if event_frame == self.state.last_counted_event_frame:
            return
        self.state.last_counted_event_frame = event_frame
        self.state.total_reps += 1
        if self._is_good_rep_label(pred_label):
            self.state.good_reps += 1
        else:
            self.state.bad_reps += 1

    async def _send_waiting_status(
        self,
        *,
        too_dark: bool,
        brightness_mean: float,
        gate_debug: Dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "ready_streak": 0,
            "needed_streak": self.ready_streak_n,
            "too_dark": too_dark,
            "brightness_mean": round(brightness_mean, 1),
            "brightness_th": self.dark_brightness_th,
        }
        if gate_debug is not None:
            payload["gate"] = gate_debug if self.debug else None
        if reason is not None:
            payload["reason"] = reason
        await self.status_sender.send_status(
            self.websocket,
            self.state,
            "waiting",
            payload,
        )

    async def _advance_ready_streak(
        self,
        *,
        gate_debug: Dict[str, Any],
        ok_message: str,
    ) -> bool:
        self.state.ready_streak += 1

        if (not self.state.ready) and (self.state.ready_streak >= self.ready_streak_n):
            self._activate_ready_phase_bottom_state()
            await self.status_sender.send_info(
                self.websocket,
                ok_message,
                {
                    "session_id": self.state.session_id,
                    "gate": gate_debug if self.debug else None,
                },
            )
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                "ready",
                {"ready_streak": self.state.ready_streak},
                force=True,
            )
        elif not self.state.ready:
            await self.status_sender.send_status(
                self.websocket,
                self.state,
                "warming_up",
                {
                    "ready_streak": self.state.ready_streak,
                    "needed_streak": self.ready_streak_n,
                    "gate": gate_debug if self.debug else None,
                },
            )

        return bool(self.state.ready)

    async def _predict_and_send_bottom(
        self,
        event_frame: int,
        phase: str,
        force: bool = False,
    ) -> None:
        start_frame = event_frame - self.pre_frames
        end_frame = event_frame + self.post_frames
        required_frame_count = self.pre_frames + self.post_frames + 1
        history_window = [
            record
            for record in self.state.history
            if start_frame <= record[0] <= end_frame
        ]

        if len(history_window) < required_frame_count and not force:
            self.state.pending_bottom_event = {
                "event_frame": event_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
            return

        if len(history_window) == 0:
            self.state.pending_bottom_event = None
            return

        self.state.pending_bottom_event = None
        await self.status_sender.send_status(
            self.websocket,
            self.state,
            "predicting",
            {
                "mode": "bottom",
                "phase": phase,
                "event_i": int(event_frame),
                "window_frames": int(len(history_window)),
                "T": int(self.model_service.bottom_T),
                "D": self.bottom_feature_dim,
            },
        )

        window_features = np.stack(
            [
                self._bottom_feature_from_history_record(record)
                for record in history_window
            ],
            axis=0,
        ).astype(np.float32)
        pred_label, prediction_confidence, _ = self.model_service.predict_bottom(
            window_features
        )
        is_good = self._is_good_rep_label(pred_label)
        self._increment_rep_counter(int(event_frame), pred_label)

        payload = {
            "type": "result",
            "mode": "bottom",
            "prediction": pred_label,
            "confidence": round(prediction_confidence, 3),
            "session_id": self.state.session_id,
            "event_i": int(event_frame),
            "window": {"pre": self.pre_frames, "post": self.post_frames},
            "T": int(self.model_service.bottom_T),
            "feature_dim": self.bottom_feature_dim,
            "reps": {
                "total": int(self.state.total_reps),
                "correct": int(self.state.good_reps),
                "incorrect": int(self.state.bad_reps),
                "goal_correct": int(self.goal_good_reps),
                "is_correct_rep": bool(is_good),
            },
        }
        if self.debug:
            print("[PRED-BOTTOM]", payload)
        if int(payload.get("event_i", -1)) != self.state.last_sent_bottom_event_frame:
            self.state.last_sent_bottom_event_frame = int(payload.get("event_i", -1))
            await self.websocket.send_text(json.dumps(payload))

    async def _resolve_pending_bottom_prediction(self, phase: str) -> None:
        if (
            self.state.pending_bottom_event is None
            or (not self.model_service.bottom_loaded)
            or self.model_service.bottom_T is None
        ):
            return

        pending_event_frame = int(self.state.pending_bottom_event["event_frame"])
        start_frame = self.state.pending_bottom_event["start_frame"]
        end_frame = self.state.pending_bottom_event["end_frame"]
        required_frame_count = self.pre_frames + self.post_frames + 1
        history_window = [
            record
            for record in self.state.history
            if start_frame <= record[0] <= end_frame
        ]
        window_complete = self.frame_index >= end_frame

        if len(history_window) >= required_frame_count or window_complete:
            self.state.pending_bottom_event = None
            await self._predict_and_send_bottom(
                pending_event_frame,
                phase,
                force=window_complete,
            )
