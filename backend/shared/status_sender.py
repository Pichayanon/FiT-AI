from __future__ import annotations

import json
from typing import Any, Dict, Optional, Protocol

from fastapi import WebSocket


class HasStatusFields(Protocol):
    status_tick: int
    last_status: str
    session_id: str


class HasPhaseFields(Protocol):
    phase_tick: int
    last_phase: str
    session_id: str


class StatusSender:
    def __init__(
        self,
        every_n_frames: int,
        phase_every_n: int = 2,
    ) -> None:
        self.every_n_frames = max(1, int(every_n_frames))
        self.phase_every_n = max(1, int(phase_every_n))

    async def send_info(
        self,
        websocket: WebSocket,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"type": "info", "message": message}
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload))

    async def send_status(
        self,
        websocket: WebSocket,
        state: Any,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        state.status_tick += 1

        if not force:
            if (state.status_tick % self.every_n_frames != 0) and (
                phase == state.last_status
            ):
                return

        payload: Dict[str, Any] = {
            "type": "status",
            "state": phase,
            "session_id": state.session_id,
        }
        if extra:
            payload.update(extra)

        state.last_status = phase
        await websocket.send_text(json.dumps(payload))

    async def send_phase(
        self,
        websocket: WebSocket,
        state: Any,
        phase: str,
        force: bool = False,
    ) -> None:
        state.phase_tick += 1
        if (
            not force
            and (state.phase_tick % self.phase_every_n != 0)
            and (phase == state.last_phase)
        ):
            return

        payload: Dict[str, Any] = {
            "type": "phase",
            "phase": phase,
            "session_id": state.session_id,
        }

        state.last_phase = phase
        await websocket.send_text(json.dumps(payload))
