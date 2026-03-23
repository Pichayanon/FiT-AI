from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


class RecordingStatusSender:
    def __init__(self) -> None:
        self.info_calls: list[dict[str, Any]] = []
        self.status_calls: list[dict[str, Any]] = []
        self.phase_calls: list[dict[str, Any]] = []

    async def send_info(
        self,
        websocket: Any,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.info_calls.append({"message": message, "extra": extra})

    async def send_status(
        self,
        websocket: Any,
        state: Any,
        phase: str,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if hasattr(state, "last_status"):
            state.last_status = phase
        self.status_calls.append(
            {"phase": phase, "extra": extra, "force": force}
        )

    async def send_phase(
        self,
        websocket: Any,
        state: Any,
        phase: str,
        force: bool = False,
    ) -> None:
        if hasattr(state, "last_phase"):
            state.last_phase = phase
        self.phase_calls.append({"phase": phase, "force": force})


class RecordingWebSocket:
    def __init__(self) -> None:
        self.sent_texts: list[str] = []
        self.sent_messages: list[Any] = []

    async def send_text(self, payload: str) -> None:
        self.sent_texts.append(payload)
        try:
            self.sent_messages.append(json.loads(payload))
        except json.JSONDecodeError:
            self.sent_messages.append(payload)


class FakePose:
    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.calls = 0

    def process(self, image: Any) -> Any:
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return SimpleNamespace(pose_landmarks=None)


def make_pose_result(landmarks: list[Any] | None) -> Any:
    if landmarks is None:
        return SimpleNamespace(pose_landmarks=None)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))
