from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PhaseBottomStreamState:
    started: bool = False
    session_id: str = ""

    ready: bool = False
    ready_streak: int = 0
    last_gate_debug: Dict[str, Any] = field(default_factory=dict)

    status_tick: int = 0
    phase_tick: int = 0
    last_status: str = ""
    last_phase: str = "unknown"

    total_reps: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    last_counted_event_frame: int = -(10**9)

    last_sent_bottom_event_frame: int = -(10**9)

    history: deque = field(default_factory=deque)
    phase_features: deque = field(default_factory=deque)
    prev_phase: str = ""
    last_bottom_event_frame: int = -(10**9)

    pending_bottom_event: Optional[Dict[str, Any]] = None

    dark_since: Optional[float] = None
    dark_alerted: bool = False
