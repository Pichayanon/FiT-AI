from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class StreamState:
    """Session-level state for a plank streaming WebSocket connection."""

    started: bool = False
    # Per-frame features: (signed_dist, hip_height_norm, body_angle)
    frame_feature_values: List[Tuple[float, float, float]] = field(default_factory=list)

    # Gate
    ready: bool = False
    ready_streak: int = 0
    chosen_side: Optional[str] = None

    # Session metadata
    session_id: str = ""

    # Status throttle
    last_status: str = ""
    status_tick: int = 0

    # Last prediction (for logs / debugging)
    last_pred_label: str = ""
    last_pred_conf: Optional[float] = None

    # Last sent (for optional deduplication)
    last_sent_label: str = ""
    last_sent_conf: Optional[float] = None

    # Frame counter
    frame_count: int = 0

    # NO_POSE watchdog
    no_pose_since: Optional[float] = None
    no_pose_alerted: bool = False

    # DARK watchdog
    dark_since: Optional[float] = None
    dark_alerted: bool = False
