from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from plank.features import aggregate_window, extract_frame_features


class PlankFeatureExtractor:
    """Thin adapter over the shared plank feature helpers."""

    def __init__(self, mp_pose: Any) -> None:
        self.mp_pose = mp_pose

    def extract_features(
        self,
        pose_result: Any,
        side: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract a single-frame feature tuple for the given side."""
        if not pose_result.pose_landmarks:
            return None
        return extract_frame_features(pose_result.pose_landmarks.landmark, side)

    @staticmethod
    def aggregate_window(
        frame_feature_values: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Aggregate a window of per-frame tuples into a 6-D feature vector."""
        return aggregate_window(frame_feature_values)


# Backward-compatible alias for existing imports.
FeatureExtractor = PlankFeatureExtractor
