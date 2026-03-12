"""
Frame quality checks for streaming video frames.

Provides simple brightness-based quality metrics used by streaming
sessions to detect poor lighting conditions.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class FrameQuality:
    """Utility for computing simple frame quality metrics (brightness)."""

    @staticmethod
    def compute_brightness_mean_bgr(frame_bgr: np.ndarray) -> float:
        """Return mean grayscale brightness in the range [0, 255].

        Args:
            frame_bgr: BGR image as numpy array.

        Returns:
            Mean brightness value (0.0 to 255.0).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def is_too_dark(frame_bgr: np.ndarray, threshold: float) -> Tuple[bool, float]:
        """Check if a frame is too dark based on a brightness threshold.

        Args:
            frame_bgr: BGR image as numpy array.
            threshold: Brightness threshold (0-255). Below this is "too dark".

        Returns:
            Tuple of (is_too_dark, brightness_mean).
        """
        mean_v = FrameQuality.compute_brightness_mean_bgr(frame_bgr)
        return (mean_v < float(threshold)), mean_v
