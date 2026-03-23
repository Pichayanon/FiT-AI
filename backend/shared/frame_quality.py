from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class FrameQuality:
    @staticmethod
    def compute_brightness_mean_bgr(frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def is_too_dark(frame_bgr: np.ndarray, threshold: float) -> Tuple[bool, float]:
        mean_v = FrameQuality.compute_brightness_mean_bgr(frame_bgr)
        return (mean_v < float(threshold)), mean_v
