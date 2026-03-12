"""
Frame decoder for base64-encoded JPEG frames.

Converts base64 JPEG strings received from iOS clients into
OpenCV BGR images for processing.
"""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


class FrameDecoder:
    """Decode base64-encoded JPEG frames into BGR images."""

    @staticmethod
    def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string into an OpenCV BGR image.

        Args:
            jpeg_b64: Base64-encoded JPEG image data.

        Returns:
            BGR image as numpy array (H, W, 3), or None on failure.
        """
        try:
            raw = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:  # pylint: disable=broad-except
            return None
