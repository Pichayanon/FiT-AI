from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


class FrameDecoder:
    @staticmethod
    def decode_jpeg_base64(jpeg_b64: str) -> Optional[np.ndarray]:
        try:
            raw = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
