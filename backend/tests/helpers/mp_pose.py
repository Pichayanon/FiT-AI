from __future__ import annotations

from types import SimpleNamespace


class FakePoseLandmark:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def make_fake_mp_pose() -> SimpleNamespace:
    return SimpleNamespace(PoseLandmark=FakePoseLandmark)
