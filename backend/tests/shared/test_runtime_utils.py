from __future__ import annotations

import base64
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from shared.app_factory import make_app
from shared.frame_decoder import FrameDecoder
from shared.frame_quality import FrameQuality
from shared.json_utils import parse_json
from shared.label_mapper import LabelMapper
from shared.phase_bottom_state import PhaseBottomStreamState
from shared.server_utils import serve
from shared.video_utils import create_video_writer, normalize_per_sample, resample_time


def test_make_app_sets_title_and_cors_middleware() -> None:
    app = make_app("Coverage App")

    assert app.title == "Coverage App"
    assert any(middleware.cls.__name__ == "CORSMiddleware" for middleware in app.user_middleware)


def test_frame_decoder_decodes_valid_jpeg_base64() -> None:
    frame = np.full((4, 4, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok is True

    decoded = FrameDecoder.decode_jpeg_base64(base64.b64encode(encoded.tobytes()).decode())

    assert decoded is not None
    assert decoded.shape == frame.shape


def test_frame_decoder_returns_none_for_invalid_base64() -> None:
    assert FrameDecoder.decode_jpeg_base64("not-base64") is None


def test_frame_quality_computes_brightness_and_dark_threshold() -> None:
    dark_frame = np.zeros((3, 3, 3), dtype=np.uint8)
    bright_frame = np.full((3, 3, 3), 255, dtype=np.uint8)

    assert FrameQuality.compute_brightness_mean_bgr(dark_frame) == 0.0
    assert FrameQuality.is_too_dark(dark_frame, threshold=10.0) == (True, 0.0)

    bright_mean = FrameQuality.compute_brightness_mean_bgr(bright_frame)
    assert bright_mean == 255.0
    assert FrameQuality.is_too_dark(bright_frame, threshold=10.0) == (False, bright_mean)


def test_parse_json_handles_dict_and_invalid_payloads() -> None:
    assert parse_json('{"x": 1}') == {"x": 1}
    assert parse_json("[1, 2, 3]") is None
    assert parse_json("{broken") is None


def test_label_mapper_returns_known_label_and_fallback() -> None:
    mapper = LabelMapper({0: "bad", 1: "good"})

    assert mapper.label_of(1) == "good"
    assert mapper.label_of(9) == "9"


def test_phase_bottom_stream_state_defaults_are_initialized() -> None:
    state = PhaseBottomStreamState()

    assert state.started is False
    assert state.ready is False
    assert state.last_phase == "unknown"
    assert list(state.history) == []
    assert list(state.phase_features) == []
    assert state.pending_bottom_event is None


def test_serve_invokes_uvicorn_run(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    app = object()

    serve(app, port=5055)

    assert calls == [
        {
            "app": app,
            "host": "0.0.0.0",
            "port": 5055,
            "reload": False,
            "log_level": "info",
        }
    ]


def test_resample_time_handles_equal_length_short_input_and_interpolation() -> None:
    frame = np.array([[1.0, 2.0]], dtype=np.float32)
    sequence = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)

    np.testing.assert_allclose(resample_time(sequence, 3), sequence)
    np.testing.assert_allclose(
        resample_time(frame, 3),
        np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        resample_time(sequence, 5),
        np.array([[0.0], [5.0], [10.0], [15.0], [20.0]], dtype=np.float32),
    )


def test_normalize_per_sample_zero_centers_and_scales_columns() -> None:
    x = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], dtype=np.float32)

    normalized = normalize_per_sample(x)

    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=0), np.ones(2), atol=1e-5)


def test_create_video_writer_tries_mp4_then_avi(monkeypatch) -> None:
    class FakeWriter:
        def __init__(self, opened: bool) -> None:
            self._opened = opened

        def isOpened(self) -> bool:
            return self._opened

    writers = iter([FakeWriter(False), FakeWriter(True)])
    fourcc_calls: list[str] = []

    monkeypatch.setattr(
        "shared.video_utils.cv2.VideoWriter_fourcc",
        lambda *args: fourcc_calls.append("".join(args)) or 1234,
    )
    monkeypatch.setattr(
        "shared.video_utils.cv2.VideoWriter",
        lambda *args, **kwargs: next(writers),
    )

    writer, path = create_video_writer("clip/output", 640, 480, 30.0)

    assert writer is not None
    assert path.endswith(".avi")
    assert fourcc_calls == ["mp4v", "MJPG"]


def test_create_video_writer_returns_mp4_when_first_writer_opens(monkeypatch) -> None:
    class FakeWriter:
        def isOpened(self) -> bool:
            return True

    monkeypatch.setattr("shared.video_utils.cv2.VideoWriter_fourcc", lambda *args: 1234)
    monkeypatch.setattr("shared.video_utils.cv2.VideoWriter", lambda *args, **kwargs: FakeWriter())

    writer, path = create_video_writer("clip/output", 640, 480, 30.0)

    assert writer is not None
    assert path.endswith(".mp4")


def test_create_video_writer_returns_none_when_all_attempts_fail(monkeypatch) -> None:
    class FakeWriter:
        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr("shared.video_utils.cv2.VideoWriter_fourcc", lambda *args: 1234)
    monkeypatch.setattr("shared.video_utils.cv2.VideoWriter", lambda *args, **kwargs: FakeWriter())

    writer, path = create_video_writer("clip/output", 640, 480, 30.0)

    assert writer is None
    assert path == ""


def test_create_video_writer_handles_writer_creation_exceptions(monkeypatch) -> None:
    monkeypatch.setattr("shared.video_utils.cv2.VideoWriter_fourcc", lambda *args: 1234)

    def raise_error(*args, **kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr("shared.video_utils.cv2.VideoWriter", raise_error)

    writer, path = create_video_writer("clip/output", 640, 480, 30.0)

    assert writer is None
    assert path == ""
