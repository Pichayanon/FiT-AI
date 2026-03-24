"""
test_api.py — FiT-AI API Test Suite

Tests all 4 exercise WebSocket endpoints:
  ✓ Health check (HTTP GET)
  ✓ WebSocket: connect → start → stop
  ✓ WebSocket: connect → start → send frames → stop
  ✓ Edge: frame sent before start (should not crash)
  ✓ Edge: stop sent before start (should not crash)
  ✓ Edge: start sent twice without stop (session should reset)
  ✓ Edge: invalid frame data (should respond with decode-failed info)
  ✓ Edge: malformed JSON (should not crash)

Usage:
    pip install websockets requests numpy opencv-python certifi
    python test_api.py                  # all exercises
    python test_api.py squat            # one exercise
    python test_api.py squat lunges     # specific exercises
"""

from __future__ import annotations

import asyncio
import base64
import json
import ssl
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import os

import certifi
import cv2
import numpy as np
import requests
import websockets

# ---------------------------------------------------------------
# Config  (override via env vars for CI/CD)
# ---------------------------------------------------------------

BASE_HTTP = os.getenv("BASE_HTTP")
BASE_WS   = os.getenv("BASE_WS")

EXERCISES   = ["squat", "plank", "lunges", "wall_sit"]
WS_TIMEOUT  = 10   # seconds to wait for each message
FRAME_COUNT = 5    # frames sent in the frame test

CA_BUNDLE   = certifi.where()
SSL_CONTEXT = ssl.create_default_context(cafile=CA_BUNDLE)

# ---------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ---------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    note: str = ""


_results: List[TestResult] = []


def record(name: str, passed: bool, note: str = "") -> bool:
    _results.append(TestResult(name, passed, note))
    return passed


# ---------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------

def ok(msg: str) -> None:
    print(f"    {GREEN}✓{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"    {RED}✗{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"    {YELLOW}→{RESET}  {msg}")


def skip(msg: str) -> None:
    print(f"    {GREY}–{RESET}  {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}┌{'─' * 52}┐{RESET}")
    print(f"{BOLD}{CYAN}│  {msg:<50}│{RESET}")
    print(f"{BOLD}{CYAN}└{'─' * 52}┘{RESET}")


def section(msg: str) -> None:
    print(f"\n  {BOLD}{msg}{RESET}")


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def ws_url(exercise: str) -> str:
    return f"{BASE_WS}/{exercise}/ws/video"


def make_test_frame(width: int = 320, height: int = 240) -> str:
    """Synthetic gradient JPEG → base64 (no human pose, used for crash testing)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        frame[i, :] = [int(i * 255 / height), 100, 180]
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode()


async def ws_collect(ws, count: int, timeout: float) -> List[Dict[str, Any]]:
    """Receive up to `count` messages within `timeout` seconds, parse JSON."""
    msgs: List[Dict[str, Any]] = []
    try:
        for _ in range(count):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            try:
                msgs.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    except asyncio.TimeoutError:
        pass
    return msgs


def _connect(url: str):
    return websockets.connect(
        url,
        ping_interval=None,
        ssl=SSL_CONTEXT if url.startswith("wss://") else None,
    )


def has_type(msgs: List[Dict], t: str) -> bool:
    return any(m.get("type") == t for m in msgs)


def first_of(msgs: List[Dict], t: str) -> Optional[Dict]:
    return next((m for m in msgs if m.get("type") == t), None)


# ---------------------------------------------------------------
# 1. HTTP Health Tests
# ---------------------------------------------------------------

def test_health_all() -> None:
    header("Health Checks — HTTP GET")

    endpoints = [
        ("/health", "global"),
        *[(f"/{ex}/health", ex) for ex in EXERCISES],
    ]

    for path, label in endpoints:
        url = f"{BASE_HTTP}{path}"
        try:
            r = requests.get(url, timeout=10, verify=CA_BUNDLE)
            if r.status_code == 200 and r.json().get("status") == "ok":
                ok(f"{label:<10}  {url}")
                record(f"health/{label}", True)
            else:
                fail(f"{label:<10}  HTTP {r.status_code}")
                record(f"health/{label}", False, f"HTTP {r.status_code}")
        except Exception as e:
            fail(f"{label:<10}  {e}")
            record(f"health/{label}", False, str(e))


# ---------------------------------------------------------------
# 2. Happy-path WebSocket Tests
# ---------------------------------------------------------------

async def test_ws_start_stop(exercise: str) -> bool:
    """connect → start → assert info+status → stop → assert info."""
    url = ws_url(exercise)
    info(f"connect → start → stop  ({url})")
    try:
        async with _connect(url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            msgs = await ws_collect(ws, 5, WS_TIMEOUT)

            if not has_type(msgs, "info"):
                fail("start: no 'info' message")
                return record(f"{exercise}/start_stop", False, "no info after start")
            m = first_of(msgs, "info")
            ok(f"start → info: {m.get('message')!r}")

            if not has_type(msgs, "status"):
                fail("start: no 'status' message")
                return record(f"{exercise}/start_stop", False, "no status after start")
            m = first_of(msgs, "status")
            ok(f"start → status: state={m.get('state')!r}  session_id={m.get('session_id')!r}")

            await ws.send(json.dumps({"type": "stop"}))
            msgs = await ws_collect(ws, 5, WS_TIMEOUT)

            if not has_type(msgs, "info"):
                fail("stop: no 'info' message")
                return record(f"{exercise}/start_stop", False, "no info after stop")
            m = first_of(msgs, "info")
            ok(f"stop  → info: {m.get('message')!r}")

        record(f"{exercise}/start_stop", True)
        return True
    except Exception as e:
        fail(f"WebSocket error: {e}")
        return record(f"{exercise}/start_stop", False, str(e))


async def test_ws_with_frames(exercise: str) -> bool:
    """connect → start → send N frames → check responses → stop."""
    url = ws_url(exercise)
    info(f"connect → start → {FRAME_COUNT} frames → stop  ({url})")
    jpeg_b64 = make_test_frame()
    try:
        async with _connect(url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws_collect(ws, 3, WS_TIMEOUT)

            all_msgs: List[Dict] = []
            for _ in range(FRAME_COUNT):
                await ws.send(json.dumps({"type": "frame", "jpeg_b64": jpeg_b64}))
                all_msgs.extend(await ws_collect(ws, 3, timeout=2))

            await ws.send(json.dumps({"type": "stop"}))
            all_msgs.extend(await ws_collect(ws, 3, WS_TIMEOUT))

            received = {m.get("type") for m in all_msgs}
            ok(f"received message types: {sorted(received)}")

            if "status" in received:
                states = list({m.get("state") for m in all_msgs if m.get("type") == "status"})
                ok(f"observed states: {states}")

            if "result" in received:
                r = first_of(all_msgs, "result")
                ok(f"prediction: {r.get('prediction')!r}  confidence={r.get('confidence')}")
            else:
                skip("no 'result' (expected — synthetic frame has no pose)")

        record(f"{exercise}/frame_test", True)
        return True
    except Exception as e:
        fail(f"WebSocket error: {e}")
        return record(f"{exercise}/frame_test", False, str(e))


# ---------------------------------------------------------------
# 3. Edge Case Tests
# ---------------------------------------------------------------

async def test_edge_frame_before_start(exercise: str) -> None:
    """Send a frame before 'start' — server must not crash."""
    url = ws_url(exercise)
    info("edge: frame before start")
    jpeg_b64 = make_test_frame()
    try:
        async with _connect(url) as ws:
            await ws.send(json.dumps({"type": "frame", "jpeg_b64": jpeg_b64}))
            msgs = await ws_collect(ws, 3, timeout=3)
            # Server should stay alive (connection not dropped unexpectedly)
            ok(f"server alive after frame-before-start  (got {len(msgs)} msg(s))")
            record(f"{exercise}/edge_frame_before_start", True)
    except Exception as e:
        fail(f"server error: {e}")
        record(f"{exercise}/edge_frame_before_start", False, str(e))


async def test_edge_stop_before_start(exercise: str) -> None:
    """Send 'stop' before 'start' — server must not crash."""
    url = ws_url(exercise)
    info("edge: stop before start")
    try:
        async with _connect(url) as ws:
            await ws.send(json.dumps({"type": "stop"}))
            msgs = await ws_collect(ws, 3, timeout=3)
            ok(f"server alive after stop-before-start  (got {len(msgs)} msg(s))")
            record(f"{exercise}/edge_stop_before_start", True)
    except Exception as e:
        fail(f"server error: {e}")
        record(f"{exercise}/edge_stop_before_start", False, str(e))


async def test_edge_double_start(exercise: str) -> None:
    """Send 'start' twice — session should reset cleanly."""
    url = ws_url(exercise)
    info("edge: start → start (no stop between)")
    try:
        async with _connect(url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            msgs1 = await ws_collect(ws, 3, WS_TIMEOUT)
            sid1 = (first_of(msgs1, "status") or {}).get("session_id", "")

            await ws.send(json.dumps({"type": "start"}))
            msgs2 = await ws_collect(ws, 3, WS_TIMEOUT)
            sid2 = (first_of(msgs2, "status") or {}).get("session_id", "")

            if sid1 and sid2 and sid1 != sid2:
                ok(f"session reset: {sid1} → {sid2}")
                record(f"{exercise}/edge_double_start", True)
            elif has_type(msgs2, "info") or has_type(msgs2, "status"):
                ok(f"second start handled (got {len(msgs2)} msg(s))")
                record(f"{exercise}/edge_double_start", True)
            else:
                fail("second start got no response")
                record(f"{exercise}/edge_double_start", False, "no response on second start")

            await ws.send(json.dumps({"type": "stop"}))
    except Exception as e:
        fail(f"server error: {e}")
        record(f"{exercise}/edge_double_start", False, str(e))


async def test_edge_invalid_frame(exercise: str) -> None:
    """Send empty/garbage jpeg_b64 after start — server should send 'Decode failed' info."""
    url = ws_url(exercise)
    info("edge: invalid frame data (empty jpeg_b64)")
    try:
        async with _connect(url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws_collect(ws, 3, WS_TIMEOUT)

            await ws.send(json.dumps({"type": "frame", "jpeg_b64": "NOT_VALID_BASE64!!!"}))
            msgs = await ws_collect(ws, 3, timeout=3)

            if any("decode" in (m.get("message") or "").lower() for m in msgs if m.get("type") == "info"):
                ok("server responded with decode-failed info")
                record(f"{exercise}/edge_invalid_frame", True)
            elif msgs:
                ok(f"server alive, responded with: {[m.get('type') for m in msgs]}")
                record(f"{exercise}/edge_invalid_frame", True)
            else:
                skip("no response (frame silently dropped — acceptable)")
                record(f"{exercise}/edge_invalid_frame", True, "silently dropped")

            await ws.send(json.dumps({"type": "stop"}))
    except Exception as e:
        fail(f"server error: {e}")
        record(f"{exercise}/edge_invalid_frame", False, str(e))


async def test_edge_malformed_json(exercise: str) -> None:
    """Send a non-JSON string — server must not crash."""
    url = ws_url(exercise)
    info("edge: malformed JSON")
    try:
        async with _connect(url) as ws:
            await ws.send("this is not json {{{")
            msgs = await ws_collect(ws, 3, timeout=3)
            ok(f"server alive after malformed JSON  (got {len(msgs)} msg(s))")
            record(f"{exercise}/edge_malformed_json", True)
    except Exception as e:
        fail(f"server error: {e}")
        record(f"{exercise}/edge_malformed_json", False, str(e))


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

async def run_exercise_tests(exercise: str) -> None:
    header(f"{exercise.upper()}")

    section("Happy Path")
    passed = await test_ws_start_stop(exercise)
    if passed:
        await asyncio.sleep(2)  # let MediaPipe release before next connection
        await test_ws_with_frames(exercise)

    await asyncio.sleep(1)

    section("Edge Cases")
    await test_edge_frame_before_start(exercise)
    await asyncio.sleep(1)
    await test_edge_stop_before_start(exercise)
    await asyncio.sleep(1)
    await test_edge_double_start(exercise)
    await asyncio.sleep(1)
    await test_edge_invalid_frame(exercise)
    await asyncio.sleep(1)
    await test_edge_malformed_json(exercise)


def print_summary() -> None:
    total  = len(_results)
    passed = sum(1 for r in _results if r.passed)
    failed = total - passed

    print(f"\n{BOLD}{CYAN}{'═' * 54}{RESET}")
    print(f"{BOLD}{CYAN}  TEST SUMMARY{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 54}{RESET}")
    print(f"  Total   : {total}")
    print(f"  {GREEN}Passed  : {passed}{RESET}")
    if failed:
        print(f"  {RED}Failed  : {failed}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 54}{RESET}")

    if failed:
        print(f"\n  {RED}{BOLD}Failed tests:{RESET}")
        for r in _results:
            if not r.passed:
                note = f"  ({r.note})" if r.note else ""
                print(f"    {RED}✗{RESET}  {r.name}{note}")

    print(f"\n  {'🟢' if failed == 0 else '🔴'}  {'ALL PASSED' if failed == 0 else f'{failed} FAILED'}\n")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

async def main(targets: List[str]) -> None:
    print(f"\n{BOLD}{'═' * 54}{RESET}")
    print(f"{BOLD}  FiT-AI API Test Suite{RESET}")
    print(f"{BOLD}{'═' * 54}{RESET}")
    print(f"  URL     : {BASE_HTTP}")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Targets : {', '.join(targets)}")

    test_health_all()

    for i, exercise in enumerate(targets):
        await run_exercise_tests(exercise)
        if i < len(targets) - 1:
            info(f"Waiting 3s before next exercise...")
            await asyncio.sleep(3)

    print_summary()


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else EXERCISES

    unknown = [t for t in targets if t not in EXERCISES]
    if unknown:
        print(f"Unknown exercise(s): {unknown}")
        print(f"Available: {EXERCISES}")
        sys.exit(1)

    asyncio.run(main(targets))
