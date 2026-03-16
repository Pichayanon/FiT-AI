"""
server_utils.py — Uvicorn server launcher utility.

Eliminates the identical argparse + uvicorn boilerplate that previously
appeared in every exercise streaming module's `main()` function.

Usage:
    from shared.server_utils import serve

    if __name__ == "__main__":
        serve("squat.squat_streaming:app", port=5051)
"""

from __future__ import annotations


def serve(module: str, port: int) -> None:
    """Start a uvicorn server for the given FastAPI module.

    Args:
        module: Dotted module path and app variable, e.g. "squat.squat_streaming:app".
        port:   Port to listen on.
    """
    import uvicorn
    uvicorn.run(module, host="0.0.0.0", port=port, reload=False, log_level="info")
