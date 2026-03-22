"""
server_utils.py — Uvicorn server launcher utility.

Eliminates the identical argparse + uvicorn boilerplate that previously
appeared in every exercise streaming module's `main()` function.

Usage:
    from shared.server_utils import serve
    from mymodule import app

    if __name__ == "__main__":
        serve(app, port=5051)
"""

from __future__ import annotations

from typing import Any


def serve(app: Any, port: int) -> None:
    """Start a uvicorn server for the given FastAPI app object.

    Accepts the app instance directly (not a string) to avoid uvicorn
    re-importing the module and loading models twice.

    Args:
        app:  FastAPI application instance.
        port: Port to listen on.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_level="info")
