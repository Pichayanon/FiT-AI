from __future__ import annotations

from typing import Any


def serve(app: Any, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, reload=False, log_level="info")
