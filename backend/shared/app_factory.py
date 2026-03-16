"""
app_factory.py — FastAPI application factory with standard CORS middleware.

Eliminates the identical FastAPI + CORS setup block that previously appeared
in every exercise streaming module (squat, lunges, plank, wall_sit).

Usage:
    from shared.app_factory import make_app
    app = make_app("FiT-AI Squat Streaming Backend")
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def make_app(title: str) -> FastAPI:
    """Create a FastAPI app pre-configured with standard CORS middleware.

    Args:
        title: Human-readable title shown in the OpenAPI docs.

    Returns:
        A FastAPI instance with CORS enabled for all origins.
    """
    app = FastAPI(title=title)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
