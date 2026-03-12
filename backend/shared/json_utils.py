"""
JSON parsing utility for WebSocket message handling.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def parse_json(msg: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON string into a dictionary.

    Args:
        msg: Raw JSON string from WebSocket message.

    Returns:
        Parsed dictionary, or None if parsing fails or result is not a dict.
    """
    try:
        obj = json.loads(msg)
        return obj if isinstance(obj, dict) else None
    except Exception:  # pylint: disable=broad-except
        return None
