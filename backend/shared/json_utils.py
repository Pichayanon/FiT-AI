from __future__ import annotations

import json
from typing import Any, Dict, Optional


def parse_json(msg: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(msg)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
