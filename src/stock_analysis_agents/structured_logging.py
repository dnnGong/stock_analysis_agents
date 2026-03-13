from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return _as_bool(os.getenv("STOCK_AGENTS_STRUCTURED_LOG", "0"))


def _log_path() -> Path:
    raw = os.getenv("STOCK_AGENTS_LOG_PATH", "stock_agents_events.jsonl").strip()
    return Path(raw)


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(x) for x in value]
    return str(value)


def log_event(event: str, **fields: Any) -> None:
    """Append one JSONL structured log event if enabled.

    Env vars:
    - STOCK_AGENTS_STRUCTURED_LOG=1 enables logging
    - STOCK_AGENTS_LOG_PATH=/path/to/file.jsonl controls output path
    """
    if not is_enabled():
        return

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "pid": os.getpid(),
        **{k: _safe(v) for k, v in fields.items()},
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

