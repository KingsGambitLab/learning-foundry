from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


_LOG_LOCK = Lock()
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_PATH = _LOG_DIR / "course_generation.log"


def log_coursegen_event(event: str, **fields: Any) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    line = json.dumps(record, sort_keys=True, default=str)
    with _LOG_LOCK:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def coursegen_log_path() -> Path:
    return _LOG_PATH
