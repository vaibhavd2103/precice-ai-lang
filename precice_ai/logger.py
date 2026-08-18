"""Structured logging for agent and MCP activity."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _log_path() -> Path:
    configured = os.environ.get("PRECICE_AI_LOG_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "logs" / "agent.jsonl").resolve()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event", {})
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                **payload,
            },
            ensure_ascii=False,
            default=str,
        )


_LOGGER = logging.getLogger("precice_ai.activity")
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False


def _ensure_handler() -> None:
    path = _log_path()
    if any(getattr(handler, "baseFilename", None) == str(path) for handler in _LOGGER.handlers):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    _LOGGER.addHandler(handler)


def log_event(event_type: str, **fields: Any) -> None:
    """Write one structured activity event without interrupting the agent."""
    try:
        _ensure_handler()
        payload = {"event": event_type, **fields}
        _LOGGER.info("activity", extra={"event": payload})
        print(f"[preCICE AI] {json.dumps(payload, ensure_ascii=False, default=str)}", flush=True)
    except Exception:
        # Logging must never break a streamed chat response.
        return


def log_file_path() -> str:
    return str(_log_path())
