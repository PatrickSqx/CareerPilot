"""Session persistence for Phase 3 demo workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.paths import SESSION_DIR, ensure_storage_dirs


def _session_path(session_id: str) -> Path:
    safe = "".join(char for char in session_id if char.isalnum() or char in {"-", "_"}) or "session"
    return SESSION_DIR / f"{safe}.json"


def save_session(payload: dict[str, Any]) -> None:
    ensure_storage_dirs()
    path = _session_path(str(payload["session_id"]))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_session(session_id: str) -> dict[str, Any] | None:
    path = _session_path(session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_session(session_id: str, **updates: Any) -> dict[str, Any] | None:
    payload = load_session(session_id)
    if not payload:
        return None
    payload.update(updates)
    save_session(payload)
    return payload

