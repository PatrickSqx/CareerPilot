"""SQLite feedback event storage for accept/reject/skip controls."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from app.services.paths import FEEDBACK_DB, ensure_storage_dirs


VALID_ACTIONS = {"accept", "reject", "skip"}


def _connect() -> sqlite3.Connection:
    ensure_storage_dirs()
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_events (
              event_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              profile_id TEXT,
              job_id TEXT NOT NULL,
              action TEXT NOT NULL,
              rank_at_action INTEGER,
              score_at_action REAL,
              timestamp TEXT NOT NULL,
              profile_snapshot_json TEXT NOT NULL,
              job_snapshot_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback_events(session_id)")


def record_feedback(
    *,
    session_id: str,
    profile: dict[str, Any],
    job: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    init_feedback_db()
    normalized_action = action.lower().strip()
    if normalized_action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported feedback action: {action}")
    event = {
        "event_id": uuid.uuid4().hex,
        "session_id": session_id,
        "profile_id": str(profile.get("profile_id", "")),
        "job_id": str(job.get("job_id", "")),
        "action": normalized_action,
        "rank_at_action": int(job.get("rank") or 0),
        "score_at_action": float(job.get("adjusted_score", job.get("final_score", 0)) or 0),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile_snapshot_json": json.dumps(profile, ensure_ascii=False),
        "job_snapshot_json": json.dumps(job, ensure_ascii=False),
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback_events (
              event_id, session_id, profile_id, job_id, action, rank_at_action,
              score_at_action, timestamp, profile_snapshot_json, job_snapshot_json
            ) VALUES (
              :event_id, :session_id, :profile_id, :job_id, :action, :rank_at_action,
              :score_at_action, :timestamp, :profile_snapshot_json, :job_snapshot_json
            )
            """,
            event,
        )
    return event


def get_feedback_events(session_id: str) -> list[dict[str, Any]]:
    init_feedback_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE session_id = ? ORDER BY timestamp, event_id",
            (session_id,),
        ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["profile_snapshot"] = json.loads(item.pop("profile_snapshot_json"))
            item["job_snapshot"] = json.loads(item.pop("job_snapshot_json"))
        except json.JSONDecodeError:
            item["profile_snapshot"] = {}
            item["job_snapshot"] = {}
        events.append(item)
    return events

