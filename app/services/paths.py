"""Project paths for the Phase 3 web layer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

APP_DIR = PROJECT_ROOT / "app"
STORAGE_DIR = APP_DIR / "storage"
SESSION_DIR = STORAGE_DIR / "sessions"
UPLOAD_DIR = STORAGE_DIR / "uploads"
FEEDBACK_DB = STORAGE_DIR / "jobpilot_feedback.sqlite"


def ensure_storage_dirs() -> None:
    for path in [STORAGE_DIR, SESSION_DIR, UPLOAD_DIR]:
        path.mkdir(parents=True, exist_ok=True)

