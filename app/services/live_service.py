"""Optional live refresh wrapper for the Phase 3 app.

The default path is dry-run/no-key safe. Real JSearch calls remain disabled from
the UI to avoid accidental quota use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.paths import PROJECT_ROOT  # noqa: F401
from jobpilot.live.adzuna_live import fetch_adzuna_live
from jobpilot.live.jsearch_live import fetch_jsearch_live
from jobpilot.live.query_builder import build_live_queries
from jobpilot.utils.text import clean_text


def run_live_refresh_preview(
    profile: dict[str, Any],
    *,
    provider: str = "adzuna",
    dry_run: bool = True,
    max_queries: int = 5,
    env_file: Path | None = None,
) -> dict[str, Any]:
    provider_name = clean_text(provider).lower() or "adzuna"
    if provider_name not in {"adzuna", "jsearch"}:
        provider_name = "adzuna"
    queries = build_live_queries(profile, max_queries=max_queries)
    if dry_run:
        return {
            "provider": provider_name,
            "mode": "dry_run",
            "queries": queries,
            "estimated_api_calls": len(queries),
            "raw_records_fetched": 0,
            "normalized_live_records": 0,
            "warnings": ["Dry run only: no live API call was made."],
        }
    if provider_name == "jsearch":
        return {
            "provider": provider_name,
            "mode": "dry_run_for_quota_safety",
            "queries": queries,
            "estimated_api_calls": len(queries),
            "raw_records_fetched": 0,
            "normalized_live_records": 0,
            "warnings": ["Real JSearch calls are disabled in the UI unless explicitly run from the CLI."],
        }
    result = fetch_adzuna_live(queries, env_file=env_file, pages_per_query=1, results_per_page=20)
    metadata = dict(result.metadata)
    metadata["mode"] = "live_attempt"
    metadata["queries"] = queries
    metadata["retained_records"] = len(result.normalized_records)
    return metadata
