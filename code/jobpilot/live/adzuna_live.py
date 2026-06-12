"""Optional live Adzuna search for Phase 2 hybrid matching."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from jobpilot.config import ADZUNA_APP_ID_ENV, ADZUNA_APP_KEY_ENV, PROCESSED_DATA_DIR
from jobpilot.ingestion.current_api import load_env_file, normalize_current_records
from jobpilot.live.provider_base import LiveFetchResult, LiveJobProvider, sanitize_live_payload
from jobpilot.utils.io import write_json


LIVE_CACHE_DIR = PROCESSED_DATA_DIR / "live_cache"


def _request_json(url: str, *, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "JobPilot live search"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _credentials(runtime_app_id: str | None = None, runtime_app_key: str | None = None) -> tuple[str, str]:
    return runtime_app_id or os.getenv(ADZUNA_APP_ID_ENV, ""), runtime_app_key or os.getenv(ADZUNA_APP_KEY_ENV, "")


class AdzunaProvider(LiveJobProvider):
    provider_name = "adzuna"

    def __init__(
        self,
        *,
        country: str = "us",
        app_id: str | None = None,
        app_key: str | None = None,
        timeout: int = 20,
    ) -> None:
        self.country = country
        self.app_id = app_id or os.getenv(ADZUNA_APP_ID_ENV, "")
        self.app_key = app_key or os.getenv(ADZUNA_APP_KEY_ENV, "")
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        location: str | None = None,
        page: int = 1,
        results_per_page: int = 20,
    ) -> list[dict[str, Any]]:
        if not self.app_id or not self.app_key:
            return []
        params = urllib.parse.urlencode(
            {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": str(max(1, min(int(results_per_page), 50))),
                "what": query,
                "where": location or "",
                "content-type": "application/json",
            }
        )
        url = f"https://api.adzuna.com/v1/api/jobs/{self.country}/search/{max(1, int(page))}?{params}"
        payload = _request_json(url, timeout=self.timeout)
        return list(payload.get("results", []))


def has_adzuna_credentials(*, runtime_app_id: str | None = None, runtime_app_key: str | None = None, env_file: Path | None = None) -> bool:
    load_env_file(env_file)
    app_id, app_key = _credentials(runtime_app_id, runtime_app_key)
    return bool(app_id and app_key)


def fetch_adzuna_live(
    queries: list[str],
    *,
    country: str = "us",
    pages_per_query: int = 1,
    results_per_page: int = 20,
    env_file: Path | None = None,
    runtime_app_id: str | None = None,
    runtime_app_key: str | None = None,
    timeout: int = 20,
) -> LiveFetchResult:
    """Fetch and normalize live Adzuna records.

    Keys are read only from env/.env/runtime args and are never returned.
    """

    load_env_file(env_file)
    app_id, app_key = _credentials(runtime_app_id, runtime_app_key)
    metadata: dict[str, Any] = {
        "provider": "adzuna",
        "has_credentials": bool(app_id and app_key),
        "api_call_count": 0,
        "raw_records_fetched": 0,
        "normalized_live_records": 0,
        "query_counts": {},
        "errors": [],
        "warnings": [],
    }
    if not app_id or not app_key:
        metadata["warnings"].append("Adzuna credentials missing; live search skipped.")
        return LiveFetchResult("adzuna", queries, [], [], metadata)

    raw_records: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    bounded_pages = max(1, min(int(pages_per_query), 2))
    bounded_results = max(1, min(int(results_per_page), 50))

    for query in queries:
        query_raw_count = 0
        query_normalized_count = 0
        for page in range(1, bounded_pages + 1):
            params = urllib.parse.urlencode(
                {
                    "app_id": app_id,
                    "app_key": app_key,
                    "results_per_page": str(bounded_results),
                    "what": query,
                    "content-type": "application/json",
                }
            )
            url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}?{params}"
            try:
                payload = _request_json(url, timeout=timeout)
                records = sanitize_live_payload(list(payload.get("results", [])), [app_id, app_key])
                metadata["api_call_count"] += 1
                raw_records.extend(records)
                query_raw_count += len(records)
                normalized, stats = normalize_current_records(records, provider="adzuna", query=query)
                for row in normalized:
                    row["source"] = "adzuna_live"
                    row["raw_source"] = "adzuna_live_api"
                    row["is_current_api"] = True
                normalized_records.extend(normalized)
                query_normalized_count += len(normalized)
                invalid = int(stats.get("invalid_records_removed", 0))
                if invalid:
                    metadata["warnings"].append(f"{query}: {invalid} invalid live records removed")
            except Exception as exc:
                error = str(sanitize_live_payload(str(exc), [app_id, app_key]))
                metadata["errors"].append(f"Adzuna fetch failed for {query} page {page}: {type(exc).__name__}: {error}")
        metadata["query_counts"][query] = {
            "raw_records": query_raw_count,
            "normalized_records": query_normalized_count,
        }

    metadata["raw_records_fetched"] = len(raw_records)
    metadata["normalized_live_records"] = len(normalized_records)
    return LiveFetchResult("adzuna", queries, raw_records, normalized_records, metadata)


def save_live_fetch_outputs(
    result: LiveFetchResult,
    *,
    timestamp: str,
    output_dir: Path = LIVE_CACHE_DIR,
) -> dict[str, str]:
    """Write live raw/normalized/report outputs without credentials."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"adzuna_live_raw_{timestamp}.json"
    report_path = output_dir / f"live_search_report_{timestamp}.json"
    write_json(raw_path, {"provider": result.provider, "queries": result.queries, "records": result.raw_records})
    write_json(report_path, result.metadata)
    return {
        "raw_path": raw_path.as_posix(),
        "report_path": report_path.as_posix(),
    }
