"""Optional current-posting ingestion with no-key cached fallback support."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from jobpilot.config import (
    ADZUNA_APP_ID_ENV,
    ADZUNA_APP_KEY_ENV,
    CACHED_CURRENT_JSONL,
    CURRENT_RAW_DIR,
    CURRENT_JOB_QUERIES,
    JSEARCH_RAPIDAPI_HOST_ENV,
    JSEARCH_API_KEY_ENV,
    RAPIDAPI_KEY_ENV,
)
from jobpilot.ingestion.cleaner import clean_normalized_record
from jobpilot.ingestion.dedup import add_dedup_fields
from jobpilot.ingestion.normalizer import normalize_adzuna_record, normalize_jsearch_record, utc_now_iso
from jobpilot.schemas import CANONICAL_COLUMNS
from jobpilot.utils.io import read_jsonl, write_csv, write_json


def load_env_file(path: Path | None) -> None:
    """Load simple KEY=VALUE entries into os.environ without external dependencies."""

    if not path or not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _request_json(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _save_raw_results(raw_output_dir: Path, provider: str, query: str, rows: list[dict[str, Any]]) -> Path:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    path = raw_output_dir / f"{provider}_{_safe_name(query)}.json"
    write_json(path, {"provider": provider, "query": query, "records": rows})
    return path


def fetch_adzuna(query: str, *, country: str = "us", results_per_page: int = 50) -> list[dict[str, Any]]:
    app_id = os.getenv(ADZUNA_APP_ID_ENV)
    app_key = os.getenv(ADZUNA_APP_KEY_ENV)
    if not app_id or not app_key:
        return []
    params = urllib.parse.urlencode(
        {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": str(results_per_page),
            "what": query,
            "content-type": "application/json",
        }
    )
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1?{params}"
    payload = _request_json(url)
    return list(payload.get("results", []))


def fetch_jsearch(query: str, *, num_pages: int = 1) -> list[dict[str, Any]]:
    api_key = os.getenv(RAPIDAPI_KEY_ENV) or os.getenv(JSEARCH_API_KEY_ENV)
    if not api_key:
        return []
    host = os.getenv(JSEARCH_RAPIDAPI_HOST_ENV, "jsearch-mega.p.rapidapi.com")
    params = urllib.parse.urlencode({"query": query, "page": "1", "num_pages": str(num_pages)})
    url = f"https://{host}/search?{params}"
    payload = _request_json(
        url,
        headers={
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host,
        },
    )
    return list(payload.get("data", []))


def load_cached_current_postings(path: Path = CACHED_CURRENT_JSONL) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path) or [])


def load_saved_current_payloads(raw_output_dir: Path = CURRENT_RAW_DIR, provider: str = "adzuna") -> list[tuple[str, list[dict[str, Any]], Path]]:
    """Load previously saved live-provider payloads without making API calls."""

    payloads: list[tuple[str, list[dict[str, Any]], Path]] = []
    if not raw_output_dir.exists():
        return payloads
    for path in sorted(raw_output_dir.glob(f"{provider}_*.json")):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue
        query = str(payload.get("query") or path.stem.removeprefix(f"{provider}_")).replace("_", " ")
        payloads.append((query, [record for record in records if isinstance(record, dict)], path))
    return payloads


def normalize_current_records(
    raw_records: list[dict[str, Any]],
    *,
    provider: str,
    query: str = "",
    ingested_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ingested_at = ingested_at or utc_now_iso()
    rows: list[dict[str, Any]] = []
    invalid = 0
    for raw in raw_records:
        if provider == "jsearch":
            normalized = normalize_jsearch_record(raw, query=query, ingested_at=ingested_at)
        else:
            normalized = normalize_adzuna_record(raw, query=query, ingested_at=ingested_at)
        cleaned, errors = clean_normalized_record(normalized)
        if errors or cleaned is None:
            invalid += 1
            continue
        rows.append(add_dedup_fields(cleaned))
    return rows, {"invalid_records_removed": invalid}


def get_current_postings(
    *,
    fetch_live: bool = False,
    provider: str = "adzuna",
    queries: list[str] | None = None,
    cached_path: Path = CACHED_CURRENT_JSONL,
    env_file: Path | None = None,
    raw_output_dir: Path = CURRENT_RAW_DIR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return normalized current postings and metadata.

    Default behavior does not call live APIs. If live fetching is disabled or no
    credentials are found, this loads a cached JSONL sample when one exists.
    """

    load_env_file(env_file)
    queries = queries or CURRENT_JOB_QUERIES
    metadata: dict[str, Any] = {
        "fetch_live_requested": fetch_live,
        "provider": provider,
        "cached_current_used": False,
        "saved_raw_current_used": False,
        "live_records_fetched": 0,
        "cached_records_loaded": 0,
        "saved_raw_records_loaded": 0,
        "errors": [],
        "query_counts": {},
        "raw_output_paths": [],
    }
    rows: list[dict[str, Any]] = []
    ingested_at = utc_now_iso()

    if fetch_live:
        for query in queries:
            try:
                raw = fetch_jsearch(query) if provider == "jsearch" else fetch_adzuna(query)
                raw_path = _save_raw_results(raw_output_dir, provider, query, raw)
                metadata["raw_output_paths"].append(str(raw_path))
                normalized, stats = normalize_current_records(raw, provider=provider, query=query, ingested_at=ingested_at)
                rows.extend(normalized)
                metadata["live_records_fetched"] += len(raw)
                metadata["query_counts"][query] = len(normalized)
                if stats["invalid_records_removed"]:
                    metadata["errors"].append(f"{query}: {stats['invalid_records_removed']} invalid current records removed")
            except Exception as exc:  # Network is optional; keep the main path alive.
                metadata["errors"].append(f"{provider} fetch failed for {query}: {type(exc).__name__}: {exc}")

    if not rows:
        cached = load_cached_current_postings(cached_path)
        metadata["cached_records_loaded"] = len(cached)
        if cached:
            metadata["cached_current_used"] = True
            provider_name = "jsearch" if any("job_title" in record for record in cached[:3]) else "adzuna"
            rows, stats = normalize_current_records(cached, provider=provider_name, query="cached_current", ingested_at=ingested_at)
            if stats["invalid_records_removed"]:
                metadata["errors"].append(f"cached current: {stats['invalid_records_removed']} invalid records removed")

    if not rows:
        saved_payloads = load_saved_current_payloads(raw_output_dir, provider=provider)
        if saved_payloads:
            metadata["saved_raw_current_used"] = True
            for query, raw, path in saved_payloads:
                normalized, stats = normalize_current_records(raw, provider=provider, query=query, ingested_at=ingested_at)
                rows.extend(normalized)
                metadata["saved_raw_records_loaded"] += len(raw)
                metadata["query_counts"][query] = len(normalized)
                metadata["raw_output_paths"].append(str(path))
                if stats["invalid_records_removed"]:
                    metadata["errors"].append(f"saved current {query}: {stats['invalid_records_removed']} invalid records removed")

    return rows, metadata


def write_current_jobs(path: Path, rows: list[dict[str, Any]]) -> int:
    return write_csv(path, rows, CANONICAL_COLUMNS)
