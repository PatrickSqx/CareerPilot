"""Optional JSearch Mega live search through RapidAPI."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from jobpilot.config import JSEARCH_API_KEY_ENV, JSEARCH_RAPIDAPI_HOST_ENV, RAPIDAPI_KEY_ENV
from jobpilot.ingestion.current_api import load_env_file, normalize_current_records
from jobpilot.live.provider_base import LiveFetchResult, LiveJobProvider, sanitize_live_payload


DEFAULT_JSEARCH_RAPIDAPI_HOST = "jsearch-mega.p.rapidapi.com"


def _credentials(runtime_api_key: str | None = None) -> tuple[str, str]:
    api_key = runtime_api_key or os.getenv(RAPIDAPI_KEY_ENV, "") or os.getenv(JSEARCH_API_KEY_ENV, "")
    host = os.getenv(JSEARCH_RAPIDAPI_HOST_ENV, DEFAULT_JSEARCH_RAPIDAPI_HOST)
    return api_key, host


def has_jsearch_credentials(*, runtime_api_key: str | None = None, env_file: Path | None = None) -> bool:
    load_env_file(env_file)
    api_key, _ = _credentials(runtime_api_key)
    return bool(api_key)


class JSearchProvider(LiveJobProvider):
    provider_name = "jsearch"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        host: str | None = None,
        timeout: int = 20,
    ) -> None:
        env_key, env_host = _credentials(api_key)
        self.api_key = env_key
        self.host = host or env_host
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        return f"https://{self.host}"

    def search(
        self,
        query: str,
        *,
        location: str | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        search_query = " ".join(part for part in [query, location or ""] if part).strip()
        params = urllib.parse.urlencode(
            {
                "query": search_query,
                "page": str(max(1, int(page))),
                "num_pages": "1",
            }
        )
        request = urllib.request.Request(
            f"{self.base_url}/search?{params}",
            headers={
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": self.host,
                "User-Agent": "JobPilot live search",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        records = list(payload.get("data", []))
        return records[: max(1, min(int(results_per_page), 20))]


def fetch_jsearch_live(
    queries: list[str],
    *,
    pages_per_query: int = 1,
    results_per_page: int = 10,
    env_file: Path | None = None,
    runtime_api_key: str | None = None,
    timeout: int = 20,
) -> LiveFetchResult:
    """Fetch and normalize live JSearch records.

    Credentials are read only from env/.env/runtime args and are never returned.
    """

    load_env_file(env_file)
    api_key, host = _credentials(runtime_api_key)
    metadata: dict[str, Any] = {
        "provider": "jsearch",
        "rapidapi_host": host,
        "has_credentials": bool(api_key),
        "api_call_count": 0,
        "estimated_api_calls": len(queries) * max(1, min(int(pages_per_query), 1)),
        "raw_records_fetched": 0,
        "normalized_live_records": 0,
        "query_counts": {},
        "errors": [],
        "warnings": [],
    }
    if not api_key:
        metadata["warnings"].append("JSearch RapidAPI credentials missing; live search skipped.")
        return LiveFetchResult("jsearch", queries, [], [], metadata)

    provider = JSearchProvider(api_key=api_key, host=host, timeout=timeout)
    raw_records: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    bounded_pages = max(1, min(int(pages_per_query), 1))
    bounded_results = max(1, min(int(results_per_page), 20))

    for query in queries:
        query_raw_count = 0
        query_normalized_count = 0
        for page in range(1, bounded_pages + 1):
            try:
                records = sanitize_live_payload(
                    provider.search(query, page=page, results_per_page=bounded_results),
                    [api_key],
                )
                metadata["api_call_count"] += 1
                raw_records.extend(records)
                query_raw_count += len(records)
                normalized, stats = normalize_current_records(records, provider="jsearch", query=query)
                for row in normalized:
                    row["source"] = "jsearch_live"
                    row["raw_source"] = "jsearch_live_api"
                    row["is_current_api"] = True
                normalized_records.extend(normalized)
                query_normalized_count += len(normalized)
                invalid = int(stats.get("invalid_records_removed", 0))
                if invalid:
                    metadata["warnings"].append(f"{query}: {invalid} invalid live records removed")
            except Exception as exc:
                error = str(sanitize_live_payload(str(exc), [api_key]))
                metadata["errors"].append(f"JSearch fetch failed for {query} page {page}: {type(exc).__name__}: {error}")
        metadata["query_counts"][query] = {
            "raw_records": query_raw_count,
            "normalized_records": query_normalized_count,
        }

    metadata["raw_records_fetched"] = len(raw_records)
    metadata["normalized_live_records"] = len(normalized_records)
    return LiveFetchResult("jsearch", queries, raw_records, normalized_records, metadata)
