"""Shared live job provider interfaces and sanitization helpers."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class LiveFetchResult:
    provider: str
    queries: list[str]
    raw_records: list[dict[str, Any]]
    normalized_records: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class LiveJobProvider(Protocol):
    provider_name: str

    def search(
        self,
        query: str,
        *,
        location: str | None = None,
        page: int = 1,
        results_per_page: int = 20,
    ) -> list[dict[str, Any]]:
        ...


def _sanitize_url(text: str, sensitive_values: list[str]) -> str:
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return text
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_pairs = [
        (key, value)
        for key, value in query_pairs
        if key.lower() not in {"app_id", "app_key", "utm_source", "rapidapi_key"}
    ]
    safe_query = urllib.parse.urlencode(safe_pairs, doseq=True)
    sanitized = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, safe_query, parsed.fragment))
    for value in sensitive_values:
        if value:
            sanitized = sanitized.replace(value, "[redacted]")
    return sanitized


def sanitize_live_payload(value: Any, sensitive_values: list[str] | None = None) -> Any:
    """Remove live API credential echoes from raw, normalized, and report payloads."""

    values = [item for item in (sensitive_values or []) if item]
    if isinstance(value, dict):
        return {key: sanitize_live_payload(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_live_payload(item, values) for item in value]
    if isinstance(value, str):
        sanitized = _sanitize_url(value, values)
        for item in values:
            sanitized = sanitized.replace(item, "[redacted]")
        return sanitized
    return value


def rel_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()
