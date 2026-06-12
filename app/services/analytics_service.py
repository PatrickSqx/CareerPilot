"""Market analytics loader for the Phase 3 web layer."""

from __future__ import annotations

import json
import csv
from functools import lru_cache
from typing import Any

from app.services.paths import PROJECT_ROOT  # noqa: F401
from jobpilot.config import (
    MARKET_ANALYTICS_JSON,
    OFFLINE_SNAPSHOT_CSV,
    OFFLINE_SNAPSHOT_SAMPLE_CSV,
    TECH_MARKET_ANALYTICS_JSON,
)


SALARY_DISPLAY_MIN = 10_000
SALARY_DISPLAY_MAX = 1_000_000


def _format_usd(value: float | int | None) -> str:
    if value is None:
        return "Not shown"
    return f"${float(value):,.0f}"


@lru_cache(maxsize=1)
def _salary_display_summary() -> dict[str, Any]:
    """Build a presentation-only salary summary from the existing snapshot.

    The persisted market analytics file keeps raw extrema, which can include
    hourly fragments or parser artifacts. The web dashboard uses screened
    annualized display bounds so obvious outliers do not become headline UI.
    """

    snapshot_path = OFFLINE_SNAPSHOT_CSV
    if not snapshot_path.exists() and OFFLINE_SNAPSHOT_SAMPLE_CSV.exists():
        snapshot_path = OFFLINE_SNAPSHOT_SAMPLE_CSV
    if not snapshot_path.exists():
        return {}

    values: list[float] = []
    excluded_low = 0
    excluded_high = 0
    with snapshot_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in ("salary_min", "salary_max"):
                raw_value = row.get(field)
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if value <= 0:
                    continue
                if value < SALARY_DISPLAY_MIN:
                    excluded_low += 1
                    continue
                if value > SALARY_DISPLAY_MAX:
                    excluded_high += 1
                    continue
                values.append(value)

    if not values:
        return {}

    average = sum(values) / len(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "count": len(values),
        "average": average,
        "min": minimum,
        "max": maximum,
        "currency": "USD",
        "screened": True,
        "excluded_low": excluded_low,
        "excluded_high": excluded_high,
        "count_label": f"{len(values):,}",
        "average_label": _format_usd(average),
        "min_label": _format_usd(minimum),
        "max_label": _format_usd(maximum),
        "screening_note": (
            f"Display range excludes {excluded_low + excluded_high:,} obvious outlier values "
            f"outside {_format_usd(SALARY_DISPLAY_MIN)}-{_format_usd(SALARY_DISPLAY_MAX)}."
        ),
    }


def load_market_analytics() -> dict[str, Any]:
    if not MARKET_ANALYTICS_JSON.exists():
        return {
            "status": "missing",
            "warning": "data/processed/market_analytics.json was not found. Run the Phase 1 pipeline first.",
        }
    payload = json.loads(MARKET_ANALYTICS_JSON.read_text(encoding="utf-8"))
    payload["status"] = "available"
    payload["salary_display_summary"] = _salary_display_summary()
    return payload


def load_tech_market_analytics() -> dict[str, Any]:
    if not TECH_MARKET_ANALYTICS_JSON.exists():
        return {"status": "missing"}
    payload = json.loads(TECH_MARKET_ANALYTICS_JSON.read_text(encoding="utf-8"))
    payload["status"] = "available"
    return payload


def as_items(value: Any, *, limit: int = 15) -> list[dict[str, Any]]:
    """Normalize common analytics dict/list shapes for templates."""

    if isinstance(value, dict):
        return [{"label": str(key), "value": item} for key, item in list(value.items())[:limit]]
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value[:limit]:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append({"label": str(item), "value": ""})
        return result
    return []


def source_rollup_counts(value: Any, *, row_count: Any = None) -> dict[str, int]:
    """Roll provider labels into the offline/current split shown on /analytics."""

    if not isinstance(value, dict):
        return {}

    offline = 0
    current_api = 0
    unknown = 0
    for key, raw_count in value.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        label = str(key or "").lower()
        if "adzuna" in label or "jsearch" in label or "api" in label:
            current_api += count
        elif label in {"", "unknown"}:
            unknown += count
        else:
            offline += count

    try:
        expected_rows = int(row_count)
    except (TypeError, ValueError):
        expected_rows = 0
    observed = offline + current_api + unknown
    if expected_rows > observed:
        unknown += expected_rows - observed

    result: dict[str, int] = {}
    if offline:
        result["Kaggle/offline snapshot"] = offline
    if current_api:
        result["Saved current API rows"] = current_api
    if unknown:
        result["Unclassified"] = unknown
    return result
