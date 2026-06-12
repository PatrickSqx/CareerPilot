"""Streaming reader and raw schema inspection for the Kaggle JSONL source."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from jobpilot.utils.text import clean_text


def stream_kaggle_jsonl(path: Path, limit: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield raw Kaggle JSONL records without loading the source file into memory."""

    yielded = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            yielded += 1
            yield line_no, record


def _get_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def inspect_kaggle_schema(path: Path, sample_rows: int = 200) -> dict[str, Any]:
    """Inspect raw fields and likely mappings using a small JSONL sample."""

    top_level_counts: Counter[str] = Counter()
    nested_examples: dict[str, dict[str, Any]] = {}
    candidate_paths = {
        "title": ["name", "position.name"],
        "company": ["orgCompany.name", "orgCompany.nameOrg", "json.schemaOrg.hiringOrganization.name"],
        "location": ["orgAddress.addressLine", "orgAddress.formatted", "orgAddress.city"],
        "country": ["orgAddress.countryCode", "orgAddress.country"],
        "state": ["orgAddress.state"],
        "city": ["orgAddress.city"],
        "description": ["text", "html"],
        "link": ["url"],
        "source_record_id": ["idInSource", "_id.$oid"],
        "date_posted_or_scraped": ["dateScraped.$date", "dateCreated.$date", "dateUploaded.$date"],
        "employment_type": ["position.workType"],
    }
    missing_counts: dict[str, int] = defaultdict(int)
    mapping_hits: dict[str, Counter[str]] = {field: Counter() for field in candidate_paths}
    records_sampled = 0
    warnings: list[str] = []

    for _, record in stream_kaggle_jsonl(path, limit=sample_rows):
        records_sampled += 1
        top_level_counts.update(record.keys())
        for nested_key in ["orgCompany", "orgAddress", "position", "json", "orgTags"]:
            value = record.get(nested_key)
            if nested_key not in nested_examples and isinstance(value, dict):
                nested_examples[nested_key] = {
                    key: clean_text(val)[:160] if not isinstance(val, (dict, list)) else type(val).__name__
                    for key, val in list(value.items())[:8]
                }
        for canonical_field, paths in candidate_paths.items():
            found_path = ""
            for candidate in paths:
                if clean_text(_get_path(record, candidate)):
                    found_path = candidate
                    break
            if found_path:
                mapping_hits[canonical_field][found_path] += 1
            else:
                missing_counts[canonical_field] += 1

    missing_rates = {
        field: round(missing_counts[field] / records_sampled, 4) if records_sampled else 1.0
        for field in candidate_paths
    }
    likely_mappings = {
        field: (hits.most_common(1)[0][0] if hits else "")
        for field, hits in mapping_hits.items()
    }
    for expected in ["title", "description", "link"]:
        if missing_rates.get(expected, 1.0) > 0.2:
            warnings.append(f"High missing rate for expected field: {expected}")
    if not path.exists():
        warnings.append(f"Kaggle JSONL path does not exist: {path}")

    return {
        "sample_rows_requested": sample_rows,
        "records_sampled": records_sampled,
        "observed_top_level_fields": dict(sorted(top_level_counts.items())),
        "nested_field_examples": nested_examples,
        "missing_rates_for_candidate_fields": missing_rates,
        "likely_mappings": likely_mappings,
        "warnings": warnings,
    }

