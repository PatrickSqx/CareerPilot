"""Inspect a generated JobPilot offline snapshot CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a JobPilot snapshot CSV.")
    parser.add_argument("snapshot_csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row_count = 0
    source_counts: Counter[str] = Counter()
    current_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    columns: list[str] = []
    with args.snapshot_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            source = row.get("source") or "unknown"
            source_counts[source] += 1
            if str(row.get("is_current_api")).lower() == "true":
                current_counts[source] += 1
            for column in columns:
                if not str(row.get(column, "") or "").strip():
                    missing_counts[column] += 1
    summary = {
        "path": str(args.snapshot_csv),
        "row_count": row_count,
        "columns": columns,
        "source_counts": dict(source_counts),
        "current_api_counts": dict(current_counts),
        "required_export_fields_present": {
            "link": "link" in columns,
            "title": "title" in columns,
            "company_or_employer": "company" in columns or "employer" in columns,
            "salary": any(column in columns for column in ["salary_min", "salary_max", "salary_raw"]),
            "location": "location" in columns,
            "full_description": "description_text" in columns or "description" in columns,
        },
        "missing_field_counts": dict(missing_counts),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

