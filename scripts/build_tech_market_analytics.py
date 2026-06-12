"""Build a focused tech/data market analytics sidecar from offline artifacts."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.config import OFFLINE_SNAPSHOT_CSV, TECH_MARKET_ANALYTICS_JSON

ROLE_FAMILY_PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "role_family_taxonomy"
    / "jobpilot_role_family_taxonomy_baseline_predictions.csv"
)

SEGMENTS = [
    {
        "key": "data_analytics_bi",
        "label": "Data analytics / BI",
        "families": {"data_analytics", "bi_analytics"},
        "patterns": [
            r"\bdata analyst\b",
            r"\bbusiness analyst\b",
            r"\banalytics engineer\b",
            r"\bbusiness intelligence\b",
            r"\bbi analyst\b",
            r"\breporting analyst\b",
            r"\btableau\b",
            r"\bpower bi\b",
            r"\bsql\b",
        ],
    },
    {
        "key": "data_engineering",
        "label": "Data engineering",
        "families": {"data_engineering"},
        "patterns": [
            r"\bdata engineer\b",
            r"\betl\b",
            r"\bdata pipeline\b",
            r"\bdata warehouse\b",
            r"\bspark\b",
            r"\bkafka\b",
            r"\bairflow\b",
            r"\bdbt\b",
        ],
    },
    {
        "key": "ml_ai",
        "label": "ML / AI",
        "families": {"ml_ai", "ml_infra", "research_ai"},
        "patterns": [
            r"\bmachine learning\b",
            r"\bdata scientist\b",
            r"\bml engineer\b",
            r"\bmlops\b",
            r"\bapplied scientist\b",
            r"\bai\b",
            r"\bdeep learning\b",
            r"\bnlp\b",
            r"\bcomputer vision\b",
            r"\bpytorch\b",
            r"\btensorflow\b",
            r"\bscikit-learn\b",
        ],
    },
    {
        "key": "software_cloud",
        "label": "Software / cloud tech",
        "families": {"software_backend", "other_tech"},
        "patterns": [
            r"\bsoftware engineer\b",
            r"\bdeveloper\b",
            r"\bdevops\b",
            r"\bsre\b",
            r"\bcloud engineer\b",
            r"\bbackend\b",
            r"\bfront.?end\b",
            r"\bfull.?stack\b",
            r"\bjava\b",
            r"\baws\b",
            r"\bazure\b",
            r"\bkubernetes\b",
        ],
    },
]

TECH_SKILL_PATTERNS = [
    r"\bsql\b",
    r"\bpython\b",
    r"\br\b",
    r"\bexcel\b",
    r"\banalytics?\b",
    r"\btableau\b",
    r"\bpower bi\b",
    r"\blooker\b",
    r"\bbusiness intelligence\b",
    r"\bdata visualization\b",
    r"\bstatistics?\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bnlp\b",
    r"\bcomputer vision\b",
    r"\bpandas\b",
    r"\bnumpy\b",
    r"\bscikit\b",
    r"\btensorflow\b",
    r"\bpytorch\b",
    r"\bdata engineering\b",
    r"\betl\b",
    r"\bdata warehouse\b",
    r"\bdata warehousing\b",
    r"\bbig data\b",
    r"\bspark\b",
    r"\bkafka\b",
    r"\bairflow\b",
    r"\bdbt\b",
    r"\baws\b",
    r"\bamazon web services\b",
    r"\bazure\b",
    r"\bgcp\b",
    r"\bgoogle cloud\b",
    r"\bdocker\b",
    r"\bkubernetes\b",
    r"\bjava\b",
    r"\bc\+\+\b",
    r"\bscala\b",
    r"\bgit\b",
    r"\bmicroservices\b",
]

TECH_SKILL_DISPLAY_RULES = [
    (r"\bsql\b", "SQL"),
    (r"\bpython\b", "Python"),
    (r"\bmicrosoft excel\b|\bexcel\b", "Excel"),
    (r"\bjava\b", "Java"),
    (r"\bamazon web services\b|\baws\b", "AWS"),
    (r"\bazure\b", "Azure"),
    (r"\bgoogle cloud\b|\bgcp\b", "GCP"),
    (r"\bmachine learning\b", "Machine learning"),
    (r"\banalytics?\b", "Analytics"),
    (r"\btableau\b", "Tableau"),
    (r"\bpower bi\b", "Power BI"),
    (r"\blooker\b", "Looker"),
    (r"\bstatistics?\b", "Statistics"),
    (r"\bpandas\b", "pandas"),
    (r"\bnumpy\b", "NumPy"),
    (r"\bscikit\b", "scikit-learn"),
    (r"\btensorflow\b", "TensorFlow"),
    (r"\bpytorch\b", "PyTorch"),
    (r"\bkafka\b", "Kafka"),
    (r"\bspark\b", "Spark"),
    (r"\bairflow\b", "Airflow"),
    (r"\bdbt\b", "dbt"),
    (r"\betl\b", "ETL"),
    (r"\bdata warehouse|\bdata warehousing\b", "Data warehousing"),
    (r"\bdata engineering\b", "Data engineering"),
    (r"\bdocker\b", "Docker"),
    (r"\bkubernetes\b", "Kubernetes"),
    (r"\bmicroservices\b", "Microservices"),
    (r"\bgit\b", "Git"),
    (r"\bc\+\+\b", "C++"),
    (r"\bscala\b", "Scala"),
    (r"\bnlp\b", "NLP"),
    (r"\bcomputer vision\b", "Computer vision"),
    (r"\bdeep learning\b", "Deep learning"),
]


def split_terms(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def is_focus_skill(skill: str) -> bool:
    text = str(skill or "").lower()
    return any(re.search(pattern, text) for pattern in TECH_SKILL_PATTERNS)


def display_skill(skill: str) -> str:
    text = str(skill or "").lower()
    for pattern, label in TECH_SKILL_DISPLAY_RULES:
        if re.search(pattern, text):
            return label
    return skill.strip()


def read_role_families() -> dict[str, set[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    if not ROLE_FAMILY_PREDICTIONS_CSV.exists():
        return families
    with ROLE_FAMILY_PREDICTIONS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            job_id = row.get("job_id", "")
            if not job_id:
                continue
            value = str(row.get("role_family_primary", "") or "").strip()
            if value:
                families[job_id].add(value)
    return families


def classify_segments(row: dict[str, str], families: set[str]) -> list[str]:
    text = " ".join(
        [
            row.get("title", ""),
            row.get("query", ""),
            row.get("normalized_role_terms", ""),
        ]
    ).lower()
    matches: list[str] = []
    for segment in SEGMENTS:
        family_hit = bool(families & segment["families"])
        text_hit = any(re.search(pattern, text) for pattern in segment["patterns"])
        if family_hit or text_hit:
            matches.append(segment["key"])
    return matches


def midpoint_salary(row: dict[str, str]) -> float | None:
    try:
        minimum = float(row.get("salary_min") or 0)
        maximum = float(row.get("salary_max") or 0)
    except ValueError:
        return None
    if minimum and maximum:
        value = (minimum + maximum) / 2
    else:
        value = minimum or maximum
    if 20_000 <= value <= 400_000:
        return value
    return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return round(ordered[idx], 2)


def summarize_salary(values: list[float]) -> dict[str, Any]:
    return {
        "listed_rows": len(values),
        "median_midpoint": round(median(values), 2) if values else None,
        "p25_midpoint": percentile(values, 0.25),
        "p75_midpoint": percentile(values, 0.75),
    }


def segment_payload(stats: dict[str, Any], total_rows: int) -> dict[str, Any]:
    count = stats["count"]
    return {
        "count": count,
        "share_of_snapshot": round(count / total_rows * 100, 2) if total_rows else 0,
        "top_skills": dict(stats["skills"].most_common(12)),
        "top_titles": dict(stats["titles"].most_common(8)),
        "top_locations": dict(stats["locations"].most_common(8)),
        "remote_distribution": dict(stats["remote"].most_common()),
        "employment_type_distribution": dict(stats["employment"].most_common(8)),
        "salary": summarize_salary(stats["salary_values"]),
    }


def build() -> dict[str, Any]:
    role_families = read_role_families()
    stats = {
        segment["key"]: {
            "count": 0,
            "skills": Counter(),
            "titles": Counter(),
            "locations": Counter(),
            "remote": Counter(),
            "employment": Counter(),
            "salary_values": [],
        }
        for segment in SEGMENTS
    }
    focus_job_ids: set[str] = set()
    focus_skills: Counter[str] = Counter()
    focus_titles: Counter[str] = Counter()
    focus_locations: Counter[str] = Counter()
    rows_seen = 0

    with OFFLINE_SNAPSHOT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_seen += 1
            job_id = row.get("job_id", "")
            matches = classify_segments(row, role_families.get(job_id, set()))
            if not matches:
                continue

            focus_job_ids.add(job_id)
            salary_value = midpoint_salary(row)
            skills = [
                display_skill(skill)
                for skill in split_terms(row.get("normalized_skills") or row.get("extracted_skills") or "")
                if is_focus_skill(skill)
            ]
            location = row.get("location") or "unknown"
            title = str(row.get("title") or "unknown").strip() or "unknown"
            remote = row.get("is_remote") or "unknown"
            employment = row.get("employment_type") or "unknown"

            for skill in skills:
                focus_skills[skill] += 1
            focus_titles[title] += 1
            focus_locations[location] += 1

            for key in matches:
                target = stats[key]
                target["count"] += 1
                target["titles"][title] += 1
                target["locations"][location] += 1
                target["remote"][remote] += 1
                target["employment"][employment] += 1
                if salary_value is not None:
                    target["salary_values"].append(salary_value)
                for skill in skills:
                    target["skills"][skill] += 1

    segments = {
        segment["label"]: segment_payload(stats[segment["key"]], rows_seen)
        for segment in SEGMENTS
    }
    return {
        "row_count": rows_seen,
        "focus_row_count": len(focus_job_ids),
        "focus_share_of_snapshot": round(len(focus_job_ids) / rows_seen * 100, 2) if rows_seen else 0,
        "segment_counts": {label: payload["count"] for label, payload in segments.items()},
        "top_focus_skills": dict(focus_skills.most_common(15)),
        "top_focus_titles": dict(focus_titles.most_common(12)),
        "top_focus_locations": dict(focus_locations.most_common(10)),
        "segments": segments,
        "source_artifacts": [
            "data/processed/jobs_offline_snapshot.csv",
            "data/processed/role_family_taxonomy/jobpilot_role_family_taxonomy_baseline_predictions.csv",
        ],
    }


def main() -> None:
    payload = build()
    TECH_MARKET_ANALYTICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(TECH_MARKET_ANALYTICS_JSON),
                "row_count": payload["row_count"],
                "focus_row_count": payload["focus_row_count"],
                "segment_counts": payload["segment_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
