"""Attach offline LCA employer activity evidence to JobPilot jobs as a sidecar.

This module intentionally does not mutate Phase 1 snapshots and does not change
Phase 2 ranking. It builds a separate `job_id`-level evidence file.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from jobpilot.config import OFFLINE_SNAPSHOT_CSV, PROCESSED_DATA_DIR
from jobpilot.evidence.lca_sponsorship import (
    DISCLOSURE_NOTE,
    TARGET_ROLE_FAMILIES,
    classify_role_family,
    normalize_employer_key,
)
from jobpilot.utils.io import ensure_parent, write_json
from jobpilot.utils.text import clean_text


LCA_CACHE_DIR = PROCESSED_DATA_DIR / "lca_cache"
EMPLOYER_ROLE_LCA_SUMMARY_CSV = LCA_CACHE_DIR / "employer_role_lca_summary.csv"
EMPLOYER_LCA_SUMMARY_CSV = LCA_CACHE_DIR / "employer_lca_summary.csv"
LCA_CACHE_MANIFEST_JSON = LCA_CACHE_DIR / "lca_cache_manifest.json"
JOB_LCA_EVIDENCE_CSV = LCA_CACHE_DIR / "job_lca_evidence.csv"
JOB_LCA_EVIDENCE_MANIFEST_JSON = LCA_CACHE_DIR / "job_lca_evidence_manifest.json"
JOB_LCA_EVIDENCE_REPORT_MD = LCA_CACHE_DIR / "job_lca_evidence_report.md"

WINDOW_METRICS = [
    "recent_2q_certified_case_count",
    "recent_2q_certified_worker_positions",
    "recent_3q_certified_case_count",
    "recent_3q_certified_worker_positions",
    "historical_8q_certified_case_count",
    "historical_8q_certified_worker_positions",
]

JOB_LCA_EVIDENCE_COLUMNS = [
    "job_id",
    "title",
    "company",
    "employer",
    "employer_key",
    "job_role_family",
    "job_role_family_basis",
    "is_focus_role_family",
    "lca_employer_match",
    "lca_role_family_match",
    "lca_match_scope",
    "lca_activity_label",
    "lca_role_activity_label",
    "lca_employer_activity_label",
    "lca_employer_display",
    "lca_latest_decision_date",
    "lca_role_recent_2q_certified_case_count",
    "lca_role_recent_2q_certified_worker_positions",
    "lca_role_recent_3q_certified_case_count",
    "lca_role_recent_3q_certified_worker_positions",
    "lca_role_historical_8q_certified_case_count",
    "lca_role_historical_8q_certified_worker_positions",
    "lca_employer_recent_2q_certified_case_count",
    "lca_employer_recent_2q_certified_worker_positions",
    "lca_employer_recent_3q_certified_case_count",
    "lca_employer_recent_3q_certified_worker_positions",
    "lca_employer_historical_8q_certified_case_count",
    "lca_employer_historical_8q_certified_worker_positions",
    "lca_evidence_note",
    "lca_interpretation_boundary",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _int_value(row: dict[str, Any] | None, key: str) -> int:
    if not row:
        return 0
    text = clean_text(row.get(key)).replace(",", "")
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _readable_activity_label(label: str) -> str:
    labels = {
        "recent_lca_activity_high": "high recent H-1B LCA activity",
        "recent_lca_activity_moderate": "moderate recent H-1B LCA activity",
        "recent_lca_activity_low": "low recent H-1B LCA activity",
        "recent_lca_activity_no_certified_cases": "recent H-1B LCA activity without certified cases",
        "historical_lca_activity_only": "historical H-1B LCA activity only",
        "no_lca_activity_in_cache": "no H-1B LCA activity in the local cache",
    }
    return labels.get(label, label.replace("_", " "))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_lca_summary_indexes(
    *,
    role_summary_path: Path = EMPLOYER_ROLE_LCA_SUMMARY_CSV,
    employer_summary_path: Path = EMPLOYER_LCA_SUMMARY_CSV,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    """Load LCA summary CSVs into exact-key lookup dictionaries."""

    role_rows = _read_csv_rows(role_summary_path)
    employer_rows = _read_csv_rows(employer_summary_path)
    role_index = {
        (clean_text(row.get("employer_key")), clean_text(row.get("role_family"))): row
        for row in role_rows
        if clean_text(row.get("employer_key")) and clean_text(row.get("role_family"))
    }
    employer_index = {
        clean_text(row.get("employer_key")): row
        for row in employer_rows
        if clean_text(row.get("employer_key"))
    }
    return role_index, employer_index


def classify_job_role_family(job: dict[str, Any]) -> tuple[str, str]:
    """Return a conservative broad role family for a JobPilot posting."""

    title_family = classify_role_family(job.get("title"))
    if title_family != "other":
        return title_family, "title"

    description_hint = " ".join(
        clean_text(job.get(key))
        for key in ["extracted_skills", "schema_org_skills", "description_text"]
        if clean_text(job.get(key))
    )
    fallback_family = classify_role_family(job.get("title"), description_hint[:1200])
    if fallback_family != "other":
        return fallback_family, "title_description"
    return "other", "unclassified"


def _evidence_note(
    *,
    role_match: bool,
    employer_match: bool,
    role_family: str,
    label: str,
) -> str:
    if role_match:
        return (
            f"Employer has {_readable_activity_label(label)} for {role_family} roles in the local DOL OFLC LCA cache. "
            "This does not confirm sponsorship for this specific posting."
        )
    if employer_match:
        return (
            "Employer has H-1B LCA activity in the local DOL OFLC cache, but no exact role-family row matched "
            "this posting. This does not confirm sponsorship for this specific posting."
        )
    return (
        "No exact normalized employer match was found in the local LCA cache. Absence of a match is not evidence "
        "that the employer will not sponsor."
    )


def build_job_lca_evidence_row(
    job: dict[str, Any],
    *,
    role_index: dict[tuple[str, str], dict[str, str]],
    employer_index: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build one job-level LCA evidence row."""

    company = clean_text(job.get("company") or job.get("employer"))
    employer = clean_text(job.get("employer") or job.get("company"))
    employer_key = normalize_employer_key(company or employer)
    role_family, role_basis = classify_job_role_family(job)
    role_row = role_index.get((employer_key, role_family))
    employer_row = employer_index.get(employer_key)
    role_match = role_row is not None
    employer_match = employer_row is not None
    match_scope = "role_family" if role_match else "employer_only" if employer_match else "none"
    label = clean_text((role_row or employer_row or {}).get("lca_activity_label")) or "no_lca_activity_in_cache"
    latest = clean_text((role_row or employer_row or {}).get("latest_decision_date"))

    row: dict[str, Any] = {
        "job_id": clean_text(job.get("job_id")),
        "title": clean_text(job.get("title")),
        "company": company,
        "employer": employer,
        "employer_key": employer_key,
        "job_role_family": role_family,
        "job_role_family_basis": role_basis,
        "is_focus_role_family": _bool_text(role_family in TARGET_ROLE_FAMILIES),
        "lca_employer_match": _bool_text(employer_match),
        "lca_role_family_match": _bool_text(role_match),
        "lca_match_scope": match_scope,
        "lca_activity_label": label,
        "lca_role_activity_label": clean_text(role_row.get("lca_activity_label")) if role_row else "",
        "lca_employer_activity_label": clean_text(employer_row.get("lca_activity_label")) if employer_row else "",
        "lca_employer_display": clean_text((role_row or employer_row or {}).get("employer_display")),
        "lca_latest_decision_date": latest,
        "lca_evidence_note": _evidence_note(
            role_match=role_match,
            employer_match=employer_match,
            role_family=role_family,
            label=label,
        ),
        "lca_interpretation_boundary": DISCLOSURE_NOTE,
    }
    for metric in WINDOW_METRICS:
        row[f"lca_role_{metric}"] = _int_value(role_row, metric)
        row[f"lca_employer_{metric}"] = _int_value(employer_row, metric)
    return row


def _iter_snapshot_rows(path: Path, *, limit: int | None = None) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield dict(row)


def _coverage_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def build_job_lca_evidence(
    *,
    snapshot_path: Path = OFFLINE_SNAPSHOT_CSV,
    role_summary_path: Path = EMPLOYER_ROLE_LCA_SUMMARY_CSV,
    employer_summary_path: Path = EMPLOYER_LCA_SUMMARY_CSV,
    output_path: Path = JOB_LCA_EVIDENCE_CSV,
    manifest_path: Path = JOB_LCA_EVIDENCE_MANIFEST_JSON,
    report_path: Path = JOB_LCA_EVIDENCE_REPORT_MD,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build and write the job-level LCA evidence sidecar."""

    role_index, employer_index = load_lca_summary_indexes(
        role_summary_path=role_summary_path,
        employer_summary_path=employer_summary_path,
    )
    total = 0
    employer_matches = 0
    role_matches = 0
    focus_row_count = 0
    focus_role_matches = 0
    labels: Counter[str] = Counter()
    role_families: Counter[str] = Counter()

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_LCA_EVIDENCE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for job in _iter_snapshot_rows(snapshot_path, limit=limit):
            row = build_job_lca_evidence_row(job, role_index=role_index, employer_index=employer_index)
            writer.writerow({column: row.get(column, "") for column in JOB_LCA_EVIDENCE_COLUMNS})

            total += 1
            employer_match = row["lca_employer_match"] == "true"
            role_match = row["lca_role_family_match"] == "true"
            focus_role = row["is_focus_role_family"] == "true"
            employer_matches += int(employer_match)
            role_matches += int(role_match)
            focus_row_count += int(focus_role)
            focus_role_matches += int(focus_role and role_match)
            labels[clean_text(row["lca_activity_label"])] += 1
            role_families[clean_text(row["job_role_family"])] += 1

    manifest: dict[str, Any] = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "snapshot_path": str(snapshot_path),
        "role_summary_path": str(role_summary_path),
        "employer_summary_path": str(employer_summary_path),
        "output_path": str(output_path),
        "rows_written": total,
        "limit": limit,
        "employer_match_count": employer_matches,
        "employer_match_rate": _coverage_rate(employer_matches, total),
        "role_family_match_count": role_matches,
        "role_family_match_rate": _coverage_rate(role_matches, total),
        "focus_role_job_count": focus_row_count,
        "focus_role_job_rate": _coverage_rate(focus_row_count, total),
        "focus_role_family_match_count": focus_role_matches,
        "focus_role_family_match_rate": _coverage_rate(focus_role_matches, focus_row_count),
        "lca_activity_label_counts": dict(sorted(labels.items())),
        "job_role_family_counts": dict(sorted(role_families.items())),
        "interpretation_note": DISCLOSURE_NOTE,
        "ranking_behavior_changed": False,
        "phase1_snapshot_modified": False,
    }
    write_json(manifest_path, manifest)
    write_job_lca_report(report_path, manifest)
    return manifest


def write_job_lca_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Job-Level H-1B/LCA Evidence Sidecar",
        "",
        DISCLOSURE_NOTE,
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Snapshot rows written: {manifest['rows_written']:,}",
        f"- Employer exact-key matches: {manifest['employer_match_count']:,} ({manifest['employer_match_rate']:.1%})",
        f"- Employer + role-family matches: {manifest['role_family_match_count']:,} ({manifest['role_family_match_rate']:.1%})",
        f"- Focus-role postings: {manifest['focus_role_job_count']:,} ({manifest['focus_role_job_rate']:.1%})",
        f"- Focus-role postings with role-family LCA match: {manifest['focus_role_family_match_count']:,} ({manifest['focus_role_family_match_rate']:.1%})",
        "",
        "## Boundaries",
        "",
        "- The Phase 1 snapshot is not modified.",
        "- Existing ranking behavior is not modified.",
        "- Exact normalized employer keys are used; no fuzzy matching is applied.",
        "- A missing LCA match is not evidence that an employer will not sponsor.",
        "",
        "## Job Role Families",
        "",
        "| Role family | Jobs |",
        "|---|---:|",
    ]
    for family, count in manifest["job_role_family_counts"].items():
        lines.append(f"| {family} | {count:,} |")
    lines.extend(["", "## LCA Activity Labels", "", "| Label | Jobs |", "|---|---:|"])
    for label, count in manifest["lca_activity_label_counts"].items():
        lines.append(f"| {label} | {count:,} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
