"""Rule-based resume-tailoring signal sidecar.

The sidecar is intentionally separate from hard-skill normalization and ranking.
It captures soft skills, professional competencies, and review-only
credential/language/tool signals for resume tailoring evidence.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobpilot.utils.io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OFFLINE_SNAPSHOT_CSV = PROCESSED_DATA_DIR / "jobs_offline_snapshot.csv"
RESUME_SIGNAL_DIR = PROCESSED_DATA_DIR / "resume_signal_sidecar"

DEFAULT_INPUT_FIELDS = [
    "title",
    "description_text",
    "raw_skills",
    "schema_org_skills",
    "raw_keywords",
    "raw_jobnames",
    "schema_org_occupational_category",
    "schema_org_experience_requirements",
]

SOFT_SKILL_SIGNALS: dict[str, list[str]] = {
    "communication": ["communication"],
    "written communication": ["written communication", "written communications"],
    "verbal communication": ["verbal communication", "verbal communications"],
    "leadership": ["leadership"],
    "teamwork": ["teamwork", "team work"],
    "collaboration": ["collaboration"],
    "attention to detail": ["attention to detail", "detail oriented", "detail-oriented"],
    "problem solving": ["problem solving", "problem-solving"],
    "critical thinking": ["critical thinking"],
    "customer service": ["customer service"],
    "time management": ["time management"],
    "organization": ["organization", "organizational skills"],
    "adaptability": ["adaptability"],
    "mentoring": ["mentoring", "coaching and mentoring"],
}

PROFESSIONAL_COMPETENCY_SIGNALS: dict[str, list[str]] = {
    "stakeholder management": ["stakeholder management"],
    "requirements gathering": ["requirements gathering", "gathering requirements"],
    "process improvement": ["process improvement"],
    "vendor coordination": ["vendor coordination", "vendor management"],
    "inventory control": ["inventory control"],
    "project coordination": ["project coordination"],
    "project management": ["project management"],
    "cross-functional collaboration": ["cross-functional collaboration", "cross functional collaboration"],
    "client-facing support": ["client-facing support", "client facing support"],
    "documentation": ["documentation"],
    "workflow optimization": ["workflow optimization", "workflow optimisation"],
    "scheduling": ["scheduling"],
    "reporting": ["reporting"],
    "training": ["training"],
    "quality assurance": ["quality assurance"],
    "risk management": ["risk management"],
    "compliance": ["compliance"],
    "data entry": ["data entry"],
    "payroll": ["payroll"],
    "merchandising": ["merchandising"],
    "recruitment": ["recruitment", "recruiting"],
    "onboarding": ["onboarding", "on boarding"],
}

CREDENTIAL_LANGUAGE_TOOL_REVIEW_SIGNALS: dict[str, list[str]] = {
    "acls": ["acls"],
    "bls": ["bls"],
    "ase": ["ase"],
    "cpa": ["cpa"],
    "cpr": ["cpr"],
    "english": ["english"],
    "spanish": ["spanish"],
    "bilingual": ["bilingual"],
    "microsoft word": ["microsoft word", "ms word"],
    "word": ["word"],
    "microsoft outlook": ["microsoft outlook", "ms outlook"],
    "outlook": ["outlook"],
    "microsoft powerpoint": ["microsoft powerpoint", "ms powerpoint"],
    "powerpoint": ["powerpoint"],
    "spreadsheets": ["spreadsheets", "spreadsheet"],
}

SIGNAL_GROUPS = {
    "soft_skill": SOFT_SKILL_SIGNALS,
    "professional_competency": PROFESSIONAL_COMPETENCY_SIGNALS,
    "credential_language_tool_review": CREDENTIAL_LANGUAGE_TOOL_REVIEW_SIGNALS,
}

SIGNAL_LIST_FIELD = {
    "soft_skill": "soft_skill_signals",
    "professional_competency": "professional_competency_signals",
    "credential_language_tool_review": "credential_language_tool_review_signals",
}


@dataclass(frozen=True)
class CompiledSignal:
    signal_type: str
    normalized_text: str
    surface: str
    surface_key: str


def _surface_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _surface_pattern_fragment(surface: str) -> str:
    escaped = re.escape(surface)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]+")
    return escaped


def compiled_signals() -> list[CompiledSignal]:
    compiled: list[CompiledSignal] = []
    for signal_type, mapping in SIGNAL_GROUPS.items():
        for normalized_text, surfaces in mapping.items():
            for surface in surfaces:
                compiled.append(
                    CompiledSignal(
                        signal_type=signal_type,
                        normalized_text=normalized_text,
                        surface=surface,
                        surface_key=_surface_key(surface),
                    )
                )
    return sorted(compiled, key=lambda item: (-len(item.surface), item.signal_type, item.normalized_text))


COMPILED_SIGNALS = compiled_signals()
SURFACE_SIGNAL_LOOKUP: dict[str, list[CompiledSignal]] = {}
for _signal in COMPILED_SIGNALS:
    SURFACE_SIGNAL_LOOKUP.setdefault(_signal.surface_key, []).append(_signal)
COMBINED_SIGNAL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(_surface_pattern_fragment(signal.surface) for signal in COMPILED_SIGNALS)
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def extract_resume_signals(row: dict[str, Any], *, input_fields: list[str] | None = None) -> dict[str, Any]:
    """Extract deduped resume-tailoring signals with one evidence span each."""

    input_fields = input_fields or DEFAULT_INPUT_FIELDS
    signal_lists: dict[str, list[str]] = {
        "soft_skill_signals": [],
        "professional_competency_signals": [],
        "credential_language_tool_review_signals": [],
    }
    seen: set[tuple[str, str]] = set()
    evidence_spans: list[dict[str, Any]] = []

    for field in input_fields:
        text = str(row.get(field) or "")
        if not text:
            continue
        for match in COMBINED_SIGNAL_PATTERN.finditer(text):
            evidence = text[match.start() : match.end()]
            for signal in SURFACE_SIGNAL_LOOKUP.get(_surface_key(evidence), []):
                key = (signal.signal_type, signal.normalized_text)
                if key in seen:
                    continue
                seen.add(key)
                list_field = SIGNAL_LIST_FIELD[signal.signal_type]
                signal_lists[list_field].append(signal.normalized_text)
                evidence_spans.append(
                    {
                        "signal_type": signal.signal_type,
                        "normalized_text": signal.normalized_text,
                        "text": evidence,
                        "source_field": field,
                        "start": match.start(),
                        "end": match.end(),
                        "evidence": evidence,
                    }
                )

    return {
        "soft_skill_signals": signal_lists["soft_skill_signals"],
        "professional_competency_signals": signal_lists["professional_competency_signals"],
        "credential_language_tool_review_signals": signal_lists["credential_language_tool_review_signals"],
        "evidence_spans": evidence_spans,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_resume_signal_sidecar(
    *,
    snapshot_path: Path = OFFLINE_SNAPSHOT_CSV,
    output_path: Path = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_sidecar.jsonl",
    manifest_path: Path = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_manifest.json",
    report_path: Path = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_report.md",
    validation_path: Path = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_validation.json",
    input_fields: list[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    input_fields = input_fields or DEFAULT_INPUT_FIELDS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    rows_with_soft = 0
    rows_with_professional = 0
    rows_with_review = 0
    evidence_span_count = 0
    signal_counts = {
        "soft_skill": Counter(),
        "professional_competency": Counter(),
        "credential_language_tool_review": Counter(),
    }
    source_field_counts: Counter[str] = Counter()

    with snapshot_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        reader = csv.DictReader(source)
        for row in reader:
            signals = extract_resume_signals(row, input_fields=input_fields)
            payload = {
                "job_id": row.get("job_id", ""),
                **signals,
                "use_for_resume_tailoring": True,
                "use_for_ranking": False,
                "soft_skills_ranking_weight": 0,
                "ranking_behavior_changed": False,
                "phase1_ingestion_modified": False,
                "phase1_snapshot_modified": False,
                "cloud_run_online_inference_enabled": False,
                "generated_at": generated_at,
            }
            out.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            rows_written += 1
            rows_with_soft += int(bool(payload["soft_skill_signals"]))
            rows_with_professional += int(bool(payload["professional_competency_signals"]))
            rows_with_review += int(bool(payload["credential_language_tool_review_signals"]))
            evidence_span_count += len(payload["evidence_spans"])
            for signal in payload["soft_skill_signals"]:
                signal_counts["soft_skill"][signal] += 1
            for signal in payload["professional_competency_signals"]:
                signal_counts["professional_competency"][signal] += 1
            for signal in payload["credential_language_tool_review_signals"]:
                signal_counts["credential_language_tool_review"][signal] += 1
            for span in payload["evidence_spans"]:
                source_field_counts[span["source_field"]] += 1

    manifest = {
        "generated_at": generated_at,
        "sidecar_type": "phase2_16i_rule_based_resume_signal_sidecar",
        "snapshot_path": str(snapshot_path),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "validation_path": str(validation_path),
        "rows_written": rows_written,
        "input_fields": input_fields,
        "use_for_resume_tailoring": True,
        "use_for_ranking": False,
        "soft_skills_ranking_weight": 0,
        "ranking_behavior_changed": False,
        "phase1_ingestion_modified": False,
        "phase1_snapshot_modified": False,
        "cloud_run_online_inference_enabled": False,
        "bert_or_escoxlmr_used": False,
        "gpu_used": False,
        "paid_apis_or_live_scraping_used": False,
        "hard_skill_fields_written": False,
        "rows_with_soft_skill_signals": rows_with_soft,
        "rows_with_soft_skill_signals_rate": _rate(rows_with_soft, rows_written),
        "rows_with_professional_competency_signals": rows_with_professional,
        "rows_with_professional_competency_signals_rate": _rate(rows_with_professional, rows_written),
        "rows_with_credential_language_tool_review_signals": rows_with_review,
        "rows_with_credential_language_tool_review_signals_rate": _rate(rows_with_review, rows_written),
        "evidence_span_count": evidence_span_count,
        "top_soft_skill_signals": dict(signal_counts["soft_skill"].most_common(30)),
        "top_professional_competency_signals": dict(signal_counts["professional_competency"].most_common(30)),
        "top_credential_language_tool_review_signals": dict(
            signal_counts["credential_language_tool_review"].most_common(30)
        ),
        "evidence_source_field_counts": dict(source_field_counts.most_common()),
    }
    write_json(manifest_path, manifest)
    write_resume_signal_report(report_path, manifest)
    validation = validate_resume_signal_sidecar(
        snapshot_path=snapshot_path,
        sidecar_path=output_path,
        manifest=manifest,
    )
    write_json(validation_path, validation)
    return manifest


def write_resume_signal_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 2.16I Resume Signal Sidecar Report",
        "",
        "Rule-based resume-tailoring signals. This sidecar is not used for ranking and does not write hard-skill fields.",
        "",
        "## Summary",
        "",
        f"- Rows written: {manifest['rows_written']:,}",
        f"- Rows with soft skill signals: {manifest['rows_with_soft_skill_signals']:,}",
        f"- Rows with professional competency signals: {manifest['rows_with_professional_competency_signals']:,}",
        f"- Rows with credential/language/tool review signals: {manifest['rows_with_credential_language_tool_review_signals']:,}",
        f"- Evidence spans: {manifest['evidence_span_count']:,}",
        "- `use_for_resume_tailoring=true`",
        "- `use_for_ranking=false`",
        "- `soft_skills_ranking_weight=0`",
        "",
        "## Top Soft Skill Signals",
        "",
        "| Signal | Rows |",
        "|---|---:|",
    ]
    for signal, count in manifest["top_soft_skill_signals"].items():
        lines.append(f"| `{signal}` | {count:,} |")
    lines.extend(["", "## Top Professional Competency Signals", "", "| Signal | Rows |", "|---|---:|"])
    for signal, count in manifest["top_professional_competency_signals"].items():
        lines.append(f"| `{signal}` | {count:,} |")
    lines.extend(["", "## Top Credential/Language/Tool Review Signals", "", "| Signal | Rows |", "|---|---:|"])
    for signal, count in manifest["top_credential_language_tool_review_signals"].items():
        lines.append(f"| `{signal}` | {count:,} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_resume_signal_sidecar(
    *,
    snapshot_path: Path,
    sidecar_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        snapshot_ids = [row["job_id"] for row in csv.DictReader(handle)]

    rows = 0
    parse_errors = []
    duplicates: Counter[str] = Counter()
    seen: set[str] = set()
    order: list[str] = []
    rows_with_soft = 0
    rows_with_professional = 0
    rows_with_review = 0
    evidence_span_count = 0
    signal_counts = {
        "soft_skill": Counter(),
        "professional_competency": Counter(),
        "credential_language_tool_review": Counter(),
    }
    source_field_counts: Counter[str] = Counter()
    invalid_rows: list[dict[str, Any]] = []
    hard_skill_field_rows = 0
    bad_flag_rows = 0

    with sidecar_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"row_index": row_index, "error": str(exc)})
                continue
            rows += 1
            job_id = str(row.get("job_id") or "")
            order.append(job_id)
            if job_id in seen:
                duplicates[job_id] += 1
            seen.add(job_id)
            if row.get("use_for_resume_tailoring") is not True or row.get("use_for_ranking") is not False:
                bad_flag_rows += 1
            if row.get("soft_skills_ranking_weight") != 0 or row.get("ranking_behavior_changed") is not False:
                bad_flag_rows += 1
            if any(field in row for field in ["normalized_hard_skills", "hard_skill_spans", "dropped_generic_terms"]):
                hard_skill_field_rows += 1
            rows_with_soft += int(bool(row.get("soft_skill_signals")))
            rows_with_professional += int(bool(row.get("professional_competency_signals")))
            rows_with_review += int(bool(row.get("credential_language_tool_review_signals")))
            for signal in row.get("soft_skill_signals") or []:
                signal_counts["soft_skill"][signal] += 1
            for signal in row.get("professional_competency_signals") or []:
                signal_counts["professional_competency"][signal] += 1
            for signal in row.get("credential_language_tool_review_signals") or []:
                signal_counts["credential_language_tool_review"][signal] += 1
            for span in row.get("evidence_spans") or []:
                evidence_span_count += 1
                source_field_counts[str(span.get("source_field") or "")] += 1
                missing = [
                    key
                    for key in ["signal_type", "normalized_text", "text", "source_field", "start", "end", "evidence"]
                    if key not in span
                ]
                if missing:
                    invalid_rows.append({"row_index": row_index, "job_id": job_id, "missing_span_keys": missing})
            for list_field in [
                "soft_skill_signals",
                "professional_competency_signals",
                "credential_language_tool_review_signals",
            ]:
                values = row.get(list_field) or []
                if len(values) != len(set(values)):
                    invalid_rows.append({"row_index": row_index, "job_id": job_id, "duplicate_signal_list": list_field})

    manifest_checks = {
        "rows_written": rows == manifest.get("rows_written"),
        "rows_with_soft_skill_signals": rows_with_soft == manifest.get("rows_with_soft_skill_signals"),
        "rows_with_professional_competency_signals": rows_with_professional
        == manifest.get("rows_with_professional_competency_signals"),
        "rows_with_credential_language_tool_review_signals": rows_with_review
        == manifest.get("rows_with_credential_language_tool_review_signals"),
        "evidence_span_count": evidence_span_count == manifest.get("evidence_span_count"),
        "top_soft_skill_signals": dict(signal_counts["soft_skill"].most_common(30))
        == manifest.get("top_soft_skill_signals"),
        "top_professional_competency_signals": dict(signal_counts["professional_competency"].most_common(30))
        == manifest.get("top_professional_competency_signals"),
        "top_credential_language_tool_review_signals": dict(
            signal_counts["credential_language_tool_review"].most_common(30)
        )
        == manifest.get("top_credential_language_tool_review_signals"),
    }
    boundary_checks = {
        "use_for_resume_tailoring_true": manifest.get("use_for_resume_tailoring") is True,
        "use_for_ranking_false": manifest.get("use_for_ranking") is False,
        "soft_skills_ranking_weight_zero": manifest.get("soft_skills_ranking_weight") == 0,
        "ranking_behavior_changed_false": manifest.get("ranking_behavior_changed") is False,
        "phase1_ingestion_modified_false": manifest.get("phase1_ingestion_modified") is False,
        "phase1_snapshot_modified_false": manifest.get("phase1_snapshot_modified") is False,
        "cloud_run_online_inference_enabled_false": manifest.get("cloud_run_online_inference_enabled") is False,
        "bert_or_escoxlmr_used_false": manifest.get("bert_or_escoxlmr_used") is False,
        "hard_skill_fields_written_false": manifest.get("hard_skill_fields_written") is False,
    }
    validation = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sidecar_path": str(sidecar_path),
        "manifest_path": str(manifest.get("manifest_path")),
        "rows": rows,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:10],
        "duplicate_job_id_count": sum(duplicates.values()),
        "job_id_order_matches_snapshot": order == snapshot_ids,
        "manifest_checks": manifest_checks,
        "boundary_checks": boundary_checks,
        "bad_flag_rows": bad_flag_rows,
        "hard_skill_field_rows": hard_skill_field_rows,
        "invalid_evidence_or_dedupe_rows": invalid_rows[:20],
        "recomputed_counts": {
            "rows_with_soft_skill_signals": rows_with_soft,
            "rows_with_professional_competency_signals": rows_with_professional,
            "rows_with_credential_language_tool_review_signals": rows_with_review,
            "evidence_span_count": evidence_span_count,
            "top_soft_skill_signals": dict(signal_counts["soft_skill"].most_common(30)),
            "top_professional_competency_signals": dict(signal_counts["professional_competency"].most_common(30)),
            "top_credential_language_tool_review_signals": dict(
                signal_counts["credential_language_tool_review"].most_common(30)
            ),
            "evidence_source_field_counts": dict(source_field_counts.most_common()),
        },
    }
    validation["all_validation_checks_passed"] = (
        rows == len(snapshot_ids)
        and len(parse_errors) == 0
        and sum(duplicates.values()) == 0
        and order == snapshot_ids
        and all(manifest_checks.values())
        and all(boundary_checks.values())
        and bad_flag_rows == 0
        and hard_skill_field_rows == 0
        and not invalid_rows
    )
    return validation
