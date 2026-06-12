"""Phase 2.16J resume-tailoring policy and job profile sidecar.

This module consumes the Phase 2.16I hard-skill and resume-signal sidecars and
writes a separate job-level tailoring profile. It is intentionally offline-only
and does not write back to Phase 1 data, ranking code, or the 16I artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jobpilot.utils.io import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
HARD_SKILL_DIR = PROCESSED_DATA_DIR / "hard_skill_sidecar"
RESUME_SIGNAL_DIR = PROCESSED_DATA_DIR / "resume_signal_sidecar"

DEFAULT_SNAPSHOT_PATH = PROCESSED_DATA_DIR / "jobs_offline_snapshot.csv"
DEFAULT_HARD_SIDECAR_PATH = HARD_SKILL_DIR / "phase2_16i_hard_skill_sidecar_reviewed_normalization.jsonl"
DEFAULT_HARD_MANIFEST_PATH = HARD_SKILL_DIR / "phase2_16i_hard_skill_manifest_reviewed_normalization.json"
DEFAULT_HARD_VALIDATION_PATH = HARD_SKILL_DIR / "phase2_16i_validation.json"
DEFAULT_RESUME_SIGNAL_PATH = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_sidecar.jsonl"
DEFAULT_RESUME_SIGNAL_MANIFEST_PATH = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_manifest.json"
DEFAULT_RESUME_SIGNAL_VALIDATION_PATH = RESUME_SIGNAL_DIR / "phase2_16i_resume_signal_validation.json"

DEFAULT_INPUT_VALIDATION_PATH = RESUME_SIGNAL_DIR / "phase2_16j_input_validation.json"
DEFAULT_POLICY_PATH = RESUME_SIGNAL_DIR / "phase2_16j_resume_signal_usage_policy.json"
DEFAULT_QUALITY_AUDIT_JSON_PATH = RESUME_SIGNAL_DIR / "phase2_16j_resume_signal_quality_audit.json"
DEFAULT_QUALITY_AUDIT_CSV_PATH = RESUME_SIGNAL_DIR / "phase2_16j_resume_signal_quality_audit.csv"
DEFAULT_PROFILE_PATH = RESUME_SIGNAL_DIR / "phase2_16j_job_tailoring_profile_sidecar.jsonl"
DEFAULT_PROFILE_MANIFEST_PATH = RESUME_SIGNAL_DIR / "phase2_16j_job_tailoring_profile_manifest.json"
DEFAULT_PROFILE_VALIDATION_PATH = RESUME_SIGNAL_DIR / "phase2_16j_job_tailoring_profile_validation.json"
DEFAULT_BOUNDARY_CHECK_PATH = RESUME_SIGNAL_DIR / "phase2_16j_hard_skill_boundary_check.json"
DEFAULT_REPORT_PATH = RESUME_SIGNAL_DIR / "phase2_16j_resume_tailoring_contract_report.md"

WATCHLIST_SIGNALS = ["word", "outlook", "powerpoint", "english", "bilingual", "ase", "bls", "cpr", "cpa"]
REVIEW_SIGNAL_TYPE = "credential_language_tool_review"

SIGNAL_LIST_FIELDS = {
    "soft_skill": "soft_skill_signals",
    "professional_competency": "professional_competency_signals",
    REVIEW_SIGNAL_TYPE: "credential_language_tool_review_signals",
}

TARGET_FIELDS = {
    "hard_skill": "hard_skill_targets",
    "soft_skill": "soft_skill_targets",
    "professional_competency": "professional_competency_targets",
    REVIEW_SIGNAL_TYPE: "review_only_targets",
}

PROFILE_REQUIRED_FIELDS = [
    "job_id",
    "hard_skill_targets",
    "soft_skill_targets",
    "professional_competency_targets",
    "review_only_targets",
    "evidence_spans",
    "use_for_resume_tailoring",
    "use_for_ranking",
]

AUDIT_CSV_COLUMNS = [
    "signal_type",
    "signal",
    "rows",
    "evidence_spans",
    "source_field_counts_json",
    "top_evidence_json",
    "policy_action",
    "use_for_bullet_rewrite",
    "use_for_skill_summary_wording",
    "human_review_only",
    "suppress",
    "resume_evidence_required",
    "false_positive_risk",
    "policy_ambiguity",
    "policy_note",
    "sample_contexts_json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def normalize_surface(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def dedupe_preserve_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def counter_records(counter: Counter[str], *, limit: int = 30) -> list[dict[str, Any]]:
    return [{"value": key, "count": count} for key, count in counter.most_common(limit)]


def counter_dict(counter: Counter[str], *, limit: int = 30) -> dict[str, int]:
    return dict(counter.most_common(limit))


def policy_for_signal(signal_type: str, signal: str, *, evidence_text: str = "") -> dict[str, Any]:
    signal = str(signal or "").strip().lower()
    surface = normalize_surface(evidence_text)

    if signal_type == "hard_skill":
        return {
            "policy_action": "bullet_rewrite_and_skill_summary",
            "allowed_uses": ["bullet_rewrite", "skill_summary_wording"],
            "use_for_bullet_rewrite": True,
            "use_for_skill_summary_wording": True,
            "human_review_only": False,
            "suppress": False,
            "resume_evidence_required": True,
            "false_positive_risk": "low",
            "policy_note": "Use as a resume target only when the candidate resume has matching evidence; never use for ranking in 16J.",
        }

    if signal_type == "soft_skill":
        if signal == "organization":
            if surface == "organization":
                return {
                    "policy_action": "suppress",
                    "allowed_uses": [],
                    "use_for_bullet_rewrite": False,
                    "use_for_skill_summary_wording": False,
                    "human_review_only": False,
                    "suppress": True,
                    "resume_evidence_required": True,
                    "false_positive_risk": "high",
                    "policy_note": "Bare organization often refers to the employer or institution, not an applicant skill; suppress unless the surface is organizational skills.",
                }
            return {
                "policy_action": "surface_guarded_bullet_rewrite",
                "allowed_uses": ["bullet_rewrite"],
                "use_for_bullet_rewrite": True,
                "use_for_skill_summary_wording": False,
                "human_review_only": False,
                "suppress": False,
                "resume_evidence_required": True,
                "false_positive_risk": "medium",
                "policy_note": "Use only for organizational-skills wording with candidate resume evidence; do not list as a standalone skill.",
            }
        return {
            "policy_action": "bullet_rewrite",
            "allowed_uses": ["bullet_rewrite"],
            "use_for_bullet_rewrite": True,
            "use_for_skill_summary_wording": False,
            "human_review_only": False,
            "suppress": False,
            "resume_evidence_required": True,
            "false_positive_risk": "medium",
            "policy_note": "Use to shape bullet wording only when the resume already supports the behavior; do not add as a standalone skill and do not rank.",
        }

    if signal_type == "professional_competency":
        high_frequency_terms = {"training", "scheduling", "compliance", "documentation", "reporting"}
        return {
            "policy_action": "bullet_rewrite_and_skill_summary",
            "allowed_uses": ["bullet_rewrite", "skill_summary_wording"],
            "use_for_bullet_rewrite": True,
            "use_for_skill_summary_wording": True,
            "human_review_only": False,
            "suppress": False,
            "resume_evidence_required": True,
            "false_positive_risk": "medium" if signal in high_frequency_terms else "low",
            "policy_note": "Use as a work-function target only with resume evidence; high-frequency terms need concrete accomplishment context.",
        }

    if signal_type == REVIEW_SIGNAL_TYPE:
        if signal in {"word", "outlook", "powerpoint"}:
            note = (
                "Bare office-tool token is policy-ambiguous and may be a common word or product mention; "
                "keep human-review-only unless resume evidence explicitly names the Microsoft tool."
            )
            risk = "high"
        elif signal in {"english", "bilingual", "spanish"}:
            note = (
                "Language requirements describe the job, not the candidate; use only after resume evidence confirms language ability."
            )
            risk = "medium"
        elif signal in {"ase", "bls", "cpr", "cpa", "acls"}:
            note = (
                "Credential/license acronyms require candidate-side proof and should not be invented from job requirements."
            )
            risk = "high"
        else:
            note = "Review-only tool/language/credential signal; require candidate evidence before any resume wording."
            risk = "medium"
        return {
            "policy_action": "human_review_only",
            "allowed_uses": ["human_review"],
            "use_for_bullet_rewrite": False,
            "use_for_skill_summary_wording": False,
            "human_review_only": True,
            "suppress": False,
            "resume_evidence_required": True,
            "false_positive_risk": risk,
            "policy_note": note,
        }

    return {
        "policy_action": "human_review_only",
        "allowed_uses": ["human_review"],
        "use_for_bullet_rewrite": False,
        "use_for_skill_summary_wording": False,
        "human_review_only": True,
        "suppress": False,
        "resume_evidence_required": True,
        "false_positive_risk": "medium",
        "policy_note": "Unknown signal type; keep review-only.",
    }


def build_policy_payload(*, generated_at: str) -> dict[str, Any]:
    records = [
        {"signal_type": "hard_skill", "signal": "*", **policy_for_signal("hard_skill", "*")},
        {"signal_type": "soft_skill", "signal": "*", **policy_for_signal("soft_skill", "*")},
        {
            "signal_type": "soft_skill",
            "signal": "organization",
            "surface_condition": "bare organization",
            **policy_for_signal("soft_skill", "organization", evidence_text="organization"),
        },
        {
            "signal_type": "soft_skill",
            "signal": "organization",
            "surface_condition": "organizational skills",
            **policy_for_signal("soft_skill", "organization", evidence_text="organizational skills"),
        },
        {
            "signal_type": "professional_competency",
            "signal": "*",
            **policy_for_signal("professional_competency", "*"),
        },
    ]
    for signal in [
        "word",
        "outlook",
        "powerpoint",
        "english",
        "bilingual",
        "spanish",
        "ase",
        "bls",
        "cpr",
        "cpa",
        "acls",
        "microsoft word",
        "microsoft outlook",
        "microsoft powerpoint",
        "spreadsheets",
    ]:
        records.append({"signal_type": REVIEW_SIGNAL_TYPE, "signal": signal, **policy_for_signal(REVIEW_SIGNAL_TYPE, signal)})

    return {
        "generated_at": generated_at,
        "phase": "phase2_16j_resume_signal_qa_tailoring_contract",
        "scope": "offline_resume_tailoring_only",
        "hard_boundaries": {
            "phase1_ingestion_modified": False,
            "phase1_snapshot_modified": False,
            "jobs_clean_generation_modified": False,
            "jobs_offline_snapshot_generation_modified": False,
            "ranking_behavior_changed": False,
            "use_for_ranking": False,
            "soft_skills_used_in_ranking": False,
            "cloud_run_online_inference_enabled": False,
            "bert_or_escoxlmr_rerun": False,
            "training_or_finetuning": False,
            "extracted_skills_used_as_gold_labels": False,
        },
        "policy_records": records,
        "watchlist_signals": WATCHLIST_SIGNALS,
    }


def validate_16i_inputs(
    *,
    snapshot_path: Path,
    hard_sidecar_path: Path,
    hard_manifest_path: Path,
    hard_validation_path: Path,
    resume_signal_path: Path,
    resume_signal_manifest_path: Path,
    resume_signal_validation_path: Path,
    output_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    hard_manifest = load_json(hard_manifest_path)
    hard_validation = load_json(hard_validation_path)
    resume_manifest = load_json(resume_signal_manifest_path)
    resume_validation = load_json(resume_signal_validation_path)

    snapshot_ids: list[str] = []
    snapshot_duplicates: Counter[str] = Counter()
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            job_id = str(row.get("job_id") or "")
            if job_id in snapshot_ids:
                snapshot_duplicates[job_id] += 1
            snapshot_ids.append(job_id)

    def scan_jsonl(path: Path, *, kind: str) -> dict[str, Any]:
        order: list[str] = []
        duplicates: Counter[str] = Counter()
        seen: set[str] = set()
        parse_errors: list[dict[str, Any]] = []
        rows = 0
        boundary_bad_rows = 0
        hard_exact_counts: Counter[str] = Counter()
        hard_tool_variant_counts: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as handle:
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
                if row.get("use_for_ranking") is not False and kind == "resume_signal":
                    boundary_bad_rows += 1
                if row.get("ranking_behavior_changed") is not False:
                    boundary_bad_rows += 1
                if row.get("phase1_ingestion_modified") is not False:
                    boundary_bad_rows += 1
                if row.get("phase1_snapshot_modified") is not False:
                    boundary_bad_rows += 1
                if kind == "hard_skill":
                    for target in row.get("normalized_hard_skills") or []:
                        target_text = str(target)
                        if target_text in WATCHLIST_SIGNALS:
                            hard_exact_counts[target_text] += 1
                        if target_text in {"microsoft word", "microsoft outlook", "microsoft powerpoint"}:
                            hard_tool_variant_counts[target_text] += 1
        return {
            "rows": rows,
            "parse_error_count": len(parse_errors),
            "parse_errors": parse_errors[:20],
            "duplicate_job_id_count": sum(duplicates.values()),
            "duplicate_job_ids_sample": counter_records(duplicates, limit=20),
            "job_id_order_matches_snapshot": order == snapshot_ids,
            "boundary_bad_rows": boundary_bad_rows,
            "hard_watchlist_exact_counts": dict(hard_exact_counts),
            "hard_microsoft_tool_variant_counts": dict(hard_tool_variant_counts),
        }

    hard_scan = scan_jsonl(hard_sidecar_path, kind="hard_skill")
    resume_scan = scan_jsonl(resume_signal_path, kind="resume_signal")

    boundary_checks = {
        "hard_validation_passed": hard_validation.get("all_validation_checks_passed") is True,
        "resume_signal_validation_passed": resume_validation.get("all_validation_checks_passed") is True,
        "hard_phase1_ingestion_modified_false": hard_manifest.get("phase1_ingestion_modified") is False,
        "hard_phase1_snapshot_modified_false": hard_manifest.get("phase1_snapshot_modified") is False,
        "hard_ranking_behavior_changed_false": hard_manifest.get("ranking_behavior_changed") is False,
        "hard_cloud_run_online_inference_enabled_false": hard_manifest.get("cloud_run_online_inference_enabled") is False,
        "hard_escoxlmr_rerun_false": hard_manifest.get("escoxlmr_rerun") is False,
        "hard_training_or_finetuning_false": hard_manifest.get("training_or_finetuning") is False,
        "hard_extracted_skills_used_as_gold_labels_false": hard_manifest.get("extracted_skills_used_as_gold_labels")
        is False,
        "resume_use_for_tailoring_true": resume_manifest.get("use_for_resume_tailoring") is True,
        "resume_use_for_ranking_false": resume_manifest.get("use_for_ranking") is False,
        "resume_ranking_behavior_changed_false": resume_manifest.get("ranking_behavior_changed") is False,
        "resume_phase1_ingestion_modified_false": resume_manifest.get("phase1_ingestion_modified") is False,
        "resume_phase1_snapshot_modified_false": resume_manifest.get("phase1_snapshot_modified") is False,
        "resume_cloud_run_online_inference_enabled_false": resume_manifest.get("cloud_run_online_inference_enabled")
        is False,
        "resume_bert_or_escoxlmr_used_false": resume_manifest.get("bert_or_escoxlmr_used") is False,
    }
    manifest_checks = {
        "snapshot_rows_50000": len(snapshot_ids) == 50000,
        "hard_rows_match_manifest": hard_scan["rows"] == hard_manifest.get("rows_written"),
        "resume_rows_match_manifest": resume_scan["rows"] == resume_manifest.get("rows_written"),
        "hard_rows_match_snapshot": hard_scan["rows"] == len(snapshot_ids),
        "resume_rows_match_snapshot": resume_scan["rows"] == len(snapshot_ids),
        "hard_no_duplicate_job_id": hard_scan["duplicate_job_id_count"] == 0,
        "resume_no_duplicate_job_id": resume_scan["duplicate_job_id_count"] == 0,
        "snapshot_no_duplicate_job_id": sum(snapshot_duplicates.values()) == 0,
        "hard_order_matches_snapshot": hard_scan["job_id_order_matches_snapshot"] is True,
        "resume_order_matches_snapshot": resume_scan["job_id_order_matches_snapshot"] is True,
        "hard_no_parse_errors": hard_scan["parse_error_count"] == 0,
        "resume_no_parse_errors": resume_scan["parse_error_count"] == 0,
        "watchlist_bare_terms_not_accepted_as_hard_skills": not hard_scan["hard_watchlist_exact_counts"],
    }
    validation = {
        "generated_at": generated_at,
        "phase": "phase2_16j_input_validation",
        "snapshot_path": str(snapshot_path),
        "hard_sidecar_path": str(hard_sidecar_path),
        "resume_signal_sidecar_path": str(resume_signal_path),
        "input_file_fingerprints": {
            "snapshot": file_fingerprint(snapshot_path),
            "hard_sidecar": file_fingerprint(hard_sidecar_path),
            "resume_signal_sidecar": file_fingerprint(resume_signal_path),
        },
        "snapshot_rows": len(snapshot_ids),
        "snapshot_duplicate_job_id_count": sum(snapshot_duplicates.values()),
        "hard_scan": hard_scan,
        "resume_signal_scan": resume_scan,
        "hard_manifest_summary": {
            "rows_written": hard_manifest.get("rows_written"),
            "normalized_hard_skill_count": hard_manifest.get("normalized_hard_skill_count"),
            "rows_with_normalized_hard_skills": hard_manifest.get("rows_with_normalized_hard_skills"),
            "go_guard": hard_manifest.get("go_guard"),
            "normalization_only_refresh": hard_manifest.get("normalization_only_refresh"),
        },
        "resume_signal_manifest_summary": {
            "rows_written": resume_manifest.get("rows_written"),
            "rows_with_soft_skill_signals": resume_manifest.get("rows_with_soft_skill_signals"),
            "rows_with_professional_competency_signals": resume_manifest.get(
                "rows_with_professional_competency_signals"
            ),
            "rows_with_credential_language_tool_review_signals": resume_manifest.get(
                "rows_with_credential_language_tool_review_signals"
            ),
            "evidence_span_count": resume_manifest.get("evidence_span_count"),
        },
        "manifest_checks": manifest_checks,
        "boundary_checks": boundary_checks,
    }
    validation["all_validation_checks_passed"] = all(manifest_checks.values()) and all(boundary_checks.values())
    write_json(output_path, validation)
    return validation


def snippet_for_span(source_text: str, start: Any, end: Any, *, window: int = 90) -> str:
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return ""
    if start_i < 0 or end_i < start_i:
        return ""
    left = max(0, start_i - window)
    right = min(len(source_text), end_i + window)
    snippet = source_text[left:right]
    return re.sub(r"\s+", " ", snippet).strip()


def audit_resume_signal_quality(
    *,
    snapshot_path: Path,
    resume_signal_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "evidence_spans": 0,
            "source_field_counts": Counter(),
            "evidence_counter": Counter(),
            "samples": [],
        }
    )
    rows_scanned = 0
    parse_errors: list[dict[str, Any]] = []

    with snapshot_path.open("r", encoding="utf-8", newline="") as snapshot_handle, resume_signal_path.open(
        "r", encoding="utf-8"
    ) as resume_handle:
        reader = csv.DictReader(snapshot_handle)
        for row_index, (snapshot_row, line) in enumerate(zip(reader, resume_handle)):
            rows_scanned += 1
            try:
                resume_row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"row_index": row_index, "error": str(exc)})
                continue

            for signal_type, list_field in SIGNAL_LIST_FIELDS.items():
                for signal in resume_row.get(list_field) or []:
                    stats[(signal_type, str(signal))]["rows"] += 1

            for span in resume_row.get("evidence_spans") or []:
                signal_type = str(span.get("signal_type") or "")
                signal = str(span.get("normalized_text") or "")
                if not signal_type or not signal:
                    continue
                key = (signal_type, signal)
                field = str(span.get("source_field") or "")
                evidence = str(span.get("evidence") or span.get("text") or "")
                stats[key]["evidence_spans"] += 1
                stats[key]["source_field_counts"][field] += 1
                stats[key]["evidence_counter"][evidence] += 1
                if len(stats[key]["samples"]) < 5:
                    source_text = str(snapshot_row.get(field) or "")
                    stats[key]["samples"].append(
                        {
                            "job_id": resume_row.get("job_id"),
                            "source_field": field,
                            "evidence": evidence,
                            "snippet": snippet_for_span(source_text, span.get("start"), span.get("end")),
                        }
                    )

    records: list[dict[str, Any]] = []
    for (signal_type, signal), item in sorted(
        stats.items(), key=lambda pair: (pair[0][0], -pair[1]["rows"], pair[0][1])
    ):
        policy = policy_for_signal(signal_type, signal)
        record = {
            "signal_type": signal_type,
            "signal": signal,
            "rows": item["rows"],
            "evidence_spans": item["evidence_spans"],
            "source_field_counts": dict(item["source_field_counts"].most_common()),
            "top_evidence": counter_records(item["evidence_counter"], limit=12),
            "policy": policy,
            "policy_ambiguity": bool(
                policy["human_review_only"] or policy["suppress"] or policy["false_positive_risk"] == "high"
            ),
            "sample_contexts": item["samples"],
        }
        records.append(record)

    csv_rows = []
    for record in records:
        policy = record["policy"]
        csv_rows.append(
            {
                "signal_type": record["signal_type"],
                "signal": record["signal"],
                "rows": record["rows"],
                "evidence_spans": record["evidence_spans"],
                "source_field_counts_json": compact_json(record["source_field_counts"]),
                "top_evidence_json": compact_json(record["top_evidence"]),
                "policy_action": policy["policy_action"],
                "use_for_bullet_rewrite": policy["use_for_bullet_rewrite"],
                "use_for_skill_summary_wording": policy["use_for_skill_summary_wording"],
                "human_review_only": policy["human_review_only"],
                "suppress": policy["suppress"],
                "resume_evidence_required": policy["resume_evidence_required"],
                "false_positive_risk": policy["false_positive_risk"],
                "policy_ambiguity": record["policy_ambiguity"],
                "policy_note": policy["policy_note"],
                "sample_contexts_json": compact_json(record["sample_contexts"]),
            }
        )

    watchlist_summary = {
        signal: next(
            (
                {
                    "rows": record["rows"],
                    "top_evidence": record["top_evidence"][:8],
                    "policy": record["policy"],
                    "policy_ambiguity": record["policy_ambiguity"],
                }
                for record in records
                if record["signal_type"] == REVIEW_SIGNAL_TYPE and record["signal"] == signal
            ),
            {
                "rows": 0,
                "top_evidence": [],
                "policy": policy_for_signal(REVIEW_SIGNAL_TYPE, signal),
                "policy_ambiguity": True,
            },
        )
        for signal in WATCHLIST_SIGNALS
    }
    audit = {
        "generated_at": generated_at,
        "phase": "phase2_16j_resume_signal_quality_audit",
        "rows_scanned": rows_scanned,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "signals_audited": len(records),
        "watchlist_summary": watchlist_summary,
        "high_risk_or_ambiguous_signals": [
            {
                "signal_type": record["signal_type"],
                "signal": record["signal"],
                "rows": record["rows"],
                "policy_action": record["policy"]["policy_action"],
                "false_positive_risk": record["policy"]["false_positive_risk"],
                "policy_note": record["policy"]["policy_note"],
            }
            for record in records
            if record["policy_ambiguity"]
        ],
        "records": records,
    }
    write_json(output_json_path, audit)
    write_csv(output_csv_path, csv_rows, AUDIT_CSV_COLUMNS)
    return audit


def enrich_span(
    span: dict[str, Any],
    *,
    target_field: str,
    source_sidecar: str,
    policy: dict[str, Any],
    signal_type: str | None = None,
) -> dict[str, Any]:
    return {
        "signal_type": signal_type or span.get("signal_type") or "hard_skill",
        "target_field": target_field,
        "normalized_text": span.get("normalized_text"),
        "text": span.get("text"),
        "source_field": span.get("source_field"),
        "start": span.get("start"),
        "end": span.get("end"),
        "evidence": span.get("evidence") or span.get("text"),
        "source_sidecar": source_sidecar,
        "policy_action": policy["policy_action"],
        "allowed_uses": policy["allowed_uses"],
        "resume_evidence_required": policy["resume_evidence_required"],
        "human_review_required": policy["human_review_only"],
        "use_for_ranking": False,
    }


def first_evidence_by_signal(resume_row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for span in resume_row.get("evidence_spans") or []:
        key = (str(span.get("signal_type") or ""), str(span.get("normalized_text") or ""))
        if key not in result:
            result[key] = span
    return result


def first_accepted_hard_skill_spans(hard_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for span in hard_row.get("hard_skill_spans") or []:
        if span.get("accepted") is not True:
            continue
        normalized = str(span.get("normalized_text") or "")
        if normalized and normalized not in result:
            result[normalized] = span
    return result


def build_job_tailoring_profile_sidecar(
    *,
    hard_sidecar_path: Path,
    resume_signal_path: Path,
    output_path: Path,
    manifest_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_with_targets = Counter()
    target_counts = {
        "hard_skill_targets": Counter(),
        "soft_skill_targets": Counter(),
        "professional_competency_targets": Counter(),
        "review_only_targets": Counter(),
    }
    suppressed_counts: Counter[str] = Counter()
    suppressed_by_signal_type: Counter[str] = Counter()
    evidence_span_count = 0
    review_policy_counts: Counter[str] = Counter()
    watchlist_profile_counts = {signal: Counter() for signal in WATCHLIST_SIGNALS}

    with hard_sidecar_path.open("r", encoding="utf-8") as hard_handle, resume_signal_path.open(
        "r", encoding="utf-8"
    ) as resume_handle, output_path.open("w", encoding="utf-8", newline="\n") as out:
        for row_index, (hard_line, resume_line) in enumerate(zip(hard_handle, resume_handle)):
            hard_row = json.loads(hard_line)
            resume_row = json.loads(resume_line)
            hard_job_id = str(hard_row.get("job_id") or "")
            resume_job_id = str(resume_row.get("job_id") or "")
            job_id = hard_job_id or resume_job_id
            hard_targets = dedupe_preserve_order(hard_row.get("normalized_hard_skills") or [])
            hard_spans = first_accepted_hard_skill_spans(hard_row)
            resume_spans = first_evidence_by_signal(resume_row)
            evidence_spans: list[dict[str, Any]] = []
            suppressed_targets: list[dict[str, Any]] = []

            for target in hard_targets:
                policy = policy_for_signal("hard_skill", target)
                span = hard_spans.get(target)
                if span:
                    evidence_spans.append(
                        enrich_span(
                            span,
                            target_field="hard_skill_targets",
                            source_sidecar="phase2_16i_hard_skill",
                            policy=policy,
                            signal_type="hard_skill",
                        )
                    )

            soft_targets: list[str] = []
            professional_targets: list[str] = []
            review_targets: list[str] = []
            for signal_type, list_field in SIGNAL_LIST_FIELDS.items():
                target_field = TARGET_FIELDS[signal_type]
                output_targets = {
                    "soft_skill_targets": soft_targets,
                    "professional_competency_targets": professional_targets,
                    "review_only_targets": review_targets,
                }[target_field]
                for signal in dedupe_preserve_order(resume_row.get(list_field) or []):
                    span = resume_spans.get((signal_type, signal), {})
                    evidence_text = str(span.get("evidence") or span.get("text") or "")
                    policy = policy_for_signal(signal_type, signal, evidence_text=evidence_text)
                    if policy["suppress"]:
                        suppressed_counts[signal] += 1
                        suppressed_by_signal_type[signal_type] += 1
                        suppressed_targets.append(
                            {
                                "signal_type": signal_type,
                                "target_field": target_field,
                                "normalized_text": signal,
                                "evidence": evidence_text,
                                "source_field": span.get("source_field"),
                                "policy_action": policy["policy_action"],
                                "policy_note": policy["policy_note"],
                            }
                        )
                        continue
                    output_targets.append(signal)
                    evidence_spans.append(
                        enrich_span(
                            span,
                            target_field=target_field,
                            source_sidecar="phase2_16i_resume_signal",
                            policy=policy,
                            signal_type=signal_type,
                        )
                    )
                    if policy["human_review_only"]:
                        review_policy_counts[signal] += 1

            payload = {
                "job_id": job_id,
                "hard_skill_targets": hard_targets,
                "soft_skill_targets": soft_targets,
                "professional_competency_targets": professional_targets,
                "review_only_targets": review_targets,
                "evidence_spans": evidence_spans,
                "suppressed_targets": suppressed_targets,
                "use_for_resume_tailoring": True,
                "use_for_ranking": False,
                "soft_skills_ranking_weight": 0,
                "resume_signal_ranking_weight": 0,
                "ranking_behavior_changed": False,
                "phase1_ingestion_modified": False,
                "phase1_snapshot_modified": False,
                "cloud_run_online_inference_enabled": False,
                "source_sidecars": {
                    "hard_skill": "phase2_16i_hard_skill_sidecar_reviewed_normalization",
                    "resume_signal": "phase2_16i_resume_signal_sidecar",
                },
                "generated_at": generated_at,
            }
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

            rows_written += 1
            evidence_span_count += len(evidence_spans)
            for field in target_counts:
                targets = payload[field]
                rows_with_targets[field] += int(bool(targets))
                for target in targets:
                    target_counts[field][target] += 1
                    if target in watchlist_profile_counts:
                        watchlist_profile_counts[target][field] += 1
            rows_with_targets["suppressed_targets"] += int(bool(suppressed_targets))

    manifest = {
        "generated_at": generated_at,
        "phase": "phase2_16j_resume_signal_qa_tailoring_contract",
        "sidecar_type": "job_tailoring_profile_sidecar",
        "input_paths": {
            "hard_sidecar": str(hard_sidecar_path),
            "resume_signal_sidecar": str(resume_signal_path),
        },
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "rows_written": rows_written,
        "profile_fields": PROFILE_REQUIRED_FIELDS,
        "target_contract": {
            "hard_skill_targets": "16I accepted normalized hard skills; resume tailoring only, no ranking.",
            "soft_skill_targets": "Soft-skill wording prompts after policy suppression; bullet rewrite only.",
            "professional_competency_targets": "Concrete work-function targets for bullet and summary wording with resume evidence.",
            "review_only_targets": "Credential/language/tool signals that require human review and candidate evidence.",
        },
        "rows_with_hard_skill_targets": rows_with_targets["hard_skill_targets"],
        "rows_with_soft_skill_targets": rows_with_targets["soft_skill_targets"],
        "rows_with_professional_competency_targets": rows_with_targets["professional_competency_targets"],
        "rows_with_review_only_targets": rows_with_targets["review_only_targets"],
        "rows_with_suppressed_targets": rows_with_targets["suppressed_targets"],
        "target_counts": {field: sum(counter.values()) for field, counter in target_counts.items()},
        "top_hard_skill_targets": counter_dict(target_counts["hard_skill_targets"]),
        "top_soft_skill_targets": counter_dict(target_counts["soft_skill_targets"]),
        "top_professional_competency_targets": counter_dict(target_counts["professional_competency_targets"]),
        "top_review_only_targets": counter_dict(target_counts["review_only_targets"]),
        "suppressed_target_counts": dict(suppressed_counts.most_common()),
        "suppressed_by_signal_type": dict(suppressed_by_signal_type.most_common()),
        "review_policy_counts": dict(review_policy_counts.most_common()),
        "watchlist_profile_counts": {signal: dict(counter) for signal, counter in watchlist_profile_counts.items()},
        "evidence_span_count": evidence_span_count,
        "use_for_resume_tailoring": True,
        "use_for_ranking": False,
        "soft_skills_ranking_weight": 0,
        "resume_signal_ranking_weight": 0,
        "ranking_behavior_changed": False,
        "phase1_ingestion_modified": False,
        "phase1_snapshot_modified": False,
        "cloud_run_online_inference_enabled": False,
        "bert_or_escoxlmr_rerun": False,
        "training_or_finetuning": False,
        "extracted_skills_used_as_gold_labels": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def validate_job_tailoring_profile_sidecar(
    *,
    snapshot_path: Path,
    profile_path: Path,
    manifest: dict[str, Any],
    output_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        snapshot_ids = [row["job_id"] for row in csv.DictReader(handle)]

    rows = 0
    order: list[str] = []
    seen: set[str] = set()
    duplicates: Counter[str] = Counter()
    parse_errors: list[dict[str, Any]] = []
    missing_required_rows: list[dict[str, Any]] = []
    bad_flag_rows = 0
    duplicate_target_rows: list[dict[str, Any]] = []
    invalid_evidence_rows: list[dict[str, Any]] = []
    target_counts = {
        "hard_skill_targets": Counter(),
        "soft_skill_targets": Counter(),
        "professional_competency_targets": Counter(),
        "review_only_targets": Counter(),
    }
    rows_with_targets = Counter()
    suppressed_counts: Counter[str] = Counter()
    evidence_span_count = 0
    watchlist_profile_counts = {signal: Counter() for signal in WATCHLIST_SIGNALS}

    with profile_path.open("r", encoding="utf-8") as handle:
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
            missing = [field for field in PROFILE_REQUIRED_FIELDS if field not in row]
            if missing:
                missing_required_rows.append({"row_index": row_index, "job_id": job_id, "missing_fields": missing})
            if row.get("use_for_resume_tailoring") is not True or row.get("use_for_ranking") is not False:
                bad_flag_rows += 1
            if row.get("ranking_behavior_changed") is not False or row.get("resume_signal_ranking_weight") != 0:
                bad_flag_rows += 1
            if row.get("phase1_ingestion_modified") is not False or row.get("phase1_snapshot_modified") is not False:
                bad_flag_rows += 1
            for field in target_counts:
                values = row.get(field) or []
                if len(values) != len(set(values)):
                    duplicate_target_rows.append({"row_index": row_index, "job_id": job_id, "field": field})
                rows_with_targets[field] += int(bool(values))
                for value in values:
                    target = str(value)
                    target_counts[field][target] += 1
                    if target in watchlist_profile_counts:
                        watchlist_profile_counts[target][field] += 1
            for suppressed in row.get("suppressed_targets") or []:
                suppressed_counts[str(suppressed.get("normalized_text") or "")] += 1
            for span in row.get("evidence_spans") or []:
                evidence_span_count += 1
                missing_span_keys = [
                    key
                    for key in [
                        "signal_type",
                        "target_field",
                        "normalized_text",
                        "source_field",
                        "start",
                        "end",
                        "evidence",
                        "policy_action",
                    ]
                    if key not in span
                ]
                if missing_span_keys:
                    invalid_evidence_rows.append(
                        {"row_index": row_index, "job_id": job_id, "missing_span_keys": missing_span_keys}
                    )
                if span.get("use_for_ranking") is not False:
                    invalid_evidence_rows.append({"row_index": row_index, "job_id": job_id, "bad_span_ranking_flag": True})

    recomputed = {
        "rows_written": rows,
        "rows_with_hard_skill_targets": rows_with_targets["hard_skill_targets"],
        "rows_with_soft_skill_targets": rows_with_targets["soft_skill_targets"],
        "rows_with_professional_competency_targets": rows_with_targets["professional_competency_targets"],
        "rows_with_review_only_targets": rows_with_targets["review_only_targets"],
        "rows_with_suppressed_targets": sum(1 for _ in []),
        "target_counts": {field: sum(counter.values()) for field, counter in target_counts.items()},
        "top_hard_skill_targets": counter_dict(target_counts["hard_skill_targets"]),
        "top_soft_skill_targets": counter_dict(target_counts["soft_skill_targets"]),
        "top_professional_competency_targets": counter_dict(target_counts["professional_competency_targets"]),
        "top_review_only_targets": counter_dict(target_counts["review_only_targets"]),
        "suppressed_target_counts": dict(suppressed_counts.most_common()),
        "watchlist_profile_counts": {signal: dict(counter) for signal, counter in watchlist_profile_counts.items()},
        "evidence_span_count": evidence_span_count,
    }
    rows_with_suppressed = 0
    with profile_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_with_suppressed += int(bool(row.get("suppressed_targets")))
    recomputed["rows_with_suppressed_targets"] = rows_with_suppressed

    manifest_checks = {
        "rows_written": rows == manifest.get("rows_written"),
        "rows_with_hard_skill_targets": recomputed["rows_with_hard_skill_targets"]
        == manifest.get("rows_with_hard_skill_targets"),
        "rows_with_soft_skill_targets": recomputed["rows_with_soft_skill_targets"]
        == manifest.get("rows_with_soft_skill_targets"),
        "rows_with_professional_competency_targets": recomputed["rows_with_professional_competency_targets"]
        == manifest.get("rows_with_professional_competency_targets"),
        "rows_with_review_only_targets": recomputed["rows_with_review_only_targets"]
        == manifest.get("rows_with_review_only_targets"),
        "rows_with_suppressed_targets": recomputed["rows_with_suppressed_targets"]
        == manifest.get("rows_with_suppressed_targets"),
        "target_counts": recomputed["target_counts"] == manifest.get("target_counts"),
        "top_hard_skill_targets": recomputed["top_hard_skill_targets"] == manifest.get("top_hard_skill_targets"),
        "top_soft_skill_targets": recomputed["top_soft_skill_targets"] == manifest.get("top_soft_skill_targets"),
        "top_professional_competency_targets": recomputed["top_professional_competency_targets"]
        == manifest.get("top_professional_competency_targets"),
        "top_review_only_targets": recomputed["top_review_only_targets"] == manifest.get("top_review_only_targets"),
        "suppressed_target_counts": recomputed["suppressed_target_counts"] == manifest.get("suppressed_target_counts"),
        "watchlist_profile_counts": recomputed["watchlist_profile_counts"] == manifest.get("watchlist_profile_counts"),
        "evidence_span_count": evidence_span_count == manifest.get("evidence_span_count"),
    }
    boundary_checks = {
        "use_for_resume_tailoring_true": manifest.get("use_for_resume_tailoring") is True,
        "use_for_ranking_false": manifest.get("use_for_ranking") is False,
        "soft_skills_ranking_weight_zero": manifest.get("soft_skills_ranking_weight") == 0,
        "resume_signal_ranking_weight_zero": manifest.get("resume_signal_ranking_weight") == 0,
        "ranking_behavior_changed_false": manifest.get("ranking_behavior_changed") is False,
        "phase1_ingestion_modified_false": manifest.get("phase1_ingestion_modified") is False,
        "phase1_snapshot_modified_false": manifest.get("phase1_snapshot_modified") is False,
        "cloud_run_online_inference_enabled_false": manifest.get("cloud_run_online_inference_enabled") is False,
        "bert_or_escoxlmr_rerun_false": manifest.get("bert_or_escoxlmr_rerun") is False,
        "training_or_finetuning_false": manifest.get("training_or_finetuning") is False,
        "extracted_skills_used_as_gold_labels_false": manifest.get("extracted_skills_used_as_gold_labels") is False,
        "watchlist_bare_terms_not_promoted_to_hard_skill_targets": all(
            recomputed["watchlist_profile_counts"][signal].get("hard_skill_targets", 0) == 0
            for signal in WATCHLIST_SIGNALS
        ),
    }
    validation = {
        "generated_at": generated_at,
        "phase": "phase2_16j_job_tailoring_profile_validation",
        "profile_path": str(profile_path),
        "manifest_path": str(manifest.get("manifest_path")),
        "rows": rows,
        "snapshot_rows": len(snapshot_ids),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "duplicate_job_id_count": sum(duplicates.values()),
        "duplicate_job_ids_sample": counter_records(duplicates, limit=20),
        "job_id_order_matches_snapshot": order == snapshot_ids,
        "missing_required_rows": missing_required_rows[:20],
        "bad_flag_rows": bad_flag_rows,
        "duplicate_target_rows": duplicate_target_rows[:20],
        "invalid_evidence_rows": invalid_evidence_rows[:20],
        "manifest_checks": manifest_checks,
        "boundary_checks": boundary_checks,
        "recomputed_counts": recomputed,
    }
    validation["all_validation_checks_passed"] = (
        rows == len(snapshot_ids)
        and len(parse_errors) == 0
        and sum(duplicates.values()) == 0
        and order == snapshot_ids
        and not missing_required_rows
        and bad_flag_rows == 0
        and not duplicate_target_rows
        and not invalid_evidence_rows
        and all(manifest_checks.values())
        and all(boundary_checks.values())
    )
    write_json(output_path, validation)
    return validation


def write_hard_skill_boundary_check(
    *,
    input_validation: dict[str, Any],
    profile_validation: dict[str, Any],
    output_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    hard_counts = input_validation["hard_scan"].get("hard_watchlist_exact_counts", {})
    profile_watchlist = profile_validation["recomputed_counts"].get("watchlist_profile_counts", {})
    boundary = {
        "generated_at": generated_at,
        "phase": "phase2_16j_hard_skill_boundary_check",
        "hard_sidecar_path": input_validation["hard_sidecar_path"],
        "profile_path": profile_validation["profile_path"],
        "bare_watchlist_terms_in_16i_hard_skills": hard_counts,
        "watchlist_profile_counts": profile_watchlist,
        "allowed_microsoft_tool_hard_skill_variants": input_validation["hard_scan"].get(
            "hard_microsoft_tool_variant_counts", {}
        ),
        "checks": {
            "bare_watchlist_terms_not_accepted_as_16i_hard_skills": not hard_counts,
            "bare_watchlist_terms_not_promoted_to_hard_skill_targets": profile_validation["boundary_checks"].get(
                "watchlist_bare_terms_not_promoted_to_hard_skill_targets"
            )
            is True,
            "profile_use_for_ranking_false": profile_validation["boundary_checks"].get("use_for_ranking_false") is True,
            "phase1_and_ranking_boundaries_preserved": (
                profile_validation["boundary_checks"].get("phase1_ingestion_modified_false") is True
                and profile_validation["boundary_checks"].get("phase1_snapshot_modified_false") is True
                and profile_validation["boundary_checks"].get("ranking_behavior_changed_false") is True
            ),
        },
    }
    boundary["all_boundary_checks_passed"] = all(boundary["checks"].values())
    write_json(output_path, boundary)
    return boundary


def write_contract_report(
    *,
    input_validation: dict[str, Any],
    quality_audit: dict[str, Any],
    policy: dict[str, Any],
    profile_manifest: dict[str, Any],
    profile_validation: dict[str, Any],
    boundary_check: dict[str, Any],
    output_path: Path,
) -> None:
    watchlist = quality_audit["watchlist_summary"]
    lines = [
        "# Phase 2.16J Resume Signal QA & Tailoring Contract",
        "",
        "Scope: offline QA, policy, and job-tailoring profile artifacts only. Phase 1 ingestion, snapshot generation, ranking, Cloud Run online inference, ESCOXLM-R/BERT extraction, training, and fine-tuning were not changed.",
        "",
        "## Input Validation",
        "",
        f"- Snapshot rows: {input_validation['snapshot_rows']:,}",
        f"- Hard-skill rows: {input_validation['hard_scan']['rows']:,}",
        f"- Resume-signal rows: {input_validation['resume_signal_scan']['rows']:,}",
        f"- 16I hard validation passed: {input_validation['boundary_checks']['hard_validation_passed']}",
        f"- 16I resume-signal validation passed: {input_validation['boundary_checks']['resume_signal_validation_passed']}",
        f"- All 16J input checks passed: {input_validation['all_validation_checks_passed']}",
        "",
        "## Watchlist Signal Review",
        "",
        "| Signal | Rows | Policy | Risk | Notes |",
        "|---|---:|---|---|---|",
    ]
    for signal in WATCHLIST_SIGNALS:
        item = watchlist[signal]
        policy_record = item["policy"]
        lines.append(
            f"| `{signal}` | {item['rows']:,} | `{policy_record['policy_action']}` | "
            f"{policy_record['false_positive_risk']} | {policy_record['policy_note']} |"
        )

    lines.extend(
        [
            "",
            "## Policy Contract",
            "",
            "- Hard-skill targets: bullet rewrite and skill-summary wording are allowed only with candidate resume evidence.",
            "- Soft-skill targets: bullet rewrite only; do not add standalone soft-skill claims.",
            "- Professional competency targets: bullet rewrite and skill-summary wording are allowed with concrete resume evidence.",
            "- Review-only targets: credential, language, and bare office-tool signals stay human-review-only.",
            "- Suppression: bare `organization` is suppressed because it often refers to the employer/institution rather than a candidate skill.",
            "",
            "## Profile Sidecar",
            "",
            f"- Rows written: {profile_manifest['rows_written']:,}",
            f"- Rows with hard-skill targets: {profile_manifest['rows_with_hard_skill_targets']:,}",
            f"- Rows with soft-skill targets: {profile_manifest['rows_with_soft_skill_targets']:,}",
            f"- Rows with professional competency targets: {profile_manifest['rows_with_professional_competency_targets']:,}",
            f"- Rows with review-only targets: {profile_manifest['rows_with_review_only_targets']:,}",
            f"- Rows with suppressed targets: {profile_manifest['rows_with_suppressed_targets']:,}",
            f"- Evidence spans: {profile_manifest['evidence_span_count']:,}",
            "- `use_for_resume_tailoring=true`",
            "- `use_for_ranking=false`",
            "",
            "## Top Targets",
            "",
            "### Hard Skills",
            "",
            "| Target | Rows |",
            "|---|---:|",
        ]
    )
    for target, count in list(profile_manifest["top_hard_skill_targets"].items())[:12]:
        lines.append(f"| `{target}` | {count:,} |")
    lines.extend(["", "### Soft Skills", "", "| Target | Rows |", "|---|---:|"])
    for target, count in list(profile_manifest["top_soft_skill_targets"].items())[:12]:
        lines.append(f"| `{target}` | {count:,} |")
    lines.extend(["", "### Professional Competencies", "", "| Target | Rows |", "|---|---:|"])
    for target, count in list(profile_manifest["top_professional_competency_targets"].items())[:12]:
        lines.append(f"| `{target}` | {count:,} |")
    lines.extend(["", "### Review Only", "", "| Target | Rows |", "|---|---:|"])
    for target, count in list(profile_manifest["top_review_only_targets"].items())[:12]:
        lines.append(f"| `{target}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Profile validation passed: {profile_validation['all_validation_checks_passed']}",
            f"- Hard-skill boundary check passed: {boundary_check['all_boundary_checks_passed']}",
            f"- Bare watchlist terms in 16I hard skills: {compact_json(boundary_check['bare_watchlist_terms_in_16i_hard_skills'])}",
            "",
            "## Artifacts",
            "",
            f"- `{DEFAULT_INPUT_VALIDATION_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_POLICY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_QUALITY_AUDIT_JSON_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_QUALITY_AUDIT_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_PROFILE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_PROFILE_MANIFEST_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_PROFILE_VALIDATION_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DEFAULT_BOUNDARY_CHECK_PATH.relative_to(PROJECT_ROOT)}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_phase2_16j_resume_tailoring_contract(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    hard_sidecar_path: Path = DEFAULT_HARD_SIDECAR_PATH,
    hard_manifest_path: Path = DEFAULT_HARD_MANIFEST_PATH,
    hard_validation_path: Path = DEFAULT_HARD_VALIDATION_PATH,
    resume_signal_path: Path = DEFAULT_RESUME_SIGNAL_PATH,
    resume_signal_manifest_path: Path = DEFAULT_RESUME_SIGNAL_MANIFEST_PATH,
    resume_signal_validation_path: Path = DEFAULT_RESUME_SIGNAL_VALIDATION_PATH,
    input_validation_path: Path = DEFAULT_INPUT_VALIDATION_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
    quality_audit_json_path: Path = DEFAULT_QUALITY_AUDIT_JSON_PATH,
    quality_audit_csv_path: Path = DEFAULT_QUALITY_AUDIT_CSV_PATH,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    profile_manifest_path: Path = DEFAULT_PROFILE_MANIFEST_PATH,
    profile_validation_path: Path = DEFAULT_PROFILE_VALIDATION_PATH,
    boundary_check_path: Path = DEFAULT_BOUNDARY_CHECK_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now()
    input_validation = validate_16i_inputs(
        snapshot_path=snapshot_path,
        hard_sidecar_path=hard_sidecar_path,
        hard_manifest_path=hard_manifest_path,
        hard_validation_path=hard_validation_path,
        resume_signal_path=resume_signal_path,
        resume_signal_manifest_path=resume_signal_manifest_path,
        resume_signal_validation_path=resume_signal_validation_path,
        output_path=input_validation_path,
        generated_at=generated_at,
    )
    policy = build_policy_payload(generated_at=generated_at)
    write_json(policy_path, policy)
    quality_audit = audit_resume_signal_quality(
        snapshot_path=snapshot_path,
        resume_signal_path=resume_signal_path,
        output_json_path=quality_audit_json_path,
        output_csv_path=quality_audit_csv_path,
        generated_at=generated_at,
    )
    profile_manifest = build_job_tailoring_profile_sidecar(
        hard_sidecar_path=hard_sidecar_path,
        resume_signal_path=resume_signal_path,
        output_path=profile_path,
        manifest_path=profile_manifest_path,
        generated_at=generated_at,
    )
    profile_validation = validate_job_tailoring_profile_sidecar(
        snapshot_path=snapshot_path,
        profile_path=profile_path,
        manifest=profile_manifest,
        output_path=profile_validation_path,
        generated_at=generated_at,
    )
    boundary_check = write_hard_skill_boundary_check(
        input_validation=input_validation,
        profile_validation=profile_validation,
        output_path=boundary_check_path,
        generated_at=generated_at,
    )
    write_contract_report(
        input_validation=input_validation,
        quality_audit=quality_audit,
        policy=policy,
        profile_manifest=profile_manifest,
        profile_validation=profile_validation,
        boundary_check=boundary_check,
        output_path=report_path,
    )
    return {
        "generated_at": generated_at,
        "input_validation_path": str(input_validation_path),
        "policy_path": str(policy_path),
        "quality_audit_json_path": str(quality_audit_json_path),
        "quality_audit_csv_path": str(quality_audit_csv_path),
        "profile_path": str(profile_path),
        "profile_manifest_path": str(profile_manifest_path),
        "profile_validation_path": str(profile_validation_path),
        "boundary_check_path": str(boundary_check_path),
        "report_path": str(report_path),
        "input_validation_passed": input_validation["all_validation_checks_passed"],
        "profile_validation_passed": profile_validation["all_validation_checks_passed"],
        "boundary_check_passed": boundary_check["all_boundary_checks_passed"],
        "rows_written": profile_manifest["rows_written"],
        "rows_with_hard_skill_targets": profile_manifest["rows_with_hard_skill_targets"],
        "rows_with_soft_skill_targets": profile_manifest["rows_with_soft_skill_targets"],
        "rows_with_professional_competency_targets": profile_manifest[
            "rows_with_professional_competency_targets"
        ],
        "rows_with_review_only_targets": profile_manifest["rows_with_review_only_targets"],
        "rows_with_suppressed_targets": profile_manifest["rows_with_suppressed_targets"],
    }
