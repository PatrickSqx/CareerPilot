"""Run JobPilot Phase 1 ingestion and offline snapshot generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from jobpilot.config import (  # noqa: E402
    CACHED_CURRENT_JSONL,
    CURRENT_JOBS_CLEAN_CSV,
    DATA_DICTIONARY_MD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEMO_ROWS,
    DEFAULT_SCHEMA_SAMPLE_ROWS,
    DEFAULT_TARGET_ROWS,
    INGESTION_DEMO_CSV,
    INGESTION_REPORT_JSON,
    JOBS_CLEAN_CSV,
    KAGGLE_JSONL,
    MARKET_ANALYTICS_JSON,
    OFFLINE_SNAPSHOT_CSV,
    ensure_phase1_dirs,
)
from jobpilot.ingestion.cleaner import clean_normalized_record  # noqa: E402
from jobpilot.ingestion.current_api import get_current_postings, write_current_jobs  # noqa: E402
from jobpilot.ingestion.dedup import ExactDeduplicator  # noqa: E402
from jobpilot.ingestion.kaggle_loader import inspect_kaggle_schema, stream_kaggle_jsonl  # noqa: E402
from jobpilot.ingestion.normalizer import normalize_kaggle_record, utc_now_iso  # noqa: E402
from jobpilot.ingestion.quality import (  # noqa: E402
    CATEGORY_CAP_SHARE,
    COMPANY_CAP_SHARE,
    MIN_CATEGORY_CAP,
    MIN_COMPANY_CAP,
    MIN_TITLE_CAP,
    TITLE_CAP_SHARE,
    USFirstRemoteSelector,
    cap_limit,
    category_key,
    company_key,
    market_eligibility,
    quality_tier,
    required_ready,
    row_quality_score,
    score_bucket,
    title_key,
)
from jobpilot.ingestion.report import (  # noqa: E402
    PHASE18_STRUCTURED_FIELDS,
    build_market_analytics,
    build_missing_field_counts,
    coverage_rate,
    current_api_counts,
    field_coverage,
    query_counts,
    source_level_coverage,
    source_counts,
    utc_now_iso as report_time,
    write_data_dictionary,
    write_report,
)
from jobpilot.ingestion.stream_demo import batched  # noqa: E402
from jobpilot.schemas import CANONICAL_COLUMNS  # noqa: E402
from jobpilot.utils.io import write_csv, write_json  # noqa: E402

DEFAULT_SCORE85_MANIFEST = PROJECT_ROOT / "data" / "processed" / "raw_quality_score85_manifest.csv"
DEFAULT_PHASE17_CANDIDATE_CACHE = PROJECT_ROOT / "data" / "processed" / "phase1_7_manifest_candidates.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JobPilot Phase 1 data ingestion.")
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--demo-rows", type=int, default=DEFAULT_DEMO_ROWS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--schema-sample-rows", type=int, default=DEFAULT_SCHEMA_SAMPLE_ROWS)
    parser.add_argument("--progress-every", type=int, default=0, help="Print ingestion progress every N Kaggle rows.")
    parser.add_argument("--kaggle-jsonl", type=Path, default=KAGGLE_JSONL)
    parser.add_argument(
        "--score85-manifest",
        type=Path,
        default=DEFAULT_SCORE85_MANIFEST,
        help="Optional row-level score>85 manifest used to avoid normalizing low-quality raw rows.",
    )
    parser.add_argument(
        "--ignore-score85-manifest",
        action="store_true",
        help="Force direct full-JSONL streaming instead of the score>85 candidate manifest.",
    )
    parser.add_argument(
        "--phase17-candidate-cache",
        type=Path,
        default=DEFAULT_PHASE17_CANDIDATE_CACHE,
        help="Optional materialized Phase 1.7 candidate JSONL cache.",
    )
    parser.add_argument(
        "--ignore-phase17-candidate-cache",
        action="store_true",
        help="Ignore the materialized Phase 1.7 candidate cache even when present.",
    )
    parser.add_argument("--fetch-current", action="store_true", help="Attempt live Adzuna/JSearch fetching if credentials exist.")
    parser.add_argument("--current-provider", choices=["adzuna", "jsearch"], default="adzuna")
    parser.add_argument("--cached-current", type=Path, default=CACHED_CURRENT_JSONL)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser.parse_args()


def rel_path(path: Path | str) -> str:
    """Return a stable project-relative path for reports when possible."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def value_counts(rows: list[dict], column: str) -> dict[str, int]:
    return dict(Counter(str(row.get(column, "") or "(blank)") for row in rows))


def load_score85_manifest(path: Path, *, min_score: int = 86) -> dict[int, dict[str, str]]:
    line_map: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                line_no = int(str(row.get("line_no", "")).strip())
                score = int(float(str(row.get("quality_score", "")).strip() or 0))
            except ValueError:
                continue
            if score >= min_score:
                line_map[line_no] = row
    return line_map


def manifest_policy_record(row: dict[str, str]) -> dict[str, str]:
    location = ", ".join(
        value
        for value in [row.get("city", ""), row.get("state", ""), row.get("country", "")]
        if str(value or "").strip()
    )
    return {
        "source": row.get("source", ""),
        "source_record_id": row.get("source_record_id", ""),
        "title": row.get("title", ""),
        "company": row.get("company", ""),
        "employer": row.get("company", ""),
        "country": row.get("country", ""),
        "state": row.get("state", ""),
        "city": row.get("city", ""),
        "location": location,
        "raw_source_country": row.get("country", ""),
        "raw_categories": row.get("category", ""),
        "raw_work_types": row.get("work_type", ""),
        "position_work_type_raw": row.get("work_type", ""),
        "description_text": "",
    }


def select_score85_manifest_lines(
    path: Path,
    *,
    target_rows: int,
    min_score: int = 86,
) -> tuple[dict[int, dict[str, str]], dict[str, object]]:
    """Preselect a bounded set of score>85 manifest line numbers.

    The manifest is an index, not the final data source. These line numbers are
    later re-read from raw JSONL and normalized through the canonical pipeline.
    """

    if target_rows <= 0:
        return {}, {
            "available_score_gt_85_manifest_rows": 0,
            "preselected_manifest_rows": 0,
            "preselected_us_rows": 0,
            "preselected_non_us_remote_rows": 0,
        }

    probe_selector = USFirstRemoteSelector(target_rows=target_rows)
    non_us_remote_target = probe_selector.non_us_remote_soft_target
    buffer_rows = min(2_000, max(250, target_rows // 25))
    us_target = target_rows - non_us_remote_target + buffer_rows
    total_target = target_rows + buffer_rows
    category_cap = cap_limit(target_rows, CATEGORY_CAP_SHARE, MIN_CATEGORY_CAP)
    title_cap = cap_limit(target_rows, TITLE_CAP_SHARE, MIN_TITLE_CAP)
    company_cap = cap_limit(target_rows, COMPANY_CAP_SHARE, MIN_COMPANY_CAP)

    selected: dict[int, dict[str, str]] = {}
    selected_us = 0
    selected_non_us_remote = 0
    available_score_rows = 0
    available_us_rows = 0
    available_non_us_remote_rows = 0
    guard_rejections: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    company_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                line_no = int(str(row.get("line_no", "")).strip())
                score = int(float(str(row.get("quality_score", "")).strip() or 0))
            except ValueError:
                continue
            if score < min_score:
                continue
            available_score_rows += 1
            record = manifest_policy_record(row)
            eligible, market_label = market_eligibility(record)
            if not eligible:
                continue

            if market_label == "non_us_remote_compatible":
                available_non_us_remote_rows += 1
                if selected_non_us_remote >= non_us_remote_target:
                    continue
            elif market_label == "us":
                available_us_rows += 1
                if selected_us >= us_target:
                    continue
            else:
                continue

            category = category_key(record)
            title = title_key(record)
            company = company_key(record)
            if category_counts[category] >= category_cap:
                guard_rejections["category_guard"] += 1
                continue
            if title_counts[title] >= title_cap:
                guard_rejections["title_guard"] += 1
                continue
            if company_counts[company] >= company_cap:
                guard_rejections["company_guard"] += 1
                continue

            selected[line_no] = row
            category_counts[category] += 1
            title_counts[title] += 1
            company_counts[company] += 1
            if market_label == "non_us_remote_compatible":
                selected_non_us_remote += 1
            else:
                selected_us += 1

            if len(selected) >= total_target and selected_non_us_remote >= non_us_remote_target:
                break

    summary = {
        "available_score_gt_85_manifest_rows": available_score_rows,
        "available_us_rows_seen_until_preselection_stop": available_us_rows,
        "available_non_us_remote_rows_seen_until_preselection_stop": available_non_us_remote_rows,
        "preselected_manifest_rows": len(selected),
        "preselected_us_rows": selected_us,
        "preselected_non_us_remote_rows": selected_non_us_remote,
        "preselection_buffer_rows": buffer_rows,
        "non_us_remote_preselection_target": non_us_remote_target,
        "preselection_guard_rejections": dict(guard_rejections),
        "preselected_max_raw_line": max(selected) if selected else 0,
    }
    return selected, summary


def iter_kaggle_candidates(
    jsonl_path: Path,
    *,
    manifest_path: Path,
    ignore_manifest: bool,
    target_rows: int,
    metadata: dict[str, object],
):
    """Yield raw Kaggle records from either the score>85 manifest or full stream."""

    if not ignore_manifest and manifest_path.exists():
        target_lines, preselection = select_score85_manifest_lines(manifest_path, target_rows=target_rows)
        target_set = set(target_lines)
        max_line = max(target_set) if target_set else 0
        metadata.update(
            {
                "mode": "score85_manifest_stream",
                "manifest_path": rel_path(manifest_path),
                "manifest_used": True,
                "target_manifest_rows": len(target_lines),
                "min_manifest_quality_score": 86,
                "preselection": preselection,
                "raw_lines_scanned": 0,
                "matched_manifest_rows": 0,
                "parse_errors": 0,
            }
        )
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                metadata["raw_lines_scanned"] = line_no
                if line_no > max_line:
                    break
                if line_no not in target_set:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    metadata["parse_errors"] = int(metadata.get("parse_errors", 0)) + 1
                    continue
                raw["_phase1_manifest_quality_score"] = target_lines[line_no].get("quality_score", "")
                metadata["matched_manifest_rows"] = int(metadata.get("matched_manifest_rows", 0)) + 1
                yield line_no, raw
        return

    metadata.update(
        {
            "mode": "full_raw_jsonl_stream",
            "manifest_path": rel_path(manifest_path),
            "manifest_used": False,
            "raw_lines_scanned": 0,
            "matched_manifest_rows": None,
            "parse_errors": None,
        }
    )
    for line_no, raw in stream_kaggle_jsonl(jsonl_path):
        metadata["raw_lines_scanned"] = line_no
        yield line_no, raw


def iter_materialized_candidates(path: Path, metadata: dict[str, object]):
    metadata.update(
        {
            "mode": "phase1_7_materialized_candidate_cache",
            "candidate_cache_path": rel_path(path),
            "candidate_cache_used": True,
            "candidate_cache_rows_seen": 0,
            "parse_errors": 0,
        }
    )
    with path.open("r", encoding="utf-8") as handle:
        for row_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                metadata["parse_errors"] = int(metadata.get("parse_errors", 0)) + 1
                continue
            metadata["candidate_cache_rows_seen"] = int(metadata.get("candidate_cache_rows_seen", 0)) + 1
            yield row_no, row


def main() -> int:
    args = parse_args()
    ensure_phase1_dirs()
    started = report_time()
    start = time.perf_counter()
    run_id = str(uuid.uuid4())
    errors_or_warnings: list[str] = []

    if not args.kaggle_jsonl.exists():
        raise FileNotFoundError(f"Kaggle JSONL source not found: {args.kaggle_jsonl}")

    raw_schema = inspect_kaggle_schema(args.kaggle_jsonl, sample_rows=args.schema_sample_rows)
    errors_or_warnings.extend(raw_schema.get("warnings", []))
    kaggle_candidate_stream_metadata: dict[str, object] = {}

    deduplicator = ExactDeduplicator()
    current_rows, current_meta = get_current_postings(
        fetch_live=args.fetch_current,
        provider=args.current_provider,
        cached_path=args.cached_current,
        env_file=args.env_file,
    )
    if "raw_output_paths" in current_meta:
        current_meta["raw_output_paths"] = [rel_path(path) for path in current_meta["raw_output_paths"]]
    errors_or_warnings.extend(current_meta.get("errors", []))

    all_rows: list[dict] = []
    demo_rows: list[dict] = []
    invalid_reasons: Counter[str] = Counter()
    invalid_records_removed_count = 0
    records_seen = 0
    records_cleaned = 0
    records_ingested = 0
    title_repaired_from_fallback = 0
    current_snapshot_market_rejections: Counter[str] = Counter()
    current_rows_retained_for_snapshot = 0
    unique_kaggle_candidates = 0
    kaggle_candidate_quality_tiers: Counter[str] = Counter()
    kaggle_candidate_quality_buckets: Counter[str] = Counter()

    for row in current_rows:
        if row.get("_title_repaired"):
            title_repaired_from_fallback += 1
        keep, row = deduplicator.keep(row)
        records_cleaned += 1
        if keep:
            eligible, market_label = market_eligibility(row)
            if not eligible:
                current_snapshot_market_rejections[market_label] += 1
                continue
            all_rows.append(row)
            current_rows_retained_for_snapshot += 1
            records_ingested += 1
            if len(demo_rows) < args.demo_rows:
                demo_rows.append(row)

    kaggle_target_rows = max(args.target_rows - len(all_rows), 0)
    selector = USFirstRemoteSelector(target_rows=kaggle_target_rows)

    stop = False
    use_candidate_cache = args.phase17_candidate_cache.exists() and not args.ignore_phase17_candidate_cache
    if use_candidate_cache:
        stream = iter_materialized_candidates(args.phase17_candidate_cache, kaggle_candidate_stream_metadata)
        for batch in batched(stream, args.batch_size):
            for _, cleaned in batch:
                records_seen += 1
                if cleaned.get("_title_repaired"):
                    title_repaired_from_fallback += 1
                records_cleaned += 1
                keep, cleaned = deduplicator.keep(cleaned)
                if not keep:
                    continue
                unique_kaggle_candidates += 1
                score = row_quality_score(cleaned)
                kaggle_candidate_quality_tiers[quality_tier(cleaned)] += 1
                kaggle_candidate_quality_buckets[score_bucket(score)] += 1
                if selector.accept(cleaned):
                    records_ingested += 1
                    if len(demo_rows) < args.demo_rows:
                        demo_rows.append(cleaned)
                if selector.complete():
                    stop = True
                    break
            if args.progress_every and records_seen % args.progress_every < args.batch_size:
                print(
                    "[phase1] cache_seen={seen:,} selected={selected:,}/{target:,} "
                    "duplicates={duplicates:,} bloom_skipped_exact={skipped:,}".format(
                        seen=records_seen,
                        selected=len(selector.selected),
                        target=kaggle_target_rows,
                        duplicates=deduplicator.duplicates_removed,
                        skipped=deduplicator.summary()["exact_membership_checks_skipped_by_bloom"],
                    ),
                    flush=True,
                )
            if stop:
                break
    else:
        stream = iter_kaggle_candidates(
            args.kaggle_jsonl,
            manifest_path=args.score85_manifest,
            ignore_manifest=args.ignore_score85_manifest,
            target_rows=kaggle_target_rows,
            metadata=kaggle_candidate_stream_metadata,
        )
        for batch in batched(stream, args.batch_size):
            for line_no, raw in batch:
                records_seen += 1
                manifest_quality_score = raw.get("_phase1_manifest_quality_score", "")
                normalized = normalize_kaggle_record(raw, line_no, ingested_at=started)
                if normalized.get("_title_repaired"):
                    title_repaired_from_fallback += 1
                cleaned, errors = clean_normalized_record(normalized)
                if cleaned is None:
                    invalid_records_removed_count += 1
                    invalid_reasons.update(errors)
                    continue
                if manifest_quality_score:
                    cleaned["_phase1_manifest_quality_score"] = manifest_quality_score
                records_cleaned += 1
                keep, cleaned = deduplicator.keep(cleaned)
                if not keep:
                    continue
                if manifest_quality_score:
                    cleaned["_phase1_manifest_quality_score"] = manifest_quality_score
                unique_kaggle_candidates += 1
                score = row_quality_score(cleaned)
                kaggle_candidate_quality_tiers[quality_tier(cleaned)] += 1
                kaggle_candidate_quality_buckets[score_bucket(score)] += 1
                if selector.accept(cleaned):
                    records_ingested += 1
                    if len(demo_rows) < args.demo_rows:
                        demo_rows.append(cleaned)
                if selector.complete():
                    stop = True
                    break
            if args.progress_every and records_seen % args.progress_every < args.batch_size:
                print(
                    "[phase1] candidate_seen={seen:,} raw_scanned={raw_scanned:,} "
                    "selected={selected:,}/{target:,} duplicates={duplicates:,} "
                    "bloom_skipped_exact={skipped:,}".format(
                        seen=records_seen,
                        raw_scanned=int(kaggle_candidate_stream_metadata.get("raw_lines_scanned", 0)),
                        selected=len(selector.selected),
                        target=kaggle_target_rows,
                        duplicates=deduplicator.duplicates_removed,
                        skipped=deduplicator.summary()["exact_membership_checks_skipped_by_bloom"],
                    ),
                    flush=True,
                )
            if stop:
                break

    kaggle_rows = selector.selected
    all_rows.extend(kaggle_rows)
    if len(all_rows) < args.target_rows:
        raise RuntimeError(
            "Phase 1.7 US-first score>85 sampling could not fill the target snapshot: "
            f"selected {len(all_rows)} of {args.target_rows} rows. "
            "Review data/processed/score85_full_distribution_audit.md and relax diversity guards if needed."
        )
    if (
        selector.non_us_remote_soft_target
        and selector.market_counts.get("non_us_remote_compatible", 0) < selector.non_us_remote_soft_target
    ):
        errors_or_warnings.append(
            "Phase 1.7 non-US remote soft target was not fully met; final snapshot remains US-first and "
            "contains only remote-compatible non-US rows found before stream exhaustion."
        )

    jobs_clean_rows = write_csv(JOBS_CLEAN_CSV, kaggle_rows, CANONICAL_COLUMNS)
    current_jobs_clean_rows = write_current_jobs(CURRENT_JOBS_CLEAN_CSV, current_rows)
    offline_snapshot_rows = write_csv(OFFLINE_SNAPSHOT_CSV, all_rows, CANONICAL_COLUMNS)
    ingestion_demo_rows = write_csv(INGESTION_DEMO_CSV, demo_rows, CANONICAL_COLUMNS)
    write_data_dictionary(DATA_DICTIONARY_MD)
    analytics = build_market_analytics(all_rows)
    write_json(MARKET_ANALYTICS_JSON, analytics)

    finished = report_time()
    runtime_seconds = round(time.perf_counter() - start, 3)
    duplicates_removed = deduplicator.duplicates_removed
    invalid_records_removed = invalid_records_removed_count
    records_ingested = len(all_rows)
    final_quality_tiers = Counter(quality_tier(row) for row in all_rows)
    final_quality_buckets = Counter(score_bucket(row_quality_score(row)) for row in all_rows)
    final_required_ready_rows = sum(1 for row in all_rows if required_ready(row))
    final_market_counts = Counter(market_eligibility(row)[1] for row in all_rows)
    report = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "runtime_seconds": runtime_seconds,
        "records_seen": records_seen,
        "records_streamed_demo": len(demo_rows),
        "records_cleaned": records_cleaned,
        "records_ingested": records_ingested,
        "invalid_records_removed": invalid_records_removed,
        "invalid_reason_counts": dict(invalid_reasons),
        "title_repaired_from_fallback": title_repaired_from_fallback,
        "duplicates_removed": duplicates_removed,
        "duplicate_rate": round(duplicates_removed / max(records_cleaned, 1), 6),
        "deduplication": deduplicator.summary(),
        "snapshot_rows": len(all_rows),
        "target_snapshot_rows": args.target_rows,
        "sampling_strategy": selector.policy()["strategy"],
        "phase1_6_sampling": {
            "enabled": False,
            "superseded_by": "phase1_7_sampling",
        },
        "phase1_7_sampling": {
            "enabled": True,
            "onepager_upper_bound_target": args.target_rows,
            "current_rows_seeded_before_kaggle_sampling": current_rows_retained_for_snapshot,
            "current_snapshot_market_rejection_counts": dict(current_snapshot_market_rejections),
            "kaggle_target_rows": kaggle_target_rows,
            "kaggle_candidate_stream": kaggle_candidate_stream_metadata,
            "unique_kaggle_candidates_evaluated": unique_kaggle_candidates,
            "candidate_quality_tier_counts": dict(kaggle_candidate_quality_tiers),
            "candidate_quality_score_buckets": dict(kaggle_candidate_quality_buckets),
            "selector_summary": selector.summary(),
            "final_snapshot_market_eligibility_counts": dict(final_market_counts.most_common()),
            "final_snapshot_required_ready_rows": final_required_ready_rows,
            "final_snapshot_required_ready_rate": round(final_required_ready_rows / max(len(all_rows), 1), 6),
            "final_snapshot_quality_tier_counts": dict(final_quality_tiers),
            "final_snapshot_quality_score_buckets": dict(final_quality_buckets),
        },
        "output_row_counts": {
            "jobs_clean_csv": jobs_clean_rows,
            "current_jobs_clean_csv": current_jobs_clean_rows,
            "offline_snapshot_csv": offline_snapshot_rows,
            "ingestion_demo_csv": ingestion_demo_rows,
        },
        "kaggle_rows_in_snapshot": sum(1 for row in all_rows if str(row.get("is_current_api")).lower() != "true"),
        "current_rows_in_snapshot": sum(1 for row in all_rows if str(row.get("is_current_api")).lower() == "true"),
        "source_counts": source_counts(all_rows),
        "current_api_counts": current_api_counts(all_rows),
        "cached_current_used": bool(current_meta.get("cached_current_used")),
        "saved_raw_current_used": bool(current_meta.get("saved_raw_current_used")),
        "current_ingestion_metadata": current_meta,
        "query_counts": query_counts(all_rows),
        "missing_field_counts": build_missing_field_counts(all_rows),
        "source_backed_field_coverage": {
            "company_id": coverage_rate(all_rows, "company_id"),
            "location_id": coverage_rate(all_rows, "location_id"),
            "raw_source_country": coverage_rate(all_rows, "raw_source_country"),
            "raw_locale": coverage_rate(all_rows, "raw_locale"),
            "company_url": coverage_rate(all_rows, "company_url"),
            "raw_categories": coverage_rate(all_rows, "raw_categories"),
            "raw_work_types": coverage_rate(all_rows, "raw_work_types"),
            "raw_qualifications": coverage_rate(all_rows, "raw_qualifications"),
            "schema_org_employment_type": coverage_rate(all_rows, "schema_org_employment_type"),
            "schema_org_skills": coverage_rate(all_rows, "schema_org_skills"),
            "schema_org_experience_requirements": coverage_rate(all_rows, "schema_org_experience_requirements"),
        },
        "phase1_8_structured_signal_preservation": {
            "enabled": True,
            "sampling_policy_changed": False,
            "raw_json_expanded_into_main_csv": False,
            "real_jsearch_run": False,
            "structured_signal_fields": PHASE18_STRUCTURED_FIELDS,
            "field_coverage": field_coverage(all_rows, PHASE18_STRUCTURED_FIELDS),
            "source_level_coverage": source_level_coverage(all_rows, PHASE18_STRUCTURED_FIELDS),
            "normalized_signal_coverage": {
                "normalized_skills": coverage_rate(all_rows, "normalized_skills"),
                "normalized_industries": coverage_rate(all_rows, "normalized_industries"),
                "normalized_role_terms": coverage_rate(all_rows, "normalized_role_terms"),
                "normalized_keywords": coverage_rate(all_rows, "normalized_keywords"),
            },
            "confidence_counts": value_counts(all_rows, "structured_signal_confidence"),
        },
        "salary_normalization_method_counts": value_counts(all_rows, "salary_normalization_method"),
        "salary_coverage_rate": coverage_rate(all_rows, "salary_min", "salary_max", "salary_raw"),
        "link_coverage_rate": coverage_rate(all_rows, "link"),
        "description_coverage_rate": coverage_rate(all_rows, "description_text", "description"),
        "output_paths": {
            "jobs_clean_csv": rel_path(JOBS_CLEAN_CSV),
            "current_jobs_clean_csv": rel_path(CURRENT_JOBS_CLEAN_CSV),
            "offline_snapshot_csv": rel_path(OFFLINE_SNAPSHOT_CSV),
            "ingestion_demo_csv": rel_path(INGESTION_DEMO_CSV),
            "ingestion_report_json": rel_path(INGESTION_REPORT_JSON),
            "data_dictionary_md": rel_path(DATA_DICTIONARY_MD),
            "market_analytics_json": rel_path(MARKET_ANALYTICS_JSON),
        },
        "sample_schema_columns": CANONICAL_COLUMNS,
        "raw_schema_inspection": raw_schema,
        "errors_or_warnings": errors_or_warnings,
    }
    write_report(INGESTION_REPORT_JSON, report)

    print(f"Phase 1 complete: {len(all_rows)} snapshot rows, {len(demo_rows)} demo rows")
    print(f"Report: {INGESTION_REPORT_JSON}")
    print(f"Snapshot: {OFFLINE_SNAPSHOT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
