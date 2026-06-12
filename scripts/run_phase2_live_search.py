"""Run optional live/hybrid Phase 2 matching with current postings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from jobpilot.config import (  # noqa: E402
    EMBEDDINGS_DIR,
    OFFLINE_SNAPSHOT_CSV,
    OFFLINE_SNAPSHOT_SAMPLE_CSV,
    PROCESSED_DATA_DIR,
)
from jobpilot.live.adzuna_live import LIVE_CACHE_DIR, fetch_adzuna_live  # noqa: E402
from jobpilot.live.hybrid_pool import build_candidate_store  # noqa: E402
from jobpilot.live.jsearch_live import fetch_jsearch_live  # noqa: E402
from jobpilot.live.provider_base import sanitize_live_payload  # noqa: E402
from jobpilot.live.query_builder import build_live_queries  # noqa: E402
from jobpilot.profile.personas import PERSONA_FIXTURES, get_persona  # noqa: E402
from jobpilot.ranking.company_signals import detect_company_signals  # noqa: E402
from jobpilot.ranking.filters import (  # noqa: E402
    apply_hard_filters,
    configured_seniority_term_hits,
    junior_level_hits,
    parse_float,
    seniority_level_hits,
)
from jobpilot.ranking.location_signals import location_violation_reason  # noqa: E402
from jobpilot.ranking.ranker import JobRanker  # noqa: E402
from jobpilot.ranking.role_signals import (  # noqa: E402
    detect_role_family_signals,
    generic_backend_devops_without_target_signal,
    matches_required_role_family,
    title_contains_profile_signal,
)
from jobpilot.ranking.scoring import skill_match_score, target_role_score  # noqa: E402
from jobpilot.schemas import CANONICAL_COLUMNS  # noqa: E402
from jobpilot.utils.io import write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional live/hybrid JobPilot Phase 2 matching.")
    parser.add_argument("--persona", choices=sorted(PERSONA_FIXTURES), default="aisha")
    parser.add_argument("--mode", choices=["offline", "live", "hybrid"], default="offline")
    parser.add_argument("--provider", choices=["adzuna", "jsearch"], default="adzuna")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--results-per-page", type=int, default=20)
    parser.add_argument("--country", default="us")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=OFFLINE_SNAPSHOT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "tfidf-svd"], default="auto")
    return parser.parse_args()


def timestamp_slug() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"


def _job_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, ""))
        for key in ["title", "company", "location", "employment_type", "company_type", "sponsorship_signal", "description_text"]
    ).lower()


def _employment_has(row: dict[str, Any], terms: set[str]) -> bool:
    employment = str(row.get("employment_type", "") or "").lower()
    text = _job_text(row)
    return any(term in employment or term in text for term in terms)


def _years(row: dict[str, Any]) -> float | None:
    return parse_float(row.get("years_required"))


def _salary_preference_matches(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    desired = parse_float(profile.get("salary_min"))
    if not desired:
        return True
    job_min = parse_float(row.get("salary_min"))
    job_max = parse_float(row.get("salary_max")) or job_min
    return bool(job_max is not None and job_max >= desired)


def _location_preference_matches(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    if not profile.get("location_preferences") and not profile.get("remote_preference") and not profile.get("us_only"):
        return True
    return location_violation_reason(row, profile) is None


def strict_persona_reasons(persona_name: str, profile: dict[str, Any], row: dict[str, Any]) -> list[str]:
    name = persona_name.lower()
    reasons: list[str] = []
    text = _job_text(row)
    level_hits = set(seniority_level_hits(row))
    junior_hits = set(junior_level_hits(row))
    seniority = str(row.get("seniority", "") or "").lower()
    company_type = str(row.get("company_type", "") or "").lower()
    sponsorship_signal = str(row.get("sponsorship_signal", "") or "").lower()
    company_signals = detect_company_signals(row)

    if name == "aisha":
        if level_hits & {"senior", "staff", "principal", "lead"} or seniority in {"senior", "staff_principal", "lead_manager"}:
            reasons.append("senior_staff_lead_principal")
        if company_signals["defense_government_contractor"] or any(term in text for term in ["defense", "military", " dod "]):
            reasons.append("defense_military")
        years = _years(row)
        if years is not None and years >= 5:
            reasons.append("years_5_plus")
        if not matches_required_role_family(profile, row):
            reasons.append("not_ml_related")
    elif name == "marcus":
        location_reason = location_violation_reason(row, profile)
        if location_reason:
            reasons.append(location_reason)
        if level_hits or seniority in {"senior", "staff_principal", "lead_manager"}:
            reasons.append("senior_or_lead_role")
        years = _years(row)
        if years is not None and years >= 3:
            reasons.append("years_3_plus")
        if _employment_has(row, {"contract", "temporary", "temp", "unpaid"}):
            reasons.append("contract_temp_unpaid")
    elif name == "priya":
        location_reason = location_violation_reason(row, profile)
        if location_reason:
            reasons.append(location_reason)
        if junior_hits or seniority in {"entry_junior", "internship"}:
            reasons.append("junior_or_intern")
        if company_type == "startup" or any(term in text for term in ["tiny startup", "small startup", "seed stage", "series a"]):
            reasons.append("startup_or_tiny_startup")
        if profile.get("avoid_defense_or_clearance") and company_signals["defense_government_contractor"]:
            reasons.append("defense_or_clearance")
        if profile.get("avoid_generic_backend_devops") and generic_backend_devops_without_target_signal(profile, row):
            reasons.append("generic_backend_devops_without_ml_infra_title_signal")
        role_signals = detect_role_family_signals(row)
        if not (set(role_signals["title_families"]) & {"ml_infra", "ml_related"} and title_contains_profile_signal(profile, row)):
            reasons.append("not_ml_ai_infra_related")
    elif name == "kenji":
        location_reason = location_violation_reason(row, profile)
        if location_reason:
            reasons.append(location_reason)
        if _employment_has(row, {"contract", "temporary", "temp", "unpaid"}):
            reasons.append("contract_temp_unpaid")
        if sponsorship_signal == "no_sponsorship":
            reasons.append("no_sponsorship")
        if configured_seniority_term_hits(profile, row, "hard_reject_seniority_terms"):
            reasons.append("overly_senior_hard_reject")
    return reasons


def evaluate_top_jobs(persona_name: str, profile: dict[str, Any], top_jobs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    denom = max(top_k, 1)
    returned = len(top_jobs)
    if not top_jobs:
        return {
            "returned_jobs": 0,
            "requested_top_k": top_k,
            "topk_completion_rate": 0.0,
            "strict_topk_pass_rate": 0.0,
            "strict_returned_pass_rate": 0.0,
            "location_preference_match_rate": 0.0,
            "salary_preference_match_rate": 0.0,
        }
    strict_pass = 0
    location_pref = 0
    salary_pref = 0
    relevant = 0
    strict_counts: Counter[str] = Counter()
    for job in top_jobs:
        strict_reasons = strict_persona_reasons(persona_name, profile, job)
        strict_counts.update(strict_reasons)
        if not strict_reasons:
            strict_pass += 1
        if _location_preference_matches(profile, job):
            location_pref += 1
        if _salary_preference_matches(profile, job):
            salary_pref += 1
        hard_filter = apply_hard_filters(profile, job)
        if hard_filter.passed and (target_role_score(profile, job) >= 0.3 or skill_match_score(profile, job) >= 0.1):
            relevant += 1
    return {
        "returned_jobs": returned,
        "requested_top_k": top_k,
        "topk_completion_rate": round(returned / denom, 4),
        "precision_at_10": round(relevant / denom, 4),
        "strict_topk_pass_rate": round(strict_pass / denom, 4),
        "strict_returned_pass_rate": round(strict_pass / max(returned, 1), 4),
        "location_preference_match_rate": round(location_pref / max(returned, 1), 4),
        "salary_preference_match_rate": round(salary_pref / max(returned, 1), 4),
        "strict_violation_counts": dict(sorted(strict_counts.items())),
    }


def source_counts(top_jobs: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(job.get("source", "") or "unknown") for job in top_jobs))


def source_group_counts(top_jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for job in top_jobs:
        source = str(job.get("source", "") or "").lower()
        raw_source = str(job.get("raw_source", "") or "").lower()
        if source.endswith("_live") or "_live" in raw_source:
            counts["live"] += 1
        else:
            counts["offline"] += 1
    return dict(counts)


def live_validation_counts(profile: dict[str, Any], top_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    hard_filter_failures = sum(1 for job in top_jobs if not apply_hard_filters(profile, job).passed)
    role_family_failures = sum(
        1
        for job in top_jobs
        if profile.get("strict_role_family") and not matches_required_role_family(profile, job)
    )
    defense_risk = sum(1 for job in top_jobs if detect_company_signals(job)["defense_government_contractor"])
    strategy_labels = sum(1 for job in top_jobs if job.get("application_strategy_label"))
    return {
        "hard_filter_violation_count_top_k": hard_filter_failures,
        "strict_role_family_violation_count_top_k": role_family_failures,
        "government_defense_risk_count_top_k": defense_risk,
        "application_strategy_label_count_top_k": strategy_labels,
        "application_strategy_label_all_present": strategy_labels == len(top_jobs),
        "possible_near_duplicate_role_count_top_k": sum(1 for job in top_jobs if job.get("possible_near_duplicate_role")),
    }


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_snapshot_path(requested: Path) -> tuple[Path, list[str]]:
    requested = requested.resolve()
    if requested.exists():
        return requested, []
    sample = requested.parent / OFFLINE_SNAPSHOT_SAMPLE_CSV.name
    if sample.exists():
        warning = (
            f"Snapshot {rel_path(requested)} not found; using review-package sample "
            f"{rel_path(sample)}. Results are for package smoke testing only."
        )
        return sample.resolve(), [warning]
    return requested, []


def main() -> int:
    args = parse_args()
    profile = get_persona(args.persona)
    timestamp = timestamp_slug()
    snapshot_path, snapshot_warnings = resolve_snapshot_path(args.snapshot)
    max_queries_limit = 5 if args.provider == "jsearch" else 10
    bounded_pages = max(1, min(args.pages_per_query, 1 if args.provider == "jsearch" else 2))
    bounded_results = max(1, min(args.results_per_page, 20 if args.provider == "jsearch" else 50))
    queries = build_live_queries(profile, max_queries=max(1, min(args.max_queries, max_queries_limit)))
    estimated_api_calls = 0 if args.mode == "offline" else len(queries) * bounded_pages
    warnings: list[str] = [*snapshot_warnings]
    live_rows: list[dict[str, Any]] = []
    live_metadata: dict[str, Any] = {
        "provider": args.provider,
        "api_call_count": 0,
        "estimated_api_calls": estimated_api_calls,
        "raw_records_fetched": 0,
        "normalized_live_records": 0,
        "warnings": [],
        "errors": [],
    }
    output_paths: dict[str, str] = {}

    if args.dry_run:
        warnings.append("Dry run requested; generated queries only and skipped live API calls.")
    elif args.mode in {"live", "hybrid"}:
        if args.provider == "jsearch":
            fetch = fetch_jsearch_live(
                queries,
                pages_per_query=bounded_pages,
                results_per_page=bounded_results,
                env_file=args.env_file,
            )
        else:
            fetch = fetch_adzuna_live(
                queries,
                country=args.country,
                pages_per_query=bounded_pages,
                results_per_page=bounded_results,
                env_file=args.env_file,
            )
        live_rows = fetch.normalized_records
        live_metadata = fetch.metadata
        output_dir = LIVE_CACHE_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / f"{args.provider}_live_raw_{timestamp}.json"
        live_jobs_path = output_dir / f"{args.provider}_live_jobs_{timestamp}.csv"
        write_json(raw_path, {"provider": fetch.provider, "queries": queries, "records": fetch.raw_records})
        write_csv(live_jobs_path, live_rows, CANONICAL_COLUMNS)
        output_paths.update({"live_raw": rel_path(raw_path), "live_jobs": rel_path(live_jobs_path)})
        warnings.extend(live_metadata.get("warnings", []))
        warnings.extend(live_metadata.get("errors", []))

    pool = build_candidate_store(
        mode=args.mode,
        live_rows=live_rows,
        snapshot_path=snapshot_path,
        cache_dir=args.cache_dir,
        embedding_backend=args.embedding_backend,
        fallback_to_offline=True,
    )
    warnings.extend(pool.metadata.get("warnings", []))
    ranker = JobRanker(pool.store)
    result = ranker.rank(profile, top_k=args.top_k, candidate_k=args.candidate_k)
    top_jobs = result["top_jobs"]
    metrics = evaluate_top_jobs(args.persona, profile, top_jobs, args.top_k)
    counts = source_counts(top_jobs)
    group_counts = source_group_counts(top_jobs)
    validation_counts = live_validation_counts(profile, top_jobs)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "persona": args.persona,
        "provider": args.provider,
        "mode_requested": args.mode,
        "mode_used": pool.mode_used,
        "dry_run": args.dry_run,
        "snapshot_requested": rel_path(args.snapshot),
        "snapshot_used": rel_path(snapshot_path),
        "snapshot_sample_fallback_used": bool(snapshot_warnings),
        "generated_queries": queries,
        "max_queries": len(queries),
        "pages_per_query": bounded_pages,
        "results_per_page": bounded_results,
        "estimated_api_calls": estimated_api_calls,
        "api_call_count": live_metadata.get("api_call_count", 0),
        "raw_live_records_fetched": live_metadata.get("raw_records_fetched", 0),
        "normalized_live_records": live_metadata.get("normalized_live_records", 0),
        "duplicates_removed_against_offline": pool.metadata.get("duplicates_against_offline", 0),
        "duplicates_removed_within_live": pool.metadata.get("duplicates_within_live", 0),
        "live_records_retained": pool.metadata.get("live_records_retained", 0),
        "candidate_pool": pool.metadata,
        "top_k_returned": len(top_jobs),
        "source_counts_top_k": counts,
        "source_group_counts_top_k": group_counts,
        "metrics": metrics,
        "validation_counts": validation_counts,
        "application_strategy": result.get("metadata", {}).get("application_strategy", {}),
        "exact_duplicate_postings_removed_by_ranker": result.get("metadata", {}).get("exact_duplicate_postings_removed", 0),
        "warnings": warnings,
        "output_paths": output_paths,
    }

    live_cache_dir = LIVE_CACHE_DIR
    live_cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = live_cache_dir / f"{args.provider}_live_search_report_{timestamp}.json"
    matching_path = PROCESSED_DATA_DIR / f"phase2_live_matching_{args.persona}.json"
    report["output_paths"]["report"] = rel_path(report_path)
    report["output_paths"]["matching"] = rel_path(matching_path)
    safe_result = sanitize_live_payload(result)
    payload = {
        "profile": profile,
        "live_search_report": report,
        **safe_result,
    }
    write_json(report_path, report)
    write_json(matching_path, payload)

    print("Phase 2 live search complete")
    print(f"Mode requested: {args.mode}")
    print(f"Mode used: {pool.mode_used}")
    print(f"Snapshot used: {snapshot_path}")
    print("Generated queries:")
    for query in queries:
        print(f"- {query}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"Returned jobs: {metrics['returned_jobs']}/{metrics['requested_top_k']}")
    print(f"Strict top-k pass rate: {metrics['strict_topk_pass_rate']}")
    print(f"Location preference match rate: {metrics['location_preference_match_rate']}")
    print(f"Salary preference match rate: {metrics['salary_preference_match_rate']}")
    print(f"Source counts top-k: {json.dumps(counts, sort_keys=True)}")
    print(f"Source group counts top-k: {json.dumps(group_counts, sort_keys=True)}")
    print(f"Report: {report_path}")
    print(f"Matching output: {matching_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
