"""Run Phase 2 retrieval and ranking benchmarks for evaluation personas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from jobpilot.config import (  # noqa: E402
    EMBEDDINGS_DIR,
    OFFLINE_SNAPSHOT_CSV,
    PERSONA_PHASE2_RESULTS_JSON,
    PHASE2_BENCHMARKS_JSON,
)
from jobpilot.profile.personas import PERSONA_FIXTURES, get_persona  # noqa: E402
from jobpilot.profile.profile_parser import profile_to_text  # noqa: E402
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
from jobpilot.retrieval.baseline import TfidfBaselineRetriever  # noqa: E402
from jobpilot.retrieval.embeddings import build_or_load_job_embeddings  # noqa: E402
from jobpilot.utils.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JobPilot Phase 2 benchmark scripts.")
    parser.add_argument("--all-personas", action="store_true", help="Run Aisha, Marcus, Priya, and Kenji.")
    parser.add_argument("--personas", nargs="*", choices=sorted(PERSONA_FIXTURES), default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--snapshot", type=Path, default=OFFLINE_SNAPSHOT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "tfidf-svd"], default="auto")
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--benchmarks-output", type=Path, default=PHASE2_BENCHMARKS_JSON)
    parser.add_argument("--persona-output", type=Path, default=PERSONA_PHASE2_RESULTS_JSON)
    return parser.parse_args()


def selected_personas(args: argparse.Namespace) -> list[str]:
    if args.all_personas or not args.personas:
        return sorted(PERSONA_FIXTURES)
    return args.personas


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


def _location_preference_matches(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    if not profile.get("location_preferences") and not profile.get("remote_preference") and not profile.get("us_only"):
        return True
    return location_violation_reason(row, profile) is None


def _salary_preference_matches(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    desired = parse_float(profile.get("salary_min"))
    if not desired:
        return True
    job_min = parse_float(row.get("salary_min"))
    job_max = parse_float(row.get("salary_max")) or job_min
    return bool(job_max is not None and job_max >= desired)


def _ml_related(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    if profile.get("strict_role_family"):
        return matches_required_role_family(profile, row)
    role_signals = detect_role_family_signals(row)
    if {"ml_related", "research_ai"} & set(role_signals["title_families"]):
        return True
    text = _job_text(row)
    terms = [
        "machine learning",
        "ml engineer",
        "mlops",
        "artificial intelligence",
        " ai ",
        "applied scientist",
        "data scientist",
        "model",
    ]
    return target_role_score(profile, row) >= 0.3 or any(term in f" {text} " for term in terms)


def _ml_infra_related(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    role_signals = detect_role_family_signals(row)
    title_families = set(role_signals["title_families"])
    if title_families & {"ml_infra", "ml_related"} and title_contains_profile_signal(profile, row):
        return True
    text = _job_text(row)
    terms = [
        "mlops",
        "platform",
        "infrastructure",
        "kubernetes",
        "spark",
        "kafka",
        "model serving",
        "machine learning",
        " ai ",
        "artificial intelligence",
    ]
    return target_role_score(profile, row) >= 0.45 and any(term in f" {text} " for term in terms)


def strict_persona_reasons(persona_name: str, profile: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """Assignment-aligned strict persona checks for benchmark reporting only."""

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
        if not _ml_infra_related(profile, row):
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


def evaluate_rows(
    persona_name: str,
    profile: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    denominator: int,
) -> dict[str, Any]:
    denom = max(denominator, 1)
    returned = len(rows)
    if not rows:
        return {
            "precision_at_10": 0.0,
            "persona_pass_rate": 0.0,
            "persona_pass_rate_loose": 0.0,
            "loose_returned_pass_rate": 0.0,
            "loose_topk_pass_rate": 0.0,
            "dealbreaker_violation_rate": 0.0,
            "dealbreaker_violation_rate_loose": 0.0,
            "dealbreaker_violation_rate_topk": 0.0,
            "strict_precision_at_10": 0.0,
            "strict_persona_pass_rate": 0.0,
            "persona_pass_rate_strict": 0.0,
            "strict_returned_pass_rate": 0.0,
            "strict_topk_pass_rate": 0.0,
            "strict_dealbreaker_violation_rate": 0.0,
            "dealbreaker_violation_rate_strict": 0.0,
            "strict_dealbreaker_violation_rate_topk": 0.0,
            "strict_violation_counts": {},
            "evaluated_rows": 0,
            "returned_jobs": 0,
            "requested_top_k": denominator,
            "topk_completion_rate": 0.0,
            "location_preference_match_rate": 0.0,
            "location_preference_topk_match_rate": 0.0,
            "salary_preference_match_rate": 0.0,
            "salary_preference_topk_match_rate": 0.0,
        }
    relevant = 0
    pass_count = 0
    violation_count = 0
    strict_relevant = 0
    strict_pass_count = 0
    strict_violation_count = 0
    location_preference_match_count = 0
    salary_preference_match_count = 0
    strict_violation_counts: dict[str, int] = {}
    for row in rows:
        filter_result = apply_hard_filters(profile, row)
        if filter_result.passed:
            pass_count += 1
        else:
            violation_count += 1
        role = target_role_score(profile, row)
        skills = skill_match_score(profile, row)
        if filter_result.passed and (role >= 0.3 or skills >= 0.1):
            relevant += 1
        if _location_preference_matches(profile, row):
            location_preference_match_count += 1
        if _salary_preference_matches(profile, row):
            salary_preference_match_count += 1

        strict_reasons = strict_persona_reasons(persona_name, profile, row)
        for reason in strict_reasons:
            strict_violation_counts[reason] = strict_violation_counts.get(reason, 0) + 1
        if strict_reasons:
            strict_violation_count += 1
        else:
            strict_pass_count += 1
        if not strict_reasons and (role >= 0.3 or skills >= 0.1 or _ml_related(profile, row)):
            strict_relevant += 1
    returned_denom = max(returned, 1)
    loose_returned_pass_rate = round(pass_count / returned_denom, 4)
    loose_topk_pass_rate = round(pass_count / denom, 4)
    strict_returned_pass_rate = round(strict_pass_count / returned_denom, 4)
    strict_topk_pass_rate = round(strict_pass_count / denom, 4)
    loose_returned_violation_rate = round(violation_count / returned_denom, 4)
    strict_returned_violation_rate = round(strict_violation_count / returned_denom, 4)
    return {
        "precision_at_10": round(relevant / denom, 4),
        "precision_at_10_loose": round(relevant / denom, 4),
        "persona_pass_rate": loose_topk_pass_rate,
        "persona_pass_rate_loose": loose_topk_pass_rate,
        "loose_returned_pass_rate": loose_returned_pass_rate,
        "loose_topk_pass_rate": loose_topk_pass_rate,
        "dealbreaker_violation_rate": loose_returned_violation_rate,
        "dealbreaker_violation_rate_loose": loose_returned_violation_rate,
        "dealbreaker_violation_rate_topk": round(violation_count / denom, 4),
        "strict_precision_at_10": round(strict_relevant / denom, 4),
        "precision_at_10_strict": round(strict_relevant / denom, 4),
        "strict_persona_pass_rate": strict_topk_pass_rate,
        "persona_pass_rate_strict": strict_topk_pass_rate,
        "strict_returned_pass_rate": strict_returned_pass_rate,
        "strict_topk_pass_rate": strict_topk_pass_rate,
        "strict_dealbreaker_violation_rate": strict_returned_violation_rate,
        "dealbreaker_violation_rate_strict": strict_returned_violation_rate,
        "strict_dealbreaker_violation_rate_topk": round(strict_violation_count / denom, 4),
        "strict_violation_counts": strict_violation_counts,
        "evaluated_rows": returned,
        "returned_jobs": returned,
        "requested_top_k": denominator,
        "topk_completion_rate": round(returned / denom, 4),
        "location_preference_match_rate": round(location_preference_match_count / returned_denom, 4),
        "location_preference_topk_match_rate": round(location_preference_match_count / denom, 4),
        "salary_preference_match_rate": round(salary_preference_match_count / returned_denom, 4),
        "salary_preference_topk_match_rate": round(salary_preference_match_count / denom, 4),
    }


def rows_from_results(job_rows: list[dict[str, Any]], results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return [job_rows[int(result["index"])] for result in results[:top_k]]


def summarize(values: list[dict[str, Any]], key: str) -> float:
    numbers = [float(item[key]) for item in values if key in item]
    return round(mean(numbers), 4) if numbers else 0.0


def merge_violation_counts(values: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        for reason, count in item.get("strict_violation_counts", {}).items():
            counts[reason] = counts.get(reason, 0) + int(count)
    return dict(sorted(counts.items()))


def slim_top_jobs(top_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slimmed: list[dict[str, Any]] = []
    for job in top_jobs:
        slimmed.append(
            {
                "rank": job.get("rank"),
                "job_id": job.get("job_id"),
                "dedup_key": job.get("dedup_key"),
                "description_hash": job.get("description_hash"),
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "salary_raw": job.get("salary_raw"),
                "link": job.get("link"),
                "source": job.get("source"),
                "raw_source": job.get("raw_source"),
                "seniority": job.get("seniority"),
                "years_required": job.get("years_required"),
                "employment_type": job.get("employment_type"),
                "company_type": job.get("company_type"),
                "company_signals": job.get("company_signals"),
                "location_signals": job.get("location_signals"),
                "role_signals": job.get("role_signals"),
                "sponsorship_signal": job.get("sponsorship_signal"),
                "final_score": job.get("final_score"),
                "embedding_similarity": job.get("embedding_similarity"),
                "hard_filter_passed": job.get("hard_filter_passed"),
                "hard_filter_violations": job.get("hard_filter_violations"),
                "matched_skills": job.get("matched_skills"),
                "score_components": job.get("score_components"),
                "penalties": job.get("penalties"),
                "why_ranked": job.get("why_ranked"),
                "application_strategy_label": job.get("application_strategy_label"),
                "same_company_rank": job.get("same_company_rank"),
                "company_application_warning": job.get("company_application_warning"),
                "possible_near_duplicate_role": job.get("possible_near_duplicate_role"),
                "recommended_apply_now": job.get("recommended_apply_now"),
                "also_consider_reason": job.get("also_consider_reason"),
            }
        )
    return slimmed


def main() -> int:
    args = parse_args()
    personas = selected_personas(args)
    store = build_or_load_job_embeddings(
        snapshot_path=args.snapshot,
        cache_dir=args.cache_dir,
        backend=args.embedding_backend,
        rebuild=args.rebuild_embeddings,
    )
    ranker = JobRanker(store)
    baseline = TfidfBaselineRetriever(store.job_rows)

    per_persona: dict[str, Any] = {}
    persona_results: dict[str, Any] = {}
    baseline_metrics: list[dict[str, Any]] = []
    ann_metrics: list[dict[str, Any]] = []
    full_metrics: list[dict[str, Any]] = []

    for persona_name in personas:
        profile = get_persona(persona_name)
        profile_text = profile_to_text(profile)

        baseline_start = time.perf_counter()
        baseline_results = baseline.search(profile_text, top_k=args.top_k)
        baseline_latency = time.perf_counter() - baseline_start
        baseline_rows = rows_from_results(store.job_rows, baseline_results, args.top_k)
        baseline_eval = evaluate_rows(persona_name, profile, baseline_rows, denominator=args.top_k)
        baseline_eval["retrieval_latency_seconds"] = round(baseline_latency, 6)
        baseline_metrics.append(baseline_eval)

        ann_start = time.perf_counter()
        query_embedding = store.embed_text(profile_text)
        ann_results = ranker.retriever.search(query_embedding, top_k=args.top_k)
        ann_latency = time.perf_counter() - ann_start
        ann_rows = rows_from_results(store.job_rows, ann_results, args.top_k)
        ann_eval = evaluate_rows(persona_name, profile, ann_rows, denominator=args.top_k)
        ann_eval["retrieval_latency_seconds"] = round(ann_latency, 6)
        ann_eval["average_similarity"] = round(mean(float(result["similarity"]) for result in ann_results[: args.top_k]), 6)
        ann_metrics.append(ann_eval)

        full_start = time.perf_counter()
        full_result = ranker.rank(profile, top_k=args.top_k, candidate_k=args.candidate_k)
        full_latency = time.perf_counter() - full_start
        full_rows = [
            {
                **job,
                "description_text": job.get("description_text", ""),
            }
            for job in full_result["top_jobs"]
        ]
        full_eval = evaluate_rows(persona_name, profile, full_rows, denominator=args.top_k)
        full_eval["ranking_latency_seconds"] = round(full_latency, 6)
        full_eval["average_score"] = round(mean(float(job["final_score"]) for job in full_result["top_jobs"]), 6) if full_result["top_jobs"] else 0.0
        full_metrics.append(full_eval)

        per_persona[persona_name] = {
            "baseline_tfidf": baseline_eval,
            "embedding_ann": ann_eval,
            "full_multistage_ranking": full_eval,
        }
        persona_results[persona_name] = {
            "profile": profile,
            "metadata": full_result["metadata"],
            "top_jobs": slim_top_jobs(full_result["top_jobs"]),
        }

    benchmarks = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_path": args.snapshot.as_posix(),
        "snapshot_rows": len(store.job_rows),
        "embedding_metadata": store.metadata,
        "ann_backend": ranker.retriever.backend,
        "baseline_fit_seconds": baseline.fit_seconds,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "method_summary": {
            "baseline_tfidf": {
                "mean_returned_jobs": summarize(baseline_metrics, "returned_jobs"),
                "mean_topk_completion_rate": summarize(baseline_metrics, "topk_completion_rate"),
                "mean_precision_at_10": summarize(baseline_metrics, "precision_at_10"),
                "mean_precision_at_10_loose": summarize(baseline_metrics, "precision_at_10_loose"),
                "mean_persona_pass_rate": summarize(baseline_metrics, "persona_pass_rate"),
                "mean_persona_pass_rate_loose": summarize(baseline_metrics, "persona_pass_rate_loose"),
                "mean_loose_returned_pass_rate": summarize(baseline_metrics, "loose_returned_pass_rate"),
                "mean_loose_topk_pass_rate": summarize(baseline_metrics, "loose_topk_pass_rate"),
                "mean_dealbreaker_violation_rate": summarize(baseline_metrics, "dealbreaker_violation_rate"),
                "mean_dealbreaker_violation_rate_loose": summarize(baseline_metrics, "dealbreaker_violation_rate_loose"),
                "mean_strict_precision_at_10": summarize(baseline_metrics, "strict_precision_at_10"),
                "mean_precision_at_10_strict": summarize(baseline_metrics, "precision_at_10_strict"),
                "mean_strict_persona_pass_rate": summarize(baseline_metrics, "strict_persona_pass_rate"),
                "mean_persona_pass_rate_strict": summarize(baseline_metrics, "persona_pass_rate_strict"),
                "mean_strict_returned_pass_rate": summarize(baseline_metrics, "strict_returned_pass_rate"),
                "mean_strict_topk_pass_rate": summarize(baseline_metrics, "strict_topk_pass_rate"),
                "mean_strict_dealbreaker_violation_rate": summarize(baseline_metrics, "strict_dealbreaker_violation_rate"),
                "mean_dealbreaker_violation_rate_strict": summarize(baseline_metrics, "dealbreaker_violation_rate_strict"),
                "mean_location_preference_match_rate": summarize(baseline_metrics, "location_preference_match_rate"),
                "mean_salary_preference_match_rate": summarize(baseline_metrics, "salary_preference_match_rate"),
                "mean_retrieval_latency_seconds": summarize(baseline_metrics, "retrieval_latency_seconds"),
                "strict_violation_counts": merge_violation_counts(baseline_metrics),
            },
            "embedding_ann": {
                "mean_returned_jobs": summarize(ann_metrics, "returned_jobs"),
                "mean_topk_completion_rate": summarize(ann_metrics, "topk_completion_rate"),
                "mean_precision_at_10": summarize(ann_metrics, "precision_at_10"),
                "mean_precision_at_10_loose": summarize(ann_metrics, "precision_at_10_loose"),
                "mean_persona_pass_rate": summarize(ann_metrics, "persona_pass_rate"),
                "mean_persona_pass_rate_loose": summarize(ann_metrics, "persona_pass_rate_loose"),
                "mean_loose_returned_pass_rate": summarize(ann_metrics, "loose_returned_pass_rate"),
                "mean_loose_topk_pass_rate": summarize(ann_metrics, "loose_topk_pass_rate"),
                "mean_dealbreaker_violation_rate": summarize(ann_metrics, "dealbreaker_violation_rate"),
                "mean_dealbreaker_violation_rate_loose": summarize(ann_metrics, "dealbreaker_violation_rate_loose"),
                "mean_strict_precision_at_10": summarize(ann_metrics, "strict_precision_at_10"),
                "mean_precision_at_10_strict": summarize(ann_metrics, "precision_at_10_strict"),
                "mean_strict_persona_pass_rate": summarize(ann_metrics, "strict_persona_pass_rate"),
                "mean_persona_pass_rate_strict": summarize(ann_metrics, "persona_pass_rate_strict"),
                "mean_strict_returned_pass_rate": summarize(ann_metrics, "strict_returned_pass_rate"),
                "mean_strict_topk_pass_rate": summarize(ann_metrics, "strict_topk_pass_rate"),
                "mean_strict_dealbreaker_violation_rate": summarize(ann_metrics, "strict_dealbreaker_violation_rate"),
                "mean_dealbreaker_violation_rate_strict": summarize(ann_metrics, "dealbreaker_violation_rate_strict"),
                "mean_location_preference_match_rate": summarize(ann_metrics, "location_preference_match_rate"),
                "mean_salary_preference_match_rate": summarize(ann_metrics, "salary_preference_match_rate"),
                "mean_retrieval_latency_seconds": summarize(ann_metrics, "retrieval_latency_seconds"),
                "mean_similarity": summarize(ann_metrics, "average_similarity"),
                "strict_violation_counts": merge_violation_counts(ann_metrics),
            },
            "full_multistage_ranking": {
                "mean_returned_jobs": summarize(full_metrics, "returned_jobs"),
                "mean_topk_completion_rate": summarize(full_metrics, "topk_completion_rate"),
                "mean_precision_at_10": summarize(full_metrics, "precision_at_10"),
                "mean_precision_at_10_loose": summarize(full_metrics, "precision_at_10_loose"),
                "mean_persona_pass_rate": summarize(full_metrics, "persona_pass_rate"),
                "mean_persona_pass_rate_loose": summarize(full_metrics, "persona_pass_rate_loose"),
                "mean_loose_returned_pass_rate": summarize(full_metrics, "loose_returned_pass_rate"),
                "mean_loose_topk_pass_rate": summarize(full_metrics, "loose_topk_pass_rate"),
                "mean_dealbreaker_violation_rate": summarize(full_metrics, "dealbreaker_violation_rate"),
                "mean_dealbreaker_violation_rate_loose": summarize(full_metrics, "dealbreaker_violation_rate_loose"),
                "mean_strict_precision_at_10": summarize(full_metrics, "strict_precision_at_10"),
                "mean_precision_at_10_strict": summarize(full_metrics, "precision_at_10_strict"),
                "mean_strict_persona_pass_rate": summarize(full_metrics, "strict_persona_pass_rate"),
                "mean_persona_pass_rate_strict": summarize(full_metrics, "persona_pass_rate_strict"),
                "mean_strict_returned_pass_rate": summarize(full_metrics, "strict_returned_pass_rate"),
                "mean_strict_topk_pass_rate": summarize(full_metrics, "strict_topk_pass_rate"),
                "mean_strict_dealbreaker_violation_rate": summarize(full_metrics, "strict_dealbreaker_violation_rate"),
                "mean_dealbreaker_violation_rate_strict": summarize(full_metrics, "dealbreaker_violation_rate_strict"),
                "mean_location_preference_match_rate": summarize(full_metrics, "location_preference_match_rate"),
                "mean_salary_preference_match_rate": summarize(full_metrics, "salary_preference_match_rate"),
                "mean_ranking_latency_seconds": summarize(full_metrics, "ranking_latency_seconds"),
                "mean_average_score": summarize(full_metrics, "average_score"),
                "strict_violation_counts": merge_violation_counts(full_metrics),
            },
        },
        "per_persona": per_persona,
    }

    write_json(args.benchmarks_output, benchmarks)
    write_json(args.persona_output, persona_results)

    print("Phase 2 benchmarks complete")
    print(f"Benchmarks: {args.benchmarks_output}")
    print(f"Persona results: {args.persona_output}")
    print(json.dumps(benchmarks["method_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
