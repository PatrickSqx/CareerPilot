"""Run Phase 2.12 backend persona diagnostics without tuning ranking rules."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_CSV, PROCESSED_DATA_DIR  # noqa: E402
from jobpilot.ingestion.dedup import add_dedup_fields  # noqa: E402
from jobpilot.live.adzuna_live import LIVE_CACHE_DIR, fetch_adzuna_live, sanitize_live_payload  # noqa: E402
from jobpilot.live.hybrid_pool import CandidatePool  # noqa: E402
from jobpilot.live.jsearch_live import fetch_jsearch_live  # noqa: E402
from jobpilot.live.query_builder import build_live_queries  # noqa: E402
from jobpilot.profile.personas import PERSONA_FIXTURES, get_persona  # noqa: E402
from jobpilot.profile.profile_parser import normalize_list  # noqa: E402
from jobpilot.ranking.company_signals import detect_company_signals  # noqa: E402
from jobpilot.ranking.filters import (  # noqa: E402
    apply_hard_filters,
    configured_seniority_term_hits,
    junior_level_hits,
    parse_float,
    seniority_level_hits,
)
from jobpilot.ranking.location_signals import is_us_location, location_violation_reason  # noqa: E402
from jobpilot.ranking.ranker import JobRanker  # noqa: E402
from jobpilot.ranking.role_signals import (  # noqa: E402
    detect_role_family_signals,
    generic_backend_devops_without_target_signal,
    role_family_match_details,
    target_role_relevance_score,
    title_contains_profile_signal,
    weak_non_ml_title_hits,
)
from jobpilot.ranking.scoring import matched_skills  # noqa: E402
from jobpilot.retrieval.embeddings import EmbeddingStore, build_or_load_job_embeddings, job_embedding_text  # noqa: E402
from jobpilot.schemas import CANONICAL_COLUMNS  # noqa: E402
from jobpilot.utils.io import write_csv, write_json  # noqa: E402
from jobpilot.utils.text import clean_text  # noqa: E402


DIAGNOSTICS_JSON = PROCESSED_DATA_DIR / "persona_backend_diagnostics.json"
FUNNELS_JSON = PROCESSED_DATA_DIR / "persona_candidate_funnels.json"
LIVE_QUERY_AUDIT_JSON = PROCESSED_DATA_DIR / "persona_live_query_audit.json"
TOP10_FLAGS_JSON = PROCESSED_DATA_DIR / "persona_top10_quality_flags.json"
DIAGNOSTICS_MD = PROCESSED_DATA_DIR / "persona_backend_diagnostics.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JobPilot Phase 2.13 backend persona diagnostics.")
    parser.add_argument("--all-personas", action="store_true", help="Run Aisha, Marcus, Priya, and Kenji.")
    parser.add_argument("--personas", nargs="*", choices=sorted(PERSONA_FIXTURES), default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--mode", choices=["offline", "live", "hybrid"], default="offline")
    parser.add_argument("--provider", choices=["adzuna", "jsearch"], default="adzuna")
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--results-per-page", type=int, default=20)
    parser.add_argument("--country", default="us")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=OFFLINE_SNAPSHOT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "tfidf-svd"], default="auto")
    return parser.parse_args()


def selected_personas(args: argparse.Namespace) -> list[str]:
    if args.all_personas or not args.personas:
        return sorted(PERSONA_FIXTURES)
    return args.personas


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def dedupe_live_rows(
    live_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    offline_keys = {str(row.get("dedup_key", "")) for row in offline_rows if row.get("dedup_key")}
    seen_live: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicate_offline = 0
    duplicate_live = 0
    for row in live_rows:
        normalized = add_dedup_fields(row)
        key = str(normalized.get("dedup_key", ""))
        if key and key in offline_keys:
            duplicate_offline += 1
            continue
        if key and key in seen_live:
            duplicate_live += 1
            continue
        if key:
            seen_live.add(key)
        unique.append(normalized)
    return unique, {
        "duplicates_against_offline": duplicate_offline,
        "duplicates_within_live": duplicate_live,
        "live_records_retained": len(unique),
    }


def store_from_rows(base_store: EmbeddingStore, rows: list[dict[str, Any]], embeddings: np.ndarray, *, mode_used: str) -> EmbeddingStore:
    metadata = dict(base_store.metadata)
    metadata.update(
        {
            "candidate_pool_mode": mode_used,
            "number_embedded": int(embeddings.shape[0]),
            "row_count": len(rows),
        }
    )
    return EmbeddingStore(
        job_rows=rows,
        job_ids=[str(row.get("job_id", "")) for row in rows],
        embeddings=embeddings.astype(np.float32),
        metadata=metadata,
        cache_dir=base_store.cache_dir,
        transformer=base_store.transformer,
        vectorizer=base_store.vectorizer,
        svd=base_store.svd,
    )


def build_candidate_pool_from_base(
    base_store: EmbeddingStore,
    *,
    mode: str,
    live_rows: list[dict[str, Any]],
    fallback_to_offline: bool = True,
) -> CandidatePool:
    mode = mode.lower()
    offline_rows = base_store.job_rows
    unique_live, dedup_metadata = dedupe_live_rows(live_rows, offline_rows)
    warnings: list[str] = []
    if mode == "offline":
        mode_used = "offline"
        store = base_store
        selected_live: list[dict[str, Any]] = []
    elif not unique_live and fallback_to_offline:
        warnings.append("No live records available; using offline fallback.")
        mode_used = "offline_fallback"
        store = base_store
        selected_live = []
    else:
        live_embeddings = np.asarray(
            [base_store.embed_text(job_embedding_text(row)) for row in unique_live],
            dtype=np.float32,
        )
        if mode == "live":
            mode_used = "live"
            store = store_from_rows(base_store, unique_live, live_embeddings, mode_used=mode_used)
        else:
            mode_used = "hybrid"
            combined_rows = [*offline_rows, *unique_live]
            combined_embeddings = np.vstack([base_store.embeddings.astype(np.float32), live_embeddings])
            store = store_from_rows(base_store, combined_rows, combined_embeddings, mode_used=mode_used)
        selected_live = unique_live
    metadata = {
        "mode_requested": mode,
        "mode_used": mode_used,
        "offline_rows": len(offline_rows),
        "live_rows_input": len(live_rows),
        **dedup_metadata,
        "candidate_rows": len(store.job_rows),
        "warnings": warnings,
    }
    return CandidatePool(mode, mode_used, store, offline_rows, selected_live, metadata)


def job_text(row: dict[str, Any]) -> str:
    return " ".join(
        clean_text(row.get(key))
        for key in ["title", "company", "employer", "location", "employment_type", "company_type", "sponsorship_signal", "description_text"]
    ).lower()


def employment_has(row: dict[str, Any], terms: set[str]) -> bool:
    employment = clean_text(row.get("employment_type")).lower()
    text = job_text(row)
    return any(term in employment or term in text for term in terms)


def years(row: dict[str, Any]) -> float | None:
    return parse_float(row.get("years_required"))


def salary_bounds(row: dict[str, Any]) -> tuple[float | None, float | None]:
    salary_min = parse_float(row.get("salary_min"))
    salary_max = parse_float(row.get("salary_max")) or salary_min
    return salary_min, salary_max


def salary_meets(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    desired = parse_float(profile.get("salary_min"))
    if not desired:
        return True
    _, salary_max = salary_bounds(row)
    return bool(salary_max is not None and salary_max >= desired)


def salary_missing(row: dict[str, Any]) -> bool:
    salary_min, salary_max = salary_bounds(row)
    return salary_min is None and salary_max is None


def location_matches(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    return location_violation_reason(row, profile) is None


def role_families(row: dict[str, Any]) -> list[str]:
    signals = detect_role_family_signals(row)
    return list(signals.get("detected_families", []))


def title_role_families(row: dict[str, Any]) -> list[str]:
    signals = detect_role_family_signals(row)
    return list(signals.get("title_families", []))


def family_hit(row: dict[str, Any], families: set[str], *, title_only: bool = True) -> bool:
    detected = set(title_role_families(row) if title_only else role_families(row))
    return bool(detected & families)


def has_senior_or_lead_risk(row: dict[str, Any]) -> bool:
    seniority = clean_text(row.get("seniority")).lower()
    return bool(seniority_level_hits(row)) or seniority in {"senior", "staff_principal", "lead_manager"}


def has_junior_risk(row: dict[str, Any]) -> bool:
    seniority = clean_text(row.get("seniority")).lower()
    return bool(junior_level_hits(row)) or seniority in {"entry_junior", "internship"}


def company_startup_risk(row: dict[str, Any]) -> bool:
    text = job_text(row)
    company_type = clean_text(row.get("company_type")).lower()
    return company_type == "startup" or any(term in text for term in ["tiny startup", "small startup", "seed stage", "series a"])


def generic_devops_backend_risk(row: dict[str, Any]) -> bool:
    title = clean_text(row.get("title")).lower()
    families = set(role_families(row))
    terms = ["devops", "backend", "cloud engineer", "software engineer", "java developer", "python developer"]
    return bool({"software_backend", "data_engineering"} & families) or any(term in title for term in terms)


def generic_devops_backend_mismatch(profile: dict[str, Any], row: dict[str, Any]) -> bool:
    return bool(profile.get("avoid_generic_backend_devops")) and generic_backend_devops_without_target_signal(profile, row)


def sponsor_or_large_company_signal(row: dict[str, Any]) -> bool:
    company_type = clean_text(row.get("company_type")).lower()
    sponsorship_signal = clean_text(row.get("sponsorship_signal")).lower()
    company_signals = detect_company_signals(row)
    return (
        sponsorship_signal == "mentions_sponsorship_or_work_auth"
        or company_type in {"large_company", "research_lab"}
        or bool(company_signals["sponsor_friendly_proxy"])
    )


def strict_persona_reasons(persona_name: str, profile: dict[str, Any], row: dict[str, Any]) -> list[str]:
    name = persona_name.lower()
    reasons: list[str] = []
    text = job_text(row)
    company_signals = detect_company_signals(row)
    seniority = clean_text(row.get("seniority")).lower()
    sponsorship_signal = clean_text(row.get("sponsorship_signal")).lower()
    company_type = clean_text(row.get("company_type")).lower()

    if name == "aisha":
        if has_senior_or_lead_risk(row) or seniority in {"senior", "staff_principal", "lead_manager"}:
            reasons.append("senior_staff_lead_principal")
        if company_signals["defense_government_contractor"] or any(term in text for term in ["defense", "military", " dod "]):
            reasons.append("defense_military")
        required_years = years(row)
        if required_years is not None and required_years >= 5:
            reasons.append("years_5_plus")
        if not family_hit(row, {"ml_related", "research_ai"}, title_only=True):
            reasons.append("not_ml_related")
    elif name == "marcus":
        location_reason = location_violation_reason(row, profile)
        if location_reason:
            reasons.append(location_reason)
        if has_senior_or_lead_risk(row):
            reasons.append("senior_or_lead_role")
        required_years = years(row)
        if required_years is not None and required_years >= 3:
            reasons.append("years_3_plus")
        if employment_has(row, {"contract", "temporary", "temp", "unpaid"}):
            reasons.append("contract_temp_unpaid")
    elif name == "priya":
        location_reason = location_violation_reason(row, profile)
        if location_reason:
            reasons.append(location_reason)
        if has_junior_risk(row):
            reasons.append("junior_or_intern")
        if company_startup_risk(row):
            reasons.append("startup_or_tiny_startup")
        if profile.get("avoid_defense_or_clearance") and company_signals["defense_government_contractor"]:
            reasons.append("defense_or_clearance")
        if generic_devops_backend_mismatch(profile, row):
            reasons.append("generic_backend_devops_without_ml_infra_title_signal")
        title_families = set(title_role_families(row))
        if not (title_families & {"ml_infra", "ml_related"} and title_contains_profile_signal(profile, row)):
            reasons.append("not_ml_ai_infra_related")
    elif name == "kenji":
        location_reason = location_violation_reason(row, profile)
        if location_reason:
            reasons.append(location_reason)
        if employment_has(row, {"contract", "temporary", "temp", "unpaid"}):
            reasons.append("contract_temp_unpaid")
        if sponsorship_signal == "no_sponsorship":
            reasons.append("no_sponsorship")
        if configured_seniority_term_hits(profile, row, "hard_reject_seniority_terms"):
            reasons.append("overly_senior_hard_reject")
    return reasons


def stage_count(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    return sum(1 for row in rows if predicate(row))


def cumulative_funnel(rows: list[dict[str, Any]], stages: list[tuple[str, Callable[[dict[str, Any]], bool]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current = list(rows)
    output: list[dict[str, Any]] = [{"stage": "total_candidate_jobs", "remaining": len(current), "removed": 0}]
    for label, predicate in stages:
        before = len(current)
        current = [row for row in current if predicate(row)]
        output.append({"stage": label, "remaining": len(current), "removed": before - len(current)})
    return output, current


def base_funnel(persona_name: str, profile: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    desired_salary = parse_float(profile.get("salary_min")) or 0.0
    if persona_name == "aisha":
        stages = [
            ("ml_research_role_family_title_match", lambda row: family_hit(row, {"ml_related", "research_ai"}, title_only=True)),
            ("no_senior_staff_lead_principal", lambda row: not has_senior_or_lead_risk(row)),
            ("no_5_plus_years_requirement", lambda row: years(row) is None or years(row) < 5),
            ("no_defense_government_contractor_signal", lambda row: not detect_company_signals(row)["defense_government_contractor"]),
        ]
        funnel, eligible = cumulative_funnel(rows, stages)
        return {
            "persona": persona_name,
            "funnel": funnel,
            "soft_preference_counts_after_default_eligible": {
                "location_remote_or_bay_area": stage_count(eligible, lambda row: location_matches(profile, row)),
                "salary_meets_min": stage_count(eligible, lambda row: salary_meets(profile, row)),
                "salary_missing_or_unknown": stage_count(eligible, salary_missing),
            },
            "final_eligible_count_default": len(eligible),
            "final_eligible_count_if_location_strict": stage_count(eligible, lambda row: location_matches(profile, row)),
            "final_eligible_count_if_salary_strict": stage_count(eligible, lambda row: salary_meets(profile, row)),
            "final_eligible_count_if_location_and_salary_strict": stage_count(eligible, lambda row: location_matches(profile, row) and salary_meets(profile, row)),
        }
    if persona_name == "marcus":
        stages = [
            ("analytics_or_bi_role_family_match", lambda row: family_hit(row, {"analytics_entry", "bi_analytics"}, title_only=False) or target_role_relevance_score(profile, row) >= 0.3),
            ("no_senior_or_lead_title", lambda row: not has_senior_or_lead_risk(row)),
            ("no_3_plus_years_requirement", lambda row: years(row) is None or years(row) < 3),
            ("no_contract_temp_unpaid", lambda row: not employment_has(row, {"contract", "temporary", "temp", "unpaid"})),
            ("us_compatible_location", lambda row: location_matches(profile, row)),
        ]
        funnel, eligible = cumulative_funnel(rows, stages)
        return {
            "persona": persona_name,
            "funnel": funnel,
            "soft_preference_counts_after_default_eligible": {
                "salary_meets_80000": stage_count(eligible, lambda row: salary_meets(profile, row)),
                "salary_missing_or_unknown": stage_count(eligible, salary_missing),
                "tech_or_healthcare_detectable": stage_count(eligible, lambda row: any(term in job_text(row) for term in ["technology", "tech", "software", "healthcare", "health care", "hospital"])),
            },
            "final_eligible_count": len(eligible),
            "desired_salary": desired_salary,
        }
    if persona_name == "priya":
        stages = [
            (
                "ml_or_ml_infra_title_family_match",
                lambda row: bool(set(title_role_families(row)) & {"ml_infra", "ml_related"})
                and title_contains_profile_signal(profile, row),
            ),
            ("no_junior_title", lambda row: not has_junior_risk(row)),
            ("us_only_location_pass", lambda row: location_matches(profile, row)),
            ("no_tiny_startup_company_size_risk", lambda row: not company_startup_risk(row)),
            ("no_defense_or_clearance_signal", lambda row: not detect_company_signals(row)["defense_government_contractor"]),
            ("no_generic_backend_devops_without_ml_signal", lambda row: not generic_devops_backend_mismatch(profile, row)),
        ]
        funnel, eligible = cumulative_funnel(rows, stages)
        return {
            "persona": persona_name,
            "funnel": funnel,
            "soft_preference_counts_after_default_eligible": {
                "generic_backend_devops_risk": stage_count(eligible, generic_devops_backend_risk),
                "salary_meets_200000": stage_count(eligible, lambda row: salary_meets(profile, row)),
                "salary_missing_or_unknown": stage_count(eligible, salary_missing),
            },
            "final_eligible_count": len(eligible),
            "desired_salary": desired_salary,
        }
    stages = [
        ("ml_research_ai_role_family_match", lambda row: family_hit(row, {"ml_related", "research_ai"}, title_only=False) or target_role_relevance_score(profile, row) >= 0.3),
        ("us_only_location_pass", lambda row: location_matches(profile, row)),
        ("no_contract_temp", lambda row: not employment_has(row, {"contract", "temporary", "temp", "unpaid"})),
        ("no_no_sponsorship_signal", lambda row: clean_text(row.get("sponsorship_signal")).lower() != "no_sponsorship"),
        ("no_staff_principal_distinguished_lead_manager", lambda row: not configured_seniority_term_hits(profile, row, "hard_reject_seniority_terms")),
    ]
    funnel, eligible = cumulative_funnel(rows, stages)
    return {
        "persona": persona_name,
        "funnel": funnel,
        "soft_preference_counts_after_default_eligible": {
            "sponsor_friendly_large_company_or_research_signal": stage_count(eligible, sponsor_or_large_company_signal),
            "sponsorship_unknown": stage_count(eligible, lambda row: clean_text(row.get("sponsorship_signal")).lower() == "unknown"),
            "overly_senior_hard_reject_risk": stage_count(eligible, lambda row: bool(configured_seniority_term_hits(profile, row, "hard_reject_seniority_terms"))),
            "seniority_realism_penalty_risk": stage_count(eligible, lambda row: bool(configured_seniority_term_hits(profile, row, "penalize_seniority_terms"))),
            "salary_meets_120000": stage_count(eligible, lambda row: salary_meets(profile, row)),
            "salary_missing_or_unknown": stage_count(eligible, salary_missing),
        },
        "final_eligible_count": len(eligible),
        "desired_salary": desired_salary,
    }


def query_audit_for_persona(persona_name: str, profile: dict[str, Any], queries: list[str], *, mode: str, dry_run: bool, live_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    joined = " | ".join(queries).lower()
    target_roles = [role.lower() for role in normalize_list(profile.get("target_roles"))]
    audit: dict[str, Any] = {
        "persona": persona_name,
        "mode": mode,
        "dry_run": dry_run,
        "queries": queries,
        "query_count": len(queries),
        "target_roles_covered": [role for role in target_roles if any(part in joined for part in role.split("/") or [role]) or role in joined],
        "contains_remote": "remote" in joined,
        "contains_us_or_city": any(term in joined for term in ["united states", "new york", "chicago", "boston", "seattle", "san francisco", "san jose", "bay area", "california"]),
        "senior_heavy_terms": [term for term in ["senior", "staff", "principal", "lead", "director"] if term in joined],
        "live_metadata": live_metadata or {},
        "warnings": [],
    }
    warnings = audit["warnings"]
    if persona_name == "aisha":
        if "remote" not in joined:
            warnings.append("Aisha queries do not include remote.")
        if not any(term in joined for term in ["san francisco", "san jose", "bay area", "california"]):
            warnings.append("Aisha queries do not cover Bay Area or California location terms.")
        if not any(term in joined for term in ["machine learning engineer", "ml engineer"]):
            warnings.append("Aisha queries miss machine learning engineer variants.")
        if "applied scientist" not in joined or "data scientist" not in joined:
            warnings.append("Aisha queries miss applied scientist or data scientist coverage.")
    elif persona_name == "marcus":
        if audit["senior_heavy_terms"]:
            warnings.append("Marcus queries include senior-heavy terms.")
        if not any(term in joined for term in ["data analyst", "business analyst", "analytics engineer", "bi analyst"]):
            warnings.append("Marcus queries may not cover entry analytics roles.")
    elif persona_name == "priya":
        if not any(term in joined for term in ["mlops", "machine learning platform", "ml infrastructure", "ai infrastructure"]):
            warnings.append("Priya queries may be too generic for ML infrastructure.")
        if any(term in joined for term in ["backend engineer", "software engineer", "devops"]) and "mlops" not in joined:
            warnings.append("Priya queries risk generic backend/DevOps drift.")
    elif persona_name == "kenji":
        if not any(term in joined for term in ["research scientist", "applied scientist", "machine learning", "data scientist"]):
            warnings.append("Kenji queries may miss AI/ML/research roles.")
        if not any(term in joined for term in ["remote", "seattle", "san francisco", "new york", "united states"]):
            warnings.append("Kenji queries may miss US/remote location coverage.")
        warnings.append("Query builder does not target sponsor-friendly companies directly; company proxy scoring happens after retrieval.")
    return audit


def diagnostic_flags(persona_name: str, profile: dict[str, Any], job: dict[str, Any], company_counts: Counter[str]) -> dict[str, bool]:
    role_details = role_family_match_details(profile, job)
    salary_min, salary_max = salary_bounds(job)
    desired_salary = parse_float(profile.get("salary_min"))
    sponsorship_signal = clean_text(job.get("sponsorship_signal")).lower()
    company = clean_text(job.get("company") or job.get("employer")).lower()
    company_signals = detect_company_signals(job)
    seniority_hard_hits = configured_seniority_term_hits(profile, job, "hard_reject_seniority_terms")
    seniority_soft_hits = configured_seniority_term_hits(profile, job, "penalize_seniority_terms")
    flags = {
        "non_target_role_family": target_role_relevance_score(profile, job) < 0.3 and not role_details["preferred_title_matches"] and not role_details["required_title_matches"],
        "seniority_risk": has_senior_or_lead_risk(job),
        "years_required_risk": False,
        "location_preference_miss": location_violation_reason(job, profile) is not None,
        "salary_missing": salary_missing(job),
        "salary_below_preference": bool(desired_salary and salary_max is not None and salary_max < desired_salary),
        "defense_or_government_contractor_risk": bool(company_signals["defense_government_contractor"]),
        "generic_devops_backend_risk": generic_devops_backend_mismatch(profile, job),
        "sponsorship_unknown": sponsorship_signal == "unknown",
        "sponsor_or_large_company_proxy": bool(company_signals["sponsor_friendly_proxy"]),
        "overly_senior_for_new_grad_or_student": bool(seniority_hard_hits or seniority_soft_hits),
        "same_company_concentration": bool(company and company_counts[company] > 1),
        "same_company_alternative": clean_text(job.get("application_strategy_label")) == "Same-company alternative",
        "possible_near_duplicate_role": bool(job.get("possible_near_duplicate_role")),
        "duplicate_or_near_duplicate_company": bool(job.get("possible_near_duplicate_role")),
    }
    required_years = years(job)
    if persona_name == "aisha":
        flags["years_required_risk"] = required_years is not None and required_years >= 5
    elif persona_name == "marcus":
        flags["years_required_risk"] = required_years is not None and required_years >= 3
    elif persona_name == "kenji":
        flags["overly_senior_for_new_grad_or_student"] = bool(seniority_hard_hits or seniority_soft_hits)
    if persona_name != "priya":
        flags["generic_devops_backend_risk"] = False
    return flags


def soft_preference_notes(profile: dict[str, Any], job: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    location_reason = location_violation_reason(job, profile)
    if location_reason:
        notes.append(f"location_preference_miss:{location_reason}")
    desired = parse_float(profile.get("salary_min"))
    _, salary_max = salary_bounds(job)
    if desired and salary_max is None:
        notes.append("salary_missing")
    elif desired and salary_max < desired:
        notes.append(f"salary_below_preference:{int(salary_max)}<{int(desired)}")
    if profile.get("needs_sponsorship") and clean_text(job.get("sponsorship_signal")).lower() == "unknown":
        if sponsor_or_large_company_signal(job):
            notes.append("sponsorship_unknown_but_company_proxy_available")
        else:
            notes.append("sponsorship_unknown_no_company_proxy")
    return notes


def top10_quality(persona_name: str, profile: dict[str, Any], top_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    company_counts = Counter(clean_text(job.get("company") or job.get("employer")).lower() for job in top_jobs)
    label_counts = Counter(clean_text(job.get("application_strategy_label")) or "unlabeled" for job in top_jobs)
    items: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    for job in top_jobs:
        flags = diagnostic_flags(persona_name, profile, job, company_counts)
        flag_counts.update([key for key, value in flags.items() if value])
        items.append(
            {
                "rank": job.get("rank"),
                "job_id": job.get("job_id", ""),
                "dedup_key": job.get("dedup_key", ""),
                "description_hash": job.get("description_hash", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "source": job.get("source", ""),
                "raw_source": job.get("raw_source", ""),
                "salary_raw": job.get("salary_raw", ""),
                "salary_min": job.get("salary_min", ""),
                "salary_max": job.get("salary_max", ""),
                "final_score": job.get("final_score", ""),
                "role_families_detected": role_families(job),
                "matched_skills": job.get("matched_skills") or matched_skills(profile, job),
                "seniority": job.get("seniority", ""),
                "years_required": job.get("years_required", ""),
                "employment_type": job.get("employment_type", ""),
                "company_type": job.get("company_type", ""),
                "sponsorship_signal": job.get("sponsorship_signal", ""),
                "hard_filter_violations": job.get("hard_filter_violations", []),
                "application_strategy_label": job.get("application_strategy_label", ""),
                "same_company_rank": job.get("same_company_rank", 0),
                "company_application_warning": job.get("company_application_warning", ""),
                "possible_near_duplicate_role": job.get("possible_near_duplicate_role", False),
                "recommended_apply_now": job.get("recommended_apply_now", False),
                "also_consider_reason": job.get("also_consider_reason", ""),
                "soft_preference_notes": soft_preference_notes(profile, job),
                "score_components": job.get("score_components", {}),
                "why_ranked": job.get("why_ranked", {}),
                "diagnostic_flags": flags,
                "strict_persona_reasons": strict_persona_reasons(persona_name, profile, job),
            }
        )
    return {
        "persona": persona_name,
        "top10": items,
        "flag_counts": dict(sorted(flag_counts.items())),
        "application_strategy_label_counts": dict(sorted(label_counts.items())),
        "same_company_counts": dict(sorted((company, count) for company, count in company_counts.items() if company)),
        "top10_count": len(items),
    }


def source_group_counts(top_jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for job in top_jobs:
        source = clean_text(job.get("source")).lower()
        raw_source = clean_text(job.get("raw_source")).lower()
        counts["live" if source.endswith("_live") or "_live" in raw_source else "offline"] += 1
    return dict(counts)


def metrics_for_top10(persona_name: str, profile: dict[str, Any], top_jobs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    strict_pass = sum(1 for job in top_jobs if not strict_persona_reasons(persona_name, profile, job))
    hard_pass = sum(1 for job in top_jobs if apply_hard_filters(profile, job).passed)
    location_pref = sum(1 for job in top_jobs if location_matches(profile, job))
    salary_pref = sum(1 for job in top_jobs if salary_meets(profile, job))
    return {
        "returned_jobs": len(top_jobs),
        "requested_top_k": top_k,
        "topk_completion_rate": round(len(top_jobs) / max(top_k, 1), 4),
        "hard_filter_pass_rate": round(hard_pass / max(len(top_jobs), 1), 4),
        "strict_topk_pass_rate": round(strict_pass / max(top_k, 1), 4),
        "strict_returned_pass_rate": round(strict_pass / max(len(top_jobs), 1), 4),
        "location_preference_match_rate": round(location_pref / max(len(top_jobs), 1), 4),
        "salary_preference_match_rate": round(salary_pref / max(len(top_jobs), 1), 4),
        "source_group_counts_top_k": source_group_counts(top_jobs),
    }


def classify_root_causes(
    persona_name: str,
    funnel: dict[str, Any],
    top_quality: dict[str, Any],
    query_audit: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    flag_counts = top_quality.get("flag_counts", {})
    final_count = int(funnel.get("final_eligible_count", funnel.get("final_eligible_count_default", 0)))

    def add(category: str, severity: str, evidence: str, action: str, timing: str) -> None:
        issues.append(
            {
                "category": category,
                "severity": severity,
                "evidence": evidence,
                "suggested_next_action": action,
                "timing": timing,
            }
        )

    if final_count < 10:
        add(
            "hard_filter_too_strict",
            "high",
            f"Only {final_count} final eligible candidates in the candidate universe.",
            "Review whether the persona's hard constraints are intended or whether more targeted live queries are needed.",
            "must_fix_before_phase3",
        )
    elif final_count < 50:
        add(
            "data_coverage_limitation",
            "medium",
            f"Only {final_count} final eligible candidates after persona constraints.",
            "Use UI copy and optional live refresh to explain limited coverage; consider targeted current-posting ingestion later.",
            "can_handle_in_phase3",
        )
    if query_audit.get("warnings"):
        add(
            "live_query_generation_gap",
            "medium",
            "; ".join(query_audit["warnings"][:3]),
            "Improve query builder only if live search is needed for this persona's demo path.",
            "can_handle_in_phase3",
        )
    if flag_counts.get("non_target_role_family"):
        add(
            "role_family_detection_issue",
            "high" if flag_counts["non_target_role_family"] >= 3 else "medium",
            f"Top 10 has {flag_counts['non_target_role_family']} non-target role-family flags.",
            "Audit title-family detection and scoring before relying on this persona in the UI.",
            "must_fix_before_phase3",
        )
    if flag_counts.get("generic_devops_backend_risk"):
        generic_count = int(flag_counts["generic_devops_backend_risk"])
        severity = "high" if persona_name == "priya" and generic_count >= 3 else "medium"
        timing = "must_fix_before_phase3" if persona_name == "priya" and generic_count >= 3 else "can_handle_in_phase3"
        add(
            "scoring_weight_issue",
            severity,
            f"Top 10 has {generic_count} generic DevOps/backend risk flags.",
            "Keep ML-infrastructure title relevance visible in the UI and consider later scoring adjustment.",
            timing,
        )
    if flag_counts.get("defense_or_government_contractor_risk"):
        defense_is_hard = persona_name == "aisha" or bool(get_persona(persona_name).get("avoid_defense_or_clearance"))
        severity = "high" if defense_is_hard else "medium"
        timing = "must_fix_before_phase3" if defense_is_hard else "can_document_as_limitation"
        add(
            "company_signal_detection_issue",
            severity,
            f"Top 10 has {flag_counts['defense_or_government_contractor_risk']} defense/government-contractor risk flags.",
            "Treat as a hard backend blocker only for personas that explicitly exclude defense/government contractors; otherwise surface as a suitability note.",
            timing,
        )
    if flag_counts.get("overly_senior_for_new_grad_or_student"):
        senior_count = int(flag_counts["overly_senior_for_new_grad_or_student"])
        severity = "high" if senior_count >= 5 else "medium"
        timing = "must_fix_before_phase3" if persona_name in {"kenji", "marcus"} and senior_count >= 5 else "can_handle_in_phase3"
        add(
            "scoring_weight_issue",
            severity,
            f"Top 10 has {senior_count} roles that look overly senior for a new graduate or student profile.",
            "Add or strengthen student/new-grad seniority preferences before using this persona as a polished UI demo.",
            timing,
        )
    if persona_name == "kenji" and flag_counts.get("sponsorship_unknown"):
        add(
            "sponsorship_metadata_sparse",
            "medium",
            f"Top 10 has {flag_counts['sponsorship_unknown']} sponsorship_unknown flags.",
            "Expose uncertainty in UI; do not claim confirmed sponsorship unless metadata says so.",
            "can_handle_in_phase3",
        )
    if flag_counts.get("possible_near_duplicate_role", 0):
        add(
            "potential_duplicate_role",
            "medium",
            f"Top 10 has {flag_counts['possible_near_duplicate_role']} possible near-duplicate same-company role flags.",
            "Review exact duplicate keys and keep lower-ranked duplicate-like roles as alternatives rather than primary apply-now items.",
            "can_handle_in_phase3",
        )
    if flag_counts.get("same_company_concentration", 0) >= 5:
        add(
            "application_strategy_note",
            "medium",
            f"Top 10 has {flag_counts['same_company_concentration']} jobs from companies that appear more than once.",
            "Use application strategy labels to apply to the best 1-2 roles per company first; repeated companies are not exact duplicate postings.",
            "can_handle_in_phase3",
        )
    if flag_counts.get("salary_missing", 0) >= max(3, metrics.get("returned_jobs", 0) // 2):
        add(
            "salary_metadata_sparse",
            "medium",
            f"Top 10 has {flag_counts['salary_missing']} jobs with missing salary.",
            "Keep salary as a preference/unknown state in UI and avoid treating missing salary as a rejection by default.",
            "can_handle_in_phase3",
        )
    if flag_counts.get("location_preference_miss") and metrics.get("strict_topk_pass_rate", 0) >= 1.0:
        add(
            "evaluator_too_lenient",
            "low",
            "Strict evaluator passes jobs while soft location preference misses remain.",
            "Document preference metrics separately from strict pass metrics.",
            "can_document_as_limitation",
        )
    if not issues:
        add(
            "no_material_issue",
            "low",
            "Top 10 has no material diagnostic flags and candidate funnel has sufficient coverage.",
            "Proceed to Phase 3 UI integration for this persona.",
            "can_handle_in_phase3",
        )
    return issues


def executive_status(root_causes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    must_fix: list[str] = []
    safe: list[str] = []
    phase3: list[str] = []
    for persona, issues in root_causes.items():
        severe = [issue for issue in issues if issue["timing"] == "must_fix_before_phase3"]
        if severe:
            must_fix.append(persona)
        else:
            safe.append(persona)
        if any(issue["timing"] == "can_handle_in_phase3" for issue in issues):
            phase3.append(persona)
    return {
        "backend_ready_for_phase3": not must_fix,
        "personas_safe_for_phase3": safe,
        "personas_with_backend_fixes_recommended": must_fix,
        "personas_with_phase3_ui_or_learning_items": phase3,
    }


def render_funnel_table(funnel: dict[str, Any]) -> list[str]:
    lines = ["| Stage | Remaining | Removed |", "|---|---:|---:|"]
    for row in funnel.get("funnel", []):
        lines.append(f"| {row['stage']} | {row['remaining']} | {row['removed']} |")
    for key, value in funnel.items():
        if key.startswith("final_eligible"):
            lines.append(f"| {key} | {value} |  |")
    return lines


def render_markdown(
    diagnostics: dict[str, Any],
    funnels: dict[str, Any],
    query_audits: dict[str, Any],
    top_quality: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# JobPilot Phase 2.13 Backend Quality Diagnostic Audit",
        "",
        "## Executive Summary",
        "",
    ]
    summary = diagnostics["executive_summary"]
    lines.append(f"- Backend ready for Phase 3: `{summary['backend_ready_for_phase3']}`")
    lines.append(f"- Personas safe for Phase 3: {', '.join(summary['personas_safe_for_phase3']) or 'none'}")
    lines.append(f"- Personas with backend fixes recommended: {', '.join(summary['personas_with_backend_fixes_recommended']) or 'none'}")
    lines.append(f"- Personas with Phase 3 UI/adaptive-learning items: {', '.join(summary['personas_with_phase3_ui_or_learning_items']) or 'none'}")
    lines.append("")
    lines.append("This audit validates Phase 2 backend ranking quality only. It does not implement Streamlit UI, resume generation, adaptive learning, Phase 1 ingestion changes, or deployment.")
    lines.append("")
    for persona in diagnostics["personas"]:
        name = persona["persona"]
        lines.extend([f"## {name.title()} Diagnosis", ""])
        lines.append(f"- Mode used: `{persona['mode_used']}`")
        lines.append(f"- Candidate rows: `{persona['candidate_pool']['candidate_rows']}`")
        lines.append(f"- Returned jobs: `{persona['metrics']['returned_jobs']}`")
        lines.append(f"- Strict top-k pass rate: `{persona['metrics']['strict_topk_pass_rate']}`")
        lines.append(f"- Location preference match rate: `{persona['metrics']['location_preference_match_rate']}`")
        lines.append(f"- Salary preference match rate: `{persona['metrics']['salary_preference_match_rate']}`")
        lines.append("")
        lines.append("### Candidate Funnel")
        lines.extend(render_funnel_table(funnels[name]))
        lines.append("")
        lines.append("### Top 10 Quality Review")
        lines.append("| Rank | Title | Company | Application label | Flags |")
        lines.append("|---:|---|---|---|---|")
        for item in top_quality[name]["top10"]:
            flags = [key for key, value in item["diagnostic_flags"].items() if value]
            lines.append(
                f"| {item['rank']} | {item['title']} | {item['company']} | "
                f"{item.get('application_strategy_label') or 'unlabeled'} | {', '.join(flags) or 'none'} |"
            )
        lines.append("")
        lines.append("### Live Query Audit")
        lines.append("- Queries:")
        for query in query_audits[name]["queries"]:
            lines.append(f"  - `{query}`")
        if query_audits[name]["warnings"]:
            lines.append("- Query warnings:")
            for warning in query_audits[name]["warnings"]:
                lines.append(f"  - {warning}")
        else:
            lines.append("- Query warnings: none")
        lines.append("")
        lines.append("### Root-Cause Classification")
        lines.append("| Category | Severity | Evidence | Timing |")
        lines.append("|---|---|---|---|")
        for issue in persona["root_causes"]:
            lines.append(f"| {issue['category']} | {issue['severity']} | {issue['evidence']} | {issue['timing']} |")
        lines.append("")
    lines.extend(
        [
            "## Recommended Next Patches",
            "",
            "### Must Fix Before Phase 3",
        ]
    )
    must_fix = [
        (persona["persona"], issue)
        for persona in diagnostics["personas"]
        for issue in persona["root_causes"]
        if issue["timing"] == "must_fix_before_phase3"
    ]
    if not must_fix:
        lines.append("- None identified by the current diagnostic run.")
    else:
        for persona, issue in must_fix:
            lines.append(f"- {persona}: {issue['category']} - {issue['suggested_next_action']}")
    lines.extend(["", "### Can Handle During Phase 3"])
    phase3 = [
        (persona["persona"], issue)
        for persona in diagnostics["personas"]
        for issue in persona["root_causes"]
        if issue["timing"] == "can_handle_in_phase3"
    ]
    if not phase3:
        lines.append("- None.")
    else:
        for persona, issue in phase3:
            lines.append(f"- {persona}: {issue['category']} - {issue['suggested_next_action']}")
    lines.extend(["", "### Can Document As Limitation"])
    limitations = [
        (persona["persona"], issue)
        for persona in diagnostics["personas"]
        for issue in persona["root_causes"]
        if issue["timing"] == "can_document_as_limitation"
    ]
    if not limitations:
        lines.append("- None.")
    else:
        for persona, issue in limitations:
            lines.append(f"- {persona}: {issue['category']} - {issue['evidence']}")
    lines.extend(
        [
            "",
            "## Evidence-Based Conclusion",
            "",
            "If a persona lacks enough strong matches, this report attributes the gap to candidate coverage, live-query generation, metadata sparsity, filtering, scoring, role-family detection, company-signal detection, or evaluator behavior. It does not describe a persona as too picky unless the funnel evidence supports that conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    personas = selected_personas(args)
    timestamp = timestamp_slug()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    live_records_by_persona: dict[str, list[dict[str, Any]]] = {}
    live_metadata_by_persona: dict[str, dict[str, Any]] = {}
    all_live_records: list[dict[str, Any]] = []
    all_raw_live_records: list[dict[str, Any]] = []
    live_warnings: list[str] = []

    query_audits: dict[str, Any] = {}
    for persona_name in personas:
        profile = get_persona(persona_name)
        queries = build_live_queries(profile, max_queries=max(1, min(args.max_queries, 10)))
        if args.mode in {"live", "hybrid"} and not args.dry_run:
            if args.provider == "jsearch":
                fetch = fetch_jsearch_live(
                    queries,
                    pages_per_query=max(1, min(args.pages_per_query, 1)),
                    results_per_page=max(1, min(args.results_per_page, 20)),
                    env_file=args.env_file,
                )
            else:
                fetch = fetch_adzuna_live(
                    queries,
                    country=args.country,
                    pages_per_query=args.pages_per_query,
                    results_per_page=args.results_per_page,
                    env_file=args.env_file,
                )
            live_records_by_persona[persona_name] = fetch.normalized_records
            live_metadata_by_persona[persona_name] = fetch.metadata
            all_live_records.extend(fetch.normalized_records)
            all_raw_live_records.extend(fetch.raw_records)
            live_warnings.extend(fetch.metadata.get("warnings", []))
            live_warnings.extend(fetch.metadata.get("errors", []))
        else:
            live_records_by_persona[persona_name] = []
            live_metadata_by_persona[persona_name] = {
                "provider": args.provider,
                "api_call_count": 0,
                "raw_records_fetched": 0,
                "normalized_live_records": 0,
                "warnings": ["Dry run requested; live API calls skipped."] if args.dry_run else [],
                "errors": [],
            }
        query_audits[persona_name] = query_audit_for_persona(
            persona_name,
            profile,
            queries,
            mode=args.mode,
            dry_run=args.dry_run,
            live_metadata=live_metadata_by_persona[persona_name],
        )

    live_output_paths: dict[str, str] = {}
    if args.mode in {"live", "hybrid"} and not args.dry_run and all_raw_live_records:
        LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = LIVE_CACHE_DIR / f"persona_live_raw_{timestamp}.json"
        jobs_path = LIVE_CACHE_DIR / f"persona_live_jobs_{timestamp}.csv"
        report_path = LIVE_CACHE_DIR / f"persona_live_search_report_{timestamp}.json"
        live_report = {
            "generated_at": generated_at,
            "provider": args.provider,
            "mode": args.mode,
            "personas": personas,
            "metadata_by_persona": live_metadata_by_persona,
            "raw_records_fetched": len(all_raw_live_records),
            "normalized_live_records": len(all_live_records),
            "warnings": live_warnings,
        }
        write_json(raw_path, {"provider": args.provider, "records": sanitize_live_payload(all_raw_live_records)})
        write_csv(jobs_path, sanitize_live_payload(all_live_records), CANONICAL_COLUMNS)
        live_report["output_paths"] = {"raw": rel_path(raw_path), "jobs": rel_path(jobs_path), "report": rel_path(report_path)}
        write_json(report_path, sanitize_live_payload(live_report))
        live_output_paths = dict(live_report["output_paths"])

    funnels: dict[str, Any] = {}
    top_quality: dict[str, Any] = {}
    root_causes: dict[str, list[dict[str, Any]]] = {}
    diagnostics_personas: list[dict[str, Any]] = []
    base_store = build_or_load_job_embeddings(
        snapshot_path=args.snapshot,
        cache_dir=args.cache_dir,
        backend=args.embedding_backend,
    )
    shared_pool = None
    shared_ranker = None
    if not any(live_records_by_persona.values()):
        shared_pool = build_candidate_pool_from_base(
            base_store,
            mode=args.mode,
            live_rows=[],
            fallback_to_offline=True,
        )
        shared_ranker = JobRanker(shared_pool.store)

    for persona_name in personas:
        profile = get_persona(persona_name)
        if shared_pool is not None and shared_ranker is not None:
            pool = shared_pool
            ranker = shared_ranker
        else:
            pool = build_candidate_pool_from_base(
                base_store,
                mode=args.mode,
                live_rows=live_records_by_persona[persona_name],
                fallback_to_offline=True,
            )
            ranker = JobRanker(pool.store)
        ranking = ranker.rank(profile, top_k=args.top_k, candidate_k=args.candidate_k)
        top_jobs = ranking["top_jobs"]
        funnel = base_funnel(persona_name, profile, pool.store.job_rows)
        quality = top10_quality(persona_name, profile, top_jobs)
        metrics = metrics_for_top10(persona_name, profile, top_jobs, args.top_k)
        causes = classify_root_causes(persona_name, funnel, quality, query_audits[persona_name], metrics)
        funnels[persona_name] = {
            **funnel,
            "mode_requested": args.mode,
            "mode_used": pool.mode_used,
            "candidate_pool": pool.metadata,
        }
        top_quality[persona_name] = quality
        root_causes[persona_name] = causes
        diagnostics_personas.append(
            {
                "persona": persona_name,
                "mode_requested": args.mode,
                "mode_used": pool.mode_used,
                "candidate_pool": pool.metadata,
                "ranking_metadata": ranking["metadata"],
                "metrics": metrics,
                "root_causes": causes,
                "top10_flag_counts": quality["flag_counts"],
            }
        )

    diagnostics = {
        "generated_at": generated_at,
        "mode_requested": args.mode,
        "dry_run": args.dry_run,
        "provider": args.provider,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "snapshot": rel_path(args.snapshot),
        "live_output_paths": live_output_paths,
        "executive_summary": executive_status(root_causes),
        "personas": diagnostics_personas,
    }

    write_json(DIAGNOSTICS_JSON, sanitize_live_payload(diagnostics))
    write_json(FUNNELS_JSON, sanitize_live_payload(funnels))
    write_json(LIVE_QUERY_AUDIT_JSON, sanitize_live_payload(query_audits))
    write_json(TOP10_FLAGS_JSON, sanitize_live_payload(top_quality))
    DIAGNOSTICS_MD.write_text(render_markdown(diagnostics, funnels, query_audits, top_quality), encoding="utf-8")

    print("Phase 2.13 persona diagnostics complete")
    print(f"Mode requested: {args.mode}")
    print(f"Dry run: {args.dry_run}")
    print(f"Personas: {', '.join(personas)}")
    print(f"Backend ready for Phase 3: {diagnostics['executive_summary']['backend_ready_for_phase3']}")
    for persona in diagnostics_personas:
        print(
            f"- {persona['persona']}: mode={persona['mode_used']} returned={persona['metrics']['returned_jobs']} "
            f"strict={persona['metrics']['strict_topk_pass_rate']} flags={json.dumps(persona['top10_flag_counts'], sort_keys=True)}"
        )
    print(f"Diagnostics JSON: {DIAGNOSTICS_JSON}")
    print(f"Diagnostics MD: {DIAGNOSTICS_MD}")
    if live_output_paths:
        print(f"Live report: {live_output_paths.get('report')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
