"""Offline Phase 2.18D evidence-aware rerank helpers.

These helpers are intentionally dry-run only. They load the compact Phase 2.18
ranking sidecar and compute non-negative evidence adjustments for exported
candidate rows. Missing or unknown evidence is neutral and never penalizes a
job.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text, normalize_for_key


LCA_ACTIVITY_WEIGHTS = {
    "recent_lca_activity_high": 0.035,
    "recent_lca_activity_moderate": 0.027,
    "recent_lca_activity_low": 0.018,
    "recent_lca_activity_no_certified_cases": 0.006,
    "historical_lca_activity_only": 0.014,
}
LARGE_SIZE_BUCKET_WEIGHTS = {
    "enterprise_10001_plus": 0.022,
    "large_1001_5000": 0.016,
    "large_5001_10000": 0.019,
}
USABLE_SIZE_POLICIES = {
    "usable_employer_context",
    "usable_franchise_operator_context",
    "usable_single_location_context",
}
MAX_EVIDENCE_ADJUSTMENT = 0.08


def parse_bool(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_float(value: Any) -> float:
    text = clean_text(value).replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_json_list(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        term = clean_text(item)
        key = normalize_for_key(term)
        if not key or key in seen:
            continue
        terms.append(term)
        seen.add(key)
    return terms


def sidecar_by_job_id(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load the 18C compact ranking sidecar keyed by job_id."""

    rows: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    fieldnames: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for raw in reader:
            row = {key: clean_text(value) for key, value in raw.items()}
            job_id = clean_text(row.get("job_id"))
            if not job_id:
                continue
            if job_id in rows:
                duplicate_ids.append(job_id)
                continue
            rows[job_id] = row
    return rows, {
        "path": path.as_posix(),
        "row_count": len(rows),
        "field_count": len(fieldnames),
        "duplicate_job_id_count": len(duplicate_ids),
        "duplicate_job_ids_sample": duplicate_ids[:10],
    }


def _profile_keys(profile: dict[str, Any], field: str) -> set[str]:
    return {normalize_for_key(value) for value in normalize_list(profile.get(field)) if normalize_for_key(value)}


def _profile_skill_keys(profile: dict[str, Any]) -> set[str]:
    keys = _profile_keys(profile, "skills")
    if "sklearn" in keys:
        keys.add("scikit learn")
    if "scikit learn" in keys:
        keys.add("sklearn")
    return keys


def _lca_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str]]:
    label = clean_text(sidecar.get("lca_activity_label")).lower()
    scope = clean_text(sidecar.get("lca_match_scope")).lower()
    if scope not in {"role_family", "employer_only"}:
        return 0.0, []
    base = LCA_ACTIVITY_WEIGHTS.get(label, 0.0)
    if base <= 0:
        return 0.0, []
    if not profile.get("needs_sponsorship"):
        base *= 0.45
    if scope == "role_family":
        base += 0.005
    return round(base, 6), [
        "LCA signal is employer historical filing activity only; it is not job-level sponsorship truth."
    ]


def _company_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str]]:
    policy = clean_text(sidecar.get("company_size_usage_policy")).lower()
    bucket = clean_text(sidecar.get("usable_employer_size_bucket")).lower()
    snapshot_type = clean_text(sidecar.get("snapshot_company_type")).lower()
    provider_type = clean_text(sidecar.get("company_type_provider")).lower()
    if policy not in USABLE_SIZE_POLICIES:
        return 0.0, []

    preferred = _profile_keys(profile, "preferred_company_types")
    boost = 0.0
    if "large company" in preferred or "large_company" in preferred or profile.get("needs_sponsorship"):
        boost = max(boost, LARGE_SIZE_BUCKET_WEIGHTS.get(bucket, 0.0))
    if "research lab" in preferred or "research_lab" in preferred:
        if snapshot_type == "research_lab" or provider_type == "research_lab":
            boost = max(boost, 0.02)
    if boost <= 0:
        return 0.0, []
    return round(boost, 6), ["Company size/type is a soft preference signal only."]


def _hard_skill_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str]]:
    profile_skills = _profile_skill_keys(profile)
    if not profile_skills:
        return 0.0, []
    scoring_terms = parse_json_list(sidecar.get("hard_skill_scoring_terms_json"))
    capped_terms = parse_json_list(sidecar.get("hard_skill_capped_terms_json"))
    scoring_matches = [term for term in scoring_terms if normalize_for_key(term) in profile_skills]
    capped_matches = [term for term in capped_terms if normalize_for_key(term) in profile_skills]
    value = min(0.036, 0.008 * len(scoring_matches) + 0.004 * len(capped_matches))
    if value <= 0:
        return 0.0, []
    matched = ", ".join(scoring_matches[:4] + capped_matches[:2])
    return round(value, 6), [f"Hard-skill evidence uses admitted scoring-safe sidecar terms: {matched}."]


def _role_family_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str]]:
    primary = normalize_for_key(sidecar.get("role_family_primary"))
    if not primary or primary == "unknown":
        return 0.0, []
    preferred = _profile_keys(profile, "preferred_role_families") | _profile_keys(profile, "required_role_families")
    if primary not in preferred:
        return 0.0, []
    confidence = parse_float(sidecar.get("role_family_confidence"))
    if confidence <= 0:
        return 0.0, []
    value = 0.026 * min(1.0, confidence)
    if parse_bool(sidecar.get("role_family_hard_filter_safe")):
        value += 0.004
    return round(min(value, 0.03), 6), ["Role-family sidecar agrees with profile target family."]


def _llm_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str]]:
    if not parse_bool(sidecar.get("llm_overlay_available")):
        return 0.0, []
    status = clean_text(sidecar.get("llm_reviewed_overlay_status")).lower()
    if status not in {"reviewed_candidate", "reviewed"}:
        return 0.0, []
    preferred = _profile_keys(profile, "preferred_role_families") | _profile_keys(profile, "required_role_families")
    candidate_family = normalize_for_key(sidecar.get("llm_reviewed_role_family_overlay_candidate"))
    action = clean_text(sidecar.get("llm_reviewed_suggested_soft_action")).lower()
    value = 0.0
    if candidate_family and candidate_family in preferred:
        value += 0.008
    if action in {"relabel_soft", "boost_soft"}:
        value += 0.004
    value = min(value, 0.012)
    if value <= 0:
        return 0.0, []
    return round(value, 6), ["LLM overlay is partial/review-only and used only as a small dry-run soft signal."]


def evidence_adjustment(
    profile: dict[str, Any],
    sidecar: dict[str, str],
    *,
    old_score: float,
) -> dict[str, Any]:
    """Return a missing-neutral evidence adjustment payload for one candidate."""

    component_fns = {
        "lca_activity": _lca_component,
        "company_size_type": _company_component,
        "hard_skill": _hard_skill_component,
        "role_family": _role_family_component,
        "llm_overlay": _llm_component,
    }
    components: dict[str, float] = {}
    notes: list[str] = []
    for name, fn in component_fns.items():
        value, component_notes = fn(profile, sidecar)
        components[name] = round(max(0.0, value), 6)
        notes.extend(component_notes)

    raw_score = round(sum(components.values()), 6)
    adjustment = round(min(MAX_EVIDENCE_ADJUSTMENT, raw_score), 6)
    if adjustment == 0:
        notes.append("Missing, unknown, or non-matching evidence was neutral; no candidate was penalized.")
    new_score = round(min(1.0, max(0.0, old_score) + adjustment), 6)
    return {
        "new_evidence_score": raw_score,
        "evidence_adjustment": adjustment,
        "new_final_score": new_score,
        "evidence_score_components": components,
        "evidence_notes": notes,
    }


def component_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize D2 evidence coverage over reranked candidates."""

    counters: Counter[str] = Counter()
    for row in rows:
        sidecar = row.get("sidecar") if isinstance(row.get("sidecar"), dict) else {}
        components = row.get("evidence_score_components") if isinstance(row.get("evidence_score_components"), dict) else {}
        if float(row.get("evidence_adjustment") or 0) > 0:
            counters["evidence_adjusted"] += 1
        if clean_text(sidecar.get("lca_match_scope")).lower() in {"role_family", "employer_only"}:
            counters["lca_match_available"] += 1
        if components.get("lca_activity", 0) > 0:
            counters["lca_activity_positive"] += 1
        if clean_text(sidecar.get("usable_employer_size_bucket")).lower() not in {"", "unknown"}:
            counters["company_size_available"] += 1
        if components.get("company_size_type", 0) > 0:
            counters["company_size_type_positive"] += 1
        if parse_json_list(sidecar.get("hard_skill_scoring_terms_json")) or parse_json_list(sidecar.get("hard_skill_capped_terms_json")):
            counters["hard_skill_terms_available"] += 1
        if components.get("hard_skill", 0) > 0:
            counters["hard_skill_positive"] += 1
        if clean_text(sidecar.get("role_family_primary")).lower() not in {"", "unknown"}:
            counters["role_family_available"] += 1
        if components.get("role_family", 0) > 0:
            counters["role_family_positive"] += 1
        if parse_bool(sidecar.get("llm_overlay_available")):
            counters["llm_overlay_available"] += 1
        if components.get("llm_overlay", 0) > 0:
            counters["llm_overlay_positive"] += 1
    total = len(rows)
    return {
        key: {"count": value, "rate": round(value / total, 6) if total else 0.0}
        for key, value in sorted(counters.items())
    } | {"candidate_rows": total}
