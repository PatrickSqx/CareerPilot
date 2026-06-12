"""Weighted scoring for multi-stage ranking."""

from __future__ import annotations

from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.ranking.company_signals import detect_company_signals
from jobpilot.ranking.filters import FilterResult, configured_seniority_term_hits, parse_float
from jobpilot.ranking.location_signals import is_remote_job, location_uncertainty_note, matches_location_preferences
from jobpilot.ranking.role_signals import generic_backend_devops_without_target_signal, target_role_relevance_score
from jobpilot.utils.text import clean_text, normalize_for_key


def split_job_skills(job: dict[str, Any]) -> list[str]:
    value = clean_text(job.get("extracted_skills"))
    if not value:
        return []
    return [part.strip().lower() for part in value.split("|") if part.strip()]


def _skill_key(value: str) -> str:
    key = normalize_for_key(value)
    if key == "sklearn":
        return "scikit learn"
    return key


def matched_skills(profile: dict[str, Any], job: dict[str, Any]) -> list[str]:
    profile_skills = {_skill_key(skill): skill for skill in normalize_list(profile.get("skills"))}
    job_skills = {_skill_key(skill): skill for skill in split_job_skills(job)}
    hits = [profile_skills[key] for key in profile_skills.keys() & job_skills.keys()]
    return sorted(hits, key=str.lower)


def skill_match_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    skills = normalize_list(profile.get("skills"))
    if not skills:
        return 0.0
    return min(1.0, len(matched_skills(profile, job)) / max(len(skills), 1))


def target_role_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    return target_role_relevance_score(profile, job)


def salary_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    desired = parse_float(profile.get("salary_min"))
    if not desired:
        return 0.5
    job_min = parse_float(job.get("salary_min"))
    job_max = parse_float(job.get("salary_max")) or job_min
    if job_max is None:
        return 0.45
    if job_max >= desired:
        return 1.0
    return max(0.0, job_max / desired)


def location_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    preferences = normalize_list(profile.get("location_preferences"))
    if not preferences:
        return 0.5
    if matches_location_preferences(job, profile):
        if location_uncertainty_note(job, profile):
            return 0.85
        return 1.0
    if is_remote_job(job):
        return 0.4
    return 0.0


def company_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    company_type = clean_text(job.get("company_type")).lower()
    company_signals = detect_company_signals(job)
    preferred = {item.lower() for item in normalize_list(profile.get("preferred_company_types"))}
    excluded = {item.lower() for item in normalize_list(profile.get("excluded_company_types"))}
    if company_type and company_type in excluded:
        return 0.0
    if company_signals["defense_government_contractor"] and "defense_military" in excluded:
        return 0.0
    if company_type and company_type in preferred:
        return 1.0
    if "research_lab" in preferred and company_signals["research_lab_proxy"]:
        return 1.0
    if "large_company" in preferred and company_signals["large_company_proxy"]:
        return 0.95
    return 0.5


def sponsorship_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    if not profile.get("needs_sponsorship"):
        return 0.5
    signal = clean_text(job.get("sponsorship_signal")).lower()
    company_type = clean_text(job.get("company_type")).lower()
    company_signals = detect_company_signals(job)
    if signal == "mentions_sponsorship_or_work_auth":
        return 1.0
    if signal == "no_sponsorship":
        return 0.0
    if company_signals["research_lab_proxy"]:
        return 0.9
    if company_signals["large_company_proxy"]:
        return 0.85
    if company_type in {"large_company", "research_lab"}:
        return 0.75
    return 0.05


def employment_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    desired = {item.lower() for item in normalize_list(profile.get("employment_types"))}
    if not desired:
        return 0.5
    employment = clean_text(job.get("employment_type")).lower()
    if any(item and item in employment for item in desired):
        return 1.0
    if employment in {"unknown", ""}:
        return 0.45
    return 0.3


def compute_score(
    profile: dict[str, Any],
    job: dict[str, Any],
    *,
    embedding_similarity: float,
    filter_result: FilterResult,
) -> dict[str, Any]:
    """Compute score components and final score."""

    components = {
        "embedding_similarity": max(0.0, min(1.0, float(embedding_similarity))),
        "skill_match": skill_match_score(profile, job),
        "target_role": target_role_score(profile, job),
        "salary": salary_score(profile, job),
        "location": location_score(profile, job),
        "company": company_score(profile, job),
        "sponsorship": sponsorship_score(profile, job),
        "employment_type": employment_score(profile, job),
    }
    weights = {
        "embedding_similarity": 0.27,
        "skill_match": 0.17,
        "target_role": 0.23,
        "salary": 0.09,
        "location": 0.08,
        "company": 0.05,
        "sponsorship": 0.06,
        "employment_type": 0.06,
    }
    company_signals = detect_company_signals(job)
    seniority_soft_hits = configured_seniority_term_hits(profile, job, "penalize_seniority_terms")
    penalties = {
        "hard_filter_violation": 0.25 * len(filter_result.violations),
        "weak_role_match": 0.08 if components["target_role"] < 0.2 else 0.0,
        "weak_skill_match": 0.05 if components["skill_match"] == 0 else 0.0,
        "generic_backend_devops_without_target_signal": 0.2
        if profile.get("avoid_generic_backend_devops") and generic_backend_devops_without_target_signal(profile, job)
        else 0.0,
        "seniority_realism": (
            0.24 if profile.get("new_grad_or_student_profile") else 0.14
        )
        if profile.get("avoid_overly_senior") and seniority_soft_hits
        else 0.0,
        "defense_or_clearance_risk": 0.18
        if profile.get("avoid_defense_or_clearance") and company_signals["defense_government_contractor"]
        else 0.0,
        "location_preference_mismatch": 0.05
        if any(note.startswith("location_preference_mismatch") for note in filter_result.notes)
        else 0.0,
        "salary_below_preference": 0.05
        if any(note.startswith("salary_below_preference") for note in filter_result.notes)
        else 0.0,
        "salary_missing": 0.03 if "salary_missing" in filter_result.notes and profile.get("salary_min") else 0.0,
        "sponsorship_unknown": (
            0.12 if company_signals["sponsor_friendly_proxy"] else 0.28
        )
        if profile.get("needs_sponsorship") and "sponsorship_unknown" in filter_result.notes
        else 0.0,
    }
    base_score = sum(components[key] * weights[key] for key in weights)
    penalty_total = min(0.75, sum(penalties.values()))
    final_score = max(0.0, min(1.0, base_score - penalty_total))
    return {
        "final_score": round(final_score, 6),
        "score_components": {key: round(value, 6) for key, value in components.items()},
        "penalties": {key: round(value, 6) for key, value in penalties.items() if value},
        "matched_skills": matched_skills(profile, job),
    }
