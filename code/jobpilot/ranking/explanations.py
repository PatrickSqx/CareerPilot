"""Why-ranked explanation generation."""

from __future__ import annotations

from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.ranking.company_signals import detect_company_signals
from jobpilot.ranking.filters import FilterResult, configured_seniority_term_hits, parse_float
from jobpilot.ranking.location_signals import location_explanation, location_violation_reason
from jobpilot.ranking.role_signals import generic_backend_devops_title_hits, role_family_match_details
from jobpilot.ranking.scoring import target_role_score
from jobpilot.utils.text import clean_text


def build_why_ranked(
    profile: dict[str, Any],
    job: dict[str, Any],
    filter_result: FilterResult,
    score_payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a structured explanation usable by UI and CSV export."""

    salary_text = clean_text(job.get("salary_raw")) or "salary not listed"
    location_text = clean_text(job.get("location")) or "location not listed"
    role_score = target_role_score(profile, job)
    matched = score_payload.get("matched_skills", [])
    positives: list[str] = []
    negatives: list[str] = []
    sponsorship_signal = clean_text(job.get("sponsorship_signal")) or "unknown"
    company_type = clean_text(job.get("company_type")) or "unknown"
    company_signals = detect_company_signals(job)
    role_family = role_family_match_details(profile, job)
    location_reason = location_violation_reason(job, profile)
    salary_preference = parse_float(profile.get("salary_min"))
    job_salary_max = parse_float(job.get("salary_max")) or parse_float(job.get("salary_min"))
    salary_matches_preference = (
        True if not salary_preference else bool(job_salary_max is not None and job_salary_max >= salary_preference)
    )
    location_matches_preference = location_reason is None

    if matched:
        positives.append("Matches skills: " + ", ".join(matched[:8]))
    if role_score >= 0.8:
        positives.append("Strong target-role/title match")
    elif role_score >= 0.3:
        positives.append("Partial target-role/title match")
    else:
        negatives.append("Weak target-role/title match")

    if role_family["required_title_matches"]:
        positives.append("Matches required role family: " + ", ".join(role_family["required_title_matches"]))
    elif role_family["preferred_title_matches"]:
        positives.append("Matches preferred role family: " + ", ".join(role_family["preferred_title_matches"]))
    elif role_family["strict_role_family"] and role_family["failure_reason"]:
        negatives.append(role_family["failure_reason"])
    elif role_family["preferred_description_matches"]:
        positives.append(
            "Weak description-level preferred role-family signal: "
            + ", ".join(role_family["preferred_description_matches"])
        )

    generic_hits = generic_backend_devops_title_hits(job)
    if profile.get("avoid_generic_backend_devops") and generic_hits and not role_family["required_title_matches"]:
        negatives.append("Generic backend/DevOps title lacks required ML/AI infrastructure signal")

    if profile.get("needs_sponsorship"):
        if sponsorship_signal == "mentions_sponsorship_or_work_auth":
            positives.insert(0, "Posting mentions sponsorship or work authorization")
        elif sponsorship_signal == "unknown" and company_signals["sponsor_friendly_proxy"]:
            positives.insert(0, "Company-name proxy suggests sponsor-friendly or large-company fit, not confirmed")
            negatives.append("Sponsorship is unknown")
        elif sponsorship_signal == "unknown":
            negatives.append("Sponsorship is unknown")

    components = score_payload.get("score_components", {})
    if salary_matches_preference and salary_preference:
        positives.append("Salary meets preference when listed")
    elif salary_preference and job_salary_max is None:
        negatives.append("Salary preference cannot be verified because salary is missing")
    elif salary_preference:
        negatives.append("Salary appears below preference")

    location_message = location_explanation(job, profile)
    if location_reason:
        negatives.append(location_message)
    elif normalize_list(profile.get("location_preferences")):
        positives.append(location_message)

    if company_signals["defense_government_contractor"]:
        negatives.append("Defense/government-contractor company signal")

    seniority_soft_hits = configured_seniority_term_hits(profile, job, "penalize_seniority_terms")
    if profile.get("avoid_overly_senior") and seniority_soft_hits:
        negatives.append("Seniority level may be high for a student or new-graduate profile: " + ", ".join(seniority_soft_hits))

    if filter_result.notes:
        negatives.extend(filter_result.notes)
    if filter_result.violations:
        negatives.extend(filter_result.violations)

    summary_parts = positives[:3] or ["Recommended mainly by semantic similarity"]
    if negatives:
        summary_parts.append("Watch: " + "; ".join(negatives[:3]))

    return {
        "matched_skills": matched,
        "target_role_score": components.get("target_role", 0),
        "location_status": location_text,
        "location_explanation": location_message,
        "salary_status": salary_text,
        "seniority_years_status": {
            "seniority": clean_text(job.get("seniority")) or "unknown",
            "years_required": clean_text(job.get("years_required")) or "not listed",
        },
        "employment_type_status": clean_text(job.get("employment_type")) or "unknown",
        "sponsorship_company_notes": {
            "sponsorship_signal": sponsorship_signal,
            "company_type": company_type,
            "company_signals": company_signals,
        },
        "role_family": role_family,
        "preference_status": {
            "location_matches_preference": location_matches_preference,
            "location_preference_reason": location_reason or "",
            "salary_matches_preference": salary_matches_preference,
            "salary_preference_min": int(salary_preference) if salary_preference else None,
        },
        "hard_filter": {
            "passed": filter_result.passed,
            "violations": filter_result.violations,
            "notes": filter_result.notes,
        },
        "score_components": components,
        "penalties": score_payload.get("penalties", {}),
        "positive_drivers": positives,
        "negative_drivers": negatives,
        "summary": ". ".join(summary_parts),
    }
