"""Hard filters and constraint checks for multi-stage ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.ranking.company_signals import detect_company_signals
from jobpilot.ranking.location_signals import (
    is_us_location,
    location_uncertainty_note,
    location_violation_reason,
    matches_location_preferences,
)
from jobpilot.ranking.role_signals import role_family_match_details
from jobpilot.ranking.role_signals import (
    generic_backend_devops_without_target_signal,
    title_contains_profile_signal,
)
from jobpilot.utils.text import clean_text


SENIORITY_LEVEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "senior": re.compile(r"(?:\bsenior\b|\bsr\.?(?=\s|$|[-,/]))", re.IGNORECASE),
    "staff": re.compile(r"\bstaff\b", re.IGNORECASE),
    "principal": re.compile(r"\bprincipal\b", re.IGNORECASE),
    "lead": re.compile(r"\blead\b", re.IGNORECASE),
    "manager": re.compile(r"\bmanager\b", re.IGNORECASE),
    "director": re.compile(r"\bdirector\b", re.IGNORECASE),
    "head": re.compile(r"\bhead(?:\s+of)?\b", re.IGNORECASE),
    "distinguished": re.compile(r"\bdistinguished\b", re.IGNORECASE),
    "iii": re.compile(r"\b(?:iii|level\s*iii|level\s*3)\b", re.IGNORECASE),
}

JUNIOR_LEVEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "junior": re.compile(r"\b(?:junior|jr\.?|jr|entry[-\s]?level|new grad|graduate)\b", re.IGNORECASE),
    "internship": re.compile(r"\b(?:intern|internship)\b", re.IGNORECASE),
}


@dataclass
class FilterResult:
    passed: bool
    violations: list[str]
    notes: list[str]


def parse_float(value: Any) -> float | None:
    text = clean_text(value).replace(",", "").replace("$", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _cell(value: Any) -> str:
    text = clean_text(value)
    return "" if text.lower() in {"nan", "none", "null"} else text


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _job_blob(job: dict[str, Any]) -> str:
    parts = [
        job.get("title", ""),
        job.get("company", ""),
        job.get("location", ""),
        job.get("employment_type", ""),
        job.get("company_type", ""),
        job.get("sponsorship_signal", ""),
        job.get("description_text", "")[:3000],
    ]
    return clean_text(" ".join(str(part) for part in parts)).lower()


def _level_scan_text(job: dict[str, Any]) -> str:
    """Use title plus short description for level detection."""

    return clean_text(f"{job.get('title', '')} {clean_text(job.get('description_text', ''))[:700]}")


def seniority_level_hits(job: dict[str, Any]) -> list[str]:
    """Return senior/lead-style level terms found in title or short description."""

    text = _level_scan_text(job)
    return [label for label, pattern in SENIORITY_LEVEL_PATTERNS.items() if pattern.search(text)]


def junior_level_hits(job: dict[str, Any]) -> list[str]:
    """Return junior/intern-style level terms found in title or short description."""

    text = _level_scan_text(job)
    return [label for label, pattern in JUNIOR_LEVEL_PATTERNS.items() if pattern.search(text)]


def excluded_level_hits(profile: dict[str, Any], job: dict[str, Any]) -> list[str]:
    """Detect excluded level terms from both precomputed and raw title text signals."""

    excluded_seniority = {item.lower() for item in normalize_list(profile.get("excluded_seniority"))}
    hits: list[str] = []
    seniority = _cell(job.get("seniority")).lower()
    if seniority and seniority in excluded_seniority:
        hits.append(seniority)
    if excluded_seniority & {"senior", "staff_principal", "lead_manager"}:
        hits.extend(seniority_level_hits(job))
    if excluded_seniority & {"entry_junior", "internship"}:
        hits.extend(junior_level_hits(job))
    deduped: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit not in seen:
            deduped.append(hit)
            seen.add(hit)
    return deduped


def configured_seniority_term_hits(profile: dict[str, Any], job: dict[str, Any], field: str) -> list[str]:
    """Detect profile-configured seniority terms in title plus short description."""

    terms = {item.lower() for item in normalize_list(profile.get(field))}
    if not terms:
        return []
    text = _level_scan_text(job)
    hits: list[str] = []
    for term in terms:
        key = "senior" if term in {"sr", "sr.", "senior"} else term
        pattern = SENIORITY_LEVEL_PATTERNS.get(key)
        if pattern and pattern.search(text):
            hits.append(term)
        elif key and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text, re.IGNORECASE):
            hits.append(term)
    deduped: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit not in seen:
            deduped.append(hit)
            seen.add(hit)
    return deduped


def location_matches(profile: dict[str, Any], job: dict[str, Any]) -> bool:
    return matches_location_preferences(job, profile)


def apply_hard_filters(profile: dict[str, Any], job: dict[str, Any]) -> FilterResult:
    """Apply explicit dealbreaker-style constraints."""

    violations: list[str] = []
    notes: list[str] = []
    blob = _job_blob(job)

    for term in normalize_list(profile.get("dealbreakers")):
        key = term.lower()
        if not key:
            continue
        if key in {"contract-only", "tiny startup", "small startup"}:
            continue
        if key in blob:
            violations.append(f"dealbreaker:{term}")

    salary_min = parse_float(profile.get("salary_min"))
    if salary_min:
        job_min = parse_float(job.get("salary_min"))
        job_max = parse_float(job.get("salary_max")) or job_min
        strict_salary = _bool(profile.get("salary_is_dealbreaker")) or _bool(profile.get("strict_salary"))
        if job_max is not None and job_max < salary_min:
            message = f"salary_below_min:{int(job_max)}<{int(salary_min)}"
            if strict_salary:
                violations.append(message)
            else:
                notes.append(f"salary_below_preference:{int(job_max)}<{int(salary_min)}")
        elif job_max is None:
            notes.append("salary_missing")

    location_reason = location_violation_reason(job, profile)
    if _bool(profile.get("strict_location")) and location_reason:
        violations.append(f"location:{location_reason}")
    elif _bool(profile.get("us_only")) and location_reason:
        violations.append(f"location:{location_reason}")
    elif location_reason:
        notes.append(f"location_preference_mismatch:{location_reason}")
    note = location_uncertainty_note(job, profile)
    if note:
        notes.append(note)

    for hit in excluded_level_hits(profile, job):
        violations.append(f"excluded_seniority:{hit}")

    for hit in configured_seniority_term_hits(profile, job, "hard_reject_seniority_terms"):
        violations.append(f"hard_reject_seniority:{hit}")

    role_family = role_family_match_details(profile, job)
    if bool(profile.get("strict_role_family")) and not role_family["strict_pass"]:
        required = "|".join(role_family["required_role_families"]) or "required"
        violations.append(f"not_required_role_family:{required}")

    if normalize_list(profile.get("title_must_include_any")) and not title_contains_profile_signal(profile, job):
        violations.append("missing_required_title_signal")

    if bool(profile.get("avoid_generic_backend_devops")) and generic_backend_devops_without_target_signal(profile, job):
        violations.append("generic_backend_devops_without_target_signal")

    years_required = parse_float(job.get("years_required"))
    max_years = parse_float(profile.get("max_years_required"))
    if years_required is not None and max_years is not None and years_required > max_years:
        violations.append(f"years_required:{int(years_required)}>{int(max_years)}")

    employment_type = _cell(job.get("employment_type")).lower()
    excluded_employment = {item.lower() for item in normalize_list(profile.get("excluded_employment_types"))}
    if any(term and term in employment_type for term in excluded_employment):
        violations.append(f"excluded_employment_type:{employment_type}")

    company_type = _cell(job.get("company_type")).lower()
    excluded_company = {item.lower() for item in normalize_list(profile.get("excluded_company_types"))}
    if company_type and company_type in excluded_company:
        violations.append(f"excluded_company_type:{company_type}")

    company_signals = detect_company_signals(job)
    excluded_terms = {item.lower() for item in normalize_list(profile.get("dealbreakers"))}
    if company_signals["defense_government_contractor"] and (
        "defense_military" in excluded_company or "defense" in excluded_terms or "military" in excluded_terms
    ):
        terms = "|".join(company_signals["defense_terms"][:3]) or "company_proxy"
        violations.append(f"defense_government_contractor:{terms}")
    elif bool(profile.get("avoid_defense_or_clearance")) and company_signals["defense_government_contractor"]:
        terms = "|".join(company_signals["defense_terms"][:3]) or "company_proxy"
        violations.append(f"defense_government_contractor:{terms}")

    if bool(profile.get("needs_sponsorship")):
        sponsorship_signal = _cell(job.get("sponsorship_signal")).lower()
        if sponsorship_signal == "no_sponsorship":
            violations.append("no_sponsorship")
        elif sponsorship_signal == "unknown":
            notes.append("sponsorship_unknown")

    return FilterResult(passed=not violations, violations=violations, notes=notes)
