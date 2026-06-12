"""Structured profile parsing for resume text and manual input."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jobpilot.utils.text import clean_text, extract_skills


PROFILE_LIST_FIELDS = {
    "skills",
    "target_roles",
    "location_preferences",
    "dealbreakers",
    "employment_types",
    "excluded_employment_types",
    "excluded_seniority",
    "preferred_company_types",
    "excluded_company_types",
    "required_role_families",
    "preferred_role_families",
    "title_must_include_any",
    "required_title_signals",
    "hard_reject_seniority_terms",
    "penalize_seniority_terms",
}


def normalize_list(value: Any) -> list[str]:
    """Normalize comma, pipe, semicolon, or newline separated values."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = re.split(r"[|,;\n]+", str(value))
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def parse_salary_preference(value: Any) -> int | None:
    """Parse salary thresholds such as 140000, $140k, or 140K."""

    if value is None or value == "":
        return None
    text = clean_text(value).lower().replace(",", "").replace("$", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)
    if match:
        return int(float(match.group(1)) * 1000)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    if number < 1000:
        number *= 1000
    return int(number)


def parse_bool(value: Any) -> bool:
    """Parse booleans from structured JSON, forms, and strings."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = clean_text(value).lower()
    return text in {"1", "true", "yes", "y", "on"}


def _extract_section(text: str, headings: list[str]) -> str:
    pattern = "|".join(re.escape(heading) for heading in headings)
    match = re.search(
        rf"(?is)\b(?:{pattern})\b\s*:?\s*(.+?)(?=\n\s*[A-Z][A-Za-z /&-]{{2,30}}\s*:?\s*\n|$)",
        text,
    )
    return clean_text(match.group(1)) if match else ""


def build_profile(**kwargs: Any) -> dict[str, Any]:
    """Build a canonical user profile dict from manual or parsed fields."""

    profile: dict[str, Any] = {
        "profile_id": clean_text(kwargs.get("profile_id") or kwargs.get("name") or "manual_profile").lower().replace(" ", "_"),
        "name": clean_text(kwargs.get("name")),
        "email": clean_text(kwargs.get("email")),
        "phone": clean_text(kwargs.get("phone")),
        "linkedin": clean_text(kwargs.get("linkedin")),
        "skills": normalize_list(kwargs.get("skills")),
        "education": clean_text(kwargs.get("education")),
        "experience_text": clean_text(kwargs.get("experience_text") or kwargs.get("experience")),
        "projects_publications": clean_text(
            kwargs.get("projects_publications")
            or " ".join(
                part
                for part in [clean_text(kwargs.get("projects")), clean_text(kwargs.get("publications"))]
                if part
            )
        ),
        "resume_source_text": clean_text(kwargs.get("resume_source_text")),
        "target_roles": normalize_list(kwargs.get("target_roles")),
        "location_preferences": normalize_list(kwargs.get("location_preferences")),
        "salary_min": parse_salary_preference(kwargs.get("salary_min") or kwargs.get("salary_preference")),
        "dealbreakers": normalize_list(kwargs.get("dealbreakers")),
        "visa_sponsorship": clean_text(kwargs.get("visa_sponsorship") or kwargs.get("visa_sponsorship_needs")),
        "needs_sponsorship": parse_bool(kwargs.get("needs_sponsorship", False)),
        "employment_types": normalize_list(kwargs.get("employment_types")),
        "excluded_employment_types": normalize_list(kwargs.get("excluded_employment_types")),
        "excluded_seniority": normalize_list(kwargs.get("excluded_seniority")),
        "max_years_required": kwargs.get("max_years_required"),
        "min_years_required": kwargs.get("min_years_required"),
        "remote_preference": clean_text(kwargs.get("remote_preference")),
        "us_only": parse_bool(kwargs.get("us_only", False)),
        "strict_location": parse_bool(kwargs.get("strict_location", False)),
        "salary_is_dealbreaker": parse_bool(kwargs.get("salary_is_dealbreaker", False)),
        "strict_salary": parse_bool(kwargs.get("strict_salary", False)),
        "preferred_company_types": normalize_list(kwargs.get("preferred_company_types")),
        "excluded_company_types": normalize_list(kwargs.get("excluded_company_types")),
        "required_role_families": normalize_list(kwargs.get("required_role_families")),
        "preferred_role_families": normalize_list(kwargs.get("preferred_role_families")),
        "strict_role_family": parse_bool(kwargs.get("strict_role_family", False)),
        "title_must_include_any": normalize_list(kwargs.get("title_must_include_any")),
        "required_title_signals": normalize_list(kwargs.get("required_title_signals")),
        "avoid_generic_backend_devops": parse_bool(kwargs.get("avoid_generic_backend_devops", False)),
        "avoid_defense_or_clearance": parse_bool(kwargs.get("avoid_defense_or_clearance", False)),
        "avoid_overly_senior": parse_bool(kwargs.get("avoid_overly_senior", False)),
        "new_grad_or_student_profile": parse_bool(kwargs.get("new_grad_or_student_profile", False)),
        "hard_reject_seniority_terms": normalize_list(kwargs.get("hard_reject_seniority_terms")),
        "penalize_seniority_terms": normalize_list(kwargs.get("penalize_seniority_terms")),
        "notes": clean_text(kwargs.get("notes")),
    }
    for key in ("max_years_required", "min_years_required"):
        value = profile.get(key)
        if value in (None, ""):
            profile[key] = None
        else:
            try:
                profile[key] = int(float(value))
            except (TypeError, ValueError):
                profile[key] = None
    return profile


def parse_profile_text(text: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse resume/profile text into the canonical profile object.

    This is intentionally lightweight. It extracts obvious sections and skill
    keywords, then lets structured overrides fill preferences.
    """

    cleaned = clean_text(text)
    overrides = overrides or {}
    extracted_skills = extract_skills("", cleaned).split("|") if cleaned else []
    profile = build_profile(
        name=overrides.get("name", ""),
        skills=overrides.get("skills") or extracted_skills,
        education=overrides.get("education") or _extract_section(cleaned, ["education"]),
        experience_text=overrides.get("experience_text")
        or _extract_section(cleaned, ["experience", "work experience", "professional experience"])
        or cleaned[:4000],
        projects_publications=overrides.get("projects_publications")
        or " ".join(
            part
            for part in [
                _extract_section(cleaned, ["projects", "selected projects"]),
                _extract_section(cleaned, ["publications", "research"]),
            ]
            if part
        ),
        **{
            key: value
            for key, value in overrides.items()
            if key
            not in {
                "name",
                "skills",
                "education",
                "experience",
                "experience_text",
                "projects",
                "publications",
                "projects_publications",
            }
        },
    )
    return profile


def load_profile_json(path: str | Path) -> dict[str, Any]:
    """Load a structured profile JSON file and normalize it."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return build_profile(**payload)


def profile_to_text(profile: dict[str, Any]) -> str:
    """Create a matching text representation for embedding and TF-IDF queries."""

    parts = [
        profile.get("name", ""),
        "Target roles: " + ", ".join(normalize_list(profile.get("target_roles"))),
        "Skills: " + ", ".join(normalize_list(profile.get("skills"))),
        "Education: " + clean_text(profile.get("education")),
        "Experience: " + clean_text(profile.get("experience_text")),
        "Projects and publications: " + clean_text(profile.get("projects_publications")),
        "Location preferences: " + ", ".join(normalize_list(profile.get("location_preferences"))),
        "Dealbreakers: " + ", ".join(normalize_list(profile.get("dealbreakers"))),
    ]
    return clean_text(" | ".join(part for part in parts if clean_text(part)))
