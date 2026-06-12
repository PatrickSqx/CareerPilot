"""Profile-aware query generation for optional live job search."""

from __future__ import annotations

import re
from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text


ROLE_FAMILY_QUERY_HINTS: dict[str, list[str]] = {
    "ml_related": ["machine learning engineer", "data scientist", "applied scientist"],
    "research_ai": ["applied scientist", "research scientist"],
    "ml_infra": ["mlops engineer", "machine learning platform engineer", "ai infrastructure engineer"],
    "analytics_entry": ["data analyst", "business analyst", "analytics engineer"],
    "bi_analytics": ["bi analyst", "business intelligence analyst", "bi engineer"],
    "data_engineering": ["data engineer", "etl engineer", "data platform engineer"],
    "software_backend": ["software engineer", "backend engineer", "python developer"],
}


def _norm_query(value: Any) -> str:
    text = re.sub(r"\s+", " ", clean_text(value).strip())
    return text


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _norm_query(value)
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _spread_terms(values: list[str], count: int) -> list[str]:
    values = _dedupe(values)
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[0]]
    step = (len(values) - 1) / (count - 1)
    indexes = [round(index * step) for index in range(count)]
    return _dedupe([values[index] for index in indexes])


def _role_terms(profile: dict[str, Any]) -> list[str]:
    roles = normalize_list(profile.get("target_roles"))
    family_terms: list[str] = []
    for family in normalize_list(profile.get("required_role_families")) + normalize_list(profile.get("preferred_role_families")):
        family_terms.extend(ROLE_FAMILY_QUERY_HINTS.get(family.lower(), []))
    return _dedupe([*roles, *family_terms])


def _location_terms(profile: dict[str, Any]) -> list[str]:
    preferences = normalize_list(profile.get("location_preferences"))
    terms: list[str] = []
    remote_preference = clean_text(profile.get("remote_preference")).lower()
    if "remote" in {pref.lower() for pref in preferences} or "remote" in remote_preference:
        terms.append("remote")
    for preference in preferences:
        lowered = preference.lower()
        if lowered in {"remote", "united states", "usa", "us"}:
            continue
        terms.append(preference)
    if remote_preference == "remote_or_bay_area":
        terms.extend(["San Francisco", "Bay Area", "San Jose", "California"])
    return _dedupe(terms)


def _seniority_terms(profile: dict[str, Any]) -> list[str]:
    excluded = {item.lower() for item in normalize_list(profile.get("excluded_seniority"))}
    if excluded & {"senior", "staff_principal", "lead_manager"}:
        return ["junior", "entry level"]
    return []


def build_live_queries(profile: dict[str, Any], *, max_queries: int = 5) -> list[str]:
    """Build lightweight live-search queries from generic profile fields."""

    roles = _role_terms(profile)
    primary_roles = roles[:4] or roles
    locations = _location_terms(profile)
    seniority_terms = _seniority_terms(profile)

    queries: list[str] = []
    remote_locations = [location for location in locations if location.lower() == "remote"]
    non_remote_locations = [location for location in locations if location.lower() != "remote"]

    has_remote = bool(remote_locations)
    has_region = bool(non_remote_locations)
    has_lower_seniority = bool(seniority_terms)
    lower_seniority_slots = 1 if has_lower_seniority and max_queries >= 3 else 0
    remaining_slots = max(max_queries - lower_seniority_slots, 0)
    if has_remote and has_region and remaining_slots >= 4:
        remote_slots = 2
        region_slots = remaining_slots - remote_slots
    elif has_remote and has_region:
        remote_slots = max(1, remaining_slots // 2)
        region_slots = remaining_slots - remote_slots
    elif has_remote:
        remote_slots = remaining_slots
        region_slots = 0
    elif has_region:
        remote_slots = 0
        region_slots = remaining_slots
    else:
        remote_slots = 0
        region_slots = 0

    remote_roles = _spread_terms(primary_roles, remote_slots)
    for role in remote_roles:
        queries.append(_norm_query(f"{role} remote"))

    region_roles = _spread_terms(primary_roles, region_slots)
    region_terms = _spread_terms(non_remote_locations, region_slots)
    for index, role in enumerate(region_roles):
        if not region_terms:
            break
        location = region_terms[index % len(region_terms)]
        queries.append(_norm_query(f"{role} {location}"))

    if lower_seniority_slots:
        lower_role = (_spread_terms(primary_roles, 2) or primary_roles)[-1]
        lower_location = "remote" if has_remote else (non_remote_locations[0] if non_remote_locations else "")
        for seniority in seniority_terms[:lower_seniority_slots]:
            queries.append(_norm_query(f"{seniority} {lower_role} {lower_location}"))

    if not queries:
        queries.extend(roles)

    return _dedupe(queries or roles)[:max_queries]
