"""Shared location matching helpers for persona-specific hard filters."""

from __future__ import annotations

import re
from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text


US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
    "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
    "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
}

US_STATE_NAMES = (
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
)

MAJOR_US_LOCATION_TERMS = (
    "chicago", "new york", "nyc", "boston", "seattle", "san francisco", "san jose",
    "oakland", "los angeles", "austin", "dallas", "houston", "atlanta", "denver",
    "washington dc", "washington, dc", "miami", "phoenix", "philadelphia",
)

BAY_AREA_TERMS = (
    "san francisco", "sf", "san jose", "oakland", "palo alto", "mountain view",
    "santa clara", "sunnyvale", "cupertino", "fremont", "menlo park", "redwood city",
    "berkeley", "bay area", "silicon valley", "california",
)

NON_US_COUNTRY_TERMS = (
    "uk", "united kingdom", "england", "scotland", "wales", "northern ireland",
    "ireland", "australia", "canada", "india", "singapore", "germany", "france",
)

NON_US_CITY_TERMS = (
    "london", "slough", "sheffield", "uxbridge", "dublin", "sydney", "toronto",
    "vancouver", "middlesex",
)

REMOTE_TERMS = (
    "remote",
    "remote us",
    "remote united states",
    "remote, united states",
    "remote - us",
    "remote within the united states",
    "anywhere in the us",
    "anywhere in the united states",
)

REMOTE_NEGATIONS = (
    "not remote",
    "no remote",
    "onsite only",
    "on-site only",
)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _contains_term(text: str, term: str) -> bool:
    normalized = f" {_norm(text)} "
    normalized_term = _norm(term)
    return bool(normalized_term and re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized))


def _field_text(job: dict[str, Any], fields: list[str]) -> str:
    return " ".join(clean_text(job.get(field)) for field in fields if clean_text(job.get(field)))


def _location_text(job: dict[str, Any]) -> str:
    return _field_text(job, ["country", "state", "city", "location"])


def _remote_text(job: dict[str, Any]) -> str:
    description = clean_text(job.get("description_text"))[:1200]
    return " ".join([_location_text(job), clean_text(job.get("title")), description])


def is_remote_job(job: dict[str, Any]) -> bool:
    status = _norm(job.get("is_remote"))
    if status in {"remote", "true", "yes", "1"}:
        return True
    text = _remote_text(job)
    if any(_contains_term(text, phrase) for phrase in REMOTE_NEGATIONS):
        return False
    return any(_contains_term(text, phrase) for phrase in REMOTE_TERMS)


def has_explicit_us_signal(job: dict[str, Any]) -> bool:
    country = _norm(job.get("country"))
    state = clean_text(job.get("state")).upper().strip()
    text = _location_text(job)
    if country in {"us", "usa", "united states", "united states of america", "u s", "u s a"}:
        return True
    if state in US_STATE_CODES:
        return True
    if any(_contains_term(text, phrase) for phrase in ("united states", "usa", "u.s.", "u.s.a.", "us")):
        return True
    if any(_contains_term(text, name) for name in US_STATE_NAMES):
        return True
    if any(re.search(rf"(?<![A-Z0-9]){code}(?![A-Z0-9])", f" {text.upper()} ") for code in US_STATE_CODES):
        return True
    return any(_contains_term(text, term) for term in MAJOR_US_LOCATION_TERMS)


def is_explicit_non_us_location(job: dict[str, Any]) -> bool:
    country = _norm(job.get("country"))
    text = _location_text(job)
    if country and country not in {"us", "usa", "united states", "united states of america", "u s", "u s a", "nan", "none", "null"}:
        return True
    if any(_contains_term(text, term) for term in NON_US_COUNTRY_TERMS):
        return True
    if any(_contains_term(text, term) for term in NON_US_CITY_TERMS):
        return not has_explicit_us_signal(job)
    return False


def is_us_location(job: dict[str, Any]) -> bool:
    if is_explicit_non_us_location(job):
        return False
    return has_explicit_us_signal(job)


def matches_bay_area(job: dict[str, Any]) -> bool:
    if is_explicit_non_us_location(job):
        return False
    text = _location_text(job)
    return any(_contains_term(text, term) for term in BAY_AREA_TERMS)


def is_remote_or_preferred_region_profile(profile: dict[str, Any]) -> bool:
    remote_preference = clean_text(profile.get("remote_preference")).lower()
    preferences = {pref.lower() for pref in normalize_list(profile.get("location_preferences"))}
    return remote_preference == "remote_or_bay_area" or (
        "remote" in preferences and "bay area" in preferences
    )


def location_violation_reason(job: dict[str, Any], profile: dict[str, Any]) -> str | None:
    if is_remote_or_preferred_region_profile(profile):
        if is_explicit_non_us_location(job):
            return "strict_location_preference_mismatch:explicit_non_us_location"
        if is_remote_job(job):
            return None
        if matches_bay_area(job):
            return None
        return "strict_location_preference_mismatch"

    if _bool(profile.get("us_only")):
        if is_explicit_non_us_location(job):
            return "explicit_non_us_location"
        if is_us_location(job):
            return None
        if is_remote_job(job):
            return None
        return "location_not_us"

    preferences = [pref.lower() for pref in normalize_list(profile.get("location_preferences"))]
    if not preferences:
        return None
    location = _norm(job.get("location"))
    if "remote" in preferences and is_remote_job(job):
        return None
    if "united states" in preferences and is_us_location(job):
        return None
    if any(pref and _contains_term(location, pref) for pref in preferences if pref != "remote"):
        return None
    return "location_preference_mismatch"


def matches_location_preferences(job: dict[str, Any], profile: dict[str, Any]) -> bool:
    return location_violation_reason(job, profile) is None


def location_uncertainty_note(job: dict[str, Any], profile: dict[str, Any]) -> str | None:
    if _bool(profile.get("us_only")) and is_remote_job(job) and not is_us_location(job) and not is_explicit_non_us_location(job):
        return "remote_location_us_uncertain"
    return None


def location_explanation(job: dict[str, Any], profile: dict[str, Any]) -> str:
    reason = location_violation_reason(job, profile)
    if reason:
        hard_location = _bool(profile.get("strict_location")) or _bool(profile.get("us_only"))
        prefix = "Rejected" if hard_location else "Location preference not matched"
        if reason == "explicit_non_us_location":
            return "Rejected: explicit non-US location for US-only profile"
        if reason == "strict_location_preference_mismatch:explicit_non_us_location":
            return f"{prefix}: explicit non-US location and does not match profile location preference"
        if reason == "strict_location_preference_mismatch":
            return f"{prefix}: does not match profile location preference"
        return f"{prefix}: {reason.replace('_', ' ')}"
    if is_remote_or_preferred_region_profile(profile):
        if is_remote_job(job):
            return "Remote role matches profile location preference"
        if matches_bay_area(job):
            return "Preferred-region location matches profile preference"
    if _bool(profile.get("us_only")):
        note = location_uncertainty_note(job, profile)
        if note:
            return "Location accepted with uncertainty: remote role without explicit US/non-US signal"
        return "Location matches US-only preference"
    return "Location or remote preference is compatible"


def detect_location_signals(job: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "is_remote": is_remote_job(job),
        "is_us_location": is_us_location(job),
        "explicit_non_us_location": is_explicit_non_us_location(job),
        "matches_bay_area": matches_bay_area(job),
    }
    if profile is not None:
        payload["matches_profile_location"] = matches_location_preferences(job, profile)
        payload["violation_reason"] = location_violation_reason(job, profile)
        payload["uncertainty_note"] = location_uncertainty_note(job, profile)
    return payload
