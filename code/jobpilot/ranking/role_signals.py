"""Role-family and title relevance signals for ranking and evaluation."""

from __future__ import annotations

import re
from typing import Any

from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text


ROLE_FAMILY_TITLE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "ml_related": (
        re.compile(r"\bmachine learning engineer\b", re.IGNORECASE),
        re.compile(r"\bml engineer\b", re.IGNORECASE),
        re.compile(r"\bdata scientist\b", re.IGNORECASE),
        re.compile(r"\bapplied scientist\b", re.IGNORECASE),
        re.compile(r"\bresearch scientist\b", re.IGNORECASE),
        re.compile(r"\bai engineer\b", re.IGNORECASE),
        re.compile(r"\bartificial intelligence engineer\b", re.IGNORECASE),
    ),
    "research_ai": (
        re.compile(r"\bresearch scientist\b", re.IGNORECASE),
        re.compile(r"\bai researcher\b", re.IGNORECASE),
        re.compile(r"\bmachine learning researcher\b", re.IGNORECASE),
        re.compile(r"\bapplied scientist\b", re.IGNORECASE),
    ),
    "ml_infra": (
        re.compile(r"\bmlops engineer\b", re.IGNORECASE),
        re.compile(r"\bmachine learning ops engineer\b", re.IGNORECASE),
        re.compile(r"\bml platform engineer\b", re.IGNORECASE),
        re.compile(r"\bml infrastructure engineer\b", re.IGNORECASE),
        re.compile(r"\bmachine learning platform engineer\b", re.IGNORECASE),
        re.compile(r"\bmachine learning infrastructure engineer\b", re.IGNORECASE),
        re.compile(r"\bai infrastructure engineer\b", re.IGNORECASE),
        re.compile(r"\bai\s*/\s*mlops engineer\b", re.IGNORECASE),
        re.compile(r"\bmlops\b", re.IGNORECASE),
    ),
    "analytics_entry": (
        re.compile(r"\bdata analyst\b", re.IGNORECASE),
        re.compile(r"\bjunior data scientist\b", re.IGNORECASE),
        re.compile(r"\banalytics analyst\b", re.IGNORECASE),
        re.compile(r"\bbusiness analyst\b", re.IGNORECASE),
        re.compile(r"\banalytics engineer\b", re.IGNORECASE),
    ),
    "bi_analytics": (
        re.compile(r"\bbi analyst\b", re.IGNORECASE),
        re.compile(r"\bbi engineer\b", re.IGNORECASE),
        re.compile(r"\bbusiness intelligence analyst\b", re.IGNORECASE),
        re.compile(r"\bbusiness intelligence engineer\b", re.IGNORECASE),
        re.compile(r"\btableau developer\b", re.IGNORECASE),
        re.compile(r"\bpower bi developer\b", re.IGNORECASE),
    ),
    "data_engineering": (
        re.compile(r"\bdata engineer\b", re.IGNORECASE),
        re.compile(r"\betl engineer\b", re.IGNORECASE),
        re.compile(r"\bbig data engineer\b", re.IGNORECASE),
        re.compile(r"\bhadoop engineer\b", re.IGNORECASE),
        re.compile(r"\bdata platform engineer\b", re.IGNORECASE),
    ),
    "software_backend": (
        re.compile(r"\bsoftware engineer\b", re.IGNORECASE),
        re.compile(r"\bbackend engineer\b", re.IGNORECASE),
        re.compile(r"\bjava developer\b", re.IGNORECASE),
        re.compile(r"\bpython developer\b", re.IGNORECASE),
        re.compile(r"\bapplication developer\b", re.IGNORECASE),
    ),
}

ROLE_FAMILY_DESCRIPTION_TERMS: dict[str, tuple[str, ...]] = {
    "ml_related": (
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "model training",
        "predictive modeling",
        "computer vision",
        "natural language processing",
    ),
    "research_ai": (
        "research scientist",
        "ai research",
        "machine learning research",
        "publish",
        "publication",
        "experiment",
    ),
    "ml_infra": (
        "mlops",
        "model serving",
        "feature platform",
        "feature store",
        "kubernetes",
        "spark",
        "kafka",
    ),
    "analytics_entry": (
        "dashboard",
        "business analytics",
        "reporting",
        "data analysis",
        "tableau",
        "power bi",
    ),
    "bi_analytics": (
        "business intelligence",
        "bi dashboard",
        "tableau",
        "power bi",
        "looker",
    ),
    "data_engineering": (
        "data pipeline",
        "etl",
        "data warehouse",
        "hadoop",
        "big data",
        "airflow",
    ),
    "software_backend": (
        "backend service",
        "microservice",
        "api development",
        "java",
        "python",
        "application development",
    ),
}

WEAK_NON_ML_TITLE_PATTERNS = (
    re.compile(r"\bbi engineer\b", re.IGNORECASE),
    re.compile(r"\bdata engineer\b", re.IGNORECASE),
    re.compile(r"\bdevops engineer\b", re.IGNORECASE),
    re.compile(r"\bcloud engineer\b", re.IGNORECASE),
    re.compile(r"\bsecurity support engineer\b", re.IGNORECASE),
    re.compile(r"\bautomation engineer\b", re.IGNORECASE),
    re.compile(r"\bmarketing engineer\b", re.IGNORECASE),
    re.compile(r"\bbusiness analyst\b", re.IGNORECASE),
    re.compile(r"\bdata analyst\b", re.IGNORECASE),
    re.compile(r"\bpython developer\b", re.IGNORECASE),
    re.compile(r"\bhadoop admin\b", re.IGNORECASE),
    re.compile(r"\bbig data admin\b", re.IGNORECASE),
    re.compile(r"\bdata labeling analyst\b", re.IGNORECASE),
)

GENERIC_BACKEND_DEVOPS_TITLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "devops": re.compile(r"\bdevops\b", re.IGNORECASE),
    "backend": re.compile(r"\bback[-\s]?end\b", re.IGNORECASE),
    "software_engineer": re.compile(r"\bsoftware engineer\b", re.IGNORECASE),
    "fullstack": re.compile(r"\bfull[-\s]?stack\b", re.IGNORECASE),
    "java": re.compile(r"\bjava (?:developer|engineer|software)\b", re.IGNORECASE),
    "cloud": re.compile(r"\bcloud engineer\b", re.IGNORECASE),
    "security": re.compile(r"\bsecurity (?:support )?engineer\b", re.IGNORECASE),
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) >= 2}


def _title(job: dict[str, Any]) -> str:
    return clean_text(job.get("title"))


def _description(job: dict[str, Any]) -> str:
    return clean_text(job.get("description_text") or job.get("description"))[:2000]


def _dedupe_families(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def detect_role_families_from_title(title: Any) -> list[str]:
    """Detect role-family labels from the job title only."""

    title_text = clean_text(title)
    hits: list[str] = []
    for family, patterns in ROLE_FAMILY_TITLE_PATTERNS.items():
        if any(pattern.search(title_text) for pattern in patterns):
            hits.append(family)
    return hits


def detect_role_families_from_description(description: Any) -> list[str]:
    """Detect weak role-family labels from short description text."""

    text = f" {_norm(description)} "
    hits: list[str] = []
    for family, terms in ROLE_FAMILY_DESCRIPTION_TERMS.items():
        if any(f" {_norm(term)} " in text for term in terms):
            hits.append(family)
    return hits


def detect_role_family_signals(job: dict[str, Any]) -> dict[str, Any]:
    """Return title-first and description-fallback role-family signals."""

    title_families = detect_role_families_from_title(_title(job))
    description_families = detect_role_families_from_description(_description(job))
    return {
        "title_families": title_families,
        "description_families": description_families,
        "detected_families": _dedupe_families(title_families + description_families),
    }


def role_family_match_details(profile: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Describe required/preferred role-family matches for a profile and job."""

    required = [item.lower() for item in normalize_list(profile.get("required_role_families"))]
    preferred = [item.lower() for item in normalize_list(profile.get("preferred_role_families"))]
    signals = detect_role_family_signals(job)
    title_families = set(signals["title_families"])
    description_families = set(signals["description_families"])
    required_title_matches = [family for family in required if family in title_families]
    required_description_matches = [family for family in required if family in description_families]
    preferred_title_matches = [family for family in preferred if family in title_families]
    preferred_description_matches = [family for family in preferred if family in description_families]
    strict = bool(profile.get("strict_role_family"))
    required_pass = not required or bool(required_title_matches)
    strict_pass = not strict or required_pass
    failure_reason = ""
    if strict and required and not required_title_matches:
        failure_reason = "not_required_role_family"
    return {
        **signals,
        "required_role_families": required,
        "preferred_role_families": preferred,
        "strict_role_family": strict,
        "required_title_matches": required_title_matches,
        "required_description_matches": required_description_matches,
        "preferred_title_matches": preferred_title_matches,
        "preferred_description_matches": preferred_description_matches,
        "required_pass": required_pass,
        "strict_pass": strict_pass,
        "failure_reason": failure_reason,
    }


def matches_required_role_family(profile: dict[str, Any], job: dict[str, Any]) -> bool:
    """Return whether a job satisfies a strict required role-family check."""

    return bool(role_family_match_details(profile, job)["strict_pass"])


def role_family_relevance_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    """Score role-family relevance, keeping description-only hits weak."""

    details = role_family_match_details(profile, job)
    if details["required_title_matches"]:
        return 1.0
    if details["preferred_title_matches"]:
        return 0.9
    if details["required_description_matches"]:
        return 0.45
    if details["preferred_description_matches"]:
        return 0.35
    return 0.0


def weak_non_ml_title_hits(job: dict[str, Any]) -> list[str]:
    title = _title(job)
    return [pattern.pattern for pattern in WEAK_NON_ML_TITLE_PATTERNS if pattern.search(title)]


def generic_backend_devops_title_hits(job: dict[str, Any]) -> list[str]:
    """Return generic software/backend/DevOps title signals."""

    title = _title(job)
    return [label for label, pattern in GENERIC_BACKEND_DEVOPS_TITLE_PATTERNS.items() if pattern.search(title)]


def has_strict_role_family_requirement(profile: dict[str, Any]) -> bool:
    return bool(profile.get("strict_role_family")) and bool(normalize_list(profile.get("required_role_families")))


def strong_required_role_title_signal(profile: dict[str, Any], job: dict[str, Any]) -> bool:
    """Return whether a title matches at least one profile-required role family."""

    return bool(role_family_match_details(profile, job)["required_title_matches"])


def title_contains_profile_signal(profile: dict[str, Any], job: dict[str, Any]) -> bool:
    """Check profile-configured title tokens such as ML, MLOps, AI, or infrastructure."""

    signals = normalize_list(profile.get("required_title_signals")) or normalize_list(profile.get("title_must_include_any"))
    if not signals:
        return True
    title = f" {_norm(_title(job))} "
    for signal in signals:
        normalized = _norm(signal)
        if normalized and re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", title):
            return True
    return False


def generic_backend_devops_without_target_signal(profile: dict[str, Any], job: dict[str, Any]) -> bool:
    """Return true when a generic backend/DevOps title lacks the configured target-role title signal."""

    if not generic_backend_devops_title_hits(job):
        return False
    if strong_required_role_title_signal(profile, job):
        return False
    return not title_contains_profile_signal(profile, job)


def strict_required_role_title_relevance(profile: dict[str, Any], job: dict[str, Any]) -> bool:
    """Generic strict role-family title check for any profile."""

    if not has_strict_role_family_requirement(profile):
        return True
    return strong_required_role_title_signal(profile, job)


def title_target_role_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    roles = normalize_list(profile.get("target_roles"))
    if not roles:
        return 0.0
    title = _norm(_title(job))
    if not title:
        return 0.0
    title_tokens = _tokens(title)
    best = 0.0
    for role in roles:
        role_norm = _norm(role)
        if not role_norm:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(role_norm)}(?![a-z0-9])", f" {title} "):
            best = max(best, 1.0)
            continue
        role_tokens = _tokens(role_norm)
        if role_tokens:
            overlap = len(role_tokens & title_tokens) / len(role_tokens)
            best = max(best, overlap)
    if has_strict_role_family_requirement(profile) and weak_non_ml_title_hits(job) and not strong_required_role_title_signal(profile, job):
        best = min(best, 0.3)
    if profile.get("avoid_generic_backend_devops") and generic_backend_devops_without_target_signal(profile, job):
        best = min(best, 0.15)
    return min(1.0, best)


def description_target_role_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    roles = normalize_list(profile.get("target_roles"))
    if not roles:
        return 0.0
    description = _norm(_description(job))
    title_score = title_target_role_score(profile, job)
    if title_score >= 0.8:
        return title_score
    role_phrase_hit = any(_norm(role) and _norm(role) in description for role in roles)
    if role_phrase_hit:
        return 0.45
    return 0.0


def target_role_relevance_score(profile: dict[str, Any], job: dict[str, Any]) -> float:
    title_score = title_target_role_score(profile, job)
    family_score = role_family_relevance_score(profile, job)
    description_score = description_target_role_score(profile, job)
    return round(max(title_score, family_score, min(description_score, 0.45)), 6)
