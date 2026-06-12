"""Phase 2.18A deterministic profile-intake parser.

The parser is intentionally local and no-key by default. It turns pasted
persona/resume text plus optional PDF/DOCX text into the same canonical profile
dict that the existing ranking path already consumes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

from fastapi import UploadFile

from app.services.paths import PROJECT_ROOT, UPLOAD_DIR, ensure_storage_dirs
from jobpilot.profile.pdf_extractor import PDFExtractionError, extract_pdf_text
from jobpilot.profile.profile_parser import build_profile, normalize_list, parse_salary_preference
from jobpilot.utils.text import clean_text, extract_skills


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "background": ("background", "profile", "summary", "candidate background"),
    "skills": ("skills", "technical skills", "core skills", "toolkit"),
    "target_roles": ("target roles", "target role", "roles", "role targets", "desired roles"),
    "preferences": ("preferences", "job preferences", "role preferences", "constraints"),
    "dealbreakers": ("dealbreakers", "deal breakers", "hard no", "hard nos", "hard rejects", "must avoid"),
    "pass_criteria": ("pass criteria", "passing criteria", "must have", "requirements", "fit criteria"),
    "education": ("education",),
    "experience": ("experience", "work experience", "professional experience"),
    "projects": ("projects", "publications", "projects and publications", "selected projects"),
}

FIELD_TO_FORM: dict[str, str] = {
    "name": "manual_name",
    "email": "manual_email",
    "phone": "manual_phone",
    "linkedin": "manual_linkedin",
    "target_roles": "manual_target_roles",
    "required_role_families": "manual_required_role_families",
    "preferred_role_families": "manual_preferred_role_families",
    "strict_role_family": "manual_strict_role_family",
    "skills": "manual_skills",
    "education": "manual_education",
    "experience_text": "manual_experience",
    "projects_publications": "manual_projects",
    "max_years_required": "manual_max_years_required",
    "excluded_seniority": "manual_excluded_seniority",
    "salary_min": "manual_salary_min",
    "salary_is_dealbreaker": "manual_salary_is_dealbreaker",
    "location_preferences": "manual_location_preferences",
    "strict_location": "manual_strict_location",
    "visa_sponsorship": "manual_visa_sponsorship",
    "needs_sponsorship": "manual_needs_sponsorship",
    "us_only": "manual_us_only",
    "preferred_company_types": "manual_preferred_company_types",
    "excluded_company_types": "manual_excluded_company_types",
    "excluded_employment_types": "manual_excluded_employment_types",
    "dealbreakers": "manual_dealbreakers",
    "hard_reject_seniority_terms": "manual_hard_reject_seniority_terms",
    "penalize_seniority_terms": "manual_penalize_seniority_terms",
    "avoid_defense_or_clearance": "manual_avoid_defense_or_clearance",
}

EXECUTABLE_FILTER_FIELDS = {
    "required_role_families",
    "preferred_role_families",
    "strict_role_family",
    "max_years_required",
    "excluded_seniority",
    "salary_min",
    "salary_is_dealbreaker",
    "location_preferences",
    "strict_location",
    "needs_sponsorship",
    "us_only",
    "preferred_company_types",
    "excluded_company_types",
    "excluded_employment_types",
    "dealbreakers",
    "hard_reject_seniority_terms",
    "avoid_defense_or_clearance",
}

PROFILE_CONTEXT_FIELDS = {
    "name",
    "email",
    "phone",
    "linkedin",
    "target_roles",
    "skills",
    "education",
    "experience_text",
    "projects_publications",
    "visa_sponsorship",
}

DEFAULT_PROVIDER_BY_MODE = {
    "llm": "gemini",
    "provider": "gemini",
    "llm_provider": "gemini",
}

DEFAULT_MODELS_BY_PROVIDER = {
    "gemini": "gemini-2.5-flash",
}

MODEL_ENV_BY_PROVIDER = {
    "gemini": (
        "JOBPILOT_PROFILE_LLM_MODEL",
        "GEMINI_MODEL",
        "GOOGLE_GENERATIVE_AI_MODEL",
        "JOBPILOT_LLM_MODEL",
    ),
}

ROLE_CANONICALS: tuple[str, ...] = (
    "Machine Learning Engineer",
    "ML Engineer",
    "Senior ML Engineer",
    "Senior Machine Learning Engineer",
    "Data Scientist",
    "Applied Scientist",
    "Research Scientist",
    "AI Engineer",
    "ML Infrastructure Engineer",
    "MLOps Engineer",
    "Machine Learning Platform Engineer",
    "AI Infrastructure Engineer",
    "Data Analyst",
    "Business Analyst",
    "BI Analyst",
    "Analytics Engineer",
    "Junior Data Scientist",
    "Data Engineer",
    "Product Manager",
    "Software Engineer",
    "Backend Engineer",
)

ROLE_FAMILY_BY_PATTERN: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:machine learning|ml engineer|data scientist|applied scientist|ai engineer)\b", re.I), ("ml_related",)),
    (re.compile(r"\b(?:research scientist|ai research|machine learning research|applied scientist)\b", re.I), ("research_ai",)),
    (re.compile(r"\b(?:mlops|ml infrastructure|ml platform|ai infrastructure|machine learning platform)\b", re.I), ("ml_infra",)),
    (re.compile(r"\b(?:data analyst|business analyst|analytics analyst|analytics engineer|junior data scientist)\b", re.I), ("analytics_entry",)),
    (re.compile(r"\b(?:bi analyst|business intelligence|tableau|power bi)\b", re.I), ("bi_analytics",)),
    (re.compile(r"\b(?:data engineer|etl|data pipeline|data platform)\b", re.I), ("data_engineering",)),
    (re.compile(r"\b(?:software engineer|backend engineer|back end|backend)\b", re.I), ("software_backend",)),
)

COMPANY_PREF_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("large_company", ("large company", "large companies", "enterprise", "big tech", "faang", "established company")),
    ("research_lab", ("research lab", "ai lab", "r&d lab", "research organization")),
    ("startup", ("startup", "start-up", "series a", "series b", "series c", "early stage")),
)

COMPANY_EXCLUDE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("defense_military", ("defense", "military", "clearance", "security clearance", "government contractor", "dod")),
    ("startup", ("avoid startup", "avoid startups", "no startup", "no startups", "tiny startup", "early stage startup")),
)


@dataclass
class ParsedProfile:
    profile: dict[str, Any]
    form_fields: dict[str, Any]
    field_sources: dict[str, str]
    notes: list[str]
    parse_method: str = "rule_fallback"
    filter_fields: dict[str, Any] = field(default_factory=dict)
    context_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMProviderSettings:
    mode: str = "rule_fallback"
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: str = ""


LLM_PROFILE_CLIENT: Callable[[str], dict[str, Any]] | None = None
LLM_PROFILE_ALLOWED_FIELDS = set(FIELD_TO_FORM)
LLM_PROFILE_LIST_FIELDS = {
    "target_roles",
    "required_role_families",
    "preferred_role_families",
    "skills",
    "location_preferences",
    "preferred_company_types",
    "excluded_company_types",
    "excluded_employment_types",
    "excluded_seniority",
    "dealbreakers",
    "hard_reject_seniority_terms",
    "penalize_seniority_terms",
}
LLM_PROFILE_BOOL_FIELDS = {
    "strict_role_family",
    "salary_is_dealbreaker",
    "strict_location",
    "needs_sponsorship",
    "us_only",
    "avoid_defense_or_clearance",
}
DETERMINISTIC_GUARDRAIL_FIELDS = {
    "required_role_families",
    "preferred_role_families",
    "strict_role_family",
    "max_years_required",
    "excluded_seniority",
    "salary_min",
    "salary_is_dealbreaker",
    "location_preferences",
    "strict_location",
    "visa_sponsorship",
    "needs_sponsorship",
    "us_only",
    "preferred_company_types",
    "excluded_company_types",
    "excluded_employment_types",
    "dealbreakers",
    "hard_reject_seniority_terms",
    "penalize_seniority_terms",
    "avoid_defense_or_clearance",
}


def _preserve_newlines(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _heading_key(line: str) -> tuple[str | None, str]:
    line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line).strip()
    for key, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            match = re.match(rf"(?i)^{re.escape(alias)}\s*:?\s*(.*)$", line)
            if match:
                return key, match.group(1).strip()
    return None, ""


def _extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        key, remainder = _heading_key(raw_line)
        if key:
            current = key
            sections.setdefault(key, [])
            if remainder:
                sections[key].append(remainder)
            continue
        if current and raw_line.strip():
            sections[current].append(raw_line.strip())
    return {key: clean_text("\n".join(lines)) for key, lines in sections.items()}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value).strip(" .:-")
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _split_items(text: str) -> list[str]:
    text = re.sub(r"(?m)^\s*(?:[-*\u2022\u00b7\u2023\u25aa\u2013\u2014]\s*)+", "", text)
    text = re.sub(r"\b(?:and|or)\b", ",", text, flags=re.I)
    return normalize_list(text)


def _find_label_value(text: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(
            rf"(?is)\b{re.escape(label)}\b\s*:?\s*(.+?)(?=\n\s*[A-Z][A-Za-z /&-]{{2,35}}\s*:|$)",
            text,
        )
        if match:
            return clean_text(match.group(1))
    return ""


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _extract_name(text: str) -> str:
    match = re.search(r"(?im)^\s*(?:name|candidate)\s*:?\s*([A-Z][A-Za-z .'-]{1,60})\s*$", text)
    if match:
        return clean_text(match.group(1))
    first_line = next((clean_text(line) for line in text.splitlines() if clean_text(line)), "")
    if first_line and len(first_line.split()) <= 5 and not _heading_key(first_line)[0] and not first_line.endswith(":"):
        if not any(char.isdigit() for char in first_line):
            return first_line
    return ""


def _extract_roles(text: str, sections: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    explicit = sections.get("target_roles") or _find_label_value(text, ("target roles", "target role", "roles"))
    if explicit:
        candidates.extend(_split_items(explicit))
    lowered = text.lower()
    for role in ROLE_CANONICALS:
        pattern = re.escape(role.lower()).replace(r"\ ", r"[-\s/]+")
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", lowered):
            candidates.append(role)
    return _dedupe(candidates)


def _extract_skills(text: str, sections: dict[str, str]) -> list[str]:
    explicit = sections.get("skills") or _find_label_value(text, ("skills", "technical skills", "core skills"))
    candidates = _split_items(explicit) if explicit else []
    keyword_hits = [skill for skill in extract_skills("", text).split("|") if skill]
    return _dedupe(candidates + keyword_hits)


def _infer_role_families(text: str, roles: list[str], sections: dict[str, str]) -> tuple[list[str], list[str], bool]:
    explicit_required = _split_items(_find_label_value(text, ("required role families", "required role family")))
    explicit_preferred = _split_items(_find_label_value(text, ("preferred role families", "preferred role family")))
    blob = " ".join([text, " ".join(roles)])
    inferred: list[str] = []
    for pattern, families in ROLE_FAMILY_BY_PATTERN:
        if pattern.search(blob):
            inferred.extend(families)
    preferred = _dedupe(explicit_preferred + inferred)
    required = _dedupe(explicit_required)
    pass_text = sections.get("pass_criteria", "")
    strict_role_family = bool(required) or _contains_any(
        pass_text + " " + sections.get("preferences", ""),
        (
            "strict role",
            "must be target",
            "must match target",
            "only target",
            "only these roles",
            "roles only",
            "role only",
            "only roles",
            "ml-related",
            "research ai",
            "required role family",
        ),
    )
    if strict_role_family and not required:
        required = preferred[:]
    return required, preferred, strict_role_family


def _extract_location(text: str, sections: dict[str, str]) -> tuple[list[str], bool, bool]:
    location_text = " ".join(
        part
        for part in [
            sections.get("preferences", ""),
            _find_label_value(text, ("location preferences", "locations", "location")),
            text,
        ]
        if part
    )
    candidates: list[str] = []
    known_locations = (
        "United States",
        "US",
        "U.S.",
        "Remote",
        "New York",
        "NYC",
        "San Francisco",
        "Bay Area",
        "Seattle",
        "Chicago",
        "Boston",
        "Austin",
        "Los Angeles",
        "California",
    )
    for location in known_locations:
        pattern = re.escape(location).replace(r"\ ", r"[-\s]+")
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", location_text, re.I):
            candidates.append("United States" if location in {"US", "U.S."} else location)
    us_only = bool(
        re.search(
            r"\b(?:us|u\.s\.|united states)\s*(?:only|based|required|roles?)\b|\bonly\s+(?:in|within)\s+(?:the\s+)?(?:us|u\.s\.|united states)\b",
            location_text,
            re.I,
        )
    )
    strict_location = us_only or bool(re.search(r"\b(?:strict|must|only)\b.{0,30}\b(?:location|remote|us|united states)\b", location_text, re.I))
    if us_only and "United States" not in candidates:
        candidates.insert(0, "United States")
    return _dedupe(candidates), us_only, strict_location


def _extract_salary_min(text: str, sections: dict[str, str]) -> tuple[int | None, bool]:
    salary_text = " ".join(
        part
        for part in [
            sections.get("preferences", ""),
            sections.get("dealbreakers", ""),
            _find_label_value(text, ("salary minimum", "minimum salary", "min salary", "compensation", "base salary")),
            text,
        ]
        if part
    )
    patterns = (
        r"\b(?:salary|compensation|base|minimum|min|at least|threshold)\b[^.\n]{0,45}?\$?\s*((?:\d{5,})|\d{2,3}(?:,\d{3})?|\d{2,3}(?:\.\d+)?\s*k)",
        r"\$\s*((?:\d{5,})|\d{2,3}(?:,\d{3})?|\d{2,3}(?:\.\d+)?\s*k)",
    )
    for pattern in patterns:
        match = re.search(pattern, salary_text, re.I)
        if match:
            parsed = parse_salary_preference(match.group(1))
            if parsed:
                strict = bool(re.search(r"\b(?:dealbreaker|hard|must|minimum|min|at least|below)\b", salary_text, re.I))
                return parsed, strict
    return None, False


def _extract_year_cap(text: str, sections: dict[str, str]) -> int | None:
    years_text = " ".join([sections.get("preferences", ""), sections.get("dealbreakers", ""), sections.get("pass_criteria", ""), text])
    plus_match = re.search(r"\b(?:no|avoid|reject|exclude)\b[^.\n]{0,25}?(\d+)\s*\+\s*(?:years|yrs)\b", years_text, re.I)
    if plus_match:
        return max(0, int(plus_match.group(1)) - 1)
    for pattern in (
        r"\b(?:no|avoid|reject|exclude)\b[^.\n]{0,45}\b(?:more than|over)\s*(\d+)\s*(?:years|yrs)\b",
        r"\b(?:max(?:imum)?|up to|no more than|at most)\s*(\d+)\s*(?:years|yrs)\b",
        r"\b0\s*[-\u2013\u2014]\s*(\d+)\s*(?:years|yrs)\b",
    ):
        match = re.search(pattern, years_text, re.I)
        if match:
            return int(match.group(1))
    negative_entry_constraint = re.search(
        r"\b(?:no|avoid|exclude|reject|not open to|dealbreaker)\b[^.\n]{0,35}\b(?:junior|jr\.?|entry[-\s]?level|intern(?:ship)?|new[-\s]?grad)\b",
        years_text,
        re.I,
    )
    if not negative_entry_constraint and re.search(r"\b(?:new grad(?:uate)?|entry[-\s]?level|early career)\b", years_text, re.I):
        return 2
    return None


def _extract_sponsorship(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    no_need = (
        "do not need sponsorship",
        "does not need sponsorship",
        "don't need sponsorship",
        "no sponsorship needed",
        "not need sponsorship",
        "us citizen",
        "u.s. citizen",
        "green card",
        "permanent resident",
    )
    if any(term in lowered for term in no_need):
        return False, ""
    needs = bool(
        re.search(
            r"\b(?:need|needs|require|requires|requiring|will need|must have).{0,35}\b(?:sponsorship|h-?1b|visa)\b|\b(?:h-?1b|opt|cpt|visa sponsorship|sponsor visa|international student)\b",
            text,
            re.I,
        )
    )
    if needs:
        sentence_match = re.search(r"(?i)([^.\n]*(?:sponsorship|h-?1b|visa)[^.\n]*)", text)
        if not sentence_match:
            sentence_match = re.search(r"(?i)([^.\n]*(?:opt|cpt|international student)[^.\n]*)", text)
        return True, clean_text(sentence_match.group(1) if sentence_match else "Needs visa sponsorship.")
    return False, ""


def _extract_employment_exclusions(text: str) -> list[str]:
    exclusions: list[str] = []
    checks = (
        ("contract", r"\b(?:no|avoid|exclude|not open to|dealbreaker).{0,30}\bcontract\b|\bfull[-\s]?time only\b|\bcontract[-\s]?only\b"),
        ("temporary", r"\b(?:no|avoid|exclude|not open to|dealbreaker).{0,30}\b(?:temporary|temp)\b|\btemporary\b"),
        ("unpaid", r"\b(?:no|avoid|exclude|not open to|dealbreaker).{0,30}\bunpaid\b|\bunpaid\b"),
        ("internship", r"\b(?:no|avoid|exclude|not open to|dealbreaker).{0,30}\b(?:intern|internship)\b"),
    )
    for label, pattern in checks:
        if re.search(pattern, text, re.I):
            exclusions.append(label)
    return _dedupe(exclusions)


def _extract_seniority_constraints(text: str, max_years_required: int | None) -> tuple[list[str], list[str], list[str], bool]:
    lowered = text.lower()
    senior_terms = ("senior", "sr", "staff", "principal", "lead", "manager", "director", "head", "distinguished")
    explicit_no_senior = bool(
        re.search(
            r"\b(?:no|avoid|exclude|reject|not open to|dealbreaker)\b[^.\n]{0,45}\b(?:senior|sr\.?|staff|principal|lead|manager|director|head|distinguished)\b",
            text,
            re.I,
        )
    )
    explicit_no_junior = bool(
        re.search(
            r"\b(?:no|avoid|exclude|reject|not open to|dealbreaker)\b[^.\n]{0,45}\b(?:junior|jr\.?|entry[-\s]?level|intern(?:ship)?|new[-\s]?grad)\b",
            text,
            re.I,
        )
    )
    entry_profile = bool(
        re.search(r"\b(?:new grad(?:uate)?|entry[-\s]?level|early career)\b", text, re.I)
        and not explicit_no_junior
    )
    excluded: list[str] = []
    hard_reject: list[str] = []
    penalize: list[str] = []
    senior_guardrail = False
    if explicit_no_senior or entry_profile or (max_years_required is not None and max_years_required <= 3):
        if "senior" in lowered or entry_profile or max_years_required is not None:
            excluded.append("senior")
            penalize.extend(["senior", "sr", "iii"])
        if any(term in lowered for term in senior_terms[2:]) or explicit_no_senior:
            excluded.extend(["staff_principal", "lead_manager"])
            hard_reject.extend(["staff", "principal", "director", "lead", "manager"])
        senior_guardrail = True
    if explicit_no_junior:
        excluded.extend(["entry_junior", "internship"])
        hard_reject.extend(["junior", "jr", "entry level", "new grad", "intern", "internship"])
    avoid_overly_senior = bool(senior_guardrail and (hard_reject or penalize))
    return _dedupe(excluded), _dedupe(hard_reject), _dedupe(penalize), avoid_overly_senior


def _extract_company_types(text: str, sections: dict[str, str]) -> tuple[list[str], list[str], bool]:
    pref_text = " ".join([sections.get("preferences", ""), sections.get("pass_criteria", ""), text])
    dealbreaker_text = " ".join([sections.get("dealbreakers", ""), sections.get("preferences", ""), text])
    preferred: list[str] = []
    excluded: list[str] = []
    for label, terms in COMPANY_PREF_TERMS:
        if any(term in pref_text.lower() for term in terms):
            if label == "startup" and re.search(
                r"\b(?:avoid|no|exclude|not|dealbreaker).{0,25}(?:startups?|start-ups?|early stage)\b",
                pref_text,
                re.I,
            ):
                excluded.append(label)
            else:
                preferred.append(label)
    large_company_size_preference = bool(
        re.search(
            r"(?:\b(?:companies?|employers?)\s+(?:with\s+)?(?:>=|=>|at least|minimum|min|over|more than)\s*100\s*\+?\s*(?:employees|people|headcount)?\s*(?:only)?\b)"
            r"|(?:\b(?:>=|=>|at least|minimum|min|over|more than)\s*100\s*\+?\s*(?:employees|people|headcount)\b)"
            r"|(?:\b100\s*\+\s*(?:employees|people|headcount)\b)"
            r"|(?:\bno\s+companies?\s+(?:with\s+)?(?:<|under|below|fewer than|less than)\s*100\s*(?:employees|people|headcount)?\b)",
            pref_text,
            re.I,
        )
    )
    if large_company_size_preference:
        preferred.append("large_company")
    for label, terms in COMPANY_EXCLUDE_TERMS:
        if any(term in dealbreaker_text.lower() for term in terms):
            excluded.append(label)
    avoid_defense = "defense_military" in excluded or bool(re.search(r"\b(?:avoid|no|exclude|dealbreaker).{0,25}\b(?:defense|military|clearance)\b", text, re.I))
    excluded = _dedupe(excluded)
    preferred = [item for item in _dedupe(preferred) if item not in set(excluded)]
    return preferred, excluded, avoid_defense


def _extract_dealbreakers(text: str, sections: dict[str, str]) -> list[str]:
    explicit = sections.get("dealbreakers") or _find_label_value(text, ("dealbreakers", "deal breakers", "hard rejects", "must avoid"))
    items = _split_items(explicit) if explicit else []
    keyword_terms = (
        "contract",
        "temporary",
        "unpaid",
        "senior",
        "staff",
        "principal",
        "defense",
        "military",
        "clearance",
        "no sponsorship",
        "startup",
    )
    lowered = text.lower()
    for term in keyword_terms:
        if re.search(rf"\b(?:no|avoid|exclude|dealbreaker|must avoid|not open to)\b[^.\n]{{0,35}}\b{re.escape(term)}\b", lowered):
            items.append(term)
    return _dedupe(items)


def _profile_to_form_fields(profile: dict[str, Any], keys: set[str] | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for profile_key, form_key in FIELD_TO_FORM.items():
        if keys is not None and profile_key not in keys:
            continue
        value = profile.get(profile_key)
        if isinstance(value, list):
            fields[form_key] = ", ".join(str(item) for item in value)
        elif isinstance(value, bool):
            fields[form_key] = value
        elif value is None:
            fields[form_key] = ""
        else:
            fields[form_key] = value
    return fields


def _parsed_profile(
    profile: dict[str, Any],
    notes: list[str],
    *,
    parse_method: str = "rule_fallback",
) -> ParsedProfile:
    form_fields = _profile_to_form_fields(profile)
    return ParsedProfile(
        profile=profile,
        form_fields=form_fields,
        field_sources=_field_sources(form_fields),
        notes=notes,
        parse_method=parse_method,
        filter_fields=_profile_to_form_fields(profile, EXECUTABLE_FILTER_FIELDS),
        context_fields=_profile_to_form_fields(profile, PROFILE_CONTEXT_FIELDS),
    )


def _field_sources(form_fields: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for key, value in form_fields.items():
        is_empty = value in ("", None, [], False)
        sources[key] = "empty" if is_empty else "inferred"
    return sources


def _apply_overrides(profile: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return profile
    payload = dict(profile)
    for key, value in overrides.items():
        if value not in (None, "", []):
            payload[key] = value
    return build_profile(**payload)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _normalize_provider(provider: Any) -> str:
    candidate = clean_text(provider or os.getenv("JOBPILOT_LLM_PROVIDER") or "gemini").lower()
    if candidate in {"google", "google_gemini"}:
        return "gemini"
    return candidate


def _provider_api_key(provider: str, api_key: str = "") -> str:
    if clean_text(api_key):
        return clean_text(api_key)
    if provider == "gemini":
        # UI-supplied parser_api_key is the only one-request override. For the
        # backend default path, prefer the Google API env file key and keep the
        # Gemini alias as compatibility fallback so it cannot accidentally
        # override the configured default.
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    return ""


def _provider_model(provider: str, model: str = "") -> str:
    override = clean_text(model)
    if override:
        return override
    for env_name in MODEL_ENV_BY_PROVIDER.get(provider, ()):
        configured = clean_text(os.getenv(env_name))
        if configured:
            return configured
    return DEFAULT_MODELS_BY_PROVIDER.get(provider, "")


def _llm_settings(
    *,
    parser_mode: str = "",
    parser_provider: str = "",
    parser_model: str = "",
    parser_api_key: str = "",
) -> LLMProviderSettings:
    mode = clean_text(parser_mode).lower() or "rule_fallback"
    provider = _normalize_provider(parser_provider or DEFAULT_PROVIDER_BY_MODE.get(mode, "gemini"))
    return LLMProviderSettings(
        mode=mode,
        provider=provider,
        model=_provider_model(provider, parser_model),
        api_key=_provider_api_key(provider, parser_api_key),
    )


def _llm_profile_requested(settings: LLMProviderSettings) -> bool:
    if settings.mode in {"llm", "provider", "llm_provider"}:
        return True
    return _truthy(os.getenv("JOBPILOT_PROFILE_LLM_ENABLED") or os.getenv("JOBPILOT_ENABLE_LLM_PROFILE_PARSE"))


def _llm_prompt(text: str) -> str:
    return (
        "Extract a JobPilot candidate profile as JSON using only these canonical fields:\n"
        + ", ".join(sorted(LLM_PROFILE_ALLOWED_FIELDS))
        + "\n\nRules:\n"
        + "- Do not invent constraints.\n"
        + "- Keep executable filters separate from background context by using the canonical fields only.\n"
        + "- Negative title constraints such as 'No Junior titles' are exclusions, not candidate identity.\n"
        + "- Return JSON only.\n\nProfile text:\n"
        + text[:12000]
    )


def _json_from_model_text(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _gemini_profile_client(text: str, settings: LLMProviderSettings) -> dict[str, Any]:
    if not settings.api_key:
        raise RuntimeError("Gemini API key is not configured.")
    if not settings.model:
        raise RuntimeError("Gemini model is not configured.")

    model = urllib.parse.quote(settings.model, safe="")
    key = urllib.parse.quote(settings.api_key, safe="")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": _llm_prompt(text)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        data = json.loads(response.read().decode("utf-8"))
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts)
    if not clean_text(content):
        raise RuntimeError("Gemini returned no parse content.")
    return _json_from_model_text(content)


def _call_llm_profile_provider(text: str, settings: LLMProviderSettings) -> dict[str, Any]:
    if LLM_PROFILE_CLIENT is not None:
        return LLM_PROFILE_CLIENT(text)
    if settings.provider == "gemini":
        return _gemini_profile_client(text, settings)
    raise RuntimeError(f"Unsupported LLM profile provider: {settings.provider}")


def _normalize_llm_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("LLM parser returned a non-object payload.")
    candidate = payload.get("profile") if isinstance(payload.get("profile"), dict) else payload
    cleaned: dict[str, Any] = {}
    for key, value in candidate.items():
        if key not in LLM_PROFILE_ALLOWED_FIELDS:
            continue
        if key in LLM_PROFILE_LIST_FIELDS:
            cleaned[key] = normalize_list(value)
        elif key in LLM_PROFILE_BOOL_FIELDS:
            cleaned[key] = _truthy(value)
        else:
            cleaned[key] = value
    return build_profile(**cleaned)


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], False)


def _extract_contact_fields(text: str) -> dict[str, str]:
    email_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.I)
    phone_match = re.search(
        r"(?:\+?1[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}",
        text,
    )
    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s|,;]+", text, flags=re.I)
    return {
        "email": clean_text(email_match.group(0)) if email_match else "",
        "phone": clean_text(phone_match.group(0)) if phone_match else "",
        "linkedin": clean_text(linkedin_match.group(0)) if linkedin_match else "",
    }


def _merge_llm_with_deterministic_guardrails(llm_profile: dict[str, Any], rule_profile: dict[str, Any]) -> dict[str, Any]:
    payload = dict(llm_profile)
    for field in DETERMINISTIC_GUARDRAIL_FIELDS:
        value = rule_profile.get(field)
        if _has_value(value):
            payload[field] = value
    return build_profile(**payload)


def _parse_profile_text_optional_llm(text: str, rule_result: ParsedProfile, settings: LLMProviderSettings) -> ParsedProfile | None:
    if not clean_text(text):
        return None
    if not _llm_profile_requested(settings):
        return None
    try:
        payload = _call_llm_profile_provider(text, settings)
        llm_profile = _normalize_llm_profile_payload(payload)
        profile = _merge_llm_with_deterministic_guardrails(llm_profile, rule_result.profile)
    except Exception as exc:
        rule_result.parse_method = "llm_failed_rule_fallback"
        status = f"; status_code={exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
        rule_result.notes.append(
            (
                f"LLM profile parser failed; provider={settings.provider}; model={settings.model or 'default'}; "
                f"error={type(exc).__name__}{status}; fallback_reason=deterministic_rule_fallback."
            )
        )
        return rule_result

    return _parsed_profile(profile, rule_result.notes[:], parse_method=f"llm_{settings.provider}")


def parse_profile_text_deterministic(text: str, *, overrides: dict[str, Any] | None = None) -> ParsedProfile:
    source_text = _preserve_newlines(text)
    sections = _extract_sections(source_text)
    parse_blob = clean_text(source_text)
    notes: list[str] = []
    if not parse_blob:
        profile = _apply_overrides(build_profile(), overrides)
        return _parsed_profile(profile, ["No profile text provided."])

    roles = _extract_roles(source_text, sections)
    skills = _extract_skills(source_text, sections)
    required_families, preferred_families, strict_role_family = _infer_role_families(source_text, roles, sections)
    locations, us_only, strict_location = _extract_location(source_text, sections)
    salary_min, salary_is_dealbreaker = _extract_salary_min(source_text, sections)
    max_years_required = _extract_year_cap(source_text, sections)
    needs_sponsorship, visa_sponsorship = _extract_sponsorship(source_text)
    excluded_employment = _extract_employment_exclusions(source_text)
    excluded_seniority, hard_reject_terms, penalize_terms, avoid_overly_senior = _extract_seniority_constraints(
        source_text,
        max_years_required,
    )
    preferred_company, excluded_company, avoid_defense = _extract_company_types(source_text, sections)
    dealbreakers = _extract_dealbreakers(source_text, sections)
    contact_fields = _extract_contact_fields(source_text)

    background = sections.get("background", "")
    experience = sections.get("experience") or background or parse_blob[:4000]
    education = sections.get("education", "")
    projects = sections.get("projects", "")

    if needs_sponsorship and "no sponsorship" not in {item.lower() for item in dealbreakers}:
        dealbreakers.append("no sponsorship")
    if "contract" in excluded_employment and "contract" not in {item.lower() for item in dealbreakers}:
        dealbreakers.append("contract")
    if "temporary" in excluded_employment and "temporary" not in {item.lower() for item in dealbreakers}:
        dealbreakers.append("temporary")
    if us_only and not locations:
        locations = ["United States"]

    profile = build_profile(
        name=_extract_name(source_text),
        email=contact_fields["email"],
        phone=contact_fields["phone"],
        linkedin=contact_fields["linkedin"],
        resume_source_text=source_text[:12000],
        target_roles=roles,
        skills=skills,
        education=education,
        experience_text=experience,
        projects_publications=projects,
        location_preferences=locations,
        salary_min=salary_min,
        salary_is_dealbreaker=salary_is_dealbreaker,
        dealbreakers=dealbreakers,
        visa_sponsorship=visa_sponsorship,
        needs_sponsorship=needs_sponsorship,
        excluded_employment_types=excluded_employment,
        excluded_seniority=excluded_seniority,
        max_years_required=max_years_required,
        us_only=us_only,
        strict_location=strict_location,
        preferred_company_types=preferred_company,
        excluded_company_types=excluded_company,
        required_role_families=required_families,
        preferred_role_families=preferred_families,
        strict_role_family=strict_role_family,
        avoid_defense_or_clearance=avoid_defense,
        avoid_overly_senior=avoid_overly_senior,
        new_grad_or_student_profile=bool(max_years_required is not None and max_years_required <= 2),
        hard_reject_seniority_terms=hard_reject_terms,
        penalize_seniority_terms=penalize_terms,
    )
    profile = _apply_overrides(profile, overrides)
    if not roles:
        notes.append("No target role was confidently inferred; edit Target Roles before matching.")
    if not skills:
        notes.append("No skills were confidently inferred; edit Skills before matching.")

    return _parsed_profile(profile, notes)


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
        text = "".join(runs).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


async def extract_upload_text(upload: UploadFile | None) -> tuple[str, list[str], Path | None]:
    if not upload or not upload.filename:
        return "", [], None
    ensure_storage_dirs()
    suffix = Path(upload.filename).suffix.lower()
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix or '.upload'}"
    with upload_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    notes: list[str] = []
    try:
        if suffix == ".pdf":
            return extract_pdf_text(upload_path), [f"Parsed uploaded PDF: {upload.filename}"], None
        if suffix == ".docx":
            text = _extract_docx_text(upload_path)
            if clean_text(text):
                return text, [f"Parsed uploaded DOCX: {upload.filename}"], upload_path
            return "", [f"DOCX contained no readable text: {upload.filename}"], upload_path
        if suffix in {".txt", ".md"}:
            return upload_path.read_text(encoding="utf-8", errors="ignore"), [f"Parsed uploaded text file: {upload.filename}"], None
        return "", [f"Unsupported upload type {suffix or '(none)'}; use PDF or DOCX."], None
    except (PDFExtractionError, KeyError, zipfile.BadZipFile, ElementTree.ParseError, OSError) as exc:
        notes.append(f"Upload text extraction failed for {upload.filename}: {exc}")
        return "", notes, upload_path if suffix == ".docx" else None


async def parse_profile_intake(
    *,
    profile_text: str = "",
    upload: UploadFile | None = None,
    overrides: dict[str, Any] | None = None,
    parser_mode: str = "",
    parser_provider: str = "",
    parser_model: str = "",
    parser_api_key: str = "",
) -> ParsedProfile:
    upload_text, notes, source_docx_path = await extract_upload_text(upload)
    combined = "\n\n".join(part for part in [upload_text, profile_text] if clean_text(part))
    result = parse_profile_text_deterministic(combined)
    settings = _llm_settings(
        parser_mode=parser_mode,
        parser_provider=parser_provider,
        parser_model=parser_model,
        parser_api_key=parser_api_key,
    )
    result = _parse_profile_text_optional_llm(combined, result, settings) or result
    if overrides:
        result.profile = _apply_overrides(result.profile, overrides)
        result.form_fields = _profile_to_form_fields(result.profile)
        result.field_sources = _field_sources(result.form_fields)
        result.filter_fields = _profile_to_form_fields(result.profile, EXECUTABLE_FILTER_FIELDS)
        result.context_fields = _profile_to_form_fields(result.profile, PROFILE_CONTEXT_FIELDS)
    if source_docx_path:
        try:
            result.profile["resume_source_docx_path"] = source_docx_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            result.profile["resume_source_docx_path"] = str(source_docx_path)
    result.notes[:0] = notes
    return result
