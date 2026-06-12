"""Profile building and matching service for the FastAPI app."""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.services.paths import PROJECT_ROOT
from app.services.profile_parse_service import parse_profile_intake
from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_CSV, OFFLINE_SNAPSHOT_SAMPLE_CSV
from jobpilot.profile.personas import PERSONA_FIXTURES, get_persona
from jobpilot.profile.profile_parser import build_profile, normalize_list
from jobpilot.ranking.ranker import JobRanker
from jobpilot.retrieval.embeddings import build_or_load_job_embeddings
from jobpilot.utils.text import clean_text


DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 200


def persona_options() -> list[dict[str, str]]:
    return [{"id": key, "name": PERSONA_FIXTURES[key]["name"]} for key in sorted(PERSONA_FIXTURES)]


def resolve_snapshot_path(snapshot_path: Path = OFFLINE_SNAPSHOT_CSV) -> tuple[Path, list[str]]:
    """Use the full snapshot, or the 500-row review-package sample when needed."""

    if snapshot_path.exists():
        return snapshot_path, []
    sample = snapshot_path.parent / OFFLINE_SNAPSHOT_SAMPLE_CSV.name
    if sample.exists():
        return sample, [
            (
                f"Full snapshot {snapshot_path.relative_to(PROJECT_ROOT).as_posix()} is missing; "
                f"using {sample.relative_to(PROJECT_ROOT).as_posix()} for smoke-test results only."
            )
        ]
    return snapshot_path, [f"Snapshot file is missing: {snapshot_path.as_posix()}"]


def snapshot_status(snapshot_path: Path = OFFLINE_SNAPSHOT_CSV) -> dict[str, str | bool]:
    """Return the current offline snapshot state for the home page."""

    if snapshot_path.exists():
        rel = snapshot_path.relative_to(PROJECT_ROOT).as_posix()
        return {
            "state": "full",
            "label": "Full dataset",
            "path": rel,
            "is_sample": False,
        }
    sample = snapshot_path.parent / OFFLINE_SNAPSHOT_SAMPLE_CSV.name
    if sample.exists():
        rel = sample.relative_to(PROJECT_ROOT).as_posix()
        return {
            "state": "sample",
            "label": "Sample dataset",
            "path": rel,
            "is_sample": True,
        }
    return {
        "state": "missing",
        "label": "Dataset missing",
        "path": snapshot_path.relative_to(PROJECT_ROOT).as_posix(),
        "is_sample": False,
    }


@lru_cache(maxsize=4)
def _ranker_for(snapshot_path: str, cache_dir: str, backend: str) -> JobRanker:
    store = build_or_load_job_embeddings(snapshot_path=snapshot_path, cache_dir=cache_dir, backend=backend)
    return JobRanker(store)


def warm_ranker_runtime() -> None:
    """Load ranking artifacts during app startup without running a full match."""

    snapshot_path, _warnings = resolve_snapshot_path()
    _ranker_for(str(snapshot_path.resolve()), str(EMBEDDINGS_DIR.resolve()), "auto")


def rank_profile(
    profile: dict[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedding_backend: str = "auto",
    session_feedback_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot_path, warnings = resolve_snapshot_path()
    ranker = _ranker_for(str(snapshot_path.resolve()), str(EMBEDDINGS_DIR.resolve()), embedding_backend)
    result = ranker.rank(
        profile,
        top_k=top_k,
        candidate_k=candidate_k,
        session_feedback_events=session_feedback_events or [],
    )
    metadata = result.setdefault("metadata", {})
    metadata["snapshot_used"] = snapshot_path.relative_to(PROJECT_ROOT).as_posix()
    metadata["snapshot_sample_fallback_used"] = snapshot_path.name == OFFLINE_SNAPSHOT_SAMPLE_CSV.name
    metadata.setdefault("warnings", [])
    metadata["warnings"].extend(warnings)
    return result


def _truthy_form_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _manual_filters_are_final(form: dict[str, Any]) -> bool:
    return _truthy_form_value(form.get("profile_filters_ready"))


def _manual_overrides(form: dict[str, Any], *, explicit: bool = False) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    mapping = {
        "manual_name": "name",
        "manual_target_roles": "target_roles",
        "manual_skills": "skills",
        "manual_location_preferences": "location_preferences",
        "manual_salary_min": "salary_min",
        "manual_dealbreakers": "dealbreakers",
        "manual_education": "education",
        "manual_experience": "experience_text",
        "manual_projects": "projects_publications",
        "manual_visa_sponsorship": "visa_sponsorship",
        "manual_excluded_seniority": "excluded_seniority",
        "manual_max_years_required": "max_years_required",
        "manual_excluded_employment_types": "excluded_employment_types",
        "manual_required_role_families": "required_role_families",
        "manual_preferred_role_families": "preferred_role_families",
        "manual_preferred_company_types": "preferred_company_types",
        "manual_excluded_company_types": "excluded_company_types",
        "manual_hard_reject_seniority_terms": "hard_reject_seniority_terms",
        "manual_penalize_seniority_terms": "penalize_seniority_terms",
    }
    list_fields = {
        "target_roles",
        "skills",
        "location_preferences",
        "dealbreakers",
        "excluded_seniority",
        "excluded_employment_types",
        "required_role_families",
        "preferred_role_families",
        "preferred_company_types",
        "excluded_company_types",
        "hard_reject_seniority_terms",
        "penalize_seniority_terms",
    }
    for source, target in mapping.items():
        value = form.get(source)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip() and not explicit:
            continue
        overrides[target] = value
    for field in list_fields:
        if field in overrides:
            overrides[field] = normalize_list(overrides[field])
    boolean_mapping = {
        "manual_needs_sponsorship": "needs_sponsorship",
        "manual_us_only": "us_only",
        "manual_strict_role_family": "strict_role_family",
        "manual_avoid_defense_or_clearance": "avoid_defense_or_clearance",
        "manual_salary_is_dealbreaker": "salary_is_dealbreaker",
        "manual_strict_location": "strict_location",
    }
    for source, target in boolean_mapping.items():
        value = _truthy_form_value(form.get(source))
        if explicit:
            overrides[target] = value
        elif value:
            overrides[target] = True
    if overrides.get("us_only") is True:
        overrides["strict_location"] = True
    return overrides


def _merge_profile(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    for key, value in overrides.items():
        if value not in (None, "", []):
            payload[key] = value
    return build_profile(**payload)


async def _profile_from_intake(
    profile_text: str,
    upload: UploadFile | None,
    overrides: dict[str, Any],
    notes: list[str],
) -> dict[str, Any] | None:
    if not clean_text(profile_text) and (not upload or not upload.filename):
        return None
    parsed = await parse_profile_intake(profile_text=profile_text, upload=upload, overrides=overrides)
    notes.extend(parsed.notes)
    return parsed.profile


async def build_profile_from_request(form: dict[str, Any], resume_pdf: UploadFile | None = None) -> tuple[dict[str, Any], list[str]]:
    """Build a profile from parsed intake text/upload, manual fields, or personas."""

    notes: list[str] = []
    manual_filters_final = _manual_filters_are_final(form)
    overrides = _manual_overrides(form, explicit=manual_filters_final)
    persona = clean_text(form.get("persona") or "aisha").lower()
    if manual_filters_final:
        profile = build_profile(**overrides)
        if persona in PERSONA_FIXTURES and persona != "manual":
            base = get_persona(persona)
            profile["profile_id"] = base.get("profile_id", persona)
            if not clean_text(profile.get("name")):
                profile["name"] = clean_text(base.get("name"))
            notes.append(f"{profile['name'] or persona.title()} demo persona submitted with reviewed filters.")
            return profile, notes
        notes.append("Manual filter fields submitted as final profile source.")
        return profile, notes

    profile_text_value = form.get("profile_text")
    profile_text = profile_text_value if isinstance(profile_text_value, str) else ""
    intake_profile = await _profile_from_intake(profile_text, resume_pdf, overrides, notes)
    if intake_profile:
        return intake_profile, notes

    if persona == "manual":
        return build_profile(**overrides), notes
    base = get_persona(persona if persona in PERSONA_FIXTURES else "aisha")
    return _merge_profile(base, overrides), notes


def build_match_payload(
    profile: dict[str, Any],
    result: dict[str, Any],
    *,
    notes: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id or uuid.uuid4().hex,
        "profile": profile,
        "notes": notes or [],
        "profile_id": profile.get("profile_id", ""),
        "top_jobs": result.get("top_jobs", []),
        "metadata": result.get("metadata", {}),
        "profile_text": result.get("profile_text", ""),
    }
