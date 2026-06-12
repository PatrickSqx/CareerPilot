"""Production-safe Phase 2.18H evidence rerank helper.

This module intentionally exposes only the evidence channels promoted by the
18G/18G.1 offline gates: admitted hard-skill sidecar terms by default, plus a
conditional company-size soft boost when the normalized profile explicitly
expresses a company-size preference. Missing evidence is neutral.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jobpilot.config import PROCESSED_DATA_DIR
from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text, normalize_for_key


RANKING_FEATURES_CSV = PROCESSED_DATA_DIR / "ranking_features" / "phase2_18_job_ranking_features.csv"

EVIDENCE_POLICY = "phase2_18h_hard_skills_only_plus_conditional_company_size_v1"
HARD_SKILLS_POLICY = "hard_skills_only"
COMPANY_SIZE_POLICY = "company_size_soft_high"

CHANNELS = ("hard_skill", "company_size_type", "lca_activity", "role_family", "llm_overlay")
HARD_SKILL_COMPONENT_CAP = 0.036
HARD_SKILLS_TOTAL_CAP = 0.05
COMPANY_SIZE_SOFT_HIGH_CAP = 0.03

USABLE_SIZE_POLICIES = {
    "usable_employer_context",
    "usable_franchise_operator_context",
    "usable_single_location_context",
}
LARGE_BUCKETS = {
    "enterprise_10001_plus",
    "enterprise_5001_10000",
    "large_5001_10000",
    "large_1001_5000",
    "large_501_1000",
}
SMALL_BUCKETS = {"micro_1_10", "small_11_50", "mid_51_200"}
MID_BUCKETS = {"mid_201_500"}

LARGE_PREFERENCE_KEYS = {"large company", "large", "enterprise", "research lab", "mature company"}
SMALL_PREFERENCE_KEYS = {"startup", "start up", "small company", "small team", "early stage"}
MEDIUM_PREFERENCE_KEYS = {"medium company", "mid size", "mid sized", "midsize"}
STARTUP_EXCLUSION_KEYS = {"startup", "start up", "early stage"}


def _zero_components() -> dict[str, float]:
    return {channel: 0.0 for channel in CHANNELS}


def _boost_item(signal: str, label: str, adjustment: float, reason: str, policy: str) -> dict[str, Any]:
    return {
        "signal": signal,
        "label": label,
        "adjustment": round(float(adjustment or 0.0), 6),
        "reason": reason,
        "policy": policy,
    }


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): clean_text(value) for key, value in row.items()}


@lru_cache(maxsize=4)
def load_ranking_sidecar(path: str | Path = RANKING_FEATURES_CSV) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load the frozen Phase 2.18 ranking sidecar keyed by job_id.

    Missing files are a supported no-op state so production matching can fall
    back to the base score/order during local demos or partial deployments.
    """

    sidecar_path = Path(path)
    if not sidecar_path.exists():
        return {}, {
            "path": sidecar_path.as_posix(),
            "available": False,
            "row_count": 0,
            "duplicate_job_id_count": 0,
            "field_count": 0,
        }

    rows: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    fieldnames: list[str] = []
    with sidecar_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for raw in reader:
            row = _clean_row(raw)
            job_id = clean_text(row.get("job_id"))
            if not job_id:
                continue
            if job_id in rows:
                duplicate_ids.append(job_id)
                continue
            rows[job_id] = row
    return rows, {
        "path": sidecar_path.as_posix(),
        "available": True,
        "row_count": len(rows),
        "duplicate_job_id_count": len(duplicate_ids),
        "duplicate_job_ids_sample": duplicate_ids[:10],
        "field_count": len(fieldnames),
    }


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


def _profile_keys(profile: dict[str, Any], field: str) -> set[str]:
    return {normalize_for_key(value) for value in normalize_list(profile.get(field)) if normalize_for_key(value)}


def _profile_skill_keys(profile: dict[str, Any]) -> set[str]:
    keys = _profile_keys(profile, "skills")
    if "sklearn" in keys:
        keys.add("scikit learn")
    if "scikit learn" in keys:
        keys.add("sklearn")
    return keys


def hard_skill_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str]]:
    """18G hard_skills_only component using admitted/scoring-safe sidecar terms."""

    profile_skills = _profile_skill_keys(profile)
    if not profile_skills:
        return 0.0, []
    scoring_terms = parse_json_list(sidecar.get("hard_skill_scoring_terms_json"))
    capped_terms = parse_json_list(sidecar.get("hard_skill_capped_terms_json"))
    scoring_matches = [term for term in scoring_terms if normalize_for_key(term) in profile_skills]
    capped_matches = [term for term in capped_terms if normalize_for_key(term) in profile_skills]
    value = min(HARD_SKILL_COMPONENT_CAP, 0.008 * len(scoring_matches) + 0.004 * len(capped_matches))
    if value <= 0:
        return 0.0, []
    matched = ", ".join(scoring_matches[:4] + capped_matches[:2])
    return round(value, 6), [f"Hard-skill evidence uses admitted scoring-safe sidecar terms: {matched}."]


def company_size_class(sidecar: dict[str, str]) -> str:
    policy = clean_text(sidecar.get("company_size_usage_policy")).lower()
    if policy not in USABLE_SIZE_POLICIES:
        return "unknown"
    bucket = clean_text(sidecar.get("usable_employer_size_bucket")).lower()
    snapshot_type = clean_text(sidecar.get("snapshot_company_type")).lower()
    provider_type = clean_text(sidecar.get("company_type_provider")).lower()
    if snapshot_type == "startup" or provider_type == "startup":
        return "small_or_startup"
    if snapshot_type == "research_lab" or provider_type == "research_lab":
        return "large_or_mature"
    if bucket in LARGE_BUCKETS:
        return "large_or_mature"
    if bucket in SMALL_BUCKETS:
        return "small_or_startup"
    if bucket in MID_BUCKETS:
        return "mid_size"
    return "unknown"


def company_size_preference_mode(profile: dict[str, Any]) -> str:
    """Return the explicit company-size preference mode for a normalized profile."""

    preferred = _profile_keys(profile, "preferred_company_types")
    excluded = _profile_keys(profile, "excluded_company_types")
    if preferred & SMALL_PREFERENCE_KEYS:
        return "prefer_small"
    if preferred & MEDIUM_PREFERENCE_KEYS:
        return "prefer_medium"
    if preferred & LARGE_PREFERENCE_KEYS:
        return "prefer_large"
    if excluded & STARTUP_EXCLUSION_KEYS:
        return "avoid_small"
    return "none"


def company_size_component(profile: dict[str, Any], sidecar: dict[str, str]) -> tuple[float, list[str], dict[str, Any]]:
    """18G.1 conditional company_size_soft_high component.

    The component activates only for explicit company-size preferences. Unknown
    or missing size remains neutral, including when excluded company-type text
    is present but the evidence sidecar lacks usable size context.
    """

    mode = company_size_preference_mode(profile)
    cls = company_size_class(sidecar)
    details = {
        "mode": mode,
        "size_class": cls,
        "unknown": cls == "unknown",
        "policy": COMPANY_SIZE_POLICY if mode != "none" else "not_triggered",
    }
    if mode == "none" or cls == "unknown":
        return 0.0, [], details
    satisfies = (
        (mode in {"prefer_large", "avoid_small"} and cls == "large_or_mature")
        or (mode == "prefer_small" and cls == "small_or_startup")
        or (mode == "prefer_medium" and cls == "mid_size")
    )
    details["satisfies"] = satisfies
    if not satisfies:
        return 0.0, [], details
    return COMPANY_SIZE_SOFT_HIGH_CAP, ["Company size/type applied only because the profile explicitly requested it."], details


def neutral_payload(old_score: float, *, reason: str, sidecar_available: bool = False) -> dict[str, Any]:
    old_score = round(max(0.0, min(1.0, float(old_score or 0.0))), 6)
    return {
        "base_score": old_score,
        "evidence_adjustment": 0.0,
        "final_score": old_score,
        "evidence_components": _zero_components(),
        "evidence_policy": EVIDENCE_POLICY,
        "hard_skill_policy": HARD_SKILLS_POLICY,
        "company_size_policy": "not_triggered",
        "company_size_policy_applied": False,
        "company_size_preference_mode": "none",
        "company_size_class": "unknown",
        "evidence_sidecar_available": sidecar_available,
        "evidence_notes": [reason],
        "ranking_boosts": [],
        "verified_boosts": [],
        "rerank_applied": False,
        "evidence_fallback_to_base": True,
        "evidence_fallback_reason": reason,
    }


def evidence_rerank_payload(
    profile: dict[str, Any],
    sidecar: dict[str, str] | None,
    *,
    old_score: float,
    hard_filter_passed: bool = True,
) -> dict[str, Any]:
    """Return the production evidence score payload for one already-scored job."""

    old_score = round(max(0.0, min(1.0, float(old_score or 0.0))), 6)
    if not hard_filter_passed:
        return neutral_payload(
            old_score,
            reason="Evidence rerank skipped for hard-filtered candidate; hard filters remain authoritative.",
            sidecar_available=sidecar is not None,
        )
    if not sidecar:
        return neutral_payload(
            old_score,
            reason="Evidence sidecar missing for job; base score/order preserved.",
            sidecar_available=False,
        )

    components = _zero_components()
    notes: list[str] = []
    boosts: list[dict[str, Any]] = []
    hard_skill_value, hard_skill_notes = hard_skill_component(profile, sidecar)
    components["hard_skill"] = hard_skill_value
    notes.extend(hard_skill_notes)
    if hard_skill_value > 0:
        boosts.append(
            _boost_item(
                "hard_skill",
                "Hard skill boost",
                hard_skill_value,
                hard_skill_notes[0] if hard_skill_notes else "Matched admitted hard-skill sidecar terms.",
                HARD_SKILLS_POLICY,
            )
        )

    company_value, company_notes, company_details = company_size_component(profile, sidecar)
    components["company_size_type"] = company_value
    notes.extend(company_notes)
    if company_value > 0:
        boosts.append(
            _boost_item(
                "company_size_type",
                "Company-size boost",
                company_value,
                company_notes[0] if company_notes else "Matched an explicit company-size preference.",
                COMPANY_SIZE_POLICY,
            )
        )

    adjustment = round(min(HARD_SKILLS_TOTAL_CAP + COMPANY_SIZE_SOFT_HIGH_CAP, sum(components.values())), 6)
    final_score = round(min(1.0, old_score + adjustment), 6)
    if adjustment == 0:
        notes.append("Missing, unknown, disabled, or non-matching evidence was neutral; no score decrease applied.")

    return {
        "base_score": old_score,
        "evidence_adjustment": adjustment,
        "final_score": final_score,
        "evidence_components": components,
        "evidence_policy": EVIDENCE_POLICY,
        "hard_skill_policy": HARD_SKILLS_POLICY,
        "company_size_policy": COMPANY_SIZE_POLICY if company_details["mode"] != "none" else "not_triggered",
        "company_size_policy_applied": company_details["mode"] != "none",
        "company_size_preference_mode": company_details["mode"],
        "company_size_class": company_details["size_class"],
        "evidence_sidecar_available": True,
        "evidence_notes": notes,
        "ranking_boosts": boosts,
        "verified_boosts": boosts,
        "rerank_applied": adjustment > 0,
        "evidence_fallback_to_base": False,
        "evidence_fallback_reason": "",
    }


class ProductionEvidenceReranker:
    """Apply the Phase 2.18H production evidence policy to scored jobs."""

    def __init__(self, sidecar_path: str | Path = RANKING_FEATURES_CSV):
        self.sidecar_path = Path(sidecar_path)

    def sidecar(self) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
        return load_ranking_sidecar(self.sidecar_path)

    def apply(
        self,
        profile: dict[str, Any],
        scored_jobs: list[dict[str, Any]],
        *,
        session_feedback_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sidecar_rows, metadata = self.sidecar()
        adjusted_count = 0
        company_policy_applied_count = 0
        hard_skill_adjusted_count = 0
        company_size_adjusted_count = 0
        missing_sidecar_job_count = 0
        fallback_job_count = 0

        for job in scored_jobs:
            old_score = float(job.get("final_score") or 0.0)
            sidecar = sidecar_rows.get(clean_text(job.get("job_id"))) if metadata["available"] else None
            if metadata["available"] and not sidecar:
                missing_sidecar_job_count += 1
            payload = evidence_rerank_payload(
                profile,
                sidecar,
                old_score=old_score,
                hard_filter_passed=bool(job.get("hard_filter_passed", True)),
            )
            job.update(payload)
            components = payload["evidence_components"]
            if payload["evidence_adjustment"] > 0:
                adjusted_count += 1
            if components.get("hard_skill", 0.0) > 0:
                hard_skill_adjusted_count += 1
            if components.get("company_size_type", 0.0) > 0:
                company_size_adjusted_count += 1
            if payload["company_size_policy_applied"]:
                company_policy_applied_count += 1
            if payload["evidence_fallback_to_base"]:
                fallback_job_count += 1

        return {
            "policy": EVIDENCE_POLICY,
            "hard_skill_policy": HARD_SKILLS_POLICY,
            "company_size_policy": COMPANY_SIZE_POLICY,
            "enabled": True,
            "sidecar_available": bool(metadata["available"]),
            "sidecar_path": metadata["path"],
            "sidecar_row_count": metadata["row_count"],
            "sidecar_duplicate_job_id_count": metadata["duplicate_job_id_count"],
            "scored_jobs": len(scored_jobs),
            "adjusted_count": adjusted_count,
            "hard_skill_adjusted_count": hard_skill_adjusted_count,
            "company_size_policy_applied_count": company_policy_applied_count,
            "company_size_adjusted_count": company_size_adjusted_count,
            "missing_sidecar_job_count": missing_sidecar_job_count,
            "fallback_job_count": fallback_job_count,
            "fallback_to_base_score": not metadata["available"],
            "disabled_channels": ["lca_activity", "role_family", "llm_overlay"],
        }
