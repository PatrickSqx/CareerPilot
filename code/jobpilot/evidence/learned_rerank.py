"""Phase 2.18J learned EBM reranker for post-hard-filter ranking."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from jobpilot.config import PROCESSED_DATA_DIR
from jobpilot.evidence.production_rerank import (
    RANKING_FEATURES_CSV,
    ProductionEvidenceReranker,
    company_size_class,
    company_size_preference_mode,
    load_ranking_sidecar,
)
from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text, normalize_for_key, stable_hash


RUNTIME_POLICY = "phase2_18j_ebm_learned_active_v1"
OLD_BASELINE_POLICY = "phase2_18h_rule_baseline_fallback"
R0_SAFE_SIDECAR_FALLBACK_POLICY = "phase2_18j_r0_safe_sidecar_fallback_v1"
DEFAULT_MODE = "learned_active"
VALID_MODES = {"old_baseline", "learned_shadow", "learned_active", "r0_safe_sidecar"}

CANDIDATE_RESERVOIR_SIZE = 200
RUNTIME_ROUND_INDEX = 2
QUALITY_MATCH_LABELS = {"Strong match", "Good match"}
MATCH_STRENGTH_ENCODING = {"Possible fit": 1, "Good match": 2, "Strong match": 3}
SAME_JOB_REJECT_SCORE = 0.0
SAME_COMPANY_REJECT_CAP = 0.45
SESSION_FEEDBACK_MIN_ADJUSTMENT = -0.08
SESSION_FEEDBACK_MAX_ADJUSTMENT = 0.08
SAFE_SIDECAR_MIN_ADJUSTMENT = -0.05
SAFE_SIDECAR_MAX_ADJUSTMENT = 0.05
POST_EBM_MIN_ADJUSTMENT = -0.10
POST_EBM_MAX_ADJUSTMENT = 0.10

RUNTIME_MODEL_ARTIFACT = PROCESSED_DATA_DIR / "phase2_18j_runtime_ebm_reranker.joblib"
INTERACTION_FEATURE_MANIFEST = PROCESSED_DATA_DIR / "phase2_18j_interaction_feature_manifest.json"
R0_COMPARATOR_SUMMARY = PROCESSED_DATA_DIR / "phase2_18j_learning_curve_summary.json"

SAFE_SIDECAR_FEATURES = (
    "company_size_bucket_visible_encoded",
    "company_size_known_flag",
    "company_size_matches_profile",
    "sponsorship_signal_visible_encoded",
    "lca_activity_visible_bucket",
    "lca_activity_available_flag",
    "lca_sponsor_proxy_for_needs_sponsorship",
    "llm_overlay_available_flag",
    "llm_role_family_matches_profile",
    "llm_suggested_soft_action_encoded",
    "llm_downstream_use_gate_encoded",
    "llm_evidence_count_bucket",
)
VISIBLE_SAFE_SIDECAR_LABELS = {
    "company_size_signal_available": "Company size signal available",
    "sponsorship_signal_available": "Sponsorship signal available",
    "h1b_activity_proxy_available": "H-1B activity proxy available",
    "llm_reviewed_role_family_signal_available": "LLM reviewed role-family signal available",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: Any) -> float | None:
    text = clean_text(value).replace(",", "").replace("$", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_remote_job(job: dict[str, Any]) -> bool:
    blob = normalize_for_key(f"{job.get('title', '')} {job.get('location', '')} {job.get('is_remote', '')}")
    return "remote" in blob.split() or clean_text(job.get("is_remote")).lower() in {"1", "true", "yes"}


def _salary_display(job: dict[str, Any]) -> str:
    salary_raw = clean_text(job.get("salary_raw"))
    if salary_raw:
        return salary_raw
    salary_min = clean_text(job.get("salary_min"))
    salary_max = clean_text(job.get("salary_max"))
    if salary_min and salary_max:
        return f"{salary_min}-{salary_max}"
    return salary_min or salary_max or "Salary not listed"


def _salary_number(text: str) -> int | None:
    lower = text.lower()
    if not lower or "not listed" in lower:
        return None
    numbers = [float(item.replace(",", "")) for item in re.findall(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", lower)]
    if not numbers:
        return None
    value = min(numbers)
    if "/hr" in lower or "hour" in lower:
        value *= 2080
    elif value < 1000:
        value *= 1000
    return int(value)


def _salary_bucket(text: str) -> int:
    value = _salary_number(text)
    if value is None:
        return 0
    if value < 80_000:
        return 1
    if value < 120_000:
        return 2
    if value < 160_000:
        return 3
    if value < 200_000:
        return 4
    return 5


def _years_bucket(job: dict[str, Any]) -> int:
    text = clean_text(job.get("years_required"))
    years = _parse_float(text)
    if years is None:
        return 0
    if years <= 1:
        return 1
    if years <= 2:
        return 2
    if years <= 4:
        return 3
    return 4


def _seniority_warning(job: dict[str, Any]) -> int:
    blob = normalize_for_key(f"{job.get('title', '')} {job.get('seniority', '')}")
    terms = {"senior", "staff", "principal", "lead", "director", "manager"}
    return int(any(term in blob.split() for term in terms))


def _profile_keys(profile: dict[str, Any], field: str) -> set[str]:
    return {normalize_for_key(value) for value in normalize_list(profile.get(field)) if normalize_for_key(value)}


def _has_any(keys: set[str], values: set[str]) -> bool:
    return bool(keys & values)


def _avoid_senior(profile: dict[str, Any]) -> int:
    terms = (
        _profile_keys(profile, "dealbreakers")
        | _profile_keys(profile, "excluded_seniority")
        | _profile_keys(profile, "hard_reject_seniority_terms")
        | _profile_keys(profile, "penalize_seniority_terms")
    )
    return int(_has_any(terms, {"senior", "staff", "principal", "lead", "manager", "director", "sr", "sr."}))


def _avoid_contract(profile: dict[str, Any]) -> int:
    terms = _profile_keys(profile, "dealbreakers") | _profile_keys(profile, "excluded_employment_types")
    return int(_has_any(terms, {"contract", "contractor", "temporary", "temp", "unpaid", "internship"}))


def _company_preference_flags(profile: dict[str, Any]) -> tuple[int, int, int]:
    preferred = _profile_keys(profile, "preferred_company_types")
    excluded = _profile_keys(profile, "excluded_company_types")
    large = {"large company", "large", "enterprise", "research lab", "mature company", "large_company", "research_lab"}
    startup = {"startup", "start up", "small company", "small team", "early stage", "small_company"}
    has_pref = bool(preferred or excluded)
    return int(has_pref), int(bool(preferred & large)), int(bool(preferred & startup))


def _employment_matches(profile: dict[str, Any], job: dict[str, Any]) -> int:
    employment = normalize_for_key(job.get("employment_type"))
    excluded = _profile_keys(profile, "excluded_employment_types")
    if ("contract" in employment or "temporary" in employment or "temp" in employment) and (
        excluded & {"contract", "contractor", "temporary", "temp"}
    ):
        return 0
    desired = _profile_keys(profile, "employment_types")
    if not desired:
        return 1
    return int(any(item and item in employment for item in desired))


def _match_strength_label(job: dict[str, Any]) -> str:
    score = _as_float(job.get("final_score"))
    if score >= 0.72:
        return "Strong match"
    if score >= 0.62:
        return "Good match"
    return "Possible fit"


def _driver_count(job: dict[str, Any], key: str) -> int:
    why_ranked = job.get("why_ranked") if isinstance(job.get("why_ranked"), dict) else {}
    items = why_ranked.get(key) if isinstance(why_ranked, dict) else []
    return len(items) if isinstance(items, list) else 0


def _matched_skill_count(job: dict[str, Any]) -> int:
    skills = job.get("matched_skills") or []
    if isinstance(skills, str):
        skills = [part.strip() for part in re.split(r"[,|]", skills) if part.strip()]
    return len(skills) if isinstance(skills, list) else 0


def _skill_keys(job: dict[str, Any]) -> set[str]:
    skills = job.get("matched_skills") or []
    if isinstance(skills, str):
        skills = [part.strip() for part in re.split(r"[,|]", skills) if part.strip()]
    return {normalize_for_key(skill) for skill in skills if normalize_for_key(skill)}


def _company_key(job: dict[str, Any]) -> str:
    return normalize_for_key(job.get("company") or job.get("employer"))


def _session_feedback_overlay(job: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    job_id = clean_text(job.get("job_id"))
    company = _company_key(job)
    skills = _skill_keys(job)
    adjustment = 0.0
    cap: float | None = None
    notes: list[str] = []
    suppress = False

    for event in events:
        action = clean_text(event.get("action")).lower()
        source = event.get("job_snapshot") if isinstance(event.get("job_snapshot"), dict) else {}
        source_job_id = clean_text(source.get("job_id") or event.get("job_id"))
        source_company = _company_key(source)
        source_skills = _skill_keys(source)
        same_job = bool(job_id and source_job_id and job_id == source_job_id)
        same_company = bool(company and source_company and company == source_company)
        skill_overlap = len(skills & source_skills)

        if action == "reject":
            if same_job:
                suppress = True
                cap = SAME_JOB_REJECT_SCORE
                notes.append("same job rejected earlier in this session")
            elif same_company:
                adjustment -= 0.06
                cap = SAME_COMPANY_REJECT_CAP if cap is None else min(cap, SAME_COMPANY_REJECT_CAP)
                notes.append("same company rejected earlier in this session")
        elif action == "skip":
            if same_job:
                adjustment -= 0.04
                notes.append("same job skipped earlier in this session")
            elif same_company:
                adjustment -= 0.02
                notes.append("same company skipped earlier in this session")
        elif action == "accept":
            if same_job:
                adjustment += 0.06
                notes.append("same job accepted earlier in this session")
            elif same_company:
                adjustment += 0.035
                notes.append("same company accepted earlier in this session")
            if skill_overlap:
                adjustment += min(0.04, 0.01 * skill_overlap)
                notes.append("shares skills with an accepted job in this session")

    adjustment = _clamp(adjustment, SESSION_FEEDBACK_MIN_ADJUSTMENT, SESSION_FEEDBACK_MAX_ADJUSTMENT)
    return {
        "adjustment": round(adjustment, 6),
        "cap": cap,
        "suppress": suppress,
        "notes": list(dict.fromkeys(notes)),
    }


def _company_size_bucket_encoded(size_class: str) -> int:
    return {
        "small_or_startup": 1,
        "mid_size": 2,
        "large_or_mature": 3,
    }.get(size_class, 0)


def _company_size_matches_profile(profile: dict[str, Any], size_class: str) -> bool:
    mode = company_size_preference_mode(profile)
    if mode == "none" or size_class == "unknown":
        return False
    return (
        (mode in {"prefer_large", "avoid_small"} and size_class == "large_or_mature")
        or (mode == "prefer_small" and size_class == "small_or_startup")
        or (mode == "prefer_medium" and size_class == "mid_size")
    )


def _sponsorship_signal(sidecar: dict[str, str] | None, job: dict[str, Any]) -> str:
    if sidecar:
        signal = clean_text(sidecar.get("snapshot_sponsorship_signal")).lower()
        if signal:
            return signal
    return clean_text(job.get("sponsorship_signal")).lower() or "unknown"


def _sponsorship_encoded(signal: str) -> int:
    if signal in {"mentions_sponsorship_or_work_auth", "sponsorship_available", "visa_sponsorship", "h1b_sponsorship"}:
        return 2
    if signal == "no_sponsorship":
        return 1
    return 0


def _lca_activity_level(label: str) -> str:
    normalized = clean_text(label).lower()
    if normalized == "recent_lca_activity_high":
        return "high"
    if normalized == "recent_lca_activity_moderate":
        return "medium"
    if normalized in {"recent_lca_activity_low", "historical_lca_activity_only"}:
        return "low"
    return "unknown"


def _lca_activity_bucket(level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(level, 0)


def _reviewed_llm_status(sidecar: dict[str, str] | None) -> bool:
    if not sidecar or not _boolish(sidecar.get("llm_overlay_available")):
        return False
    status = clean_text(sidecar.get("llm_reviewed_overlay_status")).lower()
    return status in {"reviewed_candidate", "reviewed"}


def _llm_gate_blocks_adjustment(gate: str) -> bool:
    normalized = clean_text(gate).lower()
    return normalized in {"blocked", "review_only"} or "blocked" in normalized or "review_only" in normalized


def _llm_gate_encoded(gate: str) -> int:
    normalized = clean_text(gate).lower()
    if not normalized or _llm_gate_blocks_adjustment(normalized):
        return 0
    if "allow" in normalized or "approved" in normalized:
        return 2
    return 1


def _llm_soft_action_encoded(action: str) -> int:
    normalized = clean_text(action).lower()
    return {"relabel_soft": 1, "boost_soft": 2}.get(normalized, 0)


def _llm_evidence_count_bucket(value: Any) -> int:
    count = _as_int(value)
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    if count <= 5:
        return 2
    return 3


def _profile_role_family_keys(profile: dict[str, Any]) -> set[str]:
    return _profile_keys(profile, "preferred_role_families") | _profile_keys(profile, "required_role_families")


def _safe_sidecar_payload(
    profile: dict[str, Any],
    job: dict[str, Any],
    sidecar: dict[str, str] | None,
) -> dict[str, Any]:
    size_class = company_size_class(sidecar or {})
    sponsorship = _sponsorship_signal(sidecar, job)
    lca_scope = clean_text((sidecar or {}).get("lca_match_scope")).lower()
    lca_level = _lca_activity_level((sidecar or {}).get("lca_activity_label", ""))
    lca_bucket = _lca_activity_bucket(lca_level)
    lca_available = lca_bucket > 0 and lca_scope in {"role_family", "employer_only"}

    llm_reviewed = _reviewed_llm_status(sidecar)
    llm_role_family = normalize_for_key((sidecar or {}).get("llm_reviewed_role_family_overlay_candidate"))
    llm_gate = clean_text((sidecar or {}).get("llm_downstream_use_gate")).lower()
    llm_gate_blocks = _llm_gate_blocks_adjustment(llm_gate)
    role_family_matches = bool(llm_reviewed and llm_role_family and llm_role_family in _profile_role_family_keys(profile))
    soft_action = clean_text((sidecar or {}).get("llm_reviewed_suggested_soft_action")).lower()

    features = {
        "company_size_bucket_visible_encoded": _company_size_bucket_encoded(size_class),
        "company_size_known_flag": int(size_class != "unknown"),
        "company_size_matches_profile": int(_company_size_matches_profile(profile, size_class)),
        "sponsorship_signal_visible_encoded": _sponsorship_encoded(sponsorship),
        "lca_activity_visible_bucket": lca_bucket if lca_available else 0,
        "lca_activity_available_flag": int(lca_available),
        "lca_sponsor_proxy_for_needs_sponsorship": int(bool(profile.get("needs_sponsorship")) and lca_available),
        "llm_overlay_available_flag": int(llm_reviewed and not llm_gate_blocks),
        "llm_role_family_matches_profile": int(role_family_matches and not llm_gate_blocks),
        "llm_suggested_soft_action_encoded": 0 if llm_gate_blocks or not llm_reviewed else _llm_soft_action_encoded(soft_action),
        "llm_downstream_use_gate_encoded": _llm_gate_encoded(llm_gate),
        "llm_evidence_count_bucket": _llm_evidence_count_bucket((sidecar or {}).get("llm_evidence_count")) if llm_reviewed else 0,
    }
    availability = {
        "sidecar_row_available": sidecar is not None,
        "company_size_signal_available": bool(features["company_size_known_flag"]),
        "sponsorship_signal_available": sponsorship not in {"", "unknown"},
        "h1b_activity_proxy_available": lca_available,
        "llm_reviewed_role_family_signal_available": bool(llm_reviewed and llm_role_family and not llm_gate_blocks),
        "llm_evidence_spans_used": False,
        "raw_sidecar_fields_exposed": False,
    }
    labels = [label for key, label in VISIBLE_SAFE_SIDECAR_LABELS.items() if availability.get(key)]
    context = {
        "company_size_class": size_class,
        "sponsorship_signal": sponsorship,
        "lca_activity_level": lca_level if lca_available else "unknown",
        "lca_boundary": "historical employer filing activity only; not job-level sponsorship truth",
        "llm_role_family": llm_role_family if llm_reviewed and not llm_gate_blocks else "",
        "llm_soft_action": soft_action if llm_reviewed and not llm_gate_blocks else "",
        "llm_boundary": "partial review-only overlay; not gold label",
    }
    return {
        "features": features,
        "availability": availability,
        "labels": labels,
        "context": context,
    }


def _event_safe_context(event: dict[str, Any]) -> dict[str, str]:
    snapshot = event.get("job_snapshot") if isinstance(event.get("job_snapshot"), dict) else {}
    context = snapshot.get("learned_safe_sidecar_context")
    if isinstance(context, dict):
        return {str(key): clean_text(value).lower() for key, value in context.items()}
    return {
        "company_size_class": clean_text(snapshot.get("company_size_class")).lower(),
        "llm_role_family": normalize_for_key(snapshot.get("llm_role_family")),
    }


def _safe_sidecar_adjustment(
    profile: dict[str, Any],
    safe_payload: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    features = safe_payload["features"]
    context = safe_payload["context"]
    adjustment = 0.0
    notes: list[str] = []

    size_class = clean_text(context.get("company_size_class")).lower()
    if features["company_size_matches_profile"]:
        adjustment += 0.025
        notes.append("company size matches explicit profile preference")

    accepted_size_match = False
    rejected_size_match_count = 0
    accepted_llm_role_match = False
    llm_role_family = clean_text(context.get("llm_role_family")).lower()
    for event in events:
        action = clean_text(event.get("action")).lower()
        event_context = _event_safe_context(event)
        event_size_class = clean_text(event_context.get("company_size_class")).lower()
        event_llm_role_family = clean_text(event_context.get("llm_role_family")).lower()
        if size_class and size_class != "unknown" and size_class == event_size_class:
            if action == "accept":
                accepted_size_match = True
            elif action == "reject":
                rejected_size_match_count += 1
        if llm_role_family and llm_role_family == event_llm_role_family and action == "accept":
            accepted_llm_role_match = True

    if accepted_size_match:
        adjustment += 0.015
        notes.append("company size matches an accepted pattern from prior session feedback")
    if rejected_size_match_count >= 2:
        adjustment -= 0.020
        notes.append("company size matches a repeatedly rejected prior pattern")

    needs_sponsorship = bool(profile.get("needs_sponsorship"))
    sponsorship = clean_text(context.get("sponsorship_signal")).lower()
    if needs_sponsorship and sponsorship in {
        "mentions_sponsorship_or_work_auth",
        "sponsorship_available",
        "visa_sponsorship",
        "h1b_sponsorship",
    }:
        adjustment += 0.030
        notes.append("posting visibly mentions sponsorship or work authorization")
    if needs_sponsorship:
        lca_level = clean_text(context.get("lca_activity_level")).lower()
        if lca_level == "high":
            adjustment += 0.020
            notes.append("employer has high H-1B activity proxy; not job-level sponsorship truth")
        elif lca_level == "medium":
            adjustment += 0.010
            notes.append("employer has medium H-1B activity proxy; not job-level sponsorship truth")

    if features["llm_role_family_matches_profile"]:
        adjustment += 0.020
        notes.append("reviewed LLM role-family signal matches the profile")
    if accepted_llm_role_match:
        adjustment += 0.015
        notes.append("reviewed LLM role family matches an accepted prior pattern")
    action_encoded = features["llm_suggested_soft_action_encoded"]
    if action_encoded == 2:
        adjustment += 0.010
        notes.append("reviewed LLM overlay suggested a small boost")
    elif action_encoded == 1:
        adjustment += 0.005
        notes.append("reviewed LLM overlay suggested a soft relabel")

    adjustment = _clamp(adjustment, SAFE_SIDECAR_MIN_ADJUSTMENT, SAFE_SIDECAR_MAX_ADJUSTMENT)
    return {
        "adjustment": round(adjustment, 6),
        "notes": list(dict.fromkeys(notes)),
    }


def _ui_completeness(job: dict[str, Any], salary_display: str, match_label: str) -> float:
    checks = [
        bool(clean_text(job.get("title"))),
        bool(clean_text(job.get("company") or job.get("employer"))),
        bool(clean_text(job.get("location"))),
        bool(clean_text(salary_display)),
        bool(clean_text(job.get("employment_type"))),
        bool(match_label),
        _driver_count(job, "positive_drivers") > 0,
        _driver_count(job, "negative_drivers") > 0,
        _matched_skill_count(job) > 0,
        True,
        True,
    ]
    return round(sum(1 for item in checks if item) / len(checks), 4)


def _persona_id(profile: dict[str, Any]) -> str:
    profile_id = clean_text(profile.get("profile_id")).lower()
    if profile_id:
        return profile_id
    name = clean_text(profile.get("name")).lower()
    return name or "unknown"


def _load_feature_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_hash(path: str | Path) -> str:
    payload = Path(path).read_text(encoding="utf-8")
    return stable_hash(payload)


@lru_cache(maxsize=1)
def load_r0_comparator_summary(path: str | Path = R0_COMPARATOR_SUMMARY) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.exists():
        return {
            "available": False,
            "id": "phase2_18j_r0_round0",
            "source": summary_path.as_posix(),
            "reason": "missing_r0_learning_curve_summary",
        }
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        round0 = ((payload.get("round_metrics") or {}).get("0") or {})
        overall = round0.get("overall") or {}
        per_persona = round0.get("per_persona") or {}
        assignment_personas = {}
        for persona_id, row in per_persona.items():
            if persona_id not in {"aisha", "kenji", "marcus", "priya"}:
                continue
            assignment_personas[persona_id] = {
                "mean_outcome_at_10": _as_float(row.get("mean_outcome_label_at_10")),
                "accept_at_10": _as_float(row.get("accept_rate_at_10")),
                "ndcg_at_10": _as_float(row.get("ndcg_at_10")),
            }
        return {
            "available": True,
            "id": "phase2_18j_r0_round0",
            "source": summary_path.as_posix(),
            "description": "Phase 2.18J immutable round-0 /match-equivalent Top200 reservoir simulation aggregate comparator.",
            "overall": {
                "mean_outcome_at_10": _as_float(overall.get("mean_outcome_label_at_10")),
                "accept_at_10": _as_float(overall.get("accept_rate_at_10")),
                "ndcg_at_10": _as_float(overall.get("ndcg_at_10")),
                "validator_pass_rate": _as_float(overall.get("validator_pass_rate")),
            },
            "assignment_personas": assignment_personas,
        }
    except Exception as exc:
        return {
            "available": False,
            "id": "phase2_18j_r0_round0",
            "source": summary_path.as_posix(),
            "reason": f"{type(exc).__name__}: {exc}",
        }


@lru_cache(maxsize=4)
def load_runtime_artifact(path: str | Path = RUNTIME_MODEL_ARTIFACT) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None, {
            "available": False,
            "path": artifact_path.as_posix(),
            "reason": "runtime EBM artifact not found",
        }
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - exercised by integration fallback tests.
        return None, {
            "available": False,
            "path": artifact_path.as_posix(),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(artifact, dict) or artifact.get("model") is None:
        return None, {
            "available": False,
            "path": artifact_path.as_posix(),
            "reason": "runtime EBM artifact has invalid shape",
        }
    return artifact, {
        "available": True,
        "path": artifact_path.as_posix(),
        "model_class": clean_text(artifact.get("model_class")),
        "feature_count": len(artifact.get("feature_columns") or []),
        "training_rows": artifact.get("training_rows"),
        "feature_manifest_hash": artifact.get("feature_manifest_hash"),
    }


class Phase218JLearnedReranker:
    """Default learned post-filter reranker with R0+safe-sidecar fallback."""

    def __init__(
        self,
        *,
        model_path: str | Path = RUNTIME_MODEL_ARTIFACT,
        feature_manifest_path: str | Path = INTERACTION_FEATURE_MANIFEST,
        sidecar_path: str | Path = RANKING_FEATURES_CSV,
        fallback_reranker: ProductionEvidenceReranker | None = None,
        mode: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.feature_manifest_path = Path(feature_manifest_path)
        self.sidecar_path = Path(sidecar_path)
        self.fallback_reranker = fallback_reranker or ProductionEvidenceReranker()
        self.mode = self._normalize_mode(mode or os.getenv("JOBPILOT_RERANK_MODE") or DEFAULT_MODE)

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = clean_text(mode).lower()
        return normalized if normalized in VALID_MODES else DEFAULT_MODE

    def _feature_columns(self, artifact: dict[str, Any]) -> list[str]:
        columns = artifact.get("feature_columns")
        if isinstance(columns, list) and columns:
            return [clean_text(column) for column in columns if clean_text(column)]
        manifest = _load_feature_manifest(self.feature_manifest_path)
        return [clean_text(column) for column in manifest.get("model_features") or [] if clean_text(column)]

    def _feature_row(
        self,
        profile: dict[str, Any],
        job: dict[str, Any],
        rank_shown: int,
        persona_encoding: dict[str, int],
        safe_sidecar_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        salary_display = _salary_display(job)
        match_label = _match_strength_label(job)
        why_ranked = job.get("why_ranked") if isinstance(job.get("why_ranked"), dict) else {}
        preference_status = why_ranked.get("preference_status") if isinstance(why_ranked, dict) else {}
        location_match = bool(preference_status.get("location_matches_preference")) if isinstance(preference_status, dict) else False
        has_company_pref, prefers_large, prefers_startup = _company_preference_flags(profile)
        persona = _persona_id(profile)
        row = {
            "persona_id_encoded": persona_encoding.get(persona, 0),
            "round_index": RUNTIME_ROUND_INDEX,
            "rank_shown": rank_shown,
            "prior_rank_if_seen": 0,
            "shown_before": 0,
            "salary_listed": int(_salary_number(salary_display) is not None),
            "salary_min_numeric_bucket": _salary_bucket(salary_display),
            "location_matches_profile": int(location_match),
            "remote_or_preferred_location": int(location_match or _is_remote_job(job)),
            "employment_type_matches": _employment_matches(profile, job),
            "years_required_bucket": _years_bucket(job),
            "seniority_warning_flag": _seniority_warning(job),
            "work_authorization_visible_flag": int(clean_text(job.get("sponsorship_signal")).lower() not in {"", "unknown"}),
            "match_strength_label_encoded": MATCH_STRENGTH_ENCODING.get(match_label, 0),
            "matched_skill_count": _matched_skill_count(job),
            "why_reason_count": _driver_count(job, "positive_drivers"),
            "watch_out_count": _driver_count(job, "negative_drivers"),
            "key_requirement_count": 4,
            "ui_projection_completeness_score": _ui_completeness(job, salary_display, match_label),
            "needs_sponsorship": int(bool(profile.get("needs_sponsorship"))),
            "strict_location": int(bool(profile.get("strict_location") or profile.get("us_only"))),
            "salary_is_dealbreaker": int(bool(profile.get("salary_is_dealbreaker") or profile.get("strict_salary"))),
            "avoid_senior": _avoid_senior(profile),
            "avoid_contract": _avoid_contract(profile),
            "has_company_size_preference": has_company_pref,
            "prefers_large_company": prefers_large,
            "prefers_startup": prefers_startup,
            "strict_role_family": int(bool(profile.get("strict_role_family"))),
            "prior_accept_count_persona": 0,
            "prior_skip_count_persona": 0,
            "prior_reject_count_persona": 0,
            "previously_seen_job": 0,
            "previously_rejected_job": 0,
            "previously_rejected_company": 0,
            "previously_accepted_company": 0,
            "previously_skipped_company": 0,
            "accepted_skill_overlap_prior": 0,
            "rejected_skill_overlap_prior": 0,
            "skipped_skill_overlap_prior": 0,
            "repeated_bad_recommendation_flag": 0,
        }
        row.update(safe_sidecar_features or {})
        return row

    def _quality_gate_passes(self, job: dict[str, Any]) -> bool:
        return _matched_skill_count(job) >= 1 or _match_strength_label(job) in QUALITY_MATCH_LABELS

    def _apply_session_overlay(self, scored_jobs: list[dict[str, Any]], session_feedback_events: list[dict[str, Any]]) -> dict[str, Any]:
        suppressed = 0
        adjusted = 0
        same_company_penalties = 0
        accepted_neighbor_boosts = 0
        for job in scored_jobs:
            if not bool(job.get("hard_filter_passed", True)):
                continue
            overlay = _session_feedback_overlay(job, session_feedback_events)
            if overlay["suppress"]:
                suppressed += 1
                job["session_feedback_suppressed"] = True
                job["ranking_sort_group"] = 0
            elif overlay["adjustment"] != 0 or overlay["cap"] is not None:
                adjusted += 1
                if overlay["adjustment"] < 0 or overlay["cap"] is not None:
                    same_company_penalties += 1
                if overlay["adjustment"] > 0:
                    accepted_neighbor_boosts += 1
            if overlay["notes"]:
                base_score = _as_float(job.get("learned_ebm_score"), _as_float(job.get("final_score")))
                adjusted_score = _clamp(base_score + overlay["adjustment"], 0.0, 1.0)
                if overlay["cap"] is not None:
                    adjusted_score = min(adjusted_score, float(overlay["cap"]))
                job["session_feedback_adjustment"] = overlay["adjustment"]
                job["session_feedback_score"] = round(adjusted_score, 6)
                job["session_feedback_notes"] = overlay["notes"]
                job["feedback_adjustment_explanation"] = "; ".join(overlay["notes"])
                job["final_score"] = round(adjusted_score, 6)
                job["ranking_primary_score"] = round(adjusted_score, 6)
        return {
            "enabled": bool(session_feedback_events),
            "prior_event_count": len(session_feedback_events),
            "uses_prior_session_feedback_only": True,
            "same_job_reject_suppressed_count": suppressed,
            "adjusted_count": adjusted,
            "same_company_reject_penalty_or_cap_count": same_company_penalties,
            "accept_neighborhood_boost_count": accepted_neighbor_boosts,
            "retraining_triggered": False,
        }

    def _old_baseline(
        self,
        profile: dict[str, Any],
        scored_jobs: list[dict[str, Any]],
        session_feedback_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = self.fallback_reranker.apply(profile, scored_jobs)
        for job in scored_jobs:
            job.setdefault("ranking_sort_group", 1)
            job.setdefault("ranking_primary_score", _as_float(job.get("final_score")))
        session_overlay = self._apply_session_overlay(scored_jobs, session_feedback_events)
        return {
            "policy": RUNTIME_POLICY,
            "mode": "old_baseline",
            "status": "old_baseline_active",
            "old_baseline_policy": OLD_BASELINE_POLICY,
            "old_baseline": metadata,
            "session_feedback_overlay": session_overlay,
            "fallback_reason": "",
        }

    def _shadow_old_baseline(self, profile: dict[str, Any], scored_jobs: list[dict[str, Any]]) -> dict[str, Any]:
        shadow_jobs = [dict(job) for job in scored_jobs]
        metadata = self.fallback_reranker.apply(profile, shadow_jobs)
        by_id = {clean_text(job.get("job_id")): job for job in shadow_jobs}
        for job in scored_jobs:
            shadow = by_id.get(clean_text(job.get("job_id"))) or {}
            job["old_baseline_score"] = _as_float(shadow.get("final_score"), _as_float(job.get("final_score")))
            job["old_baseline_evidence_adjustment"] = _as_float(shadow.get("evidence_adjustment"))
        return metadata

    def _r0_safe_sidecar_fallback(
        self,
        profile: dict[str, Any],
        scored_jobs: list[dict[str, Any]],
        reason: str,
        artifact_metadata: dict[str, Any],
        session_feedback_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sidecar_rows, sidecar_metadata = load_ranking_sidecar(self.sidecar_path)
        eligible = [job for job in scored_jobs if bool(job.get("hard_filter_passed", True))]
        reservoir = eligible[: CANDIDATE_RESERVOIR_SIZE]
        reservoir_ids = {id(job) for job in reservoir}
        quality_pass_count = 0
        quality_fail_count = 0
        sidecar_rows_matched = 0
        safe_sidecar_adjusted_count = 0
        safe_sidecar_label_counts: dict[str, int] = {label: 0 for label in VISIBLE_SAFE_SIDECAR_LABELS.values()}

        for rank_shown, job in enumerate(reservoir, start=1):
            sidecar = sidecar_rows.get(clean_text(job.get("job_id"))) if sidecar_metadata["available"] else None
            if sidecar:
                sidecar_rows_matched += 1
            safe_payload = _safe_sidecar_payload(profile, job, sidecar)
            for label in safe_payload["labels"]:
                safe_sidecar_label_counts[label] = safe_sidecar_label_counts.get(label, 0) + 1
            job["learned_safe_sidecar_features"] = dict(safe_payload["features"])
            job["learned_shadow_features"] = dict(safe_payload["features"])
            job["learned_feature_availability"] = dict(safe_payload["availability"])
            job["learned_safe_sidecar_context"] = dict(safe_payload["context"])
            job["safe_sidecar_signal_labels"] = list(safe_payload["labels"])

            base_score = _as_float(job.get("final_score"))
            session_overlay = _session_feedback_overlay(job, session_feedback_events)
            sidecar_overlay = _safe_sidecar_adjustment(profile, safe_payload, session_feedback_events)
            if sidecar_overlay["adjustment"]:
                safe_sidecar_adjusted_count += 1
            combined_overlay = _clamp(
                session_overlay["adjustment"] + sidecar_overlay["adjustment"],
                POST_EBM_MIN_ADJUSTMENT,
                POST_EBM_MAX_ADJUSTMENT,
            )
            final_session_score = round(_clamp(base_score + combined_overlay, 0.0, 1.0), 6)
            if session_overlay["cap"] is not None:
                final_session_score = round(min(final_session_score, float(session_overlay["cap"])), 6)
            quality_passed = self._quality_gate_passes(job)
            if quality_passed:
                quality_pass_count += 1
                job["ranking_sort_group"] = 3
            else:
                quality_fail_count += 1
                job["ranking_sort_group"] = 2
            job["final_score"] = final_session_score
            job["final_session_score"] = final_session_score
            job["ranking_primary_score"] = final_session_score
            job["rerank_applied"] = bool(sidecar_overlay["adjustment"] or session_overlay["adjustment"])
            job["learned_rerank_applied"] = False
            job["learned_rerank_fallback_reason"] = reason
            job["r0_safe_sidecar_fallback_applied"] = True
            job["r0_base_score"] = base_score
            job["learned_quality_gate_passed"] = quality_passed
            job["learned_policy"] = R0_SAFE_SIDECAR_FALLBACK_POLICY
            job["learned_tie_breaker_base_rank"] = _as_int(job.get("base_rank"))
            if session_overlay["suppress"]:
                job["session_feedback_suppressed"] = True
                job["ranking_sort_group"] = 0
                job["ranking_primary_score"] = 0.0
                job["final_score"] = 0.0
                job["rerank_applied"] = False
            if session_overlay["notes"]:
                job["session_feedback_adjustment"] = session_overlay["adjustment"]
                job["session_feedback_score"] = final_session_score
                job["session_feedback_notes"] = session_overlay["notes"]
            if sidecar_overlay["notes"]:
                job["safe_sidecar_adjustment"] = sidecar_overlay["adjustment"]
                job["safe_sidecar_adjustment_notes"] = sidecar_overlay["notes"]
            all_overlay_notes = list(dict.fromkeys((session_overlay["notes"] or []) + (sidecar_overlay["notes"] or [])))
            if all_overlay_notes:
                job["feedback_adjustment_explanation"] = "; ".join(all_overlay_notes)
            job["post_ebm_overlay_adjustment"] = round(combined_overlay, 6)

        for job in scored_jobs:
            if id(job) in reservoir_ids:
                continue
            job["learned_rerank_applied"] = False
            job["learned_rerank_fallback_reason"] = reason
            job["learned_quality_gate_passed"] = False
            job["ranking_sort_group"] = 1 if bool(job.get("hard_filter_passed", True)) else 0
            job["ranking_primary_score"] = _as_float(job.get("final_score"))

        session_overlay = {
            "enabled": bool(session_feedback_events),
            "prior_event_count": len(session_feedback_events),
            "uses_prior_session_feedback_only": True,
            "same_job_reject_suppressed_count": sum(1 for job in scored_jobs if job.get("session_feedback_suppressed")),
            "adjusted_count": sum(1 for job in scored_jobs if job.get("session_feedback_notes")),
            "same_company_reject_penalty_or_cap_count": sum(
                1
                for job in scored_jobs
                if any("rejected" in note for note in job.get("session_feedback_notes") or [])
            ),
            "accept_neighborhood_boost_count": sum(
                1
                for job in scored_jobs
                if any("accepted" in note or "accepted job" in note for note in job.get("session_feedback_notes") or [])
            ),
            "retraining_triggered": False,
        }
        safe_sidecar_overlay = {
            "enabled": bool(sidecar_metadata["available"]),
            "sidecar_path": sidecar_metadata["path"],
            "sidecar_row_count": sidecar_metadata["row_count"],
            "reservoir_rows_with_sidecar": sidecar_rows_matched,
            "adjusted_count": safe_sidecar_adjusted_count,
            "total_cap": {"min": SAFE_SIDECAR_MIN_ADJUSTMENT, "max": SAFE_SIDECAR_MAX_ADJUSTMENT},
            "combined_post_ebm_cap": {"min": POST_EBM_MIN_ADJUSTMENT, "max": POST_EBM_MAX_ADJUSTMENT},
            "signal_label_counts": safe_sidecar_label_counts,
            "lca_boundary": "historical employer filing activity only; not job-level sponsorship truth",
            "llm_boundary": "partial review-only overlay; not gold label",
            "llm_evidence_spans_used": False,
            "raw_sidecar_fields_exposed": False,
        }
        return {
            "policy": R0_SAFE_SIDECAR_FALLBACK_POLICY,
            "mode": self.mode,
            "status": "fallback_r0_safe_sidecar",
            "enabled": False,
            "fallback_reason": reason,
            "artifact": artifact_metadata,
            "candidate_reservoir_size": CANDIDATE_RESERVOIR_SIZE,
            "display_top_k_policy": "current UI Top N",
            "hard_filter_boundary": "inherited_from_JobRanker",
            "visible_hard_dealbreaker_suppression": True,
            "quality_gate": "matched_skill_count >= 1 OR match_strength_label in {Strong match, Good match}",
            "quality_gate_pass_count": quality_pass_count,
            "quality_gate_failed_count": quality_fail_count,
            "scored_jobs": len(scored_jobs),
            "eligible_jobs": len(eligible),
            "reservoir_jobs": len(reservoir),
            "primary_score": "R0-equivalent base score + session/safe-sidecar overlay",
            "tie_breaker": "base_rank",
            "session_feedback_overlay": session_overlay,
            "safe_sidecar_overlay": safe_sidecar_overlay,
            "safe_sidecar_features_added": list(SAFE_SIDECAR_FEATURES),
            "sidecar_safe_features_available": bool(sidecar_metadata["available"]),
            "sidecar_safe_features_used_by_model": [],
            "sidecar_safe_features_shadowed_only": list(SAFE_SIDECAR_FEATURES),
            "ebm_feature_columns_missing_safe_sidecar_features": list(SAFE_SIDECAR_FEATURES),
            "sidecar_safe_features_current_model_claim": "fallback_policy_not_ebm_model_inputs",
            "comparator": "phase2_18j_r0_round0",
            "r0_comparator": load_r0_comparator_summary(),
            "old_baseline_runtime_use": "not_used_for_r0_safe_sidecar_fallback",
            "old_baseline_policy": OLD_BASELINE_POLICY,
            "old_baseline": {"status": "not_run"},
        }

    def _fallback(
        self,
        profile: dict[str, Any],
        scored_jobs: list[dict[str, Any]],
        reason: str,
        artifact_metadata: dict[str, Any],
        session_feedback_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._r0_safe_sidecar_fallback(profile, scored_jobs, reason, artifact_metadata, session_feedback_events)

    def apply(
        self,
        profile: dict[str, Any],
        scored_jobs: list[dict[str, Any]],
        *,
        session_feedback_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        session_feedback_events = session_feedback_events or []
        if self.mode == "old_baseline":
            return self._old_baseline(profile, scored_jobs, session_feedback_events)
        if self.mode == "r0_safe_sidecar":
            return self._r0_safe_sidecar_fallback(
                profile,
                scored_jobs,
                "explicit_r0_safe_sidecar_mode",
                {"available": False, "reason": "explicit_r0_safe_sidecar_mode", "path": self.model_path.as_posix()},
                session_feedback_events,
            )

        artifact, artifact_metadata = load_runtime_artifact(self.model_path)
        if artifact is None:
            return self._fallback(profile, scored_jobs, artifact_metadata["reason"], artifact_metadata, session_feedback_events)

        try:
            manifest = _load_feature_manifest(self.feature_manifest_path)
            feature_columns = self._feature_columns(artifact)
            model = artifact["model"]
            persona_encoding = (manifest.get("categorical_or_encoded_features") or {}).get("persona_id_encoded") or {}
        except Exception as exc:
            return self._fallback(profile, scored_jobs, f"{type(exc).__name__}: {exc}", artifact_metadata, session_feedback_events)

        sidecar_rows, sidecar_metadata = load_ranking_sidecar(self.sidecar_path)
        feature_column_set = set(feature_columns)
        safe_features_used_by_model = [name for name in SAFE_SIDECAR_FEATURES if name in feature_column_set]
        safe_features_shadowed_only = [name for name in SAFE_SIDECAR_FEATURES if name not in feature_column_set]
        if self.mode == "learned_shadow":
            old_metadata = self._shadow_old_baseline(profile, scored_jobs)
        else:
            old_metadata = {
                "status": "not_run",
                "reason": "learned_active uses EBM score with R0 aggregate diagnostics; old baseline is reserved for fallback or explicit diagnostic mode.",
            }
        r0_comparator = load_r0_comparator_summary()
        eligible = [job for job in scored_jobs if bool(job.get("hard_filter_passed", True))]
        reservoir = eligible[: CANDIDATE_RESERVOIR_SIZE]
        reservoir_ids = {id(job) for job in reservoir}
        quality_pass_count = 0
        quality_fail_count = 0
        predictions_made = 0
        sidecar_rows_matched = 0
        safe_sidecar_adjusted_count = 0
        safe_sidecar_label_counts: dict[str, int] = {label: 0 for label in VISIBLE_SAFE_SIDECAR_LABELS.values()}

        for rank_shown, job in enumerate(reservoir, start=1):
            sidecar = sidecar_rows.get(clean_text(job.get("job_id"))) if sidecar_metadata["available"] else None
            if sidecar:
                sidecar_rows_matched += 1
            safe_payload = _safe_sidecar_payload(profile, job, sidecar)
            for label in safe_payload["labels"]:
                safe_sidecar_label_counts[label] = safe_sidecar_label_counts.get(label, 0) + 1
            job["learned_safe_sidecar_features"] = dict(safe_payload["features"])
            job["learned_shadow_features"] = dict(safe_payload["features"])
            job["learned_feature_availability"] = dict(safe_payload["availability"])
            job["learned_safe_sidecar_context"] = dict(safe_payload["context"])
            job["safe_sidecar_signal_labels"] = list(safe_payload["labels"])

            features = self._feature_row(profile, job, rank_shown, persona_encoding, safe_payload["features"])
            quality_passed = self._quality_gate_passes(job)
            row = np.asarray([[float(features.get(column, 0.0)) for column in feature_columns]], dtype=np.float64)
            ebm_score = round(max(0.0, min(1.0, float(model.predict(row)[0]))), 6)
            job["learned_ebm_score"] = ebm_score
            session_overlay = _session_feedback_overlay(job, session_feedback_events)
            sidecar_overlay = _safe_sidecar_adjustment(profile, safe_payload, session_feedback_events)
            if sidecar_overlay["adjustment"]:
                safe_sidecar_adjusted_count += 1
            combined_overlay = _clamp(
                session_overlay["adjustment"] + sidecar_overlay["adjustment"],
                POST_EBM_MIN_ADJUSTMENT,
                POST_EBM_MAX_ADJUSTMENT,
            )
            final_session_score = round(_clamp(ebm_score + combined_overlay, 0.0, 1.0), 6)
            if session_overlay["cap"] is not None:
                final_session_score = round(min(final_session_score, float(session_overlay["cap"])), 6)
            predictions_made += 1
            if quality_passed:
                quality_pass_count += 1
                job["final_score"] = final_session_score
                job["ranking_sort_group"] = 3
                job["ranking_primary_score"] = final_session_score
                job["rerank_applied"] = True
                job["learned_rerank_applied"] = self.mode == "learned_active"
            else:
                quality_fail_count += 1
                job["final_score"] = final_session_score
                job["ranking_sort_group"] = 2
                job["ranking_primary_score"] = final_session_score
                job["rerank_applied"] = False
                job["learned_rerank_applied"] = False
            if session_overlay["suppress"]:
                job["session_feedback_suppressed"] = True
                job["ranking_sort_group"] = 0
                job["ranking_primary_score"] = 0.0
                job["final_score"] = 0.0
                job["rerank_applied"] = False
            if session_overlay["notes"]:
                job["session_feedback_adjustment"] = session_overlay["adjustment"]
                job["session_feedback_score"] = final_session_score
                job["session_feedback_notes"] = session_overlay["notes"]
            if sidecar_overlay["notes"]:
                job["safe_sidecar_adjustment"] = sidecar_overlay["adjustment"]
                job["safe_sidecar_adjustment_notes"] = sidecar_overlay["notes"]
            all_overlay_notes = list(dict.fromkeys((session_overlay["notes"] or []) + (sidecar_overlay["notes"] or [])))
            if all_overlay_notes:
                job["feedback_adjustment_explanation"] = "; ".join(all_overlay_notes)
            job["post_ebm_overlay_adjustment"] = round(combined_overlay, 6)
            job["final_session_score"] = final_session_score
            job["base_score"] = _as_float(job.get("base_score"), _as_float(job.get("final_score")))
            job["learned_quality_gate_passed"] = quality_passed
            job["learned_feature_contract"] = Path(self.feature_manifest_path).name
            job["learned_policy"] = RUNTIME_POLICY
            job["learned_tie_breaker_base_rank"] = _as_int(job.get("base_rank"))

        for job in scored_jobs:
            if id(job) in reservoir_ids:
                continue
            job["learned_rerank_applied"] = False
            job["learned_quality_gate_passed"] = False
            job["ranking_sort_group"] = 1 if bool(job.get("hard_filter_passed", True)) else 0
            job["ranking_primary_score"] = _as_float(job.get("final_score"))

        if self.mode == "learned_shadow":
            for job in scored_jobs:
                job["ranking_sort_group"] = 1
                job["ranking_primary_score"] = _as_float(job.get("old_baseline_score"), _as_float(job.get("final_score")))
                if "old_baseline_score" in job:
                    job["final_score"] = job["old_baseline_score"]
                job["rerank_applied"] = bool(_as_float(job.get("old_baseline_evidence_adjustment")) > 0)

        session_overlay = {
            "enabled": bool(session_feedback_events),
            "prior_event_count": len(session_feedback_events),
            "uses_prior_session_feedback_only": True,
            "same_job_reject_suppressed_count": sum(1 for job in scored_jobs if job.get("session_feedback_suppressed")),
            "adjusted_count": sum(1 for job in scored_jobs if job.get("session_feedback_notes")),
            "same_company_reject_penalty_or_cap_count": sum(
                1
                for job in scored_jobs
                if any("rejected" in note for note in job.get("session_feedback_notes") or [])
            ),
            "accept_neighborhood_boost_count": sum(
                1
                for job in scored_jobs
                if any("accepted" in note or "accepted job" in note for note in job.get("session_feedback_notes") or [])
            ),
            "retraining_triggered": False,
        }
        sidecar_overlay = {
            "enabled": bool(sidecar_metadata["available"]),
            "sidecar_path": sidecar_metadata["path"],
            "sidecar_row_count": sidecar_metadata["row_count"],
            "reservoir_rows_with_sidecar": sidecar_rows_matched,
            "adjusted_count": safe_sidecar_adjusted_count,
            "total_cap": {"min": SAFE_SIDECAR_MIN_ADJUSTMENT, "max": SAFE_SIDECAR_MAX_ADJUSTMENT},
            "combined_post_ebm_cap": {"min": POST_EBM_MIN_ADJUSTMENT, "max": POST_EBM_MAX_ADJUSTMENT},
            "signal_label_counts": safe_sidecar_label_counts,
            "lca_boundary": "historical employer filing activity only; not job-level sponsorship truth",
            "llm_boundary": "partial review-only overlay; not gold label",
            "llm_evidence_spans_used": False,
            "raw_sidecar_fields_exposed": False,
        }

        return {
            "policy": RUNTIME_POLICY,
            "mode": self.mode,
            "status": "learned_shadow_scored" if self.mode == "learned_shadow" else "learned_active",
            "enabled": self.mode == "learned_active",
            "candidate_reservoir_size": CANDIDATE_RESERVOIR_SIZE,
            "display_top_k_policy": "current UI Top N",
            "hard_filter_boundary": "inherited_from_JobRanker",
            "visible_hard_dealbreaker_suppression": True,
            "quality_gate": "matched_skill_count >= 1 OR match_strength_label in {Strong match, Good match}",
            "quality_gate_pass_count": quality_pass_count,
            "quality_gate_failed_count": quality_fail_count,
            "scored_jobs": len(scored_jobs),
            "eligible_jobs": len(eligible),
            "reservoir_jobs": len(reservoir),
            "predictions_made": predictions_made,
            "primary_score": "EBM score" if self.mode == "learned_active" else "old baseline score",
            "tie_breaker": "base_rank",
            "session_feedback_overlay": session_overlay,
            "safe_sidecar_overlay": sidecar_overlay,
            "safe_sidecar_features_added": list(SAFE_SIDECAR_FEATURES),
            "sidecar_safe_features_available": bool(sidecar_metadata["available"]),
            "sidecar_safe_features_used_by_model": safe_features_used_by_model,
            "sidecar_safe_features_shadowed_only": safe_features_shadowed_only,
            "ebm_feature_columns_missing_safe_sidecar_features": safe_features_shadowed_only,
            "sidecar_safe_features_current_model_claim": "shadow_only_not_learned_by_current_artifact"
            if not safe_features_used_by_model
            else "artifact_feature_columns_include_some_safe_sidecar_features",
            "retraining_policy": {
                "retrain_inside_match_request": False,
                "retrain_after_every_click": False,
                "prototype_cadence": "manual or scheduled after each simulation/feedback batch",
                "future_deployed_trigger": "500 new validated rows, or 24 hours plus at least 100 new validated rows",
                "stable_production_cadence": "weekly or after 1000+ validated rows, whichever comes first",
                "course_prototype_promotion_comparator": "phase2_18j_r0_round0",
                "future_production_comparator": "current deployed model or rule fallback",
                "promotion_gates": [
                    "no hard-filter violations",
                    "validator pass rate = 1.0",
                    "no major persona regression",
                    "equal or better mean_outcome@10 / accept@10 versus Phase 2.18J R0 for the course prototype",
                ],
            },
            "forbidden_inputs_excluded": [
                "raw embeddings",
                "raw job descriptions",
                "raw sidecar fields",
                "raw LLM evidence spans",
                "backend debug fields",
                "evidence_adjustment",
                "current-row interaction_action",
            ],
            "artifact": {
                **artifact_metadata,
                "expected_feature_manifest_hash": _manifest_hash(self.feature_manifest_path)
                if self.feature_manifest_path.exists()
                else "",
            },
            "comparator": "phase2_18j_r0_round0",
            "r0_comparator": r0_comparator,
            "old_baseline_runtime_use": "fallback_only" if self.mode == "learned_active" else "explicit_diagnostic_shadow",
            "old_baseline_policy": OLD_BASELINE_POLICY,
            "old_baseline": old_metadata,
            "fallback_reason": "",
        }
