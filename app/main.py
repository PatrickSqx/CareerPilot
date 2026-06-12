"""FastAPI web app for JobPilot Phase 3."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import io
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile

from app.services.analytics_service import (
    as_items,
    load_market_analytics,
    load_tech_market_analytics,
    source_rollup_counts,
)
from app.services.feedback_service import get_feedback_events, init_feedback_db, record_feedback
from app.services.live_service import run_live_refresh_preview
from app.services.matching_service import (
    build_match_payload,
    build_profile_from_request,
    persona_options,
    rank_profile,
    snapshot_status,
    warm_ranker_runtime,
)
from app.services.paths import APP_DIR, PROJECT_ROOT, ensure_storage_dirs
from app.services.profile_parse_service import parse_profile_intake
from app.services.presentation_labels import application_strategy_display_label
from app.services.resume_service import (
    ResumeGenerationUnavailable,
    generate_api_resume,
    resume_docx_filename,
    resume_provider_status,
)
from app.services.session_service import load_session, save_session
from jobpilot.utils.text import clean_text


def _initialize_runtime() -> None:
    ensure_storage_dirs()
    init_feedback_db()
    if clean_text(os.getenv("JOBPILOT_STARTUP_WARM_RANKER")).lower() in {"1", "true", "yes", "on"}:
        try:
            warm_ranker_runtime()
        except Exception as exc:
            print(f"JobPilot startup ranker warmup skipped: {type(exc).__name__}: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _initialize_runtime()
    yield


app = FastAPI(title="JobPilot", version="3.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
templates.env.filters["json"] = lambda value: json.dumps(value, indent=2, ensure_ascii=False)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def _job_apply_url(job: dict[str, Any]) -> str:
    return clean_text(job.get("apply_url") or job.get("link"))


def _format_salary(job: dict[str, Any]) -> str:
    salary_raw = clean_text(job.get("salary_raw"))
    if salary_raw:
        return salary_raw
    salary_min = clean_text(job.get("salary_min"))
    salary_max = clean_text(job.get("salary_max"))
    if salary_min and salary_max:
        return f"{salary_min}-{salary_max}"
    if salary_min:
        return salary_min
    if salary_max:
        return salary_max
    return "Salary not listed"


def _matched_skills_text(job: dict[str, Any]) -> str:
    skills = job.get("matched_skills") or []
    if isinstance(skills, str):
        return skills
    return ", ".join(str(skill) for skill in skills if clean_text(skill))


def _why_summary(job: dict[str, Any]) -> str:
    why_ranked = job.get("why_ranked")
    if isinstance(why_ranked, dict):
        return clean_text(why_ranked.get("summary"))
    return clean_text(why_ranked)


def _match_strength(job: dict[str, Any]) -> dict[str, str]:
    score = _as_float(job.get("adjusted_score"))
    updated = score is not None
    if score is None:
        score = _as_float(job.get("final_score")) or 0.0
    if score >= 0.72:
        label = "Strong match"
    elif score >= 0.62:
        label = "Good match"
    else:
        label = "Possible fit"
    return {
        "label": label,
        "note": "Adjusted from selections" if updated else "Based on profile fit",
        "score": f"{score:.3f}",
    }


def _work_authorization_label(job: dict[str, Any]) -> str:
    signal = clean_text(job.get("sponsorship_signal")).lower()
    labels = {
        "mentions_sponsorship_or_work_auth": "Mentions sponsorship/work authorization",
        "no_sponsorship": "Says no sponsorship",
        "unknown": "Not stated in posting",
        "": "Not stated in posting",
    }
    return labels.get(signal, signal.replace("_", " ").capitalize())


def _display_value(value: Any, fallback: str = "Not listed") -> str:
    text = clean_text(value)
    if not text or text.lower() == "unknown":
        return fallback
    labels = {
        "fulltime": "Full-time",
        "full_time": "Full-time",
        "full-time": "Full-time",
        "parttime": "Part-time",
        "part_time": "Part-time",
        "part-time": "Part-time",
        "contract": "Contract",
        "temporary": "Temporary",
        "internship": "Internship",
        "entry_junior": "Entry / junior",
        "junior": "Entry / junior",
        "entry": "Entry / junior",
        "mid": "Mid-level",
        "midlevel": "Mid-level",
        "mid_level": "Mid-level",
        "senior": "Senior",
        "staff": "Staff",
        "principal": "Principal",
        "lead": "Lead",
    }
    normalized = text.lower().replace(" ", "_")
    if normalized in labels:
        return labels[normalized]
    if "_" in text and text.lower() == text:
        return text.replace("_", " ").title()
    return text


def _skill_chip_view(job: dict[str, Any], limit: int = 4) -> dict[str, Any]:
    skills = job.get("matched_skills") or []
    if isinstance(skills, str):
        skills = [skill.strip() for skill in skills.split(",")]
    visible = [clean_text(skill) for skill in skills if clean_text(skill)]
    return {"visible": visible[:limit], "extra_count": max(0, len(visible) - limit), "all": visible}


def _user_safe_text(value: Any) -> str:
    text = clean_text(value)
    replacements = {
        "Strong target-role/title match": "Strong title match for your target role",
        "Partial target-role/title match": "Partial title match for your target role",
        "Weak target-role/title match": "Weak title match for your target role",
        "Matches required role family": "Matches required role area",
        "Matches preferred role family": "Matches preferred role area",
        "Weak description-level preferred role-family signal": "Some job-description evidence matches your preferred role area",
        "Preferred-region location matches profile preference": "Location matches your preference",
        "Salary meets preference when listed": "Listed salary appears to meet your preference",
        "salary_missing": "Salary preference cannot be verified because salary is missing",
        "salary missing": "Salary preference cannot be verified because salary is missing",
        "sponsorship_unknown": "Sponsorship not stated in the posting",
        "Sponsorship is unknown": "Sponsorship not stated in the posting",
        "mentions_sponsorship_or_work_auth": "Posting mentions sponsorship or work authorization",
        "no_sponsorship": "Posting says no sponsorship",
        "ml_related": "ML / AI",
        "research_ai": "AI research",
        "ml_infra": "ML infrastructure",
        "data_engineering": "data engineering",
        "analytics_entry": "entry-level analytics",
        "bi_analytics": "BI / analytics",
        "large_company": "large company",
        "medium_company": "medium company",
        "small_company": "small company",
        "research_lab": "research lab",
        "Hard-skill evidence uses admitted scoring-safe sidecar terms": "Skills confirmed in the job text",
        "Matched admitted hard-skill sidecar terms": "Skills confirmed in the job text",
        "sidecar": "offline evidence",
        "scoring-safe": "verified",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "_" in text and text.lower() == text:
        text = text.replace("_", " ")
    return text


def _dedupe_display_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = clean_text(item).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _driver_items(job: dict[str, Any], key: str, limit: int | None = None) -> list[str]:
    why_ranked = job.get("why_ranked") if isinstance(job.get("why_ranked"), dict) else {}
    raw_items = why_ranked.get(key) if isinstance(why_ranked, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []
    items = [
        _user_safe_text(item)
        for item in raw_items
        if clean_text(item) and not clean_text(item).lower().startswith("application strategy:")
    ]
    items = _dedupe_display_items(items)
    return items[:limit] if limit else items


def _requirements_snapshot(job: dict[str, Any], profile: dict[str, Any] | None = None) -> list[dict[str, str]]:
    profile = profile or {}
    rows = [
        {"label": "Employment", "value": _display_value(job.get("employment_type"))},
        {"label": "Seniority", "value": _display_value(job.get("seniority"))},
        {"label": "Years", "value": _display_value(job.get("years_required"))},
    ]
    work_auth = _work_authorization_label(job)
    if profile.get("needs_sponsorship") or work_auth != "Not stated in posting":
        rows.append({"label": "Work authorization", "value": work_auth})
    else:
        rows.append({"label": "Work authorization", "value": "Not stated in posting"})
    return rows


def _provider_label(job: dict[str, Any]) -> str:
    source = clean_text(job.get("source") or job.get("raw_source"))
    normalized = source.lower()
    if "careerbuilder" in normalized:
        return "CareerBuilder"
    if "adzuna" in normalized:
        return "Adzuna"
    if "jsearch" in normalized:
        return "JSearch"
    if not source:
        return "Not listed"
    return source.replace("_", " ").title()


def _debug_audit_enabled(request: Request) -> bool:
    query_value = clean_text(request.query_params.get("debug_audit"))
    env_value = clean_text(os.getenv("JOBPILOT_DEBUG_AUDIT_UI"))
    return query_value.lower() in {"1", "true", "yes", "on"} or env_value.lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_breakdown(job: dict[str, Any]) -> list[dict[str, Any]]:
    why_ranked = job.get("why_ranked") if isinstance(job.get("why_ranked"), dict) else {}
    components = job.get("score_components") if isinstance(job.get("score_components"), dict) else {}
    if not components and isinstance(why_ranked, dict):
        components = why_ranked.get("score_components") if isinstance(why_ranked.get("score_components"), dict) else {}
    labels = [
        ("skill_match", "Skills", ("skills",)),
        ("target_role", "Role", ("role",)),
        ("salary", "Salary", ()),
        ("location", "Location", ()),
        ("company", "Company", ()),
        ("sponsorship", "Sponsorship", ()),
        ("employment_type", "Employment type", ("employment",)),
    ]
    breakdown: list[dict[str, Any]] = []
    for key, label, aliases in labels:
        raw_value = components.get(key)
        for alias in aliases:
            if raw_value is None:
                raw_value = components.get(alias)
        value = _as_float(raw_value)
        if value is not None:
            breakdown.append({"label": label, "value": round(value, 2)})
    semantic_value = _as_float(components.get("embedding_similarity") or components.get("embedding"))
    if semantic_value is not None:
        breakdown.insert(0, {"label": "Semantic match", "value": round(semantic_value, 2)})
    return breakdown


def _ranking_boost_items(job: dict[str, Any]) -> list[dict[str, Any]]:
    boosts = job.get("ranking_boosts") or job.get("verified_boosts") or []
    if isinstance(boosts, list) and boosts:
        return [boost for boost in boosts if isinstance(boost, dict)]
    components = job.get("evidence_components") if isinstance(job.get("evidence_components"), dict) else {}
    fallback: list[dict[str, Any]] = []
    hard_skill = _as_float(components.get("hard_skill")) or 0.0
    if hard_skill > 0:
        fallback.append(
            {
                "signal": "hard_skill",
                "label": "Hard skill boost",
                "adjustment": round(hard_skill, 6),
                "reason": "Matched admitted hard-skill sidecar terms.",
            }
        )
    company_size = _as_float(components.get("company_size_type")) or 0.0
    if company_size > 0 and job.get("company_size_policy_applied"):
        fallback.append(
            {
                "signal": "company_size_type",
                "label": "Company-size boost",
                "adjustment": round(company_size, 6),
                "reason": "Matched an explicit company-size preference.",
            }
        )
    return fallback


def _safe_sidecar_signal_items(job: dict[str, Any]) -> list[str]:
    labels = job.get("safe_sidecar_signal_labels")
    if not isinstance(labels, list):
        return []
    allowed = {
        "Company size signal available",
        "Sponsorship signal available",
        "H-1B activity proxy available",
        "LLM reviewed role-family signal available",
    }
    return [label for label in labels if isinstance(label, str) and label in allowed]


def _rank_movement_text(job: dict[str, Any]) -> str:
    base_rank = job.get("base_rank")
    final_rank = job.get("final_rank") or job.get("rank")
    try:
        base = int(base_rank)
        final = int(final_rank)
    except (TypeError, ValueError):
        return ""
    if base != final:
        return f"Base rank #{base} -> Final rank #{final}"
    if job.get("rerank_applied"):
        return "Rank unchanged after boosts"
    return ""


templates.env.globals["job_apply_url"] = _job_apply_url
templates.env.globals["format_salary"] = _format_salary
templates.env.globals["why_summary"] = _why_summary
templates.env.globals["match_strength"] = _match_strength
templates.env.globals["work_authorization_label"] = _work_authorization_label
templates.env.globals["display_value"] = _display_value
templates.env.globals["skill_chip_view"] = _skill_chip_view
templates.env.globals["driver_items"] = _driver_items
templates.env.globals["requirements_snapshot"] = _requirements_snapshot
templates.env.globals["provider_label"] = _provider_label
templates.env.globals["score_breakdown"] = _score_breakdown
templates.env.globals["ranking_boost_items"] = _ranking_boost_items
templates.env.globals["safe_sidecar_signal_items"] = _safe_sidecar_signal_items
templates.env.globals["rank_movement_text"] = _rank_movement_text


def _render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    payload = {"request": request, "project_root": PROJECT_ROOT.as_posix(), **context}
    return templates.TemplateResponse(request, template, payload)


def _session_or_404(session_id: str) -> dict[str, Any]:
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _active_jobs(session: dict[str, Any]) -> list[dict[str, Any]]:
    return session.get("adjusted_top_jobs") or session.get("top_jobs") or []


def _find_job(session: dict[str, Any], job_id: str) -> dict[str, Any]:
    normalized = clean_text(job_id)
    for job in [*session.get("adjusted_top_jobs", []), *session.get("top_jobs", [])]:
        if clean_text(job.get("job_id")) == normalized:
            return job
    raise HTTPException(status_code=404, detail="Job not found in this session")


def _result_context(session: dict[str, Any], request: Request, **extra: Any) -> HTMLResponse:
    session_id = str(session["session_id"])
    context = {
        "session": session,
        "session_id": session_id,
        "profile": session.get("profile", {}),
        "jobs": _active_jobs(session),
        "metadata": session.get("metadata", {}),
        "notes": session.get("notes", []),
        "feedback_events": get_feedback_events(session_id),
        "resume_status": resume_provider_status(),
        "debug_audit": _debug_audit_enabled(request),
        **extra,
    }
    return _render(request, "results.html", context)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "app": "jobpilot", "phase": "3"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render(
        request,
        "index.html",
        {
            "personas": persona_options(),
            "default_persona": "manual",
            "snapshot_status": snapshot_status(),
        },
    )


@app.post("/parse-profile")
async def parse_profile(request: Request) -> JSONResponse:
    form = await request.form()
    upload_value = form.get("resume_pdf") or form.get("profile_file")
    upload = upload_value if isinstance(upload_value, UploadFile) and upload_value.filename else None
    parsed = await parse_profile_intake(
        profile_text=str(form.get("profile_text") or ""),
        upload=upload,
        parser_mode=str(form.get("parser_mode") or ""),
        parser_provider=str(form.get("parser_provider") or ""),
        parser_model=str(form.get("parser_model") or ""),
        parser_api_key=str(form.get("parser_api_key") or ""),
    )
    return JSONResponse(
        {
            "profile": parsed.profile,
            "form_fields": parsed.form_fields,
            "filter_fields": parsed.filter_fields,
            "context_fields": parsed.context_fields,
            "field_sources": parsed.field_sources,
            "notes": parsed.notes,
            "parse_method": parsed.parse_method,
        }
    )


@app.post("/match", response_class=HTMLResponse)
async def match(request: Request) -> HTMLResponse:
    form = await request.form()
    resume_pdf = form.get("resume_pdf")
    upload = resume_pdf if isinstance(resume_pdf, UploadFile) and resume_pdf.filename else None
    profile, notes = await build_profile_from_request(dict(form), upload)
    top_k = int(clean_text(form.get("top_k")) or "10")
    candidate_k = int(clean_text(form.get("candidate_k")) or "1000")
    result = rank_profile(profile, top_k=max(1, min(top_k, 50)), candidate_k=max(50, min(candidate_k, 5000)))
    payload = build_match_payload(profile, result, notes=notes)
    save_session(payload)
    return _result_context(payload, request)


@app.post("/feedback", response_class=HTMLResponse)
async def feedback(request: Request) -> HTMLResponse:
    form = await request.form()
    session = _session_or_404(clean_text(form.get("session_id")))
    job = _find_job(session, clean_text(form.get("job_id")))
    action = clean_text(form.get("action")).lower()
    try:
        event = record_feedback(session_id=str(session["session_id"]), profile=session.get("profile", {}), job=job, action=action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _result_context(session, request, status_message=f"Recorded {event['action']} feedback.")


@app.post("/rerank", response_class=HTMLResponse)
async def rerank(request: Request) -> HTMLResponse:
    form = await request.form()
    session = _session_or_404(clean_text(form.get("session_id")))
    events = get_feedback_events(str(session["session_id"]))
    top_k = int(session.get("metadata", {}).get("requested_top_k") or len(session.get("top_jobs", [])) or 10)
    candidate_k = int(session.get("metadata", {}).get("candidate_k") or 1000)
    result = rank_profile(
        session.get("profile", {}),
        top_k=max(1, min(top_k, 50)),
        candidate_k=max(50, min(candidate_k, 5000)),
        session_feedback_events=events,
    )
    refreshed = build_match_payload(
        session.get("profile", {}),
        result,
        notes=session.get("notes", []),
        session_id=str(session["session_id"]),
    )
    refreshed["feedback_event_count"] = len(events)
    refreshed["profile_text"] = session.get("profile_text", refreshed.get("profile_text", ""))
    session.clear()
    session.update(refreshed)
    session.pop("adjusted_top_jobs", None)
    save_session(session)
    return _result_context(session, request, status_message=f"Results refreshed from {len(events)} selections.")


@app.post("/resume")
async def resume(request: Request):
    form = await request.form()
    session = _session_or_404(clean_text(form.get("session_id")))
    job = _find_job(session, clean_text(form.get("job_id")))
    resume_api_key = clean_text(form.get("resume_api_key"))
    resume_model = clean_text(form.get("resume_model"))
    try:
        resume_payload = generate_api_resume(
            session.get("profile", {}),
            job,
            api_key_override=resume_api_key,
            model_override=resume_model,
        )
    except ResumeGenerationUnavailable as exc:
        return _result_context(session, request, status_message=str(exc))
    except Exception as exc:
        return _result_context(
            session,
            request,
            status_message=f"Resume generation failed through the connected API: {type(exc).__name__}.",
        )
    filename = resume_docx_filename(job)
    return StreamingResponse(
        iter([resume_payload["docx_bytes"]]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request) -> HTMLResponse:
    payload = load_market_analytics()
    tech_payload = load_tech_market_analytics()
    tech_segment_counts = as_items(tech_payload.get("segment_counts"), limit=10)
    tech_largest_segment = max(
        tech_segment_counts,
        key=lambda item: float(item.get("value") or 0),
        default={"label": "", "value": 0},
    )
    return _render(
        request,
        "analytics.html",
        {
            "analytics": payload,
            "tech_analytics": tech_payload,
            "tech_segment_counts": tech_segment_counts,
            "tech_largest_segment": tech_largest_segment,
            "tech_top_skills": as_items(tech_payload.get("top_focus_skills"), limit=12),
            "tech_top_titles": as_items(tech_payload.get("top_focus_titles"), limit=10),
            "tech_top_locations": as_items(tech_payload.get("top_focus_locations"), limit=8),
            "source_rollup": as_items(
                source_rollup_counts(payload.get("source_counts"), row_count=payload.get("row_count")),
                limit=10,
            ),
            "source_counts": as_items(payload.get("source_counts"), limit=20),
            "top_skills": as_items(payload.get("top_skills") or payload.get("skill_counts"), limit=20),
            "top_locations": as_items(payload.get("top_locations") or payload.get("location_counts"), limit=20),
            "top_titles": as_items(payload.get("top_titles") or payload.get("title_counts"), limit=20),
            "remote_distribution": as_items(payload.get("remote_distribution"), limit=10),
            "employment_type_distribution": as_items(payload.get("employment_type_distribution"), limit=20),
        },
    )


@app.post("/refresh-live", response_class=HTMLResponse)
async def refresh_live(request: Request) -> HTMLResponse:
    form = await request.form()
    session = _session_or_404(clean_text(form.get("session_id")))
    provider = clean_text(form.get("provider")) or "adzuna"
    attempt_live = clean_text(form.get("attempt_live")).lower() in {"1", "true", "on", "yes"}
    report = run_live_refresh_preview(session.get("profile", {}), provider=provider, dry_run=not attempt_live)
    session["latest_live_refresh"] = report
    save_session(session)
    return _result_context(session, request, live_refresh=report, status_message="Live refresh check completed.")


@app.get("/download/top-jobs")
def download_top_jobs(session_id: str) -> StreamingResponse:
    session = _session_or_404(session_id)
    output = io.StringIO()
    fieldnames = [
        "rank",
        "job_id",
        "title",
        "company",
        "employer",
        "location",
        "salary_min",
        "salary_max",
        "salary_raw",
        "link",
        "apply_url",
        "description_text",
        "final_score",
        "adjusted_score",
        "feedback_adjustment",
        "matched_skills",
        "why_ranked_summary",
        "application_strategy_label",
        "source",
        "raw_source",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for index, job in enumerate(_active_jobs(session), start=1):
        writer.writerow(
            {
                "rank": job.get("adjusted_rank") or job.get("rank") or index,
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "employer": job.get("employer", ""),
                "location": job.get("location", ""),
                "salary_min": job.get("salary_min", ""),
                "salary_max": job.get("salary_max", ""),
                "salary_raw": job.get("salary_raw", ""),
                "link": job.get("link", ""),
                "apply_url": _job_apply_url(job),
                "description_text": job.get("description_text", ""),
                "final_score": job.get("final_score", ""),
                "adjusted_score": job.get("adjusted_score", ""),
                "feedback_adjustment": job.get("feedback_adjustment", ""),
                "matched_skills": _matched_skills_text(job),
                "why_ranked_summary": _why_summary(job),
                "application_strategy_label": application_strategy_display_label(job.get("application_strategy_label")),
                "source": job.get("source", ""),
                "raw_source": job.get("raw_source", ""),
            }
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="jobpilot_top_jobs_{session_id}.csv"'},
    )
