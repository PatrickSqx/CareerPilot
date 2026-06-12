from __future__ import annotations

import os
import json
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main
import app.services.resume_service as resume_service
from app.services.analytics_service import source_rollup_counts
from app.services.presentation_labels import application_strategy_display_label
from app.services.profile_parse_service import parse_profile_text_deterministic


FAKE_JOB = {
    "job_id": "job-1",
    "title": "Machine Learning Engineer",
    "company": "Example AI",
    "employer": "Example AI",
    "location": "Remote, US",
    "salary_min": "120000",
    "salary_max": "150000",
    "salary_raw": "$120,000-$150,000",
    "link": "https://example.com/job-1",
    "apply_url": "https://example.com/apply/job-1",
    "description_text": "Build machine learning systems with Python and SQL.",
    "matched_skills": ["python", "sql", "machine learning"],
    "final_score": 0.82,
    "rank": 1,
    "seniority": "mid",
    "years_required": "2",
    "employment_type": "full-time",
    "company_type": "large_company",
    "sponsorship_signal": "unknown",
    "score_components": {"embedding": 0.7, "skills": 0.9},
    "why_ranked": {"summary": "Strong skill and role overlap."},
    "application_strategy_label": "Apply Now",
    "source": "test_source",
    "raw_source": "test_raw_source",
}


def test_thread_env_defaults() -> None:
    expected = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in expected.items():
        assert os.environ.get(key) == value


def test_health_and_home() -> None:
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.get("/")
        assert response.status_code == 200
        assert "Run Matching" in response.text
        assert "Profile" in response.text
        assert "Aisha (demo persona)" in response.text
        assert 'data-demo="aisha"' in response.text
        assert 'data-demo="kenji"' in response.text
        assert 'data-demo="marcus"' in response.text
        assert 'data-demo="priya"' in response.text
        assert "Demo user" not in response.text
        assert "Current backend" not in response.text


def test_match_feedback_rerank_resume_and_csv(monkeypatch) -> None:
    monkeypatch.setattr(resume_service, "LLM_RESUME_CLIENT", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        job = dict(FAKE_JOB)
        if session_feedback_events:
            job["final_score"] = 0.9
            job["feedback_adjustment_explanation"] = "same job accepted earlier in this session"
        return {"top_jobs": [job], "metadata": {"embedding_backend": "test-backend"}}

    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    with TestClient(main.app) as client:
        session_dir = main.APP_DIR / "storage" / "sessions"
        before_sessions = set(session_dir.glob("*.json"))
        response = client.post(
            "/match",
            data={
                "persona": "manual",
                "manual_name": "Test User",
                "manual_target_roles": "machine learning engineer",
                "manual_skills": "python, sql, machine learning",
                "top_k": "1",
                "candidate_k": "50",
            },
        )
        assert response.status_code == 200
        assert "Machine Learning Engineer" in response.text
        assert "$120,000-$150,000" in response.text
        assert "Generate Resume" in response.text
        assert "Use the Gemini key from the profile reader to enable resume generation." in response.text
        assert "test-backend" not in response.text

        debug_response = client.post(
            "/match?debug_audit=1",
            data={
                "persona": "manual",
                "manual_name": "Test User",
                "manual_target_roles": "machine learning engineer",
                "manual_skills": "python, sql, machine learning",
                "top_k": "1",
                "candidate_k": "50",
            },
        )
        assert debug_response.status_code == 200
        assert "test-backend" in debug_response.text

        created_sessions = set(session_dir.glob("*.json")) - before_sessions
        assert created_sessions
        session_id = sorted(created_sessions, key=lambda path: path.stat().st_mtime)[-1].stem

        feedback = client.post("/feedback", data={"session_id": session_id, "job_id": "job-1", "action": "accept"})
        assert feedback.status_code == 200
        assert "Recorded accept feedback" in feedback.text

        invalid_feedback = client.post("/feedback", data={"session_id": session_id, "job_id": "job-1", "action": "maybe"})
        assert invalid_feedback.status_code == 400

        rerank = client.post("/rerank", data={"session_id": session_id})
        assert rerank.status_code == 200
        assert "Results refreshed from" in rerank.text
        assert "same job accepted earlier in this session" in rerank.text

        live_refresh = client.post("/refresh-live", data={"session_id": session_id, "provider": "jsearch"})
        assert live_refresh.status_code == 200
        assert "Job source preview completed" not in live_refresh.text
        assert "JSEARCH" not in live_refresh.text
        assert "dry_run" not in live_refresh.text
        assert "Queries:" not in live_refresh.text

        debug_refresh = client.post(f"/refresh-live?debug_audit=1", data={"session_id": session_id, "provider": "jsearch"})
        assert debug_refresh.status_code == 200
        assert "JSEARCH" in debug_refresh.text
        assert "dry_run" in debug_refresh.text
        assert "Queries:" in debug_refresh.text

        resume = client.post("/resume", data={"session_id": session_id, "job_id": "job-1"})
        assert resume.status_code == 200
        assert "Resume generation requires a connected LLM API" in resume.text
        assert "API Resume" not in resume.text

        csv_response = client.get(f"/download/top-jobs?session_id={session_id}")
        assert csv_response.status_code == 200
        assert "rank,job_id,title,company,employer,location" in csv_response.text
        assert "final_score,adjusted_score,feedback_adjustment" in csv_response.text
        assert "application_strategy_label,source,raw_source" in csv_response.text
        assert "Recommended" in csv_response.text
        assert "Apply Now" not in csv_response.text
        assert "job-1" in csv_response.text


def test_application_strategy_labels_are_user_safe() -> None:
    assert application_strategy_display_label("Apply Now") == "Recommended"
    assert application_strategy_display_label("Same-company alternative") == "Additional posting"
    assert application_strategy_display_label("Potential duplicate role") == "Similar posting"
    assert application_strategy_display_label("Custom label") == "Custom label"


def test_watch_out_display_items_are_user_safe_and_deduped() -> None:
    job = {
        "why_ranked": {
            "negative_drivers": [
                "Salary preference cannot be verified because salary is missing",
                "salary_missing",
                "Sponsorship is unknown",
                "sponsorship_unknown",
                "Application strategy: possible near-duplicate role",
            ]
        }
    }

    assert main._driver_items(job, "negative_drivers") == [
        "Salary preference cannot be verified because salary is missing",
        "Sponsorship not stated in the posting",
    ]


def test_resume_generation_requires_connected_api(monkeypatch) -> None:
    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        return {"top_jobs": [dict(FAKE_JOB)], "metadata": {"embedding_backend": "test"}}

    def fake_resume_client(prompt: str) -> str:
        assert "Machine Learning Engineer" in prompt
        assert "Example AI" in prompt
        assert "Return strict JSON content" in prompt
        assert "Required JSON schema" in prompt
        assert "Fixed template handled by backend" in prompt
        assert "User resume/profile evidence JSON" in prompt
        assert "Selected job JSON" in prompt
        assert "primary source is the user's own experience" in prompt
        assert "omit that bullet or section instead of using a placeholder" in prompt
        assert "Do not include bracketed placeholder text" in prompt
        assert '"education"' in prompt
        assert '"experience"' in prompt
        assert '"project_or_research_experience"' in prompt
        assert "Location | Target" not in prompt
        assert "Do not put the selected job title/company in the resume header" in prompt
        assert "dealbreakers" in prompt
        assert "description_text" in prompt
        assert '"application_strategy_label": "Recommended"' in prompt
        assert "Apply Now" not in prompt
        assert "Return only one JSON object" in prompt
        assert "## Professional Summary" not in prompt
        return json.dumps(
            {
                "name": "Test User",
                "contact": {"email": "model@example.com"},
                "education": [
                    {
                        "institution": "UC Davis",
                        "dates": "Aug 2025 - Sep 2026",
                        "detail": "M.S. in Business Analytics",
                    }
                ],
                "experience": [
                    {
                        "organization": "Example Project",
                        "role": "Machine Learning Engineer",
                        "location": "Remote, US",
                        "dates": "Jan 2025 - Present",
                        "bullets": ["Built machine learning systems with Python and SQL."],
                    }
                ],
                "skills": {"Data & Tools": "Python, SQL, machine learning"},
            }
        )

    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    monkeypatch.setattr(resume_service, "LLM_RESUME_CLIENT", fake_resume_client)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with TestClient(main.app) as client:
        session_dir = main.APP_DIR / "storage" / "sessions"
        before_sessions = set(session_dir.glob("*.json"))
        response = client.post(
            "/match",
            data={
                "persona": "manual",
                "manual_name": "Test User",
                "manual_target_roles": "machine learning engineer",
                "manual_skills": "python, sql, machine learning",
                "top_k": "1",
                "candidate_k": "50",
            },
        )
        assert response.status_code == 200
        assert "Generate Resume" in response.text

        created_sessions = set(session_dir.glob("*.json")) - before_sessions
        assert created_sessions
        session_id = sorted(created_sessions, key=lambda path: path.stat().st_mtime)[-1].stem
        resume = client.post("/resume", data={"session_id": session_id, "job_id": "job-1"})

    assert resume.status_code == 200
    assert resume.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert 'filename="jobpilot-machine-learning-engineer-example-ai.docx"' in resume.headers["content-disposition"]
    assert resume.content.startswith(b"PK")
    assert b"word/document.xml" in resume.content
    assert b"word/_rels/document.xml.rels" in resume.content
    with zipfile.ZipFile(BytesIO(resume.content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "UC Davis" in document_xml
    assert "Aug 2025 - Sep 2026" in document_xml
    assert "Example Project | Machine Learning Engineer" in document_xml
    assert "Remote, US | Jan 2025 - Present" in document_xml
    assert "model@example.com" not in document_xml


def test_profile_parser_extracts_resume_contact_fields() -> None:
    parsed = parse_profile_text_deterministic(
        """
        Qixiang Sun
        Phone Number: (650)-664-8580 || Email: patricksun0801@gmail.com || LinkedIn: www.linkedin.com/in/qixiangsun

        EXPERIENCE
        Built Python data workflows and analytics dashboards.
        """
    )

    assert parsed.profile["name"] == "Qixiang Sun"
    assert parsed.profile["phone"] == "(650)-664-8580"
    assert parsed.profile["email"] == "patricksun0801@gmail.com"
    assert parsed.profile["linkedin"] == "www.linkedin.com/in/qixiangsun"
    assert "Phone Number" in parsed.profile["resume_source_text"]


def test_parse_profile_upload_preserves_docx_runtime_path() -> None:
    docx_bytes = resume_service.markdown_resume_to_docx_bytes(
        "# Qixiang Sun\n"
        "(650)-664-8580 | patricksun0801@gmail.com\n\n"
        "## EXPERIENCE\n"
        "Angel Flight West | Data Analytics Intern | Santa Monica, CA | Aug 2025 - Present\n"
        "- Built Python data workflows.\n"
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/parse-profile",
            files={
                "resume_pdf": (
                    "qixiang-test.docx",
                    BytesIO(docx_bytes),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 200
    profile = response.json()["profile"]
    assert profile["resume_source_docx_path"].startswith("app/storage/uploads/")
    assert profile["resume_source_docx_path"].endswith(".docx")


def test_resume_prompt_uses_contact_and_demo_only_defaults() -> None:
    real_profile = {
        "profile_id": "qixiang_sun",
        "name": "Qixiang Sun",
        "phone": "(650)-664-8580",
        "email": "patricksun0801@gmail.com",
        "linkedin": "www.linkedin.com/in/qixiangsun",
        "education": "University of California, Davis - M.S. in Business Analytics",
        "skills": ["Python", "SQL"],
        "experience_text": "Built analytics pipelines.",
        "resume_source_text": "Qixiang Sun\nEXPERIENCE\nBuilt analytics pipelines.",
    }
    real_prompt = resume_service._resume_prompt(real_profile, FAKE_JOB)

    assert '"email": "patricksun0801@gmail.com"' in real_prompt
    assert '"phone": "(650)-664-8580"' in real_prompt
    assert '"linkedin": "www.linkedin.com/in/qixiangsun"' in real_prompt
    assert '"resume_source_text": "Qixiang Sun\\nEXPERIENCE\\nBuilt analytics pipelines."' in real_prompt
    assert "Target: Machine Learning Engineer" not in real_prompt

    demo_prompt = resume_service._resume_prompt({"profile_id": "aisha", "name": "Aisha", "email": ""}, FAKE_JOB)
    assert '"email": "aisha@gmail.com"' in demo_prompt
    assert '"phone": "(555) 010-0101"' in demo_prompt
    assert '"linkedin": "linkedin.com/in/aisha-demo"' in demo_prompt
    assert "UC Davis - M.S. in Business Analytics" in demo_prompt


def test_resume_docx_entry_lines_use_right_aligned_date_tabs() -> None:
    docx_bytes = resume_service.markdown_resume_to_docx_bytes(
        "# Qixiang Sun\n"
        "(650)-664-8580 | patricksun0801@gmail.com | www.linkedin.com/in/qixiangsun\n\n"
        "## EDUCATION\n"
        "University of California, Davis | M.S. in Business Analytics | Aug 2025 - Sep 2026\n\n"
        "## EXPERIENCE\n"
        "Angel Flight West | Data Analytics Intern | Santa Monica, CA | Aug 2025 - Present\n"
        "- Built Python data workflows.\n"
    )

    with zipfile.ZipFile(BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        styles_xml = archive.read("word/styles.xml").decode("utf-8")

    assert 'w:val="right" w:pos="11469"' in document_xml
    assert "<w:tab/>" in document_xml
    assert '<w:spacing w:afterLines="30" w:after="75" w:line="240" w:lineRule="auto"/>' in styles_xml
    assert '<w:style w:type="paragraph" w:styleId="EntryLine">' in styles_xml
    assert '<w:sz w:val="23"/>' in styles_xml
    assert "University of California, Davis | M.S. in Business Analytics" in document_xml
    assert "Aug 2025 - Sep 2026" in document_xml
    assert "Angel Flight West | Data Analytics Intern" in document_xml
    assert "Santa Monica, CA | Aug 2025 - Present" in document_xml


def test_resume_markdown_sanitizer_flattens_education_bullets() -> None:
    sanitized = resume_service._sanitize_resume_markdown(
        "# Qixiang Sun\n"
        "(650)-664-8580 | patricksun0801@gmail.com\n\n"
        "## EDUCATION\n"
        "University of California, Davis | Aug 2025 - Sep 2026\n"
        "- M.S. in Business Analytics, GPA: 3.86/4.0\n\n"
        "## EXPERIENCE\n"
        "Angel Flight West | Data Analytics Intern | Santa Monica, CA | Aug 2025 - Present\n"
        "- Built Python data workflows.\n"
    )

    assert "## EDUCATION\nUniversity of California, Davis | Aug 2025 - Sep 2026\nM.S. in Business Analytics" in sanitized
    assert "## EXPERIENCE\nAngel Flight West | Data Analytics Intern | Santa Monica, CA | Aug 2025 - Present\n- Built Python" in sanitized


def test_demo_persona_resume_expands_sparse_llm_output() -> None:
    sparse = {
        "name": "Aisha",
        "education": [{"institution": "UC Davis", "dates": "2024 - 2025", "detail": "M.S. in Business Analytics"}],
        "experience": [
            {
                "organization": "Data Scientist",
                "role": "",
                "location": "",
                "dates": "",
                "bullets": ["Built Python data workflows."],
            }
        ],
        "project_or_research_experience": [],
        "skills": {"Data & Tools": "Python, SQL"},
    }

    expanded = resume_service._expand_demo_structured_resume(
        sparse,
        {"profile_id": "aisha", "name": "Aisha"},
        FAKE_JOB,
    )
    markdown = resume_service.structured_resume_to_markdown(expanded)

    assert markdown.count("\n- ") >= 9
    assert "Applied Analytics Lab | Data Science Project Lead" in markdown
    assert "Analytics Practicum Team | Machine Learning Analyst" in markdown
    assert "Customer Churn Model | Data Scientist" in markdown
    assert "NLP Ticket Classifier | Machine Learning Project" in markdown
    assert "Machine Learning Engineer requirements" in markdown
    assert "python, sql, machine learning" in markdown.lower()
    assert "UC Davis | 2024 - 2025" not in markdown


def test_resume_preserve_source_docx_replaces_bullets_without_rebuilding_entries() -> None:
    source_docx = resume_service.markdown_resume_to_docx_bytes(
        "# Qixiang Sun\n"
        "(650)-664-8580 | patricksun0801@gmail.com\n\n"
        "## EXPERIENCE\n"
        "Angel Flight West | Data Analytics Intern | Santa Monica, CA | Aug 2025 - Present\n"
        "- Original pipeline bullet.\n"
        "- Original dashboard bullet.\n\n"
        "## SKILLS & TOOLS\n"
        "Data & Tools: SQL, Python\n"
    )
    structured = {
        "name": "Qixiang Sun",
        "contact": {"phone": "(650)-664-8580", "email": "patricksun0801@gmail.com"},
        "experience": [
            {
                "organization": "Angel Flight West",
                "role": "Data Analytics Intern",
                "location": "Santa Monica, CA",
                "dates": "Aug 2025 - Present",
                "bullets": ["Replacement pipeline bullet.", "Replacement dashboard bullet."],
            }
        ],
        "skills": {"Data & Tools": "SQL, Python, Tableau"},
    }

    output = resume_service.preserve_source_docx_layout(source_docx, structured)
    with zipfile.ZipFile(BytesIO(output)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "Angel Flight West | Data Analytics Intern" in document_xml
    assert "Santa Monica, CA | Aug 2025 - Present" in document_xml
    assert "Replacement pipeline bullet." in document_xml
    assert "Replacement dashboard bullet." in document_xml
    assert "Original pipeline bullet." not in document_xml
    assert "Original dashboard bullet." not in document_xml
    assert 'w:val="right" w:pos="11469"' in document_xml


def test_manual_profile_supports_advanced_constraints(monkeypatch) -> None:
    captured: dict[str, dict] = {}

    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        captured["profile"] = profile
        return {"top_jobs": [dict(FAKE_JOB)], "metadata": {"embedding_backend": "test"}}

    def fake_save_session(payload):
        captured["session"] = payload

    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    monkeypatch.setattr(main, "save_session", fake_save_session)

    with TestClient(main.app) as client:
        response = client.post(
            "/match?debug_audit=1",
            data={
                "persona": "manual",
                "manual_name": "Aisha Style Hidden Profile",
                "manual_target_roles": "machine learning engineer, data scientist",
                "manual_skills": "python, sql, machine learning",
                "manual_location_preferences": "remote, Bay Area",
                "manual_excluded_seniority": "senior, staff_principal, lead_manager",
                "manual_max_years_required": "4",
                "manual_required_role_families": "ml_related, research_ai",
                "manual_preferred_role_families": "ml_related, research_ai",
                "manual_excluded_company_types": "defense_military",
                "manual_strict_role_family": "on",
                "manual_avoid_defense_or_clearance": "on",
                "top_k": "1",
                "candidate_k": "50",
            },
        )

    assert response.status_code == 200
    profile = captured["profile"]
    assert profile["excluded_seniority"] == ["senior", "staff_principal", "lead_manager"]
    assert profile["max_years_required"] == 4
    assert profile["required_role_families"] == ["ml_related", "research_ai"]
    assert profile["strict_role_family"] is True
    assert profile["avoid_defense_or_clearance"] is True
    assert captured["session"]["profile"] == profile
    assert "Normalized profile JSON" in response.text
    assert "excluded_seniority" in response.text
    assert "strict_role_family" in response.text
    assert "avoid_defense_or_clearance" in response.text


def test_manual_hidden_persona_constraints_are_name_agnostic(monkeypatch) -> None:
    captured: dict[str, dict] = {}

    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        captured["profile"] = profile
        return {"top_jobs": [dict(FAKE_JOB)], "metadata": {"embedding_backend": "test"}}

    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    monkeypatch.setattr(main, "save_session", lambda payload: captured.setdefault("session", payload))

    with TestClient(main.app) as client:
        response = client.post(
            "/match",
            data={
                "persona": "manual",
                "manual_name": "Hidden Candidate",
                "manual_target_roles": "ai infrastructure engineer",
                "manual_skills": "python, kubernetes, spark",
                "manual_location_preferences": "United States, remote",
                "manual_excluded_employment_types": "contract, temporary, unpaid",
                "manual_preferred_company_types": "large_company, research_lab",
                "manual_excluded_company_types": "startup",
                "manual_hard_reject_seniority_terms": "staff, principal, director, lead",
                "manual_penalize_seniority_terms": "senior, sr, iii",
                "manual_salary_is_dealbreaker": "on",
                "manual_strict_location": "on",
                "manual_us_only": "on",
                "top_k": "1",
                "candidate_k": "50",
            },
        )

    assert response.status_code == 200
    profile = captured["profile"]
    assert profile["profile_id"] == "hidden_candidate"
    assert profile["name"] == "Hidden Candidate"
    assert profile["excluded_employment_types"] == ["contract", "temporary", "unpaid"]
    assert profile["preferred_company_types"] == ["large_company", "research_lab"]
    assert profile["excluded_company_types"] == ["startup"]
    assert profile["hard_reject_seniority_terms"] == ["staff", "principal", "director", "lead"]
    assert profile["penalize_seniority_terms"] == ["senior", "sr", "iii"]
    assert profile["salary_is_dealbreaker"] is True
    assert profile["strict_location"] is True
    assert profile["us_only"] is True
    assert profile["profile_id"] not in {"aisha", "marcus", "priya", "kenji"}


def test_analytics_page() -> None:
    with TestClient(main.app) as client:
        response = client.get("/analytics")
        assert response.status_code == 200
        assert "Full Snapshot Market Dashboard" in response.text
        assert "Target Role Lens" in response.text
        assert "Tech/Data Focus Lens" in response.text
        assert "Derived from full snapshot" in response.text
        assert "Tech/data focus postings" in response.text
        assert "Data analytics / BI" in response.text
        assert "Data engineering" in response.text
        assert "ML / AI" in response.text
        assert "Software / cloud tech" in response.text
        assert "Total rows" in response.text
        assert "Source Split" in response.text
        assert "Provider Source Counts" in response.text
        assert "Salary Summary" in response.text
        assert "Mean listed salary" in response.text
        assert "Screened observations" in response.text
        assert "$4" not in response.text
        assert "$239,200,000" not in response.text
        assert "Demand by Location" in response.text
        assert "Focus Title Frequency" in response.text
        assert "Overall Title Frequency" not in response.text
        assert "Full Snapshot Title Context" in response.text
        assert "All industries in the 50,000-row snapshot" in response.text
        assert "Remote Distribution" in response.text
        assert "Employment Type" in response.text
        assert "Dataset Caveats" not in response.text
        assert "chart-list" in response.text
        assert "chart-fill" in response.text
        assert "role-donut" in response.text
        assert "Salary Range by Segment" in response.text
        assert "range-band" in response.text
        assert "P25-P75 range" in response.text
        assert "Median" in response.text
        assert "Hybrid" in response.text
        assert "Onsite" in response.text
        assert "Kaggle/offline snapshot" in response.text
        assert "Saved current API rows" in response.text
        assert "remote" in response.text
        assert "fulltime" in response.text
        assert "SQL" in response.text
        assert "Python" in response.text
        assert "Data Scientist" in response.text
        assert response.text.index("Top Skills") < response.text.index("Tech/Data Focus Lens")


def test_analytics_source_rollup_counts() -> None:
    result = source_rollup_counts(
        {"careerbuilder_us": 49_650, "adzuna": 350},
        row_count=50_000,
    )
    assert result == {
        "Kaggle/offline snapshot": 49_650,
        "Saved current API rows": 350,
    }


def test_dockerfile_does_not_require_phase2_handoff() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "PHASE2_HANDOFF.md" not in dockerfile


def test_public_deployment_context_excludes_large_embedding_cache() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "COPY data/processed data/processed" in dockerfile
    assert "PYTHONPATH=/app/code" in dockerfile
    assert "data/processed/embeddings/" in dockerignore
    assert "data/processed/jobs_offline_snapshot.csv" in dockerignore
