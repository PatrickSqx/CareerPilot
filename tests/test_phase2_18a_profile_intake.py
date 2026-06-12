from __future__ import annotations

import asyncio
import io
import json
import re
import urllib.error
import zipfile
from xml.sax.saxutils import escape

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.services.profile_parse_service as parser_service
from jobpilot.profile.profile_parser import profile_to_text
from app.services.profile_parse_service import parse_profile_intake, parse_profile_text_deterministic


FAKE_JOB = {
    "job_id": "job-1",
    "title": "Data Analyst",
    "company": "Example Co",
    "employer": "Example Co",
    "location": "Remote, US",
    "salary_min": "90000",
    "salary_max": "130000",
    "salary_raw": "$90,000-$130,000",
    "link": "https://example.com/job-1",
    "description_text": "Analyze business data with SQL.",
    "final_score": 0.82,
    "rank": 1,
    "score_components": {},
    "matched_skills": ["SQL"],
    "why_ranked": {"summary": "Test job."},
}


AISHA_PERSONA = """Name: Aisha
Background:
Career pivoter with analytics coursework and applied machine learning projects.

Skills:
Python, SQL, pandas, scikit-learn, machine learning, analytics

Target Roles:
Machine Learning Engineer, ML Engineer, Applied Scientist, Data Scientist

Preferences:
Remote or Bay Area roles. Minimum salary $140k. ML-related or research AI roles only.

Dealbreakers:
No defense or clearance. No senior/staff/principal roles.

Pass Criteria:
Only ML-related or research AI roles. No roles requiring more than 4 years.
"""


MARCUS_PERSONA = """Name: Marcus
Background:
MSBA new graduate with internship analytics experience.

Skills:
SQL, Excel, Tableau, Power BI, Python, statistics, analytics

Target Roles:
Data Analyst, Business Analyst, BI Analyst, Junior Data Scientist, Analytics Engineer

Preferences:
US only. Remote, Chicago, New York, or Boston. Minimum salary $80k.

Dealbreakers:
Full-time only. No contract, temporary, or unpaid roles. No senior/staff/lead roles.

Pass Criteria:
Analytics entry or BI analytics roles preferred.
"""


PRIYA_PERSONA = """Name: Priya
Background:
Senior software engineer with distributed systems, cloud infrastructure, Spark, Kafka, and ML platform experience.

Skills:
Java, Python, Spark, Kafka, Kubernetes, Docker, AWS, microservices, machine learning

Target Roles:
Senior ML Engineer, ML Infrastructure Engineer, Machine Learning Platform Engineer, AI Infrastructure Engineer

Preferences:
US only. NYC, New York, or remote. Salary minimum $200k. Companies with >=100 employees only. Prefer research labs.

Dealbreakers:
No Junior titles. No entry-level, intern, or new-grad titles. No companies with <100 employees. No tiny startups.

Pass Criteria:
ML infrastructure or ML platform roles only.
"""


KENJI_PERSONA = """Name: Kenji
Background:
International CS graduate student on OPT with machine learning research experience.

Skills:
Python, PyTorch, TensorFlow, machine learning, deep learning, computer vision, NLP

Target Roles:
Research Scientist, Applied Scientist, Machine Learning Engineer, Data Scientist

Preferences:
US only. Remote, Seattle, San Francisco, or New York. Minimum salary $120k.
Prefer large companies or research labs.

Dealbreakers:
No contract/temp/unpaid roles. No senior/staff/principal/lead roles. No 3+ years.

Pass Criteria:
Needs H-1B sponsorship after OPT. 0\u20132 years. ML-related or research AI roles only.
"""


ASSIGNMENT_STYLE_LLM_DEMO = """Background:
International MSCS candidate with applied machine learning and research systems projects.

Skills:
Python, PyTorch, SQL, machine learning, deep learning, NLP, research prototyping

Target Roles:
Machine Learning Engineer, Applied Scientist, Research Scientist

Preferences:
US only. Remote, Seattle, San Francisco, or New York. Needs H-1B / visa sponsorship. Salary minimum $120000. Prefer large companies or research labs.

Dealbreakers:
No contract/temp roles. No senior/staff/principal roles. Avoid defense or clearance.

Pass Criteria:
No 3+ years required. ML-related or research AI roles only.
"""


def _docx_bytes(text: str) -> bytes:
    paragraphs = "\n".join(
        f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>"
        for line in text.splitlines()
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}</w:body>"
        "</w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def disable_real_llm_env(monkeypatch):
    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", None)
    monkeypatch.delenv("JOBPILOT_PROFILE_LLM_ENABLED", raising=False)
    monkeypatch.delenv("JOBPILOT_ENABLE_LLM_PROFILE_PARSE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JOBPILOT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("JOBPILOT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("JOBPILOT_PROFILE_LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_MODEL", raising=False)
    monkeypatch.delenv("JOBPILOT_LLM_MODEL", raising=False)


def test_aisha_persona_parser_expected_filters() -> None:
    profile = parse_profile_text_deterministic(AISHA_PERSONA).profile

    assert profile["salary_min"] == 140000
    assert profile["max_years_required"] == 4
    assert profile["excluded_seniority"] == ["senior", "staff_principal", "lead_manager"]
    assert profile["required_role_families"] == ["ml_related", "research_ai"]
    assert profile["strict_role_family"] is True
    assert profile["avoid_defense_or_clearance"] is True
    assert "Machine Learning Engineer" in profile["target_roles"]
    assert "Python" in profile["skills"]


def test_marcus_persona_parser_expected_filters() -> None:
    profile = parse_profile_text_deterministic(MARCUS_PERSONA).profile

    assert profile["us_only"] is True
    assert profile["strict_location"] is True
    assert profile["salary_min"] == 80000
    assert profile["max_years_required"] == 2
    assert profile["excluded_employment_types"] == ["contract", "temporary", "unpaid"]
    assert profile["excluded_seniority"] == ["senior", "staff_principal", "lead_manager"]
    assert set(profile["preferred_role_families"]) >= {"analytics_entry", "bi_analytics"}
    assert "Data Analyst" in profile["target_roles"]


def test_salary_parser_handles_uncommaed_five_digit_threshold() -> None:
    profile = parse_profile_text_deterministic(
        "Skills: SQL\nTarget Roles: Data Analyst\nPreferences: US only. Salary minimum $90000."
    ).profile

    assert profile["salary_min"] == 90000


def test_priya_persona_parser_does_not_self_exclude_senior_and_reads_company_size() -> None:
    parsed = parse_profile_text_deterministic(PRIYA_PERSONA)
    profile = parsed.profile

    assert profile["max_years_required"] is None
    assert "senior" not in [item.lower() for item in profile["excluded_seniority"]]
    assert "staff_principal" not in profile["excluded_seniority"]
    assert set(profile["excluded_seniority"]) == {"entry_junior", "internship"}
    assert {"junior", "entry level", "new grad", "intern"} <= set(profile["hard_reject_seniority_terms"])
    assert "Senior ML Engineer" in profile["target_roles"]
    assert profile["us_only"] is True
    assert {"New York", "NYC", "Remote", "United States"} <= set(profile["location_preferences"])
    assert profile["salary_min"] == 200000
    assert "large_company" in profile["preferred_company_types"]
    assert "research_lab" in profile["preferred_company_types"]
    assert "startup" in profile["excluded_company_types"]
    assert parsed.form_fields["manual_excluded_seniority"] == "entry_junior, internship"


def test_kenji_persona_parser_expected_filters_and_en_dash_year_range() -> None:
    profile = parse_profile_text_deterministic(KENJI_PERSONA).profile

    assert profile["needs_sponsorship"] is True
    assert "H-1B" in profile["visa_sponsorship"] or "sponsorship" in profile["visa_sponsorship"].lower()
    assert profile["us_only"] is True
    assert profile["excluded_employment_types"] == ["contract", "temporary", "unpaid"]
    assert profile["max_years_required"] == 2
    assert profile["excluded_seniority"] == ["senior", "staff_principal", "lead_manager"]
    assert profile["hard_reject_seniority_terms"] == ["staff", "principal", "director", "lead", "manager"]
    assert profile["preferred_company_types"] == ["large_company", "research_lab"]
    assert profile["required_role_families"] == ["ml_related", "research_ai"]


def test_parse_profile_endpoint_returns_parse_method_and_form_fields() -> None:
    with TestClient(main.app) as client:
        response = client.post("/parse-profile", data={"profile_text": KENJI_PERSONA})

    assert response.status_code == 200
    payload = response.json()
    assert payload["parse_method"] == "rule_fallback"
    assert payload["profile"]["needs_sponsorship"] is True
    assert payload["profile"]["max_years_required"] == 2
    assert payload["form_fields"]["manual_needs_sponsorship"] is True
    assert payload["form_fields"]["manual_excluded_employment_types"] == "contract, temporary, unpaid"
    assert payload["filter_fields"]["manual_needs_sponsorship"] is True
    assert "manual_visa_sponsorship" not in payload["filter_fields"]
    assert payload["context_fields"]["manual_visa_sponsorship"]


def test_profile_text_match_path_uses_canonical_profile_without_visible_candidate_k(monkeypatch) -> None:
    captured: dict[str, dict] = {}

    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        captured["profile"] = profile
        captured["rank_args"] = {"top_k": top_k, "candidate_k": candidate_k}
        return {"top_jobs": [dict(FAKE_JOB)], "metadata": {"embedding_backend": "test"}}

    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    monkeypatch.setattr(main, "save_session", lambda payload: captured.setdefault("session", payload))

    with TestClient(main.app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert 'type="hidden" name="candidate_k" value="200"' in home.text
        assert 'name="profile_filters_ready"' in home.text
        assert "Candidate K" not in home.text
        assert "Advanced filters" in home.text
        assert "Advanced details" not in home.text
        assert "Review summary" in home.text
        assert "Selected filters" in home.text
        assert "Parsed profile" in home.text
        assert "No filters applied." in home.text
        assert "No parsed profile yet." in home.text
        assert "Active Filters" not in home.text
        assert "Expert/debug details" not in home.text
        assert "Seniority hard rejects" not in home.text
        assert "Hard reject terms" not in home.text
        assert "Edit parsed profile details" in home.text
        assert '<details class="parsed-detail-panel">' in home.text
        assert '<details class="parsed-detail-panel" open' not in home.text
        assert "Profile context is used to" not in home.text
        assert "Empty</span>" not in home.text
        assert 'id="location_custom_text" class="location-custom-input hidden"' in home.text
        assert 'id="location_add_button"' in home.text
        assert re.search(r'name="manual_salary_min"[^>]+data-summary-field', home.text)
        assert 'placeholder="No minimum"' in home.text
        assert 'placeholder="120000"' not in home.text
        assert "Visa sponsorship</label>" not in home.text
        assert "Visa/work authorization detail" in home.text
        assert "More profile details" in home.text
        assert '<details class="context-more-panel" open' not in home.text
        assert 'id="parser_api_key"' in home.text
        assert 'name="parser_api_key"' not in home.text
        assert "Profile reader" in home.text
        assert "Interface" in home.text
        assert "Local profile reader (no API key)" in home.text
        assert "Rule fallback / Offline" not in home.text
        assert "Parser settings" not in home.text
        assert "Configured default" in home.text
        assert "Company</span>" in home.text
        assert "Company size" in home.text
        assert "Any company size" in home.text
        assert "Large companies" in home.text
        assert "Medium companies" in home.text
        assert "Small companies" in home.text
        assert "Research-focused" not in home.text
        assert "Startup-friendly" not in home.text
        assert "Avoid startups" not in home.text
        assert "Avoid defense / clearance" not in home.text
        assert 'id="company_research_focus"' not in home.text
        assert 'id="company_startup_ok"' not in home.text
        assert 'id="company_avoid_startups"' not in home.text
        assert 'id="company_avoid_defense"' not in home.text
        assert 'id="company_research_option"' not in home.text
        assert 'id="company_startup_exclusion"' not in home.text
        assert 'id="company_defense_exclusion"' not in home.text
        assert "Established / sponsorship-friendly employers" not in home.text
        assert "Established, avoid startups and defense" not in home.text
        assert "Prefer large companies / research labs" not in home.text
        assert "Large / research, no startups" not in home.text
        assert "Large / research, no startups or defense" not in home.text
        for token in (
            "ml_ai",
            "ml_infra",
            "ml_related",
            "research_ai",
            "research_lab",
            "staff_principal",
            "defense_military",
            "large_company",
            "medium_company",
            "small_company",
        ):
            assert token not in home.text

        response = client.post(
            "/match",
            data={
                "persona": "manual",
                "profile_text": KENJI_PERSONA,
                "top_k": "1",
                "candidate_k": "200",
            },
        )

    assert response.status_code == 200
    assert captured["rank_args"] == {"top_k": 1, "candidate_k": 200}
    assert captured["profile"]["needs_sponsorship"] is True
    assert captured["profile"]["max_years_required"] == 2
    assert captured["session"]["profile"] == captured["profile"]


def test_manual_filters_ready_clears_profile_text_inferred_values(monkeypatch) -> None:
    captured: dict[str, dict] = {}

    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        captured["profile"] = profile
        captured["rank_args"] = {"top_k": top_k, "candidate_k": candidate_k}
        return {"top_jobs": [dict(FAKE_JOB)], "metadata": {"embedding_backend": "test"}}

    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    monkeypatch.setattr(main, "save_session", lambda payload: captured.setdefault("session", payload))

    with TestClient(main.app) as client:
        response = client.post(
            "/match",
            data={
                "persona": "manual",
                "profile_filters_ready": "1",
                "profile_text": KENJI_PERSONA,
                "manual_name": "Edited Candidate",
                "manual_target_roles": "Data Analyst",
                "manual_skills": "SQL",
                "manual_location_preferences": "",
                "manual_salary_min": "",
                "manual_dealbreakers": "",
                "manual_education": "",
                "manual_experience": "",
                "manual_projects": "",
                "manual_visa_sponsorship": "",
                "manual_excluded_seniority": "",
                "manual_max_years_required": "",
                "manual_excluded_employment_types": "",
                "manual_required_role_families": "",
                "manual_preferred_role_families": "",
                "manual_preferred_company_types": "",
                "manual_excluded_company_types": "",
                "manual_hard_reject_seniority_terms": "",
                "manual_penalize_seniority_terms": "",
                "top_k": "1",
                "candidate_k": "200",
            },
        )

    assert response.status_code == 200
    profile = captured["profile"]
    assert profile["target_roles"] == ["Data Analyst"]
    assert profile["skills"] == ["SQL"]
    assert profile["needs_sponsorship"] is False
    assert profile["us_only"] is False
    assert profile["strict_location"] is False
    assert profile["location_preferences"] == []
    assert profile["salary_min"] is None
    assert profile["excluded_employment_types"] == []
    assert profile["excluded_seniority"] == []
    assert profile["max_years_required"] is None
    assert captured["rank_args"] == {"top_k": 1, "candidate_k": 200}


def test_llm_parser_no_key_default_path_is_rule_fallback(monkeypatch) -> None:
    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", None)
    monkeypatch.delenv("JOBPILOT_PROFILE_LLM_ENABLED", raising=False)
    monkeypatch.delenv("JOBPILOT_ENABLE_LLM_PROFILE_PARSE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JOBPILOT_OPENAI_API_KEY", raising=False)

    parsed = asyncio.run(parse_profile_intake(profile_text=KENJI_PERSONA))

    assert parsed.parse_method == "rule_fallback"
    assert parsed.profile["needs_sponsorship"] is True
    assert parsed.profile["max_years_required"] == 2


def test_llm_provider_mode_with_no_key_falls_back_cleanly() -> None:
    parsed = asyncio.run(
        parse_profile_intake(
            profile_text=KENJI_PERSONA,
            parser_mode="llm",
            parser_provider="gemini",
            parser_model="gemini-2.5-flash",
        )
    )

    assert parsed.parse_method == "llm_failed_rule_fallback"
    assert parsed.profile["needs_sponsorship"] is True
    assert parsed.profile["max_years_required"] == 2
    assert any("provider=gemini" in note for note in parsed.notes)


def test_llm_parser_fake_client_is_validated_and_normalized(monkeypatch) -> None:
    def fake_client(_: str) -> dict:
        return {
            "profile": {
                "name": "LLM Candidate",
                "target_roles": "Data Engineer, Analytics Engineer",
                "skills": ["SQL", "Python"],
                "education": "MS analytics.",
                "experience_text": "Built dashboards.",
                "visa_sponsorship": "Does not need sponsorship.",
                "location_preferences": "United States, Remote",
                "salary_min": "$95k",
                "needs_sponsorship": "false",
                "us_only": "true",
                "unknown_field": "must be ignored",
            }
        }

    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", fake_client)

    parsed = asyncio.run(
        parse_profile_intake(
            profile_text="Background: Analyst.\nSkills: SQL\nTarget Roles: Data Engineer",
            parser_mode="llm",
            parser_provider="gemini",
        )
    )

    assert parsed.parse_method == "llm_gemini"
    assert parsed.profile["name"] == "LLM Candidate"
    assert parsed.profile["target_roles"] == ["Data Engineer", "Analytics Engineer"]
    assert parsed.profile["salary_min"] == 95000
    assert parsed.profile["needs_sponsorship"] is False
    assert parsed.filter_fields["manual_salary_min"] == 95000
    assert parsed.filter_fields["manual_us_only"] is True
    assert "manual_visa_sponsorship" not in parsed.filter_fields
    assert parsed.context_fields["manual_target_roles"] == "Data Engineer, Analytics Engineer"
    assert parsed.context_fields["manual_visa_sponsorship"] == "Does not need sponsorship."
    assert "unknown_field" not in parsed.profile


def test_llm_model_uses_env_default_unless_ui_override(monkeypatch) -> None:
    captured: list[str] = []

    def fake_gemini_client(_: str, settings) -> dict:
        captured.append(settings.model)
        return {"profile": {"target_roles": ["Data Analyst"], "skills": ["SQL"]}}

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("JOBPILOT_PROFILE_LLM_MODEL", "env-default-model")
    monkeypatch.setattr(parser_service, "_gemini_profile_client", fake_gemini_client)

    parsed_env = asyncio.run(
        parse_profile_intake(
            profile_text="Skills: SQL\nTarget Roles: Data Analyst",
            parser_mode="llm",
            parser_provider="gemini",
        )
    )
    parsed_override = asyncio.run(
        parse_profile_intake(
            profile_text="Skills: SQL\nTarget Roles: Data Analyst",
            parser_mode="llm",
            parser_provider="gemini",
            parser_model="ui-selected-model",
        )
    )

    assert parsed_env.parse_method == "llm_gemini"
    assert parsed_override.parse_method == "llm_gemini"
    assert captured == ["env-default-model", "ui-selected-model"]


def test_backend_default_key_prefers_googleapi_env_and_ui_override_wins(monkeypatch) -> None:
    captured: list[str] = []

    def fake_gemini_client(_: str, settings) -> dict:
        captured.append(settings.api_key)
        return {"profile": {"target_roles": ["Data Analyst"], "skills": ["SQL"]}}

    monkeypatch.setenv("GOOGLE_API_KEY", "googleapi-default-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-alias-key")
    monkeypatch.setenv("JOBPILOT_PROFILE_LLM_MODEL", "env-default-model")
    monkeypatch.setattr(parser_service, "_gemini_profile_client", fake_gemini_client)

    default_path = asyncio.run(
        parse_profile_intake(
            profile_text="Skills: SQL\nTarget Roles: Data Analyst",
            parser_mode="llm",
            parser_provider="gemini",
        )
    )
    override_path = asyncio.run(
        parse_profile_intake(
            profile_text="Skills: SQL\nTarget Roles: Data Analyst",
            parser_mode="llm",
            parser_provider="gemini",
            parser_api_key="ui-one-request-key",
        )
    )

    assert default_path.parse_method == "llm_gemini"
    assert override_path.parse_method == "llm_gemini"
    assert captured == ["googleapi-default-key", "ui-one-request-key"]


def test_deterministic_guardrails_override_llm_high_risk_fields(monkeypatch) -> None:
    def fake_client(_: str) -> dict:
        return {
            "profile": {
                "target_roles": ["Data Analyst"],
                "salary_min": 50000,
                "needs_sponsorship": False,
                "us_only": False,
                "excluded_employment_types": [],
                "excluded_seniority": [],
                "max_years_required": None,
                "preferred_company_types": [],
            }
        }

    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", fake_client)

    parsed = asyncio.run(
        parse_profile_intake(
            profile_text=KENJI_PERSONA,
            parser_mode="llm",
            parser_provider="gemini",
        )
    )

    assert parsed.parse_method == "llm_gemini"
    assert parsed.profile["salary_min"] == 120000
    assert parsed.profile["needs_sponsorship"] is True
    assert parsed.profile["us_only"] is True
    assert parsed.profile["excluded_employment_types"] == ["contract", "temporary", "unpaid"]
    assert parsed.profile["excluded_seniority"] == ["senior", "staff_principal", "lead_manager"]
    assert parsed.profile["max_years_required"] == 2
    assert parsed.profile["preferred_company_types"] == ["large_company", "research_lab"]


def test_llm_parser_failure_falls_back_to_rules(monkeypatch) -> None:
    def failing_client(_: str) -> dict:
        raise RuntimeError("offline test failure")

    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", failing_client)

    parsed = asyncio.run(parse_profile_intake(profile_text=KENJI_PERSONA, parser_mode="llm", parser_provider="gemini"))

    assert parsed.parse_method == "llm_failed_rule_fallback"
    assert parsed.profile["needs_sponsorship"] is True
    assert parsed.profile["max_years_required"] == 2
    assert any("LLM profile parser failed" in note for note in parsed.notes)


def test_gemini_http_error_fallback_note_is_sanitized(monkeypatch) -> None:
    def failing_gemini_client(_: str, settings) -> dict:
        raise urllib.error.HTTPError("https://example.invalid/?key=SECRET", 429, "quota", {}, None)

    monkeypatch.setenv("GOOGLE_API_KEY", "SECRET")
    monkeypatch.setattr(parser_service, "_gemini_profile_client", failing_gemini_client)

    parsed = asyncio.run(
        parse_profile_intake(
            profile_text=KENJI_PERSONA,
            parser_mode="llm",
            parser_provider="gemini",
            parser_model="gemini-test-model",
        )
    )
    note = " ".join(parsed.notes)

    assert parsed.parse_method == "llm_failed_rule_fallback"
    assert "provider=gemini" in note
    assert "model=gemini-test-model" in note
    assert "error=HTTPError" in note
    assert "status_code=429" in note
    assert "SECRET" not in note
    assert "example.invalid" not in note


def test_sponsorship_toggle_source_clears_stale_no_sponsorship_dealbreaker() -> None:
    script = (main.APP_DIR / "static" / "profile_intake.js").read_text(encoding="utf-8")

    assert 'removeListItems("manual_dealbreakers", ["no sponsorship"])' in script
    assert 'setNamedValue("manual_needs_sponsorship", value === "needs")' in script
    assert 'body.append("parser_api_key", parserApiKey.value)' in script
    assert "if (parserModel?.value)" in script
    assert 'body.append("parser_model", parserModel.value)' in script


def test_review_summary_does_not_count_hidden_canonical_fields() -> None:
    script = (main.APP_DIR / "static" / "profile_intake.js").read_text(encoding="utf-8")
    home = (main.APP_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    canonical_block = home.split('<div class="canonical-hidden-fields hidden">', 1)[1].split("</div>", 1)[0]

    assert "advancedActiveCount" not in script
    assert 'label.textContent = "Advanced filters: "' not in script
    assert 'control.type !== "hidden"' in script
    assert 'form.querySelectorAll("[data-summary-field]")' in script
    assert "isSummaryControlVisible" in script
    assert "isSummaryEmpty" in script
    assert 'details:not([open])' in script
    assert 'window.addEventListener("pageshow"' in script
    assert '"Selected filters"' in script
    assert '"Parsed profile"' in script
    assert "No current selections" not in script
    assert 'data-summary-label="Role focus"' in home
    assert 'data-summary-label="Location preferences"' in home
    assert 'data-summary-label="Salary minimum"' in home
    assert 'data-summary-label="Employment rule"' in home
    assert "data-summary-field" not in canonical_block
    assert "Expert/debug details" not in home
    assert "Hard reject terms" not in home
    assert "company_mode" not in home
    assert "company_mode" not in script
    assert "company_preference_mode" not in home
    assert "company_preference_mode" not in script
    assert "company_size_mode" in home
    assert "company_research_focus" not in home
    assert "company_startup_ok" not in home
    assert "company_avoid_startups" not in home
    assert "company_avoid_defense" not in home
    assert 'applyCompanyControls("inferred")' in script
    assert 'setListPreserving("manual_excluded_company_types", ["startup", "defense_military"], [], source)' in script
    assert "company_research_option" not in home
    assert "company_startup_exclusion" not in home
    assert "company_defense_exclusion" not in home
    assert "prefer_large_research_no_startups" not in home
    assert "safe_large_research" not in home
    assert "prefer_large_research_no_startups" not in script
    assert "safe_large_research" not in script


def test_parser_api_key_is_not_returned_or_saved(monkeypatch) -> None:
    secret = "SECRET_SHOULD_NOT_APPEAR"
    captured: dict[str, dict] = {}

    def fake_client(_: str) -> dict:
        return {"profile": {"target_roles": ["Data Analyst"], "skills": ["SQL"]}}

    def fake_rank_profile(profile, *, top_k=10, candidate_k=1000, embedding_backend="auto", session_feedback_events=None):
        captured["profile"] = profile
        return {"top_jobs": [dict(FAKE_JOB)], "metadata": {"embedding_backend": "test"}}

    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", fake_client)
    monkeypatch.setattr(main, "rank_profile", fake_rank_profile)
    monkeypatch.setattr(main, "save_session", lambda payload: captured.setdefault("session", payload))

    with TestClient(main.app) as client:
        parse_response = client.post(
            "/parse-profile",
            data={
                "profile_text": "Skills: SQL\nTarget Roles: Data Analyst",
                "parser_mode": "llm",
                "parser_provider": "gemini",
                "parser_model": "gemini-2.5-flash",
                "parser_api_key": secret,
            },
        )
        match_response = client.post(
            "/match",
            data={
                "persona": "manual",
                "profile_filters_ready": "1",
                "parser_api_key": secret,
                "manual_target_roles": "Data Analyst",
                "manual_skills": "SQL",
                "top_k": "1",
            },
        )

    assert parse_response.status_code == 200
    assert match_response.status_code == 200
    assert secret not in parse_response.text
    assert secret not in match_response.text
    assert secret not in json.dumps(parse_response.json(), ensure_ascii=False)
    assert secret not in json.dumps(captured["session"], ensure_ascii=False)
    assert secret not in json.dumps(captured["profile"], ensure_ascii=False)


def test_docx_upload_text_extraction_reaches_parse_profile_rule_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(parser_service, "UPLOAD_DIR", tmp_path)

    with TestClient(main.app) as client:
        response = client.post(
            "/parse-profile",
            data={"parser_mode": "rule_fallback"},
            files={
                "resume_pdf": (
                    "assignment_profile.docx",
                    _docx_bytes(ASSIGNMENT_STYLE_LLM_DEMO),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["parse_method"] == "rule_fallback"
    assert any("Parsed uploaded DOCX" in note for note in payload["notes"])
    assert payload["profile"]["needs_sponsorship"] is True
    assert payload["profile"]["us_only"] is True
    assert payload["profile"]["salary_min"] == 120000
    assert payload["profile"]["max_years_required"] == 2
    assert payload["profile"]["excluded_employment_types"] == ["contract", "temporary"]
    assert payload["profile"]["preferred_company_types"] == ["large_company", "research_lab"]


def test_docx_upload_fake_llm_receives_extracted_text_and_guardrails_apply(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(parser_service, "UPLOAD_DIR", tmp_path)
    captured: dict[str, str] = {}

    def fake_client(text: str) -> dict:
        captured["text"] = text
        return {
            "profile": {
                "target_roles": ["Data Analyst"],
                "skills": ["Excel"],
                "salary_min": 50000,
                "needs_sponsorship": False,
                "us_only": False,
                "excluded_employment_types": [],
                "excluded_seniority": [],
                "preferred_company_types": [],
            }
        }

    monkeypatch.setattr(parser_service, "LLM_PROFILE_CLIENT", fake_client)

    with TestClient(main.app) as client:
        response = client.post(
            "/parse-profile",
            data={"parser_mode": "llm", "parser_provider": "gemini"},
            files={
                "resume_pdf": (
                    "assignment_profile.docx",
                    _docx_bytes(ASSIGNMENT_STYLE_LLM_DEMO),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["parse_method"] == "llm_gemini"
    assert "Pass Criteria" in captured["text"]
    assert payload["profile"]["salary_min"] == 120000
    assert payload["profile"]["needs_sponsorship"] is True
    assert payload["profile"]["us_only"] is True
    assert payload["profile"]["excluded_employment_types"] == ["contract", "temporary"]
    assert payload["profile"]["max_years_required"] == 2


def test_visa_context_does_not_leak_into_vector_query_text() -> None:
    profile = parse_profile_text_deterministic(KENJI_PERSONA).profile
    query_text = profile_to_text(profile)

    assert profile["visa_sponsorship"]
    assert "H-1B" in profile["visa_sponsorship"]
    assert "Visa or sponsorship" not in query_text
    assert "H-1B" not in query_text
