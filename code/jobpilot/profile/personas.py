"""Evaluation personas for Phase 2 benchmarks.

These fixtures are used by benchmark scripts only. The matching and ranking
modules consume the same generic profile fields that hidden personas can use.
"""

from __future__ import annotations

from copy import deepcopy

from jobpilot.profile.profile_parser import build_profile


PERSONA_FIXTURES: dict[str, dict] = {
    "aisha": build_profile(
        profile_id="aisha",
        name="Aisha",
        skills=["Python", "SQL", "pandas", "scikit-learn", "machine learning", "analytics"],
        education="Career pivoter with analytics coursework and applied ML projects.",
        experience_text="Built Python data workflows, predictive models, dashboards, and stakeholder analytics projects.",
        projects_publications="Customer churn model, NLP ticket classifier, and model evaluation notebooks.",
        target_roles=["Machine Learning Engineer", "ML Engineer", "Applied Scientist", "Data Scientist"],
        location_preferences=["remote", "San Francisco", "Bay Area", "Oakland", "San Jose"],
        salary_min=140000,
        dealbreakers=["defense", "military", "senior", "staff", "principal"],
        excluded_seniority=["senior", "staff_principal", "lead_manager"],
        max_years_required=4,
        remote_preference="remote_or_bay_area",
        strict_location=False,
        salary_is_dealbreaker=False,
        required_role_families=["ml_related", "research_ai"],
        preferred_role_families=["ml_related", "research_ai"],
        strict_role_family=True,
        avoid_defense_or_clearance=True,
        excluded_company_types=["defense_military"],
        employment_types=["full-time"],
    ),
    "marcus": build_profile(
        profile_id="marcus",
        name="Marcus",
        skills=["SQL", "Excel", "Tableau", "Power BI", "Python", "statistics", "analytics"],
        education="MSBA new graduate with internship analytics experience.",
        experience_text="Completed business analytics internships using SQL, Excel, dashboards, and cohort analysis.",
        projects_publications="Sales dashboard, healthcare utilization analysis, and customer segmentation project.",
        target_roles=["Data Analyst", "Business Analyst", "BI Analyst", "Junior Data Scientist", "Analytics Engineer"],
        location_preferences=["United States", "remote", "Chicago", "New York", "Boston"],
        salary_min=80000,
        dealbreakers=["unpaid", "contract-only", "contract"],
        excluded_employment_types=["unpaid", "contract", "temporary"],
        excluded_seniority=["senior", "staff_principal", "lead_manager"],
        max_years_required=2,
        us_only=True,
        strict_location=True,
        preferred_role_families=["analytics_entry", "bi_analytics"],
        strict_role_family=False,
        employment_types=["full-time"],
    ),
    "priya": build_profile(
        profile_id="priya",
        name="Priya",
        skills=["Java", "Python", "Spark", "Kafka", "Kubernetes", "Docker", "AWS", "microservices", "machine learning"],
        education="Experienced software engineer with distributed systems and cloud infrastructure background.",
        experience_text="Designed Java services, Spark pipelines, Kafka streaming systems, Kubernetes deployments, and production observability.",
        projects_publications="Real-time feature pipeline, model-serving platform prototype, and cloud migration project.",
        target_roles=[
            "ML Infrastructure Engineer",
            "MLOps Engineer",
            "Machine Learning Platform Engineer",
            "AI Infrastructure Engineer",
            "Senior ML Engineer",
            "Machine Learning Ops Engineer",
        ],
        location_preferences=["New York", "NYC", "remote", "United States"],
        salary_min=200000,
        dealbreakers=["junior", "tiny startup", "small startup"],
        excluded_seniority=["internship", "entry_junior"],
        preferred_company_types=["large_company", "research_lab"],
        excluded_company_types=["startup"],
        us_only=True,
        strict_location=True,
        required_role_families=["ml_infra", "ml_related"],
        preferred_role_families=["ml_infra", "ml_related"],
        strict_role_family=True,
        title_must_include_any=["machine learning", "ml", "mlops", "ai", "platform", "infrastructure"],
        required_title_signals=["machine learning", "ml", "mlops", "ai", "platform", "infrastructure"],
        avoid_generic_backend_devops=True,
        avoid_defense_or_clearance=True,
        employment_types=["full-time"],
    ),
    "kenji": build_profile(
        profile_id="kenji",
        name="Kenji",
        skills=["Python", "PyTorch", "TensorFlow", "machine learning", "deep learning", "computer vision", "NLP"],
        education="International CS graduate student on OPT with machine learning research experience.",
        experience_text="Research assistant building deep learning models, evaluation pipelines, and reproducible experiments.",
        projects_publications="Computer vision publication, transformer experiment suite, and research poster.",
        target_roles=["Research Scientist", "Applied Scientist", "Machine Learning Engineer", "Data Scientist"],
        location_preferences=["United States", "remote", "Seattle", "San Francisco", "New York"],
        salary_min=120000,
        dealbreakers=["contract", "temporary", "no sponsorship"],
        excluded_employment_types=["contract", "temporary", "unpaid"],
        needs_sponsorship=True,
        visa_sponsorship="Needs H-1B sponsorship after OPT.",
        us_only=True,
        strict_location=True,
        preferred_company_types=["large_company", "research_lab"],
        required_role_families=["ml_related", "research_ai"],
        preferred_role_families=["ml_related", "research_ai"],
        strict_role_family=True,
        title_must_include_any=["machine learning", "ml", "ai", "data scientist", "applied scientist", "research scientist"],
        required_title_signals=["machine learning", "ml", "ai", "data scientist", "applied scientist", "research scientist"],
        avoid_overly_senior=True,
        avoid_defense_or_clearance=True,
        new_grad_or_student_profile=True,
        hard_reject_seniority_terms=["staff", "principal", "distinguished", "director", "lead", "manager", "head"],
        penalize_seniority_terms=["senior", "sr", "iii"],
        employment_types=["full-time"],
    ),
}


def get_persona(name: str) -> dict:
    key = name.strip().lower()
    if key not in PERSONA_FIXTURES:
        valid = ", ".join(sorted(PERSONA_FIXTURES))
        raise KeyError(f"Unknown persona {name!r}. Valid personas: {valid}")
    return deepcopy(PERSONA_FIXTURES[key])
