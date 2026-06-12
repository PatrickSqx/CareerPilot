from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.ingestion.quality import USFirstRemoteSelector, market_eligibility, row_quality_score


def _high_score_record(
    *,
    title: str = "Data Scientist",
    company: str = "Acme Analytics",
    source: str = "careerbuilder_us",
    country: str = "US",
    state: str = "CA",
    city: str = "San Francisco",
    location: str = "San Francisco, CA, US",
    description: str = "Build data products with Python, SQL, and machine learning for business teams.",
) -> dict[str, str]:
    return {
        "title": title,
        "company": company,
        "employer": company,
        "link": f"https://example.com/{title.lower().replace(' ', '-')}",
        "description_text": description,
        "description": description,
        "source_record_id": f"{source}-{title}-{company}",
        "source": source,
        "country": country,
        "state": state,
        "city": city,
        "location": location,
        "raw_source_country": country,
        "raw_locale": "en-US" if country == "US" else "en",
        "salary_raw": "$120,000 - $150,000",
        "employment_type": "full-time",
        "company_url": "https://example.com",
        "raw_categories": "Technology",
        "raw_work_types": "Full-time",
        "raw_qualifications": "Bachelor degree",
        "schema_org_employment_type": "FULL_TIME",
        "schema_org_skills": "Python|SQL|Machine Learning",
        "schema_org_experience_requirements": "3 years of experience",
    }


def test_phase1_7_accepts_us_score_gt_85_records() -> None:
    row = _high_score_record()
    selector = USFirstRemoteSelector(target_rows=5)

    assert row_quality_score(row) > 85
    assert market_eligibility(row) == (True, "us")
    assert selector.accept(row) is True
    assert selector.summary()["selected_market_eligibility_counts"] == {"us": 1}


def test_phase1_7_rejects_non_us_onsite_records() -> None:
    row = _high_score_record(
        source="cvlibrary_uk",
        country="GB",
        state="London",
        city="London",
        location="London, United Kingdom",
        description="On-site analytics role with Python and SQL.",
    )
    selector = USFirstRemoteSelector(target_rows=5)

    assert row_quality_score(row) > 85
    assert market_eligibility(row) == (False, "non_us_not_remote_compatible")
    assert selector.accept(row) is False
    assert selector.summary()["rejection_counts"]["non_us_not_remote_compatible"] == 1


def test_phase1_7_accepts_non_us_remote_compatible_records() -> None:
    row = _high_score_record(
        source="linkedin_ie",
        country="IE",
        state="",
        city="Dublin",
        location="Remote",
        description="Remote anywhere data science role with Python, SQL, and machine learning.",
    )
    selector = USFirstRemoteSelector(target_rows=5)

    assert market_eligibility(row) == (True, "non_us_remote_compatible")
    assert selector.accept(row) is True
    assert selector.summary()["selected_market_eligibility_counts"] == {"non_us_remote_compatible": 1}


def test_phase1_7_rejects_non_us_remote_restricted_records() -> None:
    row = _high_score_record(
        source="linkedin_uk",
        country="GB",
        state="London",
        city="London",
        location="Remote, United Kingdom",
        description="Remote within the UK data role with Python, SQL, and analytics.",
    )
    selector = USFirstRemoteSelector(target_rows=5)

    assert market_eligibility(row) == (False, "non_us_remote_restricted")
    assert selector.accept(row) is False
    assert selector.summary()["rejection_counts"]["non_us_remote_restricted"] == 1


def test_phase1_7_rejects_non_us_applications_only_records() -> None:
    row = _high_score_record(
        source="careerbuilder_uk",
        country="UK",
        state="Leeds",
        city="Leeds",
        location="Remote, UK",
        description="Fully remote data engineering role. UK applications only.",
    )

    assert market_eligibility(row) == (False, "non_us_remote_restricted")


def test_phase1_7_does_not_hard_cap_source() -> None:
    selector = USFirstRemoteSelector(target_rows=5)

    for index in range(5):
        row = _high_score_record(
            title=f"Analytics Engineer {index}",
            company=f"Acme {index}",
            source="careerbuilder_us",
            location=f"Austin, TX, US {index}",
            state="TX",
            city="Austin",
            description=f"Build analytics systems with Python, SQL, and dbt for team {index}.",
        )
        assert selector.accept(row) is True

    summary = selector.summary()
    assert summary["selected_rows"] == 5
    assert summary["selected_source_counts"] == {"careerbuilder_us": 5}
    assert summary["policy"]["source_hard_cap_enabled"] is False
