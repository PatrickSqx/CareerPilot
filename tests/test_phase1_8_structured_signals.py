from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.ingestion.normalizer import (  # noqa: E402
    normalize_adzuna_record,
    normalize_jsearch_record,
    normalize_kaggle_record,
)


def test_kaggle_structured_signals_are_preserved_and_normalized() -> None:
    raw = {
        "source": "careerbuilder_us",
        "idInSource": "abc-123",
        "name": "Data Scientist",
        "companyID": {"$oid": "company-1"},
        "locationID": {"$oid": "location-1"},
        "sourceCC": "us",
        "locale": "en_US",
        "dateCreated": {"$date": "2021-09-01T00:00:00Z"},
        "dateScraped": {"$date": "2021-09-03T00:00:00Z"},
        "dateUploaded": {"$date": "2021-09-04T00:00:00Z"},
        "dateExpired": {"$date": "2021-10-01T00:00:00Z"},
        "referenceID": "ref-abc",
        "orgCompany": {
            "name": "Acme Analytics",
            "description": "Enterprise analytics software company.",
            "info": {"companySize": "1001-5000"},
            "urls": {"url": "https://example.com"},
        },
        "orgAddress": {
            "addressLine": "San Francisco, CA",
            "city": "San Francisco",
            "state": "CA",
            "countryCode": "US",
            "country": "US",
            "postCode": "94105",
            "county": "San Francisco County",
            "geoPoint": {"lat": 37.789, "lng": -122.394},
        },
        "position": {
            "workType": "fulltime",
            "careerLevel": "Entry Level",
            "contractType": "permanent",
            "department": "Data Science",
        },
        "orgTags": {
            "CATEGORIES": ["Information Technology"],
            "WORK_TYPES": ["Full Time"],
            "QUALIFICATIONS": ["Bachelor degree"],
            "SKILLS": ["Python", "SQL"],
            "INDUSTRIES": ["Software"],
            "JOBNAMES": ["Data Scientist"],
            "KEYWORDS": ["analytics", "experimentation"],
            "REQUIREMENTS": ["Bachelor degree"],
            "LANGUAGES": ["English"],
            "COMPANY_BENEFITS": ["401k"],
        },
        "json": {
            "schemaOrg": {
                "title": "Data Scientist",
                "datePosted": "2021-08-30",
                "validThrough": "2021-10-15",
                "identifier": {"name": "CareerBuilder", "value": "schema-abc"},
                "employmentType": "FULL_TIME",
                "skills": ["Python", "Machine Learning"],
                "industry": "Technology",
                "occupationalCategory": "Data Science",
                "educationRequirements": {"credentialCategory": "bachelor degree"},
                "experienceRequirements": {"monthsOfExperience": 24},
                "directApply": True,
                "baseSalary": {
                    "currency": "USD",
                    "value": {"minValue": 120000, "maxValue": 150000, "unitText": "YEAR"},
                },
            },
            "pageData": {"keywords": ["ml", "analytics"]},
        },
        "text": (
            "Build data products with Python, SQL, and machine learning for analytics teams. "
            "Requires two years of relevant experience."
        ),
        "url": "https://jobs.example.com/abc-123",
    }

    row = normalize_kaggle_record(raw, line_no=99, ingested_at="2026-06-05T00:00:00Z")

    assert row["raw_skills"] == "Python|SQL"
    assert row["raw_industries"] == "Software"
    assert row["raw_jobnames"] == "Data Scientist"
    assert row["raw_keywords"] == "analytics|experimentation"
    assert row["raw_requirements"] == "Bachelor degree"
    assert row["raw_languages"] == "English"
    assert row["raw_benefits"] == "401k"
    assert row["position_career_level_raw"] == "Entry Level"
    assert row["position_contract_type_raw"] == "permanent"
    assert row["company_description_raw"] == "Enterprise analytics software company."
    assert row["company_size_raw"] == "1001-5000"
    assert row["schema_org_industry"] == "Technology"
    assert row["schema_org_occupational_category"] == "Data Science"
    assert row["schema_org_valid_through"] == "2021-10-15"
    assert row["reference_id"] == "ref-abc"
    assert row["postal_code"] == "94105"
    assert row["county"] == "San Francisco County"
    assert row["latitude"] == "37.789"
    assert row["longitude"] == "-122.394"
    assert row["date_posted_or_scraped"] == "2021-09-03T00:00:00Z"
    assert row["posting_date_raw"] == "2021-08-30"
    assert row["expiration_date_raw"] == "2021-10-15"
    assert row["direct_apply_raw"] == "true"
    assert row["salary_currency"] == "USD"
    assert "Machine Learning" in row["normalized_skills"].split("|")
    assert "Technology" in row["normalized_industries"].split("|")
    assert "Data Scientist" in row["normalized_role_terms"].split("|")
    assert "page_data_keywords" in row["structured_signal_sources"]
    assert row["structured_signal_confidence"] == "high"


def test_adzuna_structured_adapter_uses_category_geo_contract_and_parser_fallback() -> None:
    raw = {
        "id": "5747083406",
        "title": "Data Scientist",
        "company": {"display_name": "Procter & Gamble"},
        "created": "2026-06-01T01:21:49Z",
        "description": "Build Python and SQL machine learning models for consumer analytics teams.",
        "location": {
            "display_name": "Cincinnati, Hamilton County",
            "area": ["US", "Ohio", "Hamilton County", "Cincinnati"],
        },
        "category": {"label": "IT Jobs", "tag": "it-jobs"},
        "contract_time": "full_time",
        "contract_type": "permanent",
        "salary_min": 100000,
        "salary_max": 120000,
        "salary_is_predicted": "1",
        "latitude": 39.105431,
        "longitude": -84.502154,
        "redirect_url": "https://www.adzuna.com/land/ad/5747083406",
    }

    row = normalize_adzuna_record(raw, query="data scientist", ingested_at="2026-06-05T00:00:00Z")

    assert row["posting_date_raw"] == "2026-06-01T01:21:49Z"
    assert row["raw_industries"] == "IT Jobs|it-jobs"
    assert row["normalized_industries"] == "IT Jobs|it-jobs"
    assert row["position_contract_type_raw"] == "full_time|permanent"
    assert row["employment_type"] == "full-time"
    assert row["county"] == "Hamilton County"
    assert row["latitude"] == "39.105431"
    assert row["longitude"] == "-84.502154"
    assert row["salary_is_predicted"] == "1"
    assert row["raw_skills"] == ""
    assert "python" in row["normalized_skills"].split("|")
    assert "adzuna_category" in row["structured_signal_sources"]


def test_jsearch_structured_adapter_maps_highlights_and_direct_apply_without_live_call() -> None:
    raw = {
        "job_id": "j-1",
        "job_title": "Machine Learning Engineer",
        "employer_name": "Acme AI",
        "job_city": "Austin",
        "job_state": "TX",
        "job_country": "US",
        "job_description": "Build Python ML systems with SQL analytics and production model monitoring.",
        "job_apply_link": "https://jobs.example.com/j-1",
        "job_posted_at_datetime_utc": "2026-06-02T12:00:00Z",
        "job_employment_type": "FULLTIME",
        "job_employment_types": ["FULLTIME", "CONTRACTOR"],
        "job_required_skills": ["Python", "SQL"],
        "job_highlights": {
            "Qualifications": ["3 years machine learning experience"],
            "Responsibilities": ["Build model pipelines"],
        },
        "job_required_experience": {"required_experience_in_months": 36},
        "job_required_education": {"postgraduate_degree": False, "bachelors_degree": True},
        "job_latitude": 30.2672,
        "job_longitude": -97.7431,
        "job_salary_currency": "USD",
        "job_apply_is_direct": True,
    }

    row = normalize_jsearch_record(raw, query="machine learning engineer", ingested_at="2026-06-05T00:00:00Z")

    assert row["posting_date_raw"] == "2026-06-02T12:00:00Z"
    assert row["position_contract_type_raw"] == "FULLTIME|CONTRACTOR"
    assert row["raw_skills"] == "Python|SQL|3 years machine learning experience"
    assert "job_highlights" in row["raw_requirements"]
    assert "bachelors_degree" in row["schema_org_education_requirements"]
    assert row["latitude"] == "30.2672"
    assert row["longitude"] == "-97.7431"
    assert row["salary_currency"] == "USD"
    assert row["direct_apply_raw"] == "true"
    assert "jsearch_required_skills" in row["structured_signal_sources"]
    assert "jsearch_highlights" in row["structured_signal_sources"]
