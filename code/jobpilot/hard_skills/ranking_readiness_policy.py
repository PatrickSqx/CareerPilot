"""Offline policy labels for Phase 2.16K hard-skill ranking readiness.

These labels are advisory only. They do not connect hard skills to the current
ranking implementation.
"""

from __future__ import annotations

from typing import Any


POLICY_CATEGORIES = [
    "ranking_ready_core",
    "ranking_ready_low_weight",
    "ranking_contextual_only",
    "resume_tailoring_only",
    "human_review_required",
    "suppress_before_ranking",
]


PROGRAMMING_LANGUAGES = {
    "bash",
    "c",
    "c#",
    "c++",
    "go",
    "java",
    "javascript",
    "kotlin",
    "perl",
    "php",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "sql",
    "swift",
    "typescript",
}

CONCRETE_TOOLS_AND_SYSTEMS = {
    ".net",
    "abap",
    "airflow",
    "alteryx",
    "android",
    "angular",
    "ansible",
    "apache",
    "apache spark",
    "arcgis",
    "autocad",
    "aws",
    "azure",
    "bigquery",
    "cad",
    "cassandra",
    "ci/cd",
    "civil 3d",
    "css",
    "cuda",
    "databricks",
    "dbt",
    "django",
    "docker",
    "dynamodb",
    "elasticsearch",
    "etl",
    "fastapi",
    "figma",
    "flask",
    "forklift truck",
    "gcp",
    "git",
    "github actions",
    "google analytics",
    "graphql",
    "hadoop",
    "html",
    "ios",
    "jenkins",
    "jira",
    "kafka",
    "keras",
    "kubernetes",
    "linux",
    "looker",
    "mathematica",
    "matlab",
    "microservices",
    "mlops",
    "mongodb",
    "mysql",
    "next.js",
    "node.js",
    "nosql",
    "numpy",
    "oracle",
    "pandas",
    "postgresql",
    "power bi",
    "pyspark",
    "pytorch",
    "react",
    "redis",
    "rest api",
    "salesforce",
    "sap",
    "sas",
    "scikit-learn",
    "snowflake",
    "solidity",
    "spark",
    "sql server",
    "stata",
    "tableau",
    "tensorflow",
    "terraform",
    "transformers",
    "unix",
    "vue",
}

WELL_DEFINED_DOMAIN_METHODS = {
    "a/b testing",
    "acceptance testing",
    "api design",
    "biochemistry",
    "biology",
    "blockchain",
    "chemistry",
    "computer vision",
    "data engineering",
    "data modeling",
    "data science",
    "data visualization",
    "data warehousing",
    "deep learning",
    "electrical systems",
    "electrical wiring",
    "financial modeling",
    "machine learning",
    "natural language processing",
    "nlp",
    "test automation",
    "test planning",
}

CREDENTIALS_LICENSES_AND_STANDARDS = {
    "cdl",
    "generally accepted accounting principles",
    "hvac",
    "medical terminology",
    "professional engineer",
    "project management professional",
}

LOW_WEIGHT_TERMS = {
    "accounting",
    "auditing",
    "business intelligence",
    "cybersecurity",
    "data analysis",
    "excel",
    "financial statements",
    "forecasting",
    "general ledger",
    "microsoft access",
    "microsoft outlook",
    "microsoft powerpoint",
    "microsoft windows",
    "microsoft word",
}

CONTEXTUAL_ONLY_TERMS = {
    "agile",
    "customer relationship management",
    "data management",
    "data quality",
    "microsoft office",
    "product quality assurance",
    "project management",
    "quality assurance",
    "quality management",
    "risk management",
    "scrum",
    "software quality assurance",
}

RESUME_TAILORING_ONLY_TERMS: set[str] = set()
HUMAN_REVIEW_REQUIRED_TERMS = {"spring"}
SUPPRESS_BEFORE_RANKING_TERMS: set[str] = set()

WATCHLIST_TERMS = {
    "agile",
    "c#",
    "c++",
    "excel",
    "go",
    "microsoft access",
    "microsoft office",
    "microsoft outlook",
    "microsoft powerpoint",
    "microsoft word",
    "project management",
    "quality assurance",
    "r",
    "risk management",
    "scrum",
    "spring",
}


def classify_hard_skill_for_ranking(term: str) -> dict[str, Any]:
    """Return the advisory Phase 2.16K policy label for a normalized hard skill."""

    normalized = str(term or "").strip().lower()

    if normalized in SUPPRESS_BEFORE_RANKING_TERMS:
        category = "suppress_before_ranking"
        rationale = "Accepted by normalization but too ambiguous for ranking without a stricter future rule."
    elif normalized in HUMAN_REVIEW_REQUIRED_TERMS:
        category = "human_review_required"
        rationale = "Accepted by normalization but requires human adjudication before any ranking use."
    elif normalized in RESUME_TAILORING_ONLY_TERMS:
        category = "resume_tailoring_only"
        rationale = "Useful for resume wording only; do not use as a ranking feature."
    elif normalized in CONTEXTUAL_ONLY_TERMS:
        category = "ranking_contextual_only"
        rationale = "Broad method or suite-level term; only useful as context beside stronger concrete evidence."
    elif normalized in LOW_WEIGHT_TERMS:
        category = "ranking_ready_low_weight"
        rationale = "Useful hard-skill signal, but broad or common enough that future ranking should cap its weight."
    elif (
        normalized in PROGRAMMING_LANGUAGES
        or normalized in CONCRETE_TOOLS_AND_SYSTEMS
        or normalized in WELL_DEFINED_DOMAIN_METHODS
        or normalized in CREDENTIALS_LICENSES_AND_STANDARDS
    ):
        category = "ranking_ready_core"
        rationale = "Specific language, concrete tool/system, credential/license, standard, or well-defined technical method."
    else:
        category = "human_review_required"
        rationale = "No explicit 16K policy rule matched; keep out of ranking until reviewed."

    return {
        "normalized_hard_skill": normalized,
        "policy_category": category,
        "future_ranking_weight_guidance": {
            "ranking_ready_core": "eligible_core_feature_after_future_integration",
            "ranking_ready_low_weight": "eligible_low_or_capped_weight_after_future_integration",
            "ranking_contextual_only": "context_only_no_standalone_positive_boost",
            "resume_tailoring_only": "resume_tailoring_only_no_ranking_weight",
            "human_review_required": "no_ranking_weight_until_reviewed",
            "suppress_before_ranking": "suppress_from_ranking_inputs",
        }[category],
        "likely_too_generic": normalized in {"microsoft office"},
        "likely_ambiguous": normalized in WATCHLIST_TERMS or category in {"human_review_required", "suppress_before_ranking"},
        "broad_category": normalized in CONTEXTUAL_ONLY_TERMS or normalized in LOW_WEIGHT_TERMS,
        "concrete_tool": normalized in CONCRETE_TOOLS_AND_SYSTEMS
        or normalized.startswith("microsoft ")
        or normalized in {"excel", "salesforce", "sap", "jira", "aws"},
        "credential_or_license": normalized in CREDENTIALS_LICENSES_AND_STANDARDS,
        "programming_language": normalized in PROGRAMMING_LANGUAGES,
        "domain_method": normalized in WELL_DEFINED_DOMAIN_METHODS
        or normalized in {"accounting", "auditing", "cybersecurity", "data analysis", "risk management", "quality assurance"},
        "watchlist_term": normalized in WATCHLIST_TERMS,
        "policy_rationale": rationale,
    }
