"""Hard-skill normalization helpers for Phase 2.16B.

The normalizer keeps raw model spans available for audit while deciding which
surface forms are safe to expose as normalized hard skills.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from jobpilot.utils.text import clean_text, normalize_for_key


GENERIC_TERMS = {
    "ability",
    "account",
    "administr",
    "administration",
    "analysis",
    "analys",
    "analytics",
    "application",
    "applications",
    "architecture",
    "bank",
    "banking",
    "best practic",
    "billing",
    "business",
    "business administration",
    "business development",
    "business process",
    "cloud",
    "code",
    "coding",
    "com",
    "communication",
    "computer",
    "computer literacy",
    "computer litera",
    "computer science",
    "computer systems",
    "con",
    "construction",
    "consulting",
    "control",
    "customer",
    "data",
    "database",
    "de",
    "design",
    "development",
    "dis",
    "distribu",
    "distribution business",
    "document",
    "documentation",
    "dr",
    "emer",
    "eng",
    "engineering",
    "finance",
    "finan",
    "for",
    "framework",
    "general",
    "governance",
    "gowning",
    "handling",
    "health",
    "health care",
    "hou",
    "in",
    "information",
    "infrastructure",
    "information technology",
    "installation",
    "insurance",
    "interface",
    "inter",
    "language",
    "law",
    "learning",
    "le",
    "ling",
    "litera",
    "los",
    "main",
    "man",
    "management",
    "manufa",
    "manufacturing",
    "marketing",
    "me",
    "medic",
    "methodologie",
    "microsoft",
    "model",
    "models",
    "mor",
    "operation",
    "operations",
    "pack",
    "payable",
    "platform",
    "prevention",
    "pro",
    "process",
    "programming",
    "programming language",
    "project",
    "quality",
    "re",
    "report",
    "reporting",
    "reports",
    "research",
    "requirements",
    "restaurant operation",
    "retail",
    "retailing",
    "sales",
    "sch",
    "sche",
    "scheduling",
    "science",
    "se",
    "service",
    "software",
    "software development",
    "software packaging",
    "solution",
    "solutions",
    "spanish",
    "spring",
    "statistics",
    "storage warehousing",
    "system",
    "systems",
    "tech",
    "technical",
    "technic",
    "technique",
    "technology",
    "test",
    "testing",
    "tools",
    "training",
    "turing",
    "ver",
    "war",
    "warehou",
    "warehousing",
    "word processo",
    "word processor",
    "work",
}

SOFT_SKILL_TERMS = {
    "adaptability",
    "attention to detail",
    "business acumen",
    "client rapport",
    "collaboration",
    "communication",
    "complex problem solving",
    "critical thinking",
    "customer service",
    "flexible",
    "leadership",
    "mentoring",
    "organization",
    "problem solving",
    "self motivated",
    "team player",
    "teamwork",
    "time management",
    "written communication",
}

CANONICAL_HARD_SKILLS = {
    ".net",
    "a/b testing",
    "abap",
    "acceptance testing",
    "accounting",
    "airflow",
    "agile",
    "alteryx",
    "android",
    "angular",
    "ansible",
    "apache",
    "apache spark",
    "api design",
    "arcgis",
    "auditing",
    "autocad",
    "aws",
    "azure",
    "bash",
    "bigquery",
    "biochemistry",
    "biology",
    "blockchain",
    "business intelligence",
    "c",
    "c#",
    "c++",
    "cad",
    "cae",
    "cassandra",
    "cdl",
    "cfd",
    "ci/cd",
    "chemistry",
    "civil 3d",
    "computer vision",
    "css",
    "customer relationship management",
    "cuda",
    "cybersecurity",
    "databricks",
    "data analysis",
    "data engineering",
    "data management",
    "data modeling",
    "data quality",
    "data science",
    "data visualization",
    "data warehousing",
    "dbt",
    "deep learning",
    "django",
    "docker",
    "dynamodb",
    "electrical systems",
    "electrical wiring",
    "elasticsearch",
    "etl",
    "excel",
    "fastapi",
    "figma",
    "financial modeling",
    "financial statements",
    "flask",
    "forecasting",
    "forklift truck",
    "gcp",
    "general ledger",
    "generally accepted accounting principles",
    "git",
    "github actions",
    "go",
    "google analytics",
    "graphql",
    "hadoop",
    "html",
    "hvac",
    "ios",
    "java",
    "javascript",
    "jenkins",
    "jira",
    "kafka",
    "keras",
    "kotlin",
    "kubernetes",
    "linux",
    "looker",
    "machine learning",
    "mathematica",
    "matlab",
    "medical terminology",
    "microservices",
    "microsoft access",
    "microsoft office",
    "microsoft outlook",
    "microsoft powerpoint",
    "microsoft windows",
    "microsoft word",
    "mlops",
    "mongodb",
    "mysql",
    "natural language processing",
    "next.js",
    "nlp",
    "node.js",
    "nosql",
    "numpy",
    "oracle",
    "pandas",
    "perl",
    "php",
    "postgresql",
    "power bi",
    "professional engineer",
    "product quality assurance",
    "project management professional",
    "pyspark",
    "python",
    "pytorch",
    "quality assurance",
    "quality management",
    "r",
    "react",
    "redis",
    "rest api",
    "risk management",
    "ruby",
    "rust",
    "salesforce",
    "sap",
    "sas",
    "scala",
    "scikit-learn",
    "scrum",
    "snowflake",
    "software quality assurance",
    "solidity",
    "spark",
    "spring batch",
    "spring boot",
    "spring framework",
    "spring mvc",
    "sql",
    "sql server",
    "stata",
    "swift",
    "tableau",
    "tensorflow",
    "terraform",
    "test automation",
    "test planning",
    "transformers",
    "typescript",
    "unix",
    "vue",
}

ALIASES = {
    "accepted accounting principles": "generally accepted accounting principles",
    "amazon web services": "aws",
    "agile software development": "agile",
    "apache airflow": "airflow",
    "apache kafka": "kafka",
    "apache spark": "apache spark",
    "asp net": ".net",
    "automated testing": "test automation",
    "bi": "business intelligence",
    "c# programming language": "c#",
    "c++ programming language": "c++",
    "c programming language": "c",
    "c plus plus": "c++",
    "c sharp": "c#",
    "c sharp programming language": "c#",
    "cascading style sheets css": "css",
    "cial driver s license cdl": "cdl",
    "cpp": "c++",
    "ci cd": "ci/cd",
    "commercial driver s license cdl": "cdl",
    "crm": "customer relationship management",
    "data analys": "data analysis",
    "gaap": "generally accepted accounting principles",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "go programming language": "go",
    "golang": "go",
    "heating ventilation and air conditioning": "hvac",
    "html5": "html",
    "hypertext markup language": "html",
    "information security": "cybersecurity",
    "java programming language": "java",
    "java spring": "spring framework",
    "javascript programming language": "javascript",
    "js": "javascript",
    "k8s": "kubernetes",
    "machine learning operations": "mlops",
    "ms access": "microsoft access",
    "ms excel": "excel",
    "ms office": "microsoft office",
    "ms powerpoint": "microsoft powerpoint",
    "microsoft excel": "excel",
    "microsoft power bi": "power bi",
    "microsoft sql server": "sql server",
    "natural language processing": "natural language processing",
    "net framework": ".net",
    "node": "node.js",
    "node js": "node.js",
    "nodejs": "node.js",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "postgresql": "postgresql",
    "powerbi": "power bi",
    "pmp": "project management professional",
    "py torch": "pytorch",
    "react js": "react",
    "reactjs": "react",
    "rest": "rest api",
    "restful api": "rest api",
    "r programming language": "r",
    "scrum software development": "scrum",
    "sci kit learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "software quality assurance sqa": "software quality assurance",
    "sqa": "software quality assurance",
    "sql programming language": "sql",
    "tf": "tensorflow",
    "torch": "pytorch",
    "ts": "typescript",
}


@dataclass(frozen=True)
class NormalizationResult:
    raw_text: str
    normalized_text: str
    accepted: bool
    drop_reason: str
    normalization_status: str


def _clean_surface(value: Any) -> str:
    text = clean_text(value)
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    text = text.replace("Ġ", " ")
    text = text.strip(" \t\r\n.,;:()[]{}")
    return clean_text(text)


def normalize_skill_surface(value: Any) -> str:
    """Return a stable canonical surface key for a candidate hard skill."""

    text = _clean_surface(value).lower()
    if not text:
        return ""

    direct = {
        "c++": "c++",
        "c#": "c#",
        ".net": ".net",
        "node.js": "node.js",
        "next.js": "next.js",
        "a/b testing": "a/b testing",
        "ci/cd": "ci/cd",
    }
    if text in direct:
        return direct[text]

    text = text.replace("&", " and ")
    text = re.sub(r"[\\/]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in direct:
        return direct[text]

    if text in ALIASES:
        return ALIASES[text]
    key = normalize_for_key(text)
    if key in ALIASES:
        return ALIASES[key]
    return key


def normalize_hard_skill(value: Any, *, min_chars: int = 2) -> NormalizationResult:
    """Normalize and classify a raw candidate span.

    Dropped candidates are preserved by the caller as audit evidence, but they
    should not be added to normalized_hard_skills.
    """

    raw_text = _clean_surface(value)
    normalized = normalize_skill_surface(raw_text)
    if not raw_text or not normalized:
        return NormalizationResult(raw_text, normalized, False, "empty", "empty")

    if len(normalized) < min_chars:
        raw_key = normalize_for_key(raw_text.lower())
        if normalized not in {"c", "r"} or raw_key in {"c", "r"}:
            return NormalizationResult(raw_text, normalized, False, "term_too_short", "surface_only")

    if normalized == "go" and normalize_for_key(raw_text.lower()) == "go":
        return NormalizationResult(raw_text, normalized, False, "term_too_short", "surface_only")

    if normalized in SOFT_SKILL_TERMS:
        return NormalizationResult(raw_text, normalized, False, "soft_skill", "soft_skill_suppressed")

    if normalized in GENERIC_TERMS:
        return NormalizationResult(raw_text, normalized, False, "too_generic", "generic_suppressed")

    if normalized in CANONICAL_HARD_SKILLS:
        return NormalizationResult(raw_text, normalized, True, "", "canonical_dictionary")

    return NormalizationResult(raw_text, normalized, True, "", "surface_only")


def dictionary_terms() -> list[str]:
    """Return known hard-skill surfaces for the local no-model baseline."""

    terms = set(CANONICAL_HARD_SKILLS)
    terms.update(ALIASES)
    terms.update(ALIASES.values())
    return sorted(terms, key=lambda item: (-len(item), item))
