"""Text normalization and lightweight feature extraction helpers."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_FOR_KEY_RE = re.compile(r"[^a-z0-9]+")

SKILL_KEYWORDS = [
    "python",
    "sql",
    "tableau",
    "power bi",
    "excel",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "pytorch",
    "tensorflow",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "spark",
    "pyspark",
    "kafka",
    "kubernetes",
    "docker",
    "aws",
    "gcp",
    "azure",
    "airflow",
    "dbt",
    "snowflake",
    "databricks",
    "java",
    "c++",
    "microservices",
    "statistics",
    "a/b testing",
    "experimentation",
    "analytics",
    "data engineering",
    "mlops",
]


def clean_text(value: Any) -> str:
    """Decode HTML entities, strip tags, and collapse whitespace."""

    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x00", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_key(value: Any) -> str:
    """Normalize a field for use inside stable hash keys."""

    text = clean_text(value).lower()
    text = PUNCT_FOR_KEY_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_hash(value: Any, *, length: int | None = None) -> str:
    """Return a stable SHA-256 hash."""

    digest = hashlib.sha256(clean_text(value).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:length] if length else digest


def first_nonempty(*values: Any) -> str:
    """Return the first non-empty cleaned string."""

    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def parse_salary_from_text(*values: Any) -> tuple[str, str, str]:
    """Extract rough salary min/max from common US salary text patterns."""

    text = " ".join(clean_text(value) for value in values if clean_text(value))
    if not text:
        return "", "", ""

    lowered = text.lower()
    salary_context = any(token in lowered for token in ["salary", "compensation", "$", "usd", "per year", "annually"])
    if not salary_context:
        return "", "", ""

    # Match "$120,000 - $150,000", "$120k-$150k", "120k to 150k".
    range_pattern = re.compile(
        r"(?:\$?\s*)?(\d{2,3}(?:,\d{3})?|\d{2,3}\s*k)"
        r"\s*(?:-|to|through|\u2013|\u2014)\s*"
        r"(?:\$?\s*)?(\d{2,3}(?:,\d{3})?|\d{2,3}\s*k)",
        re.IGNORECASE,
    )
    single_pattern = re.compile(r"\$\s*(\d{2,3}(?:,\d{3})?|\d{2,3}\s*k)", re.IGNORECASE)

    def to_number(raw: str) -> int | None:
        raw = raw.lower().replace(",", "").replace(" ", "")
        if raw.endswith("k"):
            raw = raw[:-1]
            multiplier = 1000
        else:
            multiplier = 1
        try:
            value = int(float(raw) * multiplier)
        except ValueError:
            return None
        if value < 1_000:
            value *= 1000
        return value

    match = range_pattern.search(text)
    if match:
        lo = to_number(match.group(1))
        hi = to_number(match.group(2))
        if lo and hi:
            if lo > hi:
                lo, hi = hi, lo
            return str(lo), str(hi), match.group(0)

    match = single_pattern.search(text)
    if match:
        value = to_number(match.group(1))
        if value:
            return str(value), "", match.group(0)

    return "", "", ""


def extract_skills(title: Any, description: Any) -> str:
    """Return pipe-separated skill hits from a practical keyword list."""

    original_text = f"{clean_text(title)} {clean_text(description)[:5000]}"
    text = original_text.lower()
    hits: list[str] = []
    for skill in SKILL_KEYWORDS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])"
        try:
            matched = bool(re.search(pattern, text))
        except RuntimeError:
            matched = skill.lower() in text
        if matched and skill not in hits:
            hits.append("scikit-learn" if skill == "sklearn" else skill)

    r_context_pattern = re.compile(
        r"\b(?:r programming|r language|programming in r|statistical computing in r|"
        r"experience (?:with|using) r|using r (?:for|to|and))\b",
        re.IGNORECASE,
    )
    if r_context_pattern.search(original_text) and "r" not in hits:
        hits.append("r")
    return "|".join(hits)


def infer_seniority(title: Any, description: Any) -> str:
    text = f"{clean_text(title)} {clean_text(description)[:1500]}".lower()
    if any(token in text for token in ["intern", "internship"]):
        return "internship"
    if any(token in text for token in ["staff", "principal", "distinguished"]):
        return "staff_principal"
    if any(token in text for token in ["senior", "sr.", "sr "]):
        return "senior"
    if any(token in text for token in ["junior", "jr.", "entry level", "new grad", "graduate"]):
        return "entry_junior"
    if any(token in text for token in ["lead", "manager"]):
        return "lead_manager"
    return "unknown"


def infer_years_required(description: Any) -> str:
    """Infer required experience years only from explicit experience contexts."""

    text = clean_text(description)[:5000].lower()
    number = r"(?P<years>1[0-5]|[1-9])"
    years_word = r"(?:years?|yrs?)"
    experience_words = (
        r"(?:experience|exp\.?|background|expertise|professional experience|"
        r"work experience|industry experience|hands-on experience)"
    )
    patterns = [
        rf"\b{number}\s*(?:\+|plus)?\s*{years_word}\s+(?:of\s+)?"
        rf"(?:relevant\s+|professional\s+|work\s+|industry\s+|hands-on\s+)?{experience_words}\b",
        rf"\b(?:minimum|min\.?|at least|requires?|required|requirement|must have|have|"
        rf"possess|with)\s+(?:of\s+)?{number}\s*(?:\+|plus)?\s*{years_word}\s+"
        rf"(?:of\s+)?(?:relevant\s+|professional\s+|work\s+|industry\s+|hands-on\s+)?"
        rf"{experience_words}\b",
        rf"\b{number}\s*(?:\+|plus)?\s*{years_word}\s+(?:required|preferred)\b",
        rf"\b{experience_words}\s*(?:of|:|with)?\s*{number}\s*(?:\+|plus)?\s*{years_word}\b",
    ]
    years: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                value = int(match.group("years"))
            except ValueError:
                continue
            if 1 <= value <= 15:
                years.append(value)
    return str(min(years)) if years else ""


def infer_remote(title: Any, description: Any, location: Any) -> str:
    text = f"{clean_text(title)} {clean_text(location)} {clean_text(description)[:1500]}".lower()
    if "remote" in text:
        return "remote"
    if "hybrid" in text:
        return "hybrid"
    if "on-site" in text or "onsite" in text:
        return "onsite"
    return "unknown"


def infer_employment_type(title: Any, description: Any, raw_type: Any = "") -> str:
    text = f"{clean_text(raw_type)} {clean_text(title)} {clean_text(description)[:2000]}".lower()
    if any(token in text for token in ["unpaid", "volunteer"]):
        return "unpaid"
    if any(token in text for token in ["contract", "contractor", "c2c", "w2 contract"]):
        return "contract"
    if any(token in text for token in ["temporary", "temp "]):
        return "temporary"
    if any(token in text for token in ["internship", "intern "]):
        return "internship"
    if "part-time" in text or "part time" in text or "part_time" in text:
        return "part-time"
    if "full-time" in text or "full time" in text or "full_time" in text:
        return "full-time"
    raw_clean = normalize_for_key(raw_type)
    return raw_clean or "unknown"


def infer_company_type(company: Any, description: Any) -> str:
    text = f"{clean_text(company)} {clean_text(description)[:2500]}".lower()
    if any(token in text for token in ["defense", "military", "army", "navy", "air force", "dod "]):
        return "defense_military"
    if any(token in text for token in ["startup", "seed stage", "series a", "venture-backed"]):
        return "startup"
    if any(token in text for token in ["research lab", "laboratory", "university", "institute"]):
        return "research_lab"
    if any(token in text for token in ["fortune 500", "global leader", "large enterprise", "multinational"]):
        return "large_company"
    return "unknown"


def infer_sponsorship_signal(description: Any) -> str:
    text = clean_text(description)[:5000].lower()
    if any(token in text for token in ["will not sponsor", "unable to sponsor", "no sponsorship"]):
        return "no_sponsorship"
    if any(token in text for token in ["h-1b", "h1b", "visa sponsorship", "sponsor visa", "work authorization"]):
        return "mentions_sponsorship_or_work_auth"
    return "unknown"


def make_embedding_text(record: dict[str, Any]) -> str:
    """Construct downstream text for later embedding generation."""

    parts = [
        record.get("title", ""),
        record.get("company", ""),
        record.get("location", ""),
        record.get("employment_type", ""),
        record.get("seniority", ""),
        record.get("normalized_role_terms", ""),
        record.get("normalized_industries", ""),
        record.get("normalized_skills", ""),
        record.get("normalized_keywords", ""),
        record.get("extracted_skills", ""),
        record.get("schema_org_occupational_category", ""),
        record.get("company_description_raw", ""),
        record.get("description_text", ""),
    ]
    return clean_text(" | ".join(part for part in parts if clean_text(part)))
