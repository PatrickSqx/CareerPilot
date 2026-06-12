"""API-gated resume tailoring for Phase 3.

Resume generation is intentionally disabled unless an LLM API is connected.
The local helper below remains useful for prompt scaffolding, but the web route
does not expose no-key resume generation.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from app.services.paths import PROJECT_ROOT, UPLOAD_DIR
from app.services.presentation_labels import application_strategy_display_label
from jobpilot.profile.profile_parser import normalize_list
from jobpilot.utils.text import clean_text


DEFAULT_RESUME_PROVIDER = "gemini"
DEFAULT_GEMINI_RESUME_MODEL = "gemini-2.5-flash"
RESUME_MODEL_ENV_NAMES = (
    "JOBPILOT_RESUME_LLM_MODEL",
    "JOBPILOT_PROFILE_LLM_MODEL",
    "GEMINI_MODEL",
    "GOOGLE_GENERATIVE_AI_MODEL",
    "JOBPILOT_LLM_MODEL",
)

LLM_RESUME_CLIENT: Callable[[str], str] | None = None
DEMO_RESUME_EDUCATION = "UC Davis - M.S. in Business Analytics"
DEMO_RESUME_CONTEXT: dict[str, dict[str, Any]] = {
    "aisha": {
        "education": [{"institution": "UC Davis", "dates": "", "detail": "M.S. in Business Analytics"}],
        "experience": [
            {
                "organization": "Applied Analytics Lab",
                "role": "Data Science Project Lead",
                "location": "Davis, CA",
                "dates": "2025 - Present",
                "bullets": [
                    "Built Python and SQL data workflows to clean, join, and analyze product and customer datasets.",
                    "Developed predictive modeling notebooks with pandas and scikit-learn for churn, segmentation, and business decision support.",
                    "Created stakeholder-facing dashboards and summaries to translate model findings into practical recommendations.",
                    "Documented feature logic, validation checks, and modeling assumptions so analysis could be reviewed and reused.",
                ],
            },
            {
                "organization": "Analytics Practicum Team",
                "role": "Machine Learning Analyst",
                "location": "Remote",
                "dates": "2024 - 2025",
                "bullets": [
                    "Evaluated model performance using cross-validation, error analysis, and documented assumptions.",
                    "Prepared SQL extracts and feature tables to support repeatable analysis across multiple project milestones.",
                    "Presented tradeoffs between model accuracy, interpretability, and operational use to non-technical stakeholders.",
                    "Wrote concise reporting notes that compared candidate models and identified next-step data collection needs.",
                ],
            },
        ],
        "project_or_research_experience": [
            {
                "organization": "Customer Churn Model",
                "role": "Data Scientist",
                "location": "",
                "dates": "2025",
                "bullets": [
                    "Trained classification models to identify at-risk customers and explain drivers behind churn risk.",
                    "Compared feature importance and model metrics to recommend retention actions for business teams.",
                    "Validated train/test splits and confusion-matrix results to compare model tradeoffs before summarizing findings.",
                ],
            },
            {
                "organization": "NLP Ticket Classifier",
                "role": "Machine Learning Project",
                "location": "",
                "dates": "2024",
                "bullets": [
                    "Built an NLP workflow to categorize support tickets and summarize recurring issue patterns.",
                    "Created model evaluation notebooks covering data quality, precision/recall, and failure cases.",
                    "Reviewed misclassified examples to identify labeling gaps and improve model handoff notes.",
                ],
            },
        ],
        "skills": {
            "Analytics & Research": "Predictive modeling, classification, customer analytics, model evaluation, stakeholder analysis",
            "Data & Tools": "Python, SQL, pandas, scikit-learn, dashboards, notebooks",
            "Communication": "Stakeholder reporting, model documentation, dashboard storytelling, practical recommendations",
        },
    },
    "marcus": {
        "education": [{"institution": "UC Davis", "dates": "", "detail": "M.S. in Business Analytics"}],
        "experience": [
            {
                "organization": "Business Analytics Practicum",
                "role": "Data Analyst",
                "location": "Davis, CA",
                "dates": "2025 - Present",
                "bullets": [
                    "Built SQL and Excel analyses to monitor business trends, cohort behavior, and operational performance.",
                    "Created Tableau and Power BI dashboards for weekly reporting and stakeholder decision-making.",
                    "Translated ambiguous business questions into data pulls, metric definitions, and concise findings.",
                    "Validated source tables and dashboard calculations before sharing updates with business stakeholders.",
                ],
            },
            {
                "organization": "Analytics Internship Project",
                "role": "Business Analyst Intern",
                "location": "Remote",
                "dates": "2024 - 2025",
                "bullets": [
                    "Analyzed customer segments and utilization patterns to identify opportunities for process improvement.",
                    "Prepared data quality checks, summary tables, and management-ready reporting notes.",
                    "Partnered with non-technical stakeholders to prioritize metrics and clarify dashboard requirements.",
                    "Documented metric definitions and assumptions so recurring reports could be interpreted consistently.",
                ],
            },
        ],
        "project_or_research_experience": [
            {
                "organization": "Sales Dashboard",
                "role": "BI Analyst",
                "location": "",
                "dates": "2025",
                "bullets": [
                    "Designed an interactive dashboard covering revenue trends, pipeline stages, and account-level drilldowns.",
                    "Used SQL and Excel validation checks to reconcile source data before publishing insights.",
                    "Organized filters, KPIs, and summary views to support fast comparison across teams and time periods.",
                ],
            },
            {
                "organization": "Customer Segmentation Project",
                "role": "Analytics Project",
                "location": "",
                "dates": "2024",
                "bullets": [
                    "Clustered customer behavior patterns and summarized segment profiles for targeting recommendations.",
                    "Documented assumptions, limitations, and next-step analyses for business review.",
                    "Translated segment findings into practical reporting notes for stakeholder discussion.",
                ],
            },
        ],
        "skills": {
            "Analytics & Research": "Business analysis, cohort analysis, segmentation, statistics, KPI definition",
            "Data & Tools": "SQL, Excel, Tableau, Power BI, Python, dashboards",
            "Communication": "Requirements gathering, executive summaries, stakeholder reporting, metric documentation",
        },
    },
    "priya": {
        "education": [{"institution": "UC Davis", "dates": "", "detail": "M.S. in Business Analytics"}],
        "experience": [
            {
                "organization": "Cloud ML Platform Team",
                "role": "Software Engineer",
                "location": "New York, NY",
                "dates": "2023 - Present",
                "bullets": [
                    "Designed Java and Python services supporting feature pipelines, model-serving workflows, and production observability.",
                    "Built Spark and Kafka data processing jobs to prepare model inputs for downstream machine learning systems.",
                    "Improved deployment reliability using Docker, Kubernetes, AWS services, and service-level monitoring.",
                    "Coordinated interface requirements with data science users to keep platform outputs usable for model experiments.",
                ],
            },
            {
                "organization": "Infrastructure Modernization Project",
                "role": "Backend Engineer",
                "location": "Remote",
                "dates": "2021 - 2023",
                "bullets": [
                    "Migrated legacy services into containerized deployments with repeatable CI/CD and monitoring practices.",
                    "Partnered with data science users to translate experimentation needs into scalable platform features.",
                    "Documented operational runbooks and incident response workflows for production services.",
                    "Reviewed deployment logs and monitoring signals to identify reliability issues before user-facing impact.",
                ],
            },
        ],
        "project_or_research_experience": [
            {
                "organization": "Real-Time Feature Pipeline",
                "role": "ML Infrastructure Project",
                "location": "",
                "dates": "2025",
                "bullets": [
                    "Prototyped streaming feature transformations with Kafka and Spark for low-latency model inputs.",
                    "Defined validation checks for data freshness, schema drift, and serving consistency.",
                    "Mapped feature pipeline dependencies and handoff requirements for model deployment review.",
                ],
            }
            ,
            {
                "organization": "Model Serving Reliability Review",
                "role": "Platform Engineering Project",
                "location": "",
                "dates": "2024",
                "bullets": [
                    "Audited service health checks, deployment notes, and monitoring dashboards for model-serving workflows.",
                    "Summarized reliability gaps and recommended runbook updates for cross-functional engineering review.",
                    "Connected operational findings to model iteration needs so data science users could diagnose serving issues faster.",
                ],
            },
        ],
        "skills": {
            "Analytics & Research": "ML infrastructure, feature pipelines, production monitoring, system design",
            "Data & Tools": "Java, Python, Spark, Kafka, Kubernetes, Docker, AWS, microservices",
            "Communication": "Platform documentation, cross-functional requirements, runbooks, technical design reviews",
        },
    },
    "kenji": {
        "education": [{"institution": "UC Davis", "dates": "", "detail": "M.S. in Business Analytics"}],
        "experience": [
            {
                "organization": "Machine Learning Research Lab",
                "role": "Research Assistant",
                "location": "Davis, CA",
                "dates": "2024 - Present",
                "bullets": [
                    "Built deep learning experiments with PyTorch and TensorFlow to evaluate model accuracy and robustness.",
                    "Created reproducible training and evaluation pipelines covering data preprocessing, metrics, and error analysis.",
                    "Summarized experiment results for research discussions, posters, and model improvement planning.",
                    "Maintained experiment logs and comparison tables to support repeatable review of modeling tradeoffs.",
                ],
            },
            {
                "organization": "Applied AI Project Team",
                "role": "Machine Learning Engineer",
                "location": "Remote",
                "dates": "2023 - 2024",
                "bullets": [
                    "Developed NLP and computer vision prototypes using Python notebooks and structured experiment tracking.",
                    "Compared transformer and baseline models to identify performance, latency, and data quality tradeoffs.",
                    "Documented research assumptions and reproducibility steps for handoff to collaborators.",
                    "Prepared concise model evaluation notes that connected error patterns to next-step experiment design.",
                ],
            },
        ],
        "project_or_research_experience": [
            {
                "organization": "Computer Vision Publication",
                "role": "Research Project",
                "location": "",
                "dates": "2025",
                "bullets": [
                    "Prepared model experiments, ablation notes, and performance summaries for a computer vision research paper.",
                    "Reviewed qualitative model failures to guide dataset cleaning and follow-up experiments.",
                    "Tracked experiment variants and evaluation outcomes to support publication-ready comparison tables.",
                ],
            },
            {
                "organization": "Transformer Experiment Suite",
                "role": "NLP Project",
                "location": "",
                "dates": "2024",
                "bullets": [
                    "Implemented transformer experiments and evaluation scripts for NLP classification tasks.",
                    "Tracked accuracy, precision, recall, and failure cases to support model selection decisions.",
                    "Summarized misclassification examples and model limitations for research discussion.",
                ],
            },
        ],
        "skills": {
            "Analytics & Research": "Deep learning, model evaluation, experiment design, NLP, computer vision",
            "Data & Tools": "Python, PyTorch, TensorFlow, scikit-learn, notebooks, evaluation pipelines",
            "Communication": "Research summaries, poster preparation, reproducibility notes, collaborator handoff",
        },
    },
}

FIXED_RESUME_STYLE_TEMPLATE = """Qixiang-style fixed DOCX template:
- Times New Roman, narrow margins, black text, no color accents.
- Centered name and contact header with thin underline.
- Section headings in all caps with thin underline.
- Education entry: institution on the left, dates on the right; degree/GPA on the next plain line.
- Experience/project entry: organization and role on the left, location and dates on the right.
- Bullets use the user's evidence only; no summary, target, objective, or placeholder text.
- Skills render as three compact text lines.
"""

RESUME_JSON_SCHEMA = {
    "name": "Candidate Name",
    "contact": {"phone": "", "email": "", "linkedin": ""},
    "education": [{"institution": "", "dates": "", "detail": ""}],
    "experience": [{"organization": "", "role": "", "location": "", "dates": "", "bullets": []}],
    "project_or_research_experience": [
        {"organization": "", "role": "", "location": "", "dates": "", "bullets": []}
    ],
    "skills": {
        "Analytics & Research": "",
        "Data & Tools": "",
        "Communication": "",
    },
}


class ResumeGenerationUnavailable(RuntimeError):
    """Raised when resume generation is requested without a connected API."""


def _as_lines(values: list[str], *, fallback: str = "") -> list[str]:
    lines = [clean_text(value) for value in values if clean_text(value)]
    return lines or ([fallback] if fallback else [])


def _job_skills(job: dict[str, Any]) -> list[str]:
    skills = job.get("matched_skills") or []
    if isinstance(skills, str):
        skills = [part.strip() for part in skills.split("|")]
    return [clean_text(skill) for skill in skills if clean_text(skill)]


def _is_demo_profile(profile: dict[str, Any]) -> bool:
    profile_id = clean_text(profile.get("profile_id")).lower()
    return profile_id in {"aisha", "marcus", "priya", "kenji"} or bool(profile.get("demo_persona"))


def _demo_email(profile: dict[str, Any]) -> str:
    name = clean_text(profile.get("name")) or clean_text(profile.get("profile_id")) or "candidate"
    slug = re.sub(r"[^a-z0-9]+", "", name.lower())
    return f"{slug or 'candidate'}@gmail.com"


def _demo_phone(profile: dict[str, Any]) -> str:
    profile_id = clean_text(profile.get("profile_id")).lower()
    suffix_by_profile = {"aisha": "0101", "marcus": "0102", "priya": "0103", "kenji": "0104"}
    suffix = suffix_by_profile.get(profile_id, "0100")
    return f"(555) 010-{suffix[-4:]}"


def _demo_linkedin(profile: dict[str, Any]) -> str:
    name = clean_text(profile.get("name")) or clean_text(profile.get("profile_id")) or "candidate"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"linkedin.com/in/{slug or 'candidate'}-demo"


def _demo_context(profile: dict[str, Any]) -> dict[str, Any]:
    profile_id = clean_text(profile.get("profile_id")).lower()
    return DEMO_RESUME_CONTEXT.get(profile_id, {})


def _resume_display_profile(profile: dict[str, Any]) -> dict[str, Any]:
    payload = dict(profile)
    if _is_demo_profile(profile):
        if not clean_text(payload.get("email")):
            payload["email"] = _demo_email(profile)
        if not clean_text(payload.get("phone")):
            payload["phone"] = _demo_phone(profile)
        if not clean_text(payload.get("linkedin")):
            payload["linkedin"] = _demo_linkedin(profile)
        if not clean_text(payload.get("education")):
            payload["education"] = DEMO_RESUME_EDUCATION
        elif clean_text(payload.get("education")).lower() in {
            "career pivoter with analytics coursework and applied ml projects.",
            "analytics coursework and applied ml projects.",
        }:
            payload["education"] = DEMO_RESUME_EDUCATION
        context = _demo_context(payload)
        if context:
            payload["demo_resume_context"] = context
    return payload


def _clean_resume_markdown(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def _resume_source_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    lines = [clean_text(line) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _xml_text(value: Any) -> str:
    return escape(_strip_inline_markdown(str(value)), {'"': "&quot;"})


def _resume_clean_text(value: Any) -> str:
    text = clean_text(value)
    # Keep date ranges ASCII and avoid mojibake-like separators in generated output.
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace(" \u6bcf ", " - ")
    return text


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def _is_entry_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("- "):
        return False
    return " | " in stripped


def _looks_like_date_segment(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|present|\d{4})\b",
            text,
            flags=re.I,
        )
    )


def _entry_line_parts(text: str) -> tuple[str, str]:
    parts = [clean_text(part) for part in text.split("|") if clean_text(part)]
    if len(parts) >= 4:
        return " | ".join(parts[:2]), " | ".join(parts[2:])
    if len(parts) == 3 and _looks_like_date_segment(parts[-1]):
        return " | ".join(parts[:2]), parts[-1]
    if len(parts) == 2 and _looks_like_date_segment(parts[-1]):
        return parts[0], parts[1]
    return clean_text(text), ""


def _paragraph(text: str, *, style: str = "") -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t xml:space=\"preserve\">{_xml_text(text)}</w:t></w:r></w:p>"


def _entry_line_paragraph(text: str) -> str:
    left, right = _entry_line_parts(text)
    if not right:
        return _paragraph(text, style="EntryLine")
    return (
        "<w:p>"
        "<w:pPr>"
        '<w:pStyle w:val="EntryLine"/>'
        '<w:tabs><w:tab w:val="right" w:pos="11469"/></w:tabs>'
        "</w:pPr>"
        f"<w:r><w:t xml:space=\"preserve\">{_xml_text(left)}</w:t></w:r>"
        "<w:r><w:tab/></w:r>"
        f"<w:r><w:t xml:space=\"preserve\">{_xml_text(right)}</w:t></w:r>"
        "</w:p>"
    )


def _markdown_line_to_docx_paragraph(line: str, *, subtitle: bool = False) -> str:
    stripped = line.strip()
    if not stripped:
        return "<w:p/>"
    if stripped.startswith("# "):
        return _paragraph(stripped[2:].strip(), style="Heading1")
    if stripped.startswith("## "):
        return _paragraph(stripped[3:].strip(), style="Heading2")
    if stripped.startswith("- "):
        return _list_paragraph(stripped[2:].strip())
    if subtitle:
        return _paragraph(stripped, style="Subtitle")
    if _is_entry_line(stripped):
        return _entry_line_paragraph(stripped)
    return _paragraph(stripped)


DISALLOWED_RESUME_SECTIONS = {
    "professional summary",
    "summary",
    "objective",
    "target",
    "profile",
    "tailoring notes",
    "job-specific tailoring notes",
}


def _sanitize_resume_markdown(markdown: str) -> str:
    cleaned_lines: list[str] = []
    skipping_disallowed_section = False
    current_section = ""
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        heading_match = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading_match:
            heading = _strip_inline_markdown(heading_match.group(1)).strip().rstrip(":").lower()
            if heading in DISALLOWED_RESUME_SECTIONS or heading.startswith("target:"):
                skipping_disallowed_section = True
                continue
            skipping_disallowed_section = False
            current_section = heading
        if skipping_disallowed_section:
            continue
        if re.match(r"(?i)^target\s*:", line):
            continue
        if current_section == "education" and line.startswith("- "):
            cleaned_lines.append(line[2:].strip())
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines).strip() + "\n"


def _list_paragraph(text: str) -> str:
    return (
        "<w:p>"
        "<w:pPr>"
        '<w:pStyle w:val="ListParagraph"/>'
        "<w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr>"
        "</w:pPr>"
        f"<w:r><w:t xml:space=\"preserve\">{_xml_text(text)}</w:t></w:r>"
        "</w:p>"
    )


def _resume_contact_line(contact: dict[str, Any]) -> str:
    parts = []
    phone = _resume_clean_text(contact.get("phone"))
    email = _resume_clean_text(contact.get("email"))
    linkedin = _resume_clean_text(contact.get("linkedin"))
    if phone:
        parts.append(phone)
    if email:
        parts.append(email)
    if linkedin:
        parts.append(linkedin)
    return " | ".join(parts)


def _extract_json_payload(raw: str) -> dict[str, Any] | None:
    text = _clean_resume_markdown(raw)
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    elif "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _clean_bullets(values: Any, *, limit: int = 5) -> list[str]:
    if isinstance(values, str):
        candidates = [line.strip(" -\u2022\t") for line in values.splitlines()]
    elif isinstance(values, list):
        candidates = values
    else:
        candidates = []
    bullets: list[str] = []
    for value in candidates:
        bullet = _resume_clean_text(value)
        if not bullet or "[" in bullet or "]" in bullet:
            continue
        if re.search(r"(?i)\b(add|insert|placeholder|tailoring note|missing evidence)\b", bullet):
            continue
        if bullet.lower() not in {item.lower() for item in bullets}:
            bullets.append(bullet)
        if len(bullets) >= limit:
            break
    return bullets


def _normalize_education(values: Any) -> list[dict[str, str]]:
    rows = values if isinstance(values, list) else []
    education: list[dict[str, str]] = []
    for value in rows:
        if isinstance(value, dict):
            institution = _resume_clean_text(value.get("institution") or value.get("school") or value.get("university"))
            dates = _resume_clean_text(value.get("dates") or value.get("date_range"))
            detail = _resume_clean_text(value.get("detail") or value.get("degree") or value.get("description"))
        else:
            parts = [_resume_clean_text(part) for part in str(value).split("|") if _resume_clean_text(part)]
            institution = parts[0] if parts else ""
            dates = parts[-1] if len(parts) > 1 and _looks_like_date_segment(parts[-1]) else ""
            detail = " | ".join(parts[1:-1] if dates else parts[1:])
        if institution:
            education.append({"institution": institution, "dates": dates, "detail": detail})
    return education[:4]


def _normalize_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    organization = _resume_clean_text(value.get("organization") or value.get("company") or value.get("project"))
    role = _resume_clean_text(value.get("role") or value.get("title"))
    location = _resume_clean_text(value.get("location"))
    dates = _resume_clean_text(value.get("dates") or value.get("date_range"))
    bullets = _clean_bullets(value.get("bullets"), limit=5)
    if not organization and not role and not bullets:
        return None
    return {
        "organization": organization,
        "role": role,
        "location": location,
        "dates": dates,
        "bullets": bullets,
    }


def _normalize_entries(values: Any, *, limit: int = 6) -> list[dict[str, Any]]:
    rows = values if isinstance(values, list) else []
    entries: list[dict[str, Any]] = []
    for value in rows:
        entry = _normalize_entry(value)
        if entry:
            entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def _normalize_skills(value: Any) -> dict[str, str]:
    groups = {"Analytics & Research": "", "Data & Tools": "", "Communication": ""}
    if isinstance(value, dict):
        for group in groups:
            groups[group] = _resume_clean_text(value.get(group) or value.get(group.lower()) or "")
        for key, raw_value in value.items():
            clean_key = _resume_clean_text(key)
            if clean_key and clean_key not in groups and len(groups) < 6:
                groups[clean_key] = _resume_clean_text(raw_value)
    elif isinstance(value, list):
        groups["Data & Tools"] = ", ".join(_resume_clean_text(item) for item in value if _resume_clean_text(item))
    elif isinstance(value, str):
        groups["Data & Tools"] = _resume_clean_text(value)
    return {key: item for key, item in groups.items() if item}


def _profile_contact(profile: dict[str, Any]) -> dict[str, str]:
    display_profile = _resume_display_profile(profile)
    return {
        "phone": _resume_clean_text(display_profile.get("phone")),
        "email": _resume_clean_text(display_profile.get("email")),
        "linkedin": _resume_clean_text(display_profile.get("linkedin")),
    }


def _normalize_structured_resume(data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    display_profile = _resume_display_profile(profile)
    profile_contact = _profile_contact(display_profile)
    contact = profile_contact
    name = _resume_clean_text(display_profile.get("name")) or _resume_clean_text(data.get("name")) or "Candidate"
    projects = data.get("project_or_research_experience")
    if projects is None:
        projects = data.get("projects") or data.get("research_experience")
    return {
        "name": name,
        "contact": contact,
        "education": _normalize_education(data.get("education")),
        "experience": _normalize_entries(data.get("experience"), limit=6),
        "project_or_research_experience": _normalize_entries(projects, limit=4),
        "skills": _normalize_skills(data.get("skills")),
    }


def _merge_skill_line(existing: str, additions: list[str], *, limit: int = 10) -> str:
    values: list[str] = []
    for chunk in [existing, ", ".join(additions)]:
        for item in chunk.split(","):
            clean = _resume_clean_text(item)
            if clean and clean.lower() not in {value.lower() for value in values}:
                values.append(clean)
            if len(values) >= limit:
                break
    return ", ".join(values)


def _demo_entry_is_generic(entry: dict[str, Any]) -> bool:
    org = _resume_clean_text(entry.get("organization")).lower()
    role = _resume_clean_text(entry.get("role"))
    location = _resume_clean_text(entry.get("location"))
    dates = _resume_clean_text(entry.get("dates"))
    bullets = _clean_bullets(entry.get("bullets"), limit=5)
    role_like_orgs = {
        "analytics analyst",
        "business analyst",
        "data analyst",
        "data scientist",
        "machine learning engineer",
        "ml engineer",
    }
    if org in role_like_orgs and not role and not location and not dates:
        return True
    return not org and not role and not location and not dates and len(bullets) <= 1


def _append_missing_demo_entries(existing: list[dict[str, Any]], demo_entries: list[dict[str, Any]], *, min_entries: int) -> list[dict[str, Any]]:
    payload = [dict(item) for item in existing if isinstance(item, dict) and not _demo_entry_is_generic(item)]
    for entry in demo_entries:
        if len(payload) >= min_entries:
            break
        org = _resume_clean_text(entry.get("organization"))
        if org and org.lower() in {_resume_clean_text(item.get("organization")).lower() for item in payload}:
            continue
        payload.append(dict(entry))
    return payload


def _ensure_demo_bullet_depth(entries: list[dict[str, Any]], demo_entries: list[dict[str, Any]], *, minimum: int) -> list[dict[str, Any]]:
    demo_by_org = {_resume_clean_text(entry.get("organization")).lower(): entry for entry in demo_entries}
    payload: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        bullets = _clean_bullets(item.get("bullets"), limit=5)
        demo = demo_by_org.get(_resume_clean_text(item.get("organization")).lower(), {})
        for bullet in _clean_bullets(demo.get("bullets"), limit=5):
            if len(bullets) >= minimum:
                break
            if bullet.lower() not in {value.lower() for value in bullets}:
                bullets.append(bullet)
        item["bullets"] = bullets
        payload.append(item)
    return payload


def _apply_demo_entry_guardrails(entries: list[dict[str, Any]], demo_entries: list[dict[str, Any]], *, bullet_limit: int) -> list[dict[str, Any]]:
    demo_by_org = {_resume_clean_text(entry.get("organization")).lower(): entry for entry in demo_entries}
    payload: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        demo = demo_by_org.get(_resume_clean_text(item.get("organization")).lower())
        if demo:
            item["organization"] = _resume_clean_text(demo.get("organization"))
            item["role"] = _resume_clean_text(demo.get("role"))
            item["location"] = _resume_clean_text(demo.get("location"))
            item["dates"] = _resume_clean_text(demo.get("dates"))
            item["bullets"] = _clean_bullets(demo.get("bullets"), limit=bullet_limit)
        payload.append(item)
    return payload


def _tailor_demo_bullets_for_job(
    entries: list[dict[str, Any]],
    job: dict[str, Any],
    *,
    add_title_alignment: bool = True,
    add_skill_alignment: bool = True,
) -> list[dict[str, Any]]:
    title = _resume_clean_text(job.get("title"))
    matched = [skill for skill in _job_skills(job)[:4] if skill]
    if not title and not matched:
        return entries
    payload: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        item = dict(entry)
        bullets = _clean_bullets(item.get("bullets"), limit=5)
        if add_title_alignment and index == 0 and title and bullets:
            bullets[0] = re.sub(
                r"\.$",
                f", with emphasis on {title} requirements.",
                bullets[0],
            )
        if add_skill_alignment and index == 0 and matched and bullets:
            skill_phrase = ", ".join(matched[:3])
            if skill_phrase.lower() not in " ".join(bullets).lower():
                bullets.append(f"Applied {skill_phrase} in project work aligned with the selected job requirements.")
        item["bullets"] = bullets[:5]
        payload.append(item)
    return payload


def _expand_demo_structured_resume(resume: dict[str, Any], profile: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    context = _demo_context(profile)
    if not context:
        return resume
    payload = dict(resume)
    if context.get("education"):
        payload["education"] = _normalize_education(context.get("education"))
    elif not payload.get("education"):
        payload["education"] = _normalize_education(context.get("education"))
    demo_experience = _normalize_entries(context.get("experience"), limit=4)
    demo_projects = _normalize_entries(context.get("project_or_research_experience"), limit=4)
    experience = _append_missing_demo_entries(
        payload.get("experience") if isinstance(payload.get("experience"), list) else [],
        demo_experience,
        min_entries=2,
    )
    experience = _apply_demo_entry_guardrails(experience, demo_experience, bullet_limit=4)
    experience = _ensure_demo_bullet_depth(experience, demo_experience, minimum=3)
    payload["experience"] = _tailor_demo_bullets_for_job(experience, job)
    projects = _append_missing_demo_entries(
        payload.get("project_or_research_experience") if isinstance(payload.get("project_or_research_experience"), list) else [],
        demo_projects,
        min_entries=2 if len(demo_projects) > 1 else 1,
    )
    projects = _apply_demo_entry_guardrails(projects, demo_projects, bullet_limit=3)
    projects = _ensure_demo_bullet_depth(projects, demo_projects, minimum=2)
    payload["project_or_research_experience"] = _tailor_demo_bullets_for_job(
        projects,
        job,
        add_title_alignment=False,
        add_skill_alignment=False,
    )
    demo_skills = context.get("skills") if isinstance(context.get("skills"), dict) else {}
    skills: dict[str, str] = {}
    matched = _job_skills(job)
    for key, value in demo_skills.items():
        skills[key] = _merge_skill_line("", [_resume_clean_text(value), *matched])
    payload["skills"] = skills
    return payload


def _markdown_resume_to_structured(markdown: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Compatibility fallback for old tests or provider responses."""

    sanitized = _sanitize_resume_markdown(markdown)
    data: dict[str, Any] = {
        "name": "",
        "contact": {},
        "education": [],
        "experience": [],
        "project_or_research_experience": [],
        "skills": {},
    }
    current_section = ""
    current_entry: dict[str, Any] | None = None
    for raw_line in sanitized.splitlines():
        line = raw_line.strip()
        if not line:
            current_entry = None
            continue
        if line.startswith("# "):
            data["name"] = line[2:].strip()
            continue
        if line.startswith("## "):
            current_section = line[3:].strip().lower()
            current_entry = None
            continue
        if not current_section and "|" in line:
            parts = [part.strip() for part in line.split("|")]
            data["contact"] = {
                "phone": parts[0] if len(parts) > 0 else "",
                "email": parts[1] if len(parts) > 1 else "",
                "linkedin": parts[2] if len(parts) > 2 else "",
            }
            continue
        if current_section == "education":
            if line.startswith("- "):
                line = line[2:].strip()
            if "|" in line:
                parts = [part.strip() for part in line.split("|") if part.strip()]
                data["education"].append(
                    {
                        "institution": parts[0] if parts else "",
                        "dates": parts[-1] if len(parts) > 1 and _looks_like_date_segment(parts[-1]) else "",
                        "detail": " | ".join(parts[1:-1] if len(parts) > 2 else parts[1:]),
                    }
                )
            elif data["education"]:
                data["education"][-1]["detail"] = line
            continue
        if current_section in {"experience", "project or research experience", "research experience", "projects"}:
            target = "experience" if current_section == "experience" else "project_or_research_experience"
            if line.startswith("- "):
                if current_entry is not None:
                    current_entry.setdefault("bullets", []).append(line[2:].strip())
                continue
            parts = [part.strip() for part in line.split("|") if part.strip()]
            current_entry = {
                "organization": parts[0] if parts else "",
                "role": parts[1] if len(parts) > 1 else "",
                "location": parts[2] if len(parts) > 2 else "",
                "dates": parts[3] if len(parts) > 3 else "",
                "bullets": [],
            }
            data[target].append(current_entry)
            continue
        if current_section == "skills & tools" and ":" in line:
            key, value = line.split(":", 1)
            data["skills"][key.strip()] = value.strip()
    return _normalize_structured_resume(data, profile)


def _resume_response_to_structured(raw: str, profile: dict[str, Any]) -> dict[str, Any]:
    payload = _extract_json_payload(raw)
    if payload is not None:
        return _normalize_structured_resume(payload, profile)
    return _markdown_resume_to_structured(raw, profile)


def _provider_api_key(provider: str = DEFAULT_RESUME_PROVIDER, api_key_override: str = "") -> str:
    if provider != "gemini":
        return ""
    if clean_text(api_key_override):
        return clean_text(api_key_override)
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _provider_model(provider: str = DEFAULT_RESUME_PROVIDER, model_override: str = "") -> str:
    if provider != "gemini":
        return ""
    if clean_text(model_override):
        return clean_text(model_override)
    for env_name in RESUME_MODEL_ENV_NAMES:
        configured = clean_text(os.getenv(env_name))
        if configured:
            return configured
    return DEFAULT_GEMINI_RESUME_MODEL


def resume_provider_status(*, api_key_override: str = "", model_override: str = "") -> dict[str, Any]:
    """Return API-connection state without exposing credentials."""

    provider = DEFAULT_RESUME_PROVIDER
    model = _provider_model(provider, model_override)
    api_key = _provider_api_key(provider, api_key_override)
    return {
        "available": bool(LLM_RESUME_CLIENT is not None or api_key),
        "provider": provider,
        "model": model,
        "reason": "" if (LLM_RESUME_CLIENT is not None or api_key) else "LLM API key is not configured.",
    }


def resume_docx_filename(job: dict[str, Any]) -> str:
    title = clean_text(job.get("title")) or "resume"
    company = clean_text(job.get("company") or job.get("employer"))
    raw = "-".join(part for part in [title, company] if part)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    return f"jobpilot-{slug or 'resume'}.docx"


def _profile_skills(profile: dict[str, Any], job: dict[str, Any]) -> list[str]:
    profile_skills = normalize_list(profile.get("skills"))
    matched = _job_skills(job)
    ordered: list[str] = []
    for skill in [*matched, *profile_skills]:
        if skill and skill.lower() not in {item.lower() for item in ordered}:
            ordered.append(skill)
    return ordered[:12]


def _summary(profile: dict[str, Any], job: dict[str, Any]) -> str:
    role = clean_text(job.get("title")) or "target role"
    company = clean_text(job.get("company") or job.get("employer")) or "the employer"
    roles = normalize_list(profile.get("target_roles"))
    role_phrase = roles[0] if roles else role
    skills = _profile_skills(profile, job)[:4]
    skill_phrase = ", ".join(skills) if skills else "data-driven problem solving"
    return (
        f"{role_phrase} candidate targeting {role} at {company}, with strengths in "
        f"{skill_phrase}. Background tailored to the posting's requirements while avoiding "
        "unsupported claims."
    )


def _resume_prompt(profile: dict[str, Any], job: dict[str, Any]) -> str:
    display_profile = _resume_display_profile(profile)
    profile_payload = {
        "name": clean_text(display_profile.get("name")) or "Candidate",
        "email": clean_text(display_profile.get("email")),
        "phone": clean_text(display_profile.get("phone")),
        "linkedin": clean_text(display_profile.get("linkedin")),
        "resume_source_text": _resume_source_text(display_profile.get("resume_source_text"))[:12000],
        "target_roles": normalize_list(display_profile.get("target_roles")),
        "skills": normalize_list(display_profile.get("skills")),
        "education": normalize_list(display_profile.get("education")),
        "experience": normalize_list(display_profile.get("experience_text")),
        "projects_publications": normalize_list(display_profile.get("projects_publications")),
        "location_preferences": normalize_list(display_profile.get("location_preferences")),
        "salary_min": display_profile.get("salary_min"),
        "salary_is_dealbreaker": bool(display_profile.get("salary_is_dealbreaker")),
        "dealbreakers": normalize_list(display_profile.get("dealbreakers")),
        "excluded_seniority": normalize_list(display_profile.get("excluded_seniority")),
        "excluded_employment_types": normalize_list(display_profile.get("excluded_employment_types")),
        "preferred_company_types": normalize_list(display_profile.get("preferred_company_types")),
        "excluded_company_types": normalize_list(display_profile.get("excluded_company_types")),
        "required_role_families": normalize_list(display_profile.get("required_role_families")),
        "preferred_role_families": normalize_list(display_profile.get("preferred_role_families")),
        "demo_resume_context": display_profile.get("demo_resume_context") if _is_demo_profile(display_profile) else {},
        "strict_role_family": bool(display_profile.get("strict_role_family")),
        "max_years_required": display_profile.get("max_years_required"),
        "needs_sponsorship": bool(display_profile.get("needs_sponsorship")),
        "visa_sponsorship": clean_text(display_profile.get("visa_sponsorship")),
        "avoid_defense_or_clearance": bool(display_profile.get("avoid_defense_or_clearance")),
    }
    job_payload = {
        "job_id": clean_text(job.get("job_id")),
        "title": clean_text(job.get("title")),
        "company": clean_text(job.get("company") or job.get("employer")),
        "location": clean_text(job.get("location")),
        "salary_min": clean_text(job.get("salary_min")),
        "salary_max": clean_text(job.get("salary_max")),
        "salary_raw": clean_text(job.get("salary_raw")),
        "employment_type": clean_text(job.get("employment_type")),
        "seniority": clean_text(job.get("seniority")),
        "years_required": clean_text(job.get("years_required")),
        "company_type": clean_text(job.get("company_type")),
        "sponsorship_signal": clean_text(job.get("sponsorship_signal")),
        "matched_skills": _job_skills(job),
        "why_ranked": job.get("why_ranked") if isinstance(job.get("why_ranked"), dict) else clean_text(job.get("why_ranked")),
        "application_strategy_label": application_strategy_display_label(job.get("application_strategy_label")),
        "source": clean_text(job.get("source")),
        "raw_source": clean_text(job.get("raw_source")),
        "description_text": clean_text(job.get("description_text"))[:5000],
    }
    return (
        "Return strict JSON content for a concise, ATS-friendly resume draft for this candidate and selected job.\n"
        "The backend will render the DOCX with a fixed Qixiang-style template; do not return Markdown or formatting instructions. "
        "The selected job is used only to decide emphasis, ordering, and wording.\n\n"
        "Format rules:\n"
        "- Return only one JSON object matching the schema below; no commentary, no code fences, no Markdown.\n"
        "- Use these content sections only: education, experience, project_or_research_experience, skills.\n"
        "- Do not add Professional Summary, Summary, Objective, Target, Profile, Additional Details, or Tailoring Notes.\n"
        "- Header/contact fields should contain only available phone, email, and LinkedIn. Skip missing contact fields.\n"
        "- Keep company, role, location, dates, school, degree, GPA, and metrics only when present in the user's evidence.\n"
        "- In education, put school in institution, date range in dates, and degree/GPA/coursework in detail.\n"
        "- In experience/project entries, split organization, role, location, and dates into separate JSON fields.\n"
        "- If a section has no evidence, omit the section; do not create a placeholder section.\n\n"
        "Tailoring rules:\n"
        "- The primary source is the user's own experience, projects, education, and skills.\n"
        "- For demo personas only, demo_resume_context is allowed evidence for a richer synthetic demo resume.\n"
        "- For demo personas without uploaded resume text, include enough content for a realistic one-page draft: usually 2 experience entries, 1-2 project/research entries, and 3-5 bullets per major experience when supported by demo_resume_context.\n"
        "- Prefer the raw source resume text when available; preserve the user's existing resume structure, employers, role names, and education facts.\n"
        "- Use the selected job title, company, matched skills, why-ranked evidence, and job description to decide what to emphasize, reorder, and rephrase.\n"
        "- Do not put the selected job title/company in the resume header as a target line.\n"
        "- The header should contain only contact information that is present in the profile evidence. Skip missing contact fields instead of inventing them.\n"
        "- Rewrite the user's existing evidence to better match the job; do not create new experience.\n"
        "- Keep bullets specific and evidence-backed; avoid generic filler that could apply to any candidate.\n"
        "- Treat filters, dealbreakers, sponsorship needs, salary, seniority limits, and company preferences as guardrails only. Do not turn them into resume content unless they are factual profile evidence.\n"
        "- Use only facts present in the candidate profile and job data.\n"
        "- Do not invent employers, degrees, metrics, certifications, publications, work authorization, or dates.\n"
        "- Do not mention persona labels, profile IDs, dealbreakers, or exclusions in the resume.\n"
        "- If evidence is missing for a bullet or section, omit that bullet or section instead of using a placeholder.\n"
        "- Do not include bracketed placeholder text or instructions in the final resume.\n"
        "- Do not include tailoring notes, verification reminders, commentary about missing evidence, or instructions to the user.\n"
        "- Use ASCII hyphen date ranges like 'Aug 2025 - Present'.\n\n"
        f"Fixed template handled by backend:\n{FIXED_RESUME_STYLE_TEMPLATE}\n\n"
        f"Required JSON schema:\n{json.dumps(RESUME_JSON_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"User resume/profile evidence JSON:\n{json.dumps(profile_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Selected job JSON:\n{json.dumps(job_payload, ensure_ascii=False, indent=2)}\n"
    )


def _gemini_resume_client(prompt: str, *, model: str, api_key: str) -> str:
    if not api_key:
        raise ResumeGenerationUnavailable("LLM API key is not configured.")
    if not model:
        raise ResumeGenerationUnavailable("LLM model is not configured.")
    encoded_model = urllib.parse.quote(model, safe="")
    encoded_key = urllib.parse.quote(api_key, safe="")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:generateContent?key={encoded_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return _clean_resume_markdown("\n".join(str(part.get("text", "")) for part in parts))


def structured_resume_to_markdown(resume: dict[str, Any]) -> str:
    """Convert normalized resume content into the fixed internal template."""

    lines: list[str] = []
    name = _resume_clean_text(resume.get("name")) or "Candidate"
    contact = resume.get("contact") if isinstance(resume.get("contact"), dict) else {}
    contact_line = _resume_contact_line(contact)
    lines.append(f"# {name}")
    if contact_line:
        lines.append(contact_line)
    lines.append("")

    education = resume.get("education") if isinstance(resume.get("education"), list) else []
    if education:
        lines.append("## EDUCATION")
        for item in education:
            if not isinstance(item, dict):
                continue
            institution = _resume_clean_text(item.get("institution"))
            dates = _resume_clean_text(item.get("dates"))
            detail = _resume_clean_text(item.get("detail"))
            if institution:
                lines.append(f"{institution} | {dates}" if dates else institution)
            if detail:
                lines.append(detail)
        lines.append("")

    experience = resume.get("experience") if isinstance(resume.get("experience"), list) else []
    if experience:
        lines.append("## EXPERIENCE")
        for item in experience:
            if not isinstance(item, dict):
                continue
            left = " | ".join(
                part for part in [_resume_clean_text(item.get("organization")), _resume_clean_text(item.get("role"))] if part
            )
            right = " | ".join(
                part for part in [_resume_clean_text(item.get("location")), _resume_clean_text(item.get("dates"))] if part
            )
            if left:
                lines.append(f"{left} | {right}" if right else left)
            for bullet in _clean_bullets(item.get("bullets"), limit=5):
                lines.append(f"- {bullet}")
        lines.append("")

    projects = (
        resume.get("project_or_research_experience")
        if isinstance(resume.get("project_or_research_experience"), list)
        else []
    )
    if projects:
        lines.append("## PROJECT OR RESEARCH EXPERIENCE")
        for item in projects:
            if not isinstance(item, dict):
                continue
            left = " | ".join(
                part for part in [_resume_clean_text(item.get("organization")), _resume_clean_text(item.get("role"))] if part
            )
            right = " | ".join(
                part for part in [_resume_clean_text(item.get("location")), _resume_clean_text(item.get("dates"))] if part
            )
            if left:
                lines.append(f"{left} | {right}" if right else left)
            for bullet in _clean_bullets(item.get("bullets"), limit=5):
                lines.append(f"- {bullet}")
        lines.append("")

    skills = resume.get("skills") if isinstance(resume.get("skills"), dict) else {}
    skill_lines = [(key, _resume_clean_text(value)) for key, value in skills.items() if _resume_clean_text(value)]
    if skill_lines:
        lines.append("## SKILLS & TOOLS")
        for key, value in skill_lines:
            lines.append(f"{_resume_clean_text(key)}: {value}")

    return "\n".join(lines).strip() + "\n"


def structured_resume_to_docx_bytes(resume: dict[str, Any]) -> bytes:
    """Render normalized resume content with the fixed Qixiang-style DOCX template."""

    return _qixiang_style_docx_bytes(resume)


RPR_NAME = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:bCs/>'
    '<w:sz w:val="31"/><w:szCs w:val="31"/></w:rPr>'
)
RPR_CONTACT = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
    '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
)
RPR_SECTION = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:bCs/>'
    '<w:color w:val="000000"/></w:rPr>'
)
RPR_ENTRY = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:bCs/>'
    '<w:sz w:val="23"/><w:szCs w:val="23"/></w:rPr>'
)
RPR_BODY = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
    '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
)
RPR_SKILLS = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
    '<w:sz w:val="23"/><w:szCs w:val="23"/></w:rPr>'
)

PPR_NAME = (
    "<w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"1\" "
    'w:color="000000"/></w:pBdr><w:ind w:right="50"/><w:jc w:val="center"/>'
    f"{RPR_NAME}</w:pPr>"
)
PPR_CONTACT = (
    "<w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"1\" "
    'w:color="000000"/></w:pBdr><w:ind w:right="50"/><w:jc w:val="center"/>'
    f"{RPR_CONTACT}</w:pPr>"
)
PPR_SECTION = (
    "<w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"1\" "
    'w:color="000000"/></w:pBdr><w:tabs><w:tab w:val="left" w:pos="360"/>'
    '<w:tab w:val="right" w:pos="10080"/></w:tabs>'
    '<w:spacing w:before="150" w:after="65" w:line="240" w:lineRule="auto"/><w:ind w:right="51"/>'
    f"{RPR_SECTION}</w:pPr>"
)
PPR_ENTRY = (
    '<w:pPr><w:tabs><w:tab w:val="left" w:pos="360"/>'
    '<w:tab w:val="right" w:pos="11469"/></w:tabs><w:spacing w:after="45" w:line="240" w:lineRule="auto"/><w:ind w:right="50"/>'
    f"{RPR_ENTRY}</w:pPr>"
)
PPR_EDU_DETAIL = (
    "<w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"1\" "
    'w:color="000000"/></w:pBdr><w:tabs><w:tab w:val="left" w:pos="360"/>'
    '<w:tab w:val="right" w:pos="10080"/></w:tabs>'
    '<w:spacing w:after="75" w:line="240" w:lineRule="auto"/><w:ind w:right="51"/>'
    f"{RPR_BODY}</w:pPr>"
)
PPR_BULLET = (
    '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="7"/></w:numPr>'
    '<w:tabs><w:tab w:val="left" w:pos="360"/><w:tab w:val="right" w:pos="11469"/></w:tabs>'
    '<w:spacing w:afterLines="30" w:after="75" w:line="240" w:lineRule="auto"/><w:ind w:right="51"/>'
    f"{RPR_BODY}</w:pPr>"
)
PPR_SKILLS = (
    '<w:pPr><w:tabs><w:tab w:val="right" w:pos="10080"/></w:tabs>'
    '<w:spacing w:after="30" w:line="240" w:lineRule="auto"/><w:ind w:right="50"/>'
    f"{RPR_SKILLS}</w:pPr>"
)
PPR_GROUP_GAP = '<w:pPr><w:spacing w:before="0" w:after="80" w:line="1" w:lineRule="exact"/><w:ind w:right="50"/></w:pPr>'


def _direct_paragraph(text: str = "", *, ppr: str = "", rpr: str = "") -> str:
    run = ""
    if text:
        run = f"<w:r>{rpr}<w:t xml:space=\"preserve\">{_xml_text(text)}</w:t></w:r>"
    return f"<w:p>{ppr}{run}</w:p>"


def _direct_tabbed_paragraph(left: str, right: str, *, ppr: str = PPR_ENTRY, rpr: str = RPR_ENTRY) -> str:
    left = _resume_clean_text(left)
    right = _resume_clean_text(right)
    if not right:
        return _direct_paragraph(left, ppr=ppr, rpr=rpr)
    return (
        f"<w:p>{ppr}"
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{_xml_text(left)}</w:t></w:r>"
        "<w:r><w:tab/></w:r>"
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{_xml_text(right)}</w:t></w:r>"
        "</w:p>"
    )


def _qixiang_style_document_xml(resume: dict[str, Any]) -> str:
    paragraphs: list[str] = []
    name = _resume_clean_text(resume.get("name")) or "Candidate"
    contact = resume.get("contact") if isinstance(resume.get("contact"), dict) else {}
    contact_line = _resume_contact_line(contact)

    paragraphs.append(_direct_paragraph(name, ppr=PPR_NAME, rpr=RPR_NAME))
    if contact_line:
        paragraphs.append(_direct_paragraph(contact_line, ppr=PPR_CONTACT, rpr=RPR_CONTACT))
    paragraphs.append(_direct_paragraph(ppr=PPR_CONTACT))

    education = resume.get("education") if isinstance(resume.get("education"), list) else []
    if education:
        paragraphs.append(_direct_paragraph("EDUCATION", ppr=PPR_SECTION, rpr=RPR_SECTION))
        for item in education:
            if not isinstance(item, dict):
                continue
            institution = _resume_clean_text(item.get("institution"))
            dates = _resume_clean_text(item.get("dates"))
            detail = _resume_clean_text(item.get("detail"))
            if institution:
                paragraphs.append(_direct_tabbed_paragraph(institution, dates, ppr=PPR_ENTRY, rpr=RPR_ENTRY))
            if detail:
                paragraphs.append(_direct_paragraph(detail, ppr=PPR_EDU_DETAIL, rpr=RPR_BODY))

    experience = resume.get("experience") if isinstance(resume.get("experience"), list) else []
    if experience:
        paragraphs.append(_direct_paragraph("EXPERIENCE", ppr=PPR_SECTION, rpr=RPR_SECTION))
        for index, item in enumerate(experience):
            if not isinstance(item, dict):
                continue
            left = " | ".join(
                part for part in [_resume_clean_text(item.get("organization")), _resume_clean_text(item.get("role"))] if part
            )
            right = " | ".join(
                part for part in [_resume_clean_text(item.get("location")), _resume_clean_text(item.get("dates"))] if part
            )
            if left:
                paragraphs.append(_direct_tabbed_paragraph(left, right, ppr=PPR_ENTRY, rpr=RPR_ENTRY))
            for bullet in _clean_bullets(item.get("bullets"), limit=5):
                paragraphs.append(_direct_paragraph(bullet, ppr=PPR_BULLET, rpr=RPR_BODY))
            if index < len(experience) - 1:
                paragraphs.append(_direct_paragraph(ppr=PPR_GROUP_GAP))

    projects = (
        resume.get("project_or_research_experience")
        if isinstance(resume.get("project_or_research_experience"), list)
        else []
    )
    if projects:
        paragraphs.append(_direct_paragraph("PROJECT OR RESEARCH EXPERIENCE", ppr=PPR_SECTION, rpr=RPR_SECTION))
        for index, item in enumerate(projects):
            if not isinstance(item, dict):
                continue
            left = " | ".join(
                part for part in [_resume_clean_text(item.get("organization")), _resume_clean_text(item.get("role"))] if part
            )
            right = " | ".join(
                part for part in [_resume_clean_text(item.get("location")), _resume_clean_text(item.get("dates"))] if part
            )
            if left:
                paragraphs.append(_direct_tabbed_paragraph(left, right, ppr=PPR_ENTRY, rpr=RPR_ENTRY))
            for bullet in _clean_bullets(item.get("bullets"), limit=5):
                paragraphs.append(_direct_paragraph(bullet, ppr=PPR_BULLET, rpr=RPR_BODY))
            if index < len(projects) - 1:
                paragraphs.append(_direct_paragraph(ppr=PPR_GROUP_GAP))

    skills = resume.get("skills") if isinstance(resume.get("skills"), dict) else {}
    skill_lines = [(key, _resume_clean_text(value)) for key, value in skills.items() if _resume_clean_text(value)]
    if skill_lines:
        paragraphs.append(_direct_paragraph("SKILLS & TOOLS", ppr=PPR_SECTION, rpr=RPR_SECTION))
        for key, value in skill_lines:
            paragraphs.append(_direct_paragraph(f"{_resume_clean_text(key)}: {value}", ppr=PPR_SKILLS, rpr=RPR_SKILLS))

    body = "\n".join(paragraphs)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="454" w:right="454" w:bottom="454" w:left="454" w:header="720" w:footer="720" w:gutter="0"/>
      <w:pgNumType w:start="1"/>
      <w:cols w:space="720"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def _qixiang_numbering_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="3">
    <w:multiLevelType w:val="multilevel"/>
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="&#61548;"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="420" w:hanging="420"/></w:pPr>
      <w:rPr><w:rFonts w:ascii="Wingdings" w:hAnsi="Wingdings" w:hint="default"/><w:sz w:val="15"/><w:szCs w:val="15"/></w:rPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="7"><w:abstractNumId w:val="3"/></w:num>
</w:numbering>
"""


def _qixiang_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="21"/>
        <w:szCs w:val="21"/>
        <w:color w:val="000000"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
</w:styles>
"""


def _docx_package_bytes(document_xml: str, styles_xml: str, numbering_xml: str) -> bytes:
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>
"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels_xml)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/numbering.xml", numbering_xml)
    return buffer.getvalue()


def _qixiang_style_docx_bytes(resume: dict[str, Any]) -> bytes:
    return _docx_package_bytes(_qixiang_style_document_xml(resume), _qixiang_styles_xml(), _qixiang_numbering_xml())


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
ElementTree.register_namespace("w", W_NS)


def _resolve_resume_source_docx_path(profile: dict[str, Any]) -> Path | None:
    raw_path = _resume_clean_text(profile.get("resume_source_docx_path"))
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        candidate = candidate.resolve()
        upload_root = UPLOAD_DIR.resolve()
    except OSError:
        return None
    if candidate.suffix.lower() != ".docx" or not candidate.exists():
        return None
    if candidate != upload_root and upload_root not in candidate.parents:
        return None
    return candidate


def _source_paragraph_text(paragraph: ElementTree.Element) -> str:
    namespace = f"{{{W_NS}}}"
    parts: list[str] = []
    for run in paragraph.findall(f"{namespace}r"):
        if run.find(f"{namespace}tab") is not None:
            parts.append("\t")
        for node in run.findall(f"{namespace}t"):
            parts.append(node.text or "")
    return "".join(parts)


def _source_paragraph_has_num(paragraph: ElementTree.Element) -> bool:
    return paragraph.find(f"./{{{W_NS}}}pPr/{{{W_NS}}}numPr") is not None


def _source_paragraph_is_entry(text: str) -> bool:
    cleaned = _resume_clean_text(text)
    return " | " in cleaned and _looks_like_date_segment(cleaned)


def _replace_source_paragraph_text(paragraph: ElementTree.Element, text: str, *, prefix: str = "") -> None:
    namespace = f"{{{W_NS}}}"
    text = _resume_clean_text(text)
    first_run_rpr = paragraph.find(f"./{namespace}r/{namespace}rPr")
    paragraph_rpr = paragraph.find(f"./{namespace}pPr/{namespace}rPr")
    template_rpr = first_run_rpr if first_run_rpr is not None else paragraph_rpr
    for child in list(paragraph):
        if child.tag == f"{namespace}r":
            paragraph.remove(child)
    run = ElementTree.Element(f"{namespace}r")
    if template_rpr is not None:
        run.append(deepcopy(template_rpr))
    text_node = ElementTree.SubElement(run, f"{namespace}t")
    text_node.set(XML_SPACE, "preserve")
    text_node.text = f"{prefix}{text}" if text else ""
    insert_at = 1 if len(paragraph) and paragraph[0].tag == f"{namespace}pPr" else 0
    paragraph.insert(insert_at, run)


def _resume_bullets_by_section(resume: dict[str, Any]) -> dict[str, list[str]]:
    experience: list[str] = []
    for item in resume.get("experience") if isinstance(resume.get("experience"), list) else []:
        if isinstance(item, dict):
            experience.extend(_clean_bullets(item.get("bullets"), limit=5))
    research: list[str] = []
    projects = (
        resume.get("project_or_research_experience")
        if isinstance(resume.get("project_or_research_experience"), list)
        else []
    )
    for item in projects:
        if isinstance(item, dict):
            research.extend(_clean_bullets(item.get("bullets"), limit=5))
    return {"experience": experience, "research": research}


def _resume_skill_lines_for_source(resume: dict[str, Any], target_count: int) -> list[str]:
    skills = resume.get("skills") if isinstance(resume.get("skills"), dict) else {}
    lines = [f"{_resume_clean_text(key)}: {_resume_clean_text(value)}" for key, value in skills.items() if _resume_clean_text(value)]
    if target_count == 2 and len(lines) >= 3:
        return [lines[0], f"{lines[1]} {lines[2]}"]
    return lines[:target_count]


def preserve_source_docx_layout(source_docx_bytes: bytes, resume: dict[str, Any]) -> bytes:
    """Patch text into the uploaded DOCX while preserving original Word layout."""

    source_buffer = BytesIO(source_docx_bytes)
    with zipfile.ZipFile(source_buffer) as source_archive:
        root = ElementTree.fromstring(source_archive.read("word/document.xml"))
        paragraphs = root.findall(f".//{{{W_NS}}}body/{{{W_NS}}}p")
        section = ""
        bullet_targets: list[tuple[ElementTree.Element, str, bool]] = []
        skill_targets: list[ElementTree.Element] = []
        for paragraph in paragraphs:
            text = _resume_clean_text(_source_paragraph_text(paragraph))
            upper = text.upper()
            if upper == "EXPERIENCE":
                section = "experience"
                continue
            if upper in {"RESEARCH EXPERIENCE", "PROJECT OR RESEARCH EXPERIENCE"}:
                section = "research"
                continue
            if upper == "SKILLS & TOOLS":
                section = "skills"
                continue
            if upper == "EDUCATION":
                section = "education"
                continue
            if not text:
                continue
            if section in {"experience", "research"} and not _source_paragraph_is_entry(text):
                bullet_targets.append((paragraph, section, _source_paragraph_has_num(paragraph)))
            elif section == "skills":
                skill_targets.append(paragraph)

        bullets = _resume_bullets_by_section(resume)
        bullet_indexes = {"experience": 0, "research": 0}
        for paragraph, target_section, has_num in bullet_targets:
            candidates = bullets[target_section]
            index = bullet_indexes[target_section]
            if index >= len(candidates):
                continue
            prefix = "" if has_num else "\u2022 "
            _replace_source_paragraph_text(paragraph, candidates[index], prefix=prefix)
            bullet_indexes[target_section] += 1

        for paragraph, line in zip(skill_targets, _resume_skill_lines_for_source(resume, len(skill_targets))):
            _replace_source_paragraph_text(paragraph, line)

        updated_document_xml = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
        output = BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as output_archive:
            for info in source_archive.infolist():
                if info.filename == "word/document.xml":
                    output_archive.writestr(info, updated_document_xml)
                else:
                    output_archive.writestr(info, source_archive.read(info.filename))
        return output.getvalue()


def markdown_resume_to_docx_bytes(markdown: str) -> bytes:
    """Convert fixed-style resume Markdown into a lightweight DOCX file."""

    paragraphs_xml: list[str] = []
    subtitle_pending = False
    for line in markdown.splitlines():
        stripped = line.strip()
        is_subtitle = bool(subtitle_pending and stripped and not stripped.startswith("#") and not stripped.startswith("- "))
        paragraphs_xml.append(_markdown_line_to_docx_paragraph(line, subtitle=is_subtitle))
        if stripped.startswith("# "):
            subtitle_pending = True
        elif is_subtitle:
            subtitle_pending = False
    paragraphs = "\n".join(paragraphs_xml)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {paragraphs}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="454" w:right="454" w:bottom="454" w:left="454" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="21"/>
        <w:color w:val="000000"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr><w:spacing w:after="40" w:line="240" w:lineRule="auto"/><w:ind w:right="50"/></w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:after="40" w:line="240" w:lineRule="auto"/><w:ind w:right="50"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="21"/><w:color w:val="000000"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="0" w:after="20"/>
      <w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="000000"/></w:pBdr>
      <w:ind w:right="50"/>
    </w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="31"/><w:color w:val="000000"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="80" w:line="240" w:lineRule="auto"/>
      <w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="000000"/></w:pBdr>
      <w:ind w:right="50"/>
    </w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="21"/><w:color w:val="000000"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="150" w:after="65" w:line="240" w:lineRule="auto"/>
      <w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="000000"/></w:pBdr>
      <w:ind w:right="50"/>
    </w:pPr>
    <w:rPr><w:b/><w:caps/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:afterLines="30" w:after="75" w:line="240" w:lineRule="auto"/>
      <w:ind w:left="300" w:hanging="180" w:right="50"/>
    </w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="21"/><w:color w:val="000000"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="EntryLine">
    <w:name w:val="Entry Line"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:after="45" w:line="240" w:lineRule="auto"/>
      <w:ind w:right="50"/>
    </w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="23"/><w:color w:val="000000"/></w:rPr>
  </w:style>
</w:styles>
"""
    numbering_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0">
    <w:multiLevelType w:val="singleLevel"/>
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="&#8226;"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="300" w:hanging="180"/></w:pPr>
      <w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol" w:hint="default"/></w:rPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>
"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels_xml)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/numbering.xml", numbering_xml)
    return buffer.getvalue()


def generate_api_resume(
    profile: dict[str, Any],
    job: dict[str, Any],
    *,
    api_key_override: str = "",
    model_override: str = "",
) -> dict[str, Any]:
    """Generate a resume only when an LLM API is connected."""

    status = resume_provider_status(api_key_override=api_key_override, model_override=model_override)
    if not status["available"]:
        raise ResumeGenerationUnavailable("Resume generation requires a connected LLM API.")
    prompt = _resume_prompt(profile, job)
    provider = str(status["provider"])
    model = str(status["model"])
    if LLM_RESUME_CLIENT is not None:
        draft = _clean_resume_markdown(LLM_RESUME_CLIENT(prompt))
    elif provider == "gemini":
        draft = _gemini_resume_client(prompt, model=model, api_key=_provider_api_key(provider, api_key_override))
    else:
        raise ResumeGenerationUnavailable(f"Unsupported resume provider: {provider}")
    if not draft:
        raise RuntimeError("LLM resume provider returned an empty draft.")
    structured = _resume_response_to_structured(draft, profile)
    structured = _expand_demo_structured_resume(structured, _resume_display_profile(profile), job)
    draft = structured_resume_to_markdown(structured)
    mode = "fixed_template"
    source_docx_path = _resolve_resume_source_docx_path(profile)
    docx_bytes: bytes
    if source_docx_path:
        try:
            docx_bytes = preserve_source_docx_layout(source_docx_path.read_bytes(), structured)
            mode = "source_docx_preserve"
        except (KeyError, zipfile.BadZipFile, ElementTree.ParseError, OSError):
            docx_bytes = structured_resume_to_docx_bytes(structured)
    else:
        docx_bytes = structured_resume_to_docx_bytes(structured)
    return {
        "draft": draft.strip() + "\n",
        "docx_bytes": docx_bytes,
        "provider": provider,
        "model": model,
        "mode": mode,
    }


def generate_rule_based_resume(profile: dict[str, Any], job: dict[str, Any]) -> str:
    """Return a local markdown scaffold; not exposed by the web route."""

    display_profile = _resume_display_profile(profile)
    name = clean_text(display_profile.get("name")) or "Candidate"
    contact_line = " | ".join(
        part
        for part in [
            clean_text(display_profile.get("phone")),
            clean_text(display_profile.get("email")),
            clean_text(display_profile.get("linkedin")),
        ]
        if part
    )
    title = clean_text(job.get("title")) or "Selected Role"
    company = clean_text(job.get("company") or job.get("employer")) or "Selected Employer"
    skills = _profile_skills(display_profile, job)
    education = _as_lines(normalize_list(display_profile.get("education")))
    experience = _as_lines(normalize_list(display_profile.get("experience_text")))
    projects = _as_lines(normalize_list(display_profile.get("projects_publications")))
    matched = _job_skills(job)

    sections: list[str] = [
        f"# {name}",
        contact_line,
        "",
        "## EDUCATION",
    ]

    for line in education[:3]:
        sections.append(f"- {line}")

    sections.extend(["", "## EXPERIENCE"])
    for line in experience[:5]:
        sections.append(f"- {line}")

    sections.extend(["", "## PROJECT OR RESEARCH EXPERIENCE"])
    for line in projects[:5]:
        sections.append(f"- {line}")

    sections.extend(
        [
            "",
            "## SKILLS & TOOLS",
            f"Data & Tools: {', '.join(skills) if skills else 'Add profile-supported tools'}",
        ]
    )
    return "\n".join(part for part in sections if part is not None).strip() + "\n"
