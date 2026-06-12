# CareerPilot

CareerPilot is a portfolio-safe export of my JobPilot final project: a local-first job matching and resume-support web app that turns job postings and a candidate profile into ranked matches, explanations, feedback-aware reranking, analytics, CSV export, and optional LLM-assisted resume drafting.

The repository is intended for GitHub review. It includes source code, runnable app paths, tests, documentation, benchmark artifacts, and a small public sample dataset. It intentionally excludes private API keys, raw provider payloads, local runtime sessions, full embedding caches, and the full 50,000-row offline snapshot.

## What It Demonstrates

- End-to-end product workflow from profile intake to ranked job cards, feedback, export, analytics, and resume generation.
- Full-stack app architecture using FastAPI, Jinja templates, Python services, static frontend assets, and local storage boundaries.
- Data ingestion and normalization for multi-source job postings, including schema fidelity, deduplication, source-backed fields, and data dictionary outputs.
- Retrieval and ranking architecture with dense/vector-style retrieval, sklearn fallback, hard filters, explanation fields, and a runtime-safe learned reranker artifact.
- Evaluation discipline through benchmark reports, ranking comparison artifacts, persona simulations, and smoke tests.
- Packaging judgment: this public export keeps the app runnable with sample data while excluding private/raw/heavy artifacts.

## Repository Layout

```text
app/                         FastAPI routes, templates, static assets, services
code/jobpilot/               Core ingestion, profile, retrieval, ranking, evidence, and utility modules
data/processed/              Public-safe sample data, reports, manifests, and benchmark outputs
docs/                        Architecture notes for portfolio reviewers
scripts/                     Reproducible pipeline, benchmark, analytics, and readiness scripts
tests/                       Focused smoke and behavior tests
brief.md / brief.pdf         Final project write-up
Dockerfile                   Container runtime for the web app
```

## Local Run

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "code"
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

If the full offline snapshot is not present, the app falls back to `data/processed/jobs_offline_snapshot_sample_500.csv` for smoke-test matching and analytics. Optional Gemini/API-powered profile reading, live refresh, and resume generation require user-provided keys at runtime; the default matching path does not require private credentials.

## Validation

The public export was checked with:

```powershell
$env:PYTHONPATH = "code"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q tests\test_ingestion_dedup.py tests\test_phase2_18a_profile_intake.py tests\test_phase3_app.py
```

Current result: `43 passed`.

## Public Data Boundary

Included:

- 500-row public-safe processed snapshot sample for local smoke tests.
- Ingestion report and data dictionary.
- Ranking and reranking benchmark artifacts.
- Company metadata and sponsorship-evidence manifests.
- Feedback simulation and persona simulation summaries.

Excluded:

- Raw provider payloads.
- Full 50,000-row snapshot.
- Embedding cache directories.
- Local session/upload storage.
- API keys, `.env` files, ZIP packages, logs, and private handoff material.

## Portfolio Framing

This project is best described as a course final project and personal portfolio project for applied AI/product automation, full-stack software engineering, and data pipeline work. The strongest interview angle is the system design tradeoff: keep the default path offline and reproducible, while allowing optional live APIs and LLM features without making private services mandatory for review.

