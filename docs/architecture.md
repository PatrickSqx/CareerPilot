# CareerPilot Architecture

CareerPilot is organized around a local-first review workflow:

1. Ingest and normalize job postings into a processed snapshot.
2. Parse or manually build a candidate profile.
3. Retrieve candidate jobs from the processed snapshot.
4. Apply hard filters, ranking features, reranking, and explanation builders.
5. Show ranked jobs in a FastAPI web app with feedback, CSV export, analytics, and optional resume generation.

## Main Layers

| Layer | Purpose | Representative files |
| --- | --- | --- |
| Web app | Routes, forms, result cards, analytics, feedback, resume download | `app/main.py`, `app/templates/`, `app/static/` |
| App services | Matching, profile parsing, analytics loading, feedback/rerank, resume service | `app/services/` |
| Data ingestion | Source normalization, cleaning, deduplication, quality checks | `code/jobpilot/ingestion/`, `scripts/run_phase1_pipeline.py` |
| Profile logic | Persona fixtures, profile parser, structured filters | `code/jobpilot/profile/`, `app/services/profile_parse_service.py` |
| Retrieval/ranking | Embeddings, fallback retrieval, hard filters, scoring, explanations | `code/jobpilot/retrieval/`, `code/jobpilot/ranking/` |
| Reranking evidence | Runtime-safe learned reranker artifacts and evidence manifests | `code/jobpilot/evidence/`, `data/processed/phase2_18j_*` |
| Evaluation | Benchmarks, persona simulations, feedback simulation | `scripts/run_phase2_benchmarks.py`, `data/processed/ranking_eval/`, `data/processed/phase3_feedback_simulation.json` |
| Packaging | Docker app, public sample data, readiness checks | `Dockerfile`, `.dockerignore`, `.gitignore`, `scripts/check_final_readiness.py` |

## Public Export Boundary

The GitHub version is designed to be reviewable without private dependencies. It keeps code and evidence artifacts, but excludes raw provider data, credentials, the full offline snapshot, local sessions, upload outputs, and embedding caches.

When the full snapshot is absent, matching and analytics use `data/processed/jobs_offline_snapshot_sample_500.csv` so reviewers can still run the app and smoke tests locally.

## Design Tradeoffs

- The default workflow is offline and reproducible; live job refresh and LLM features are optional.
- User-visible explanations are separated from hidden ranking internals.
- Hard filters remain authoritative before learned reranking so dealbreakers are not overridden by model scores.
- Runtime feedback affects later session ranking but is not used as a same-row training label.
- Public packaging favors verifiable artifacts and small sample data over shipping heavy or private caches.

