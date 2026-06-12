"""Run Phase 2 matching for one persona, PDF, or structured profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_CSV, PROCESSED_DATA_DIR  # noqa: E402
from jobpilot.profile.pdf_extractor import PDFExtractionError, extract_pdf_text  # noqa: E402
from jobpilot.profile.personas import PERSONA_FIXTURES, get_persona  # noqa: E402
from jobpilot.profile.profile_parser import load_profile_json, parse_profile_text  # noqa: E402
from jobpilot.ranking.ranker import rank_jobs_for_profile  # noqa: E402
from jobpilot.utils.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JobPilot Phase 2 matching.")
    parser.add_argument("--persona", choices=sorted(PERSONA_FIXTURES), help="Built-in evaluation persona.")
    parser.add_argument("--profile-json", type=Path, help="Structured profile JSON fallback/input.")
    parser.add_argument("--resume-pdf", type=Path, help="Resume PDF to extract and parse.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--snapshot", type=Path, default=OFFLINE_SNAPSHOT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "tfidf-svd"], default="auto")
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_profile(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    if args.persona:
        return get_persona(args.persona), notes
    if args.profile_json:
        return load_profile_json(args.profile_json), notes
    if args.resume_pdf:
        try:
            text = extract_pdf_text(args.resume_pdf)
            return parse_profile_text(text), notes
        except PDFExtractionError as exc:
            notes.append(str(exc))
            raise SystemExit("PDF extraction failed and no structured --profile-json fallback was provided.") from exc
    raise SystemExit("Provide --persona, --profile-json, or --resume-pdf.")


def default_output_path(args: argparse.Namespace, profile: dict[str, Any]) -> Path:
    if args.output:
        return args.output
    profile_id = str(profile.get("profile_id") or args.persona or "profile").replace(" ", "_")
    return PROCESSED_DATA_DIR / f"phase2_matching_{profile_id}.json"


def main() -> int:
    args = parse_args()
    profile, notes = load_profile(args)
    result = rank_jobs_for_profile(
        profile,
        snapshot_path=args.snapshot,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        embedding_backend=args.embedding_backend,
        rebuild_embeddings=args.rebuild_embeddings,
    )
    payload = {
        "profile": profile,
        "notes": notes,
        **result,
    }
    output_path = default_output_path(args, profile)
    write_json(output_path, payload)

    print(f"Phase 2 matching complete for profile: {profile.get('profile_id')}")
    print(f"Output: {output_path}")
    print(f"Embedding backend: {result['metadata'].get('embedding_backend')}")
    print(f"ANN backend: {result['metadata'].get('ann_backend')}")
    for job in result["top_jobs"][: args.top_k]:
        print(
            f"{job['rank']:>2}. {job['title']} | {job['company']} | "
            f"score={job['final_score']:.3f} | {job['why_ranked']['summary']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
