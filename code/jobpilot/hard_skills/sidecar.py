"""Build Phase 2.16B hard-skill extraction sidecars.

The module is deliberately offline/batch-only. It does not alter the canonical
Phase 1 snapshot and it does not wire model output into ranking.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Protocol

from jobpilot.config import OFFLINE_SNAPSHOT_CSV, PROCESSED_DATA_DIR
from jobpilot.hard_skills.normalization import (
    NormalizationResult,
    dictionary_terms,
    normalize_hard_skill,
)
from jobpilot.utils.io import ensure_parent, write_json
from jobpilot.utils.text import clean_text, stable_hash


PARSER_VERSION = "phase2_16b_hard_skill_sidecar_v0"
DEFAULT_MODEL_NAME = "jjzha/escoxlmr_knowledge_extraction"
DEFAULT_MODEL_REVISION = "11ff79181beba00102094005d0408a2d894e3092"
JOBBERT_KNOWLEDGE_MODEL = "jjzha/jobbert_knowledge_extraction"
JOBBERT_KNOWLEDGE_REVISION = "9dea9759b7e290413cb4d92e8cae338affab9be4"

HARD_SKILL_SIDECAR_DIR = PROCESSED_DATA_DIR / "hard_skill_sidecar"
HARD_SKILL_SIDECAR_JSONL = HARD_SKILL_SIDECAR_DIR / "phase2_16b_hard_skill_sidecar.jsonl"
HARD_SKILL_SIDECAR_MANIFEST_JSON = HARD_SKILL_SIDECAR_DIR / "phase2_16b_hard_skill_manifest.json"
HARD_SKILL_SIDECAR_REPORT_MD = HARD_SKILL_SIDECAR_DIR / "phase2_16b_hard_skill_report.md"

DEFAULT_INPUT_FIELDS = [
    "title",
    "raw_skills",
    "schema_org_skills",
    "raw_keywords",
    "raw_requirements",
    "description_text",
]
DEFAULT_CONTEXT_FIELDS = [
    "raw_jobnames",
    "schema_org_occupational_category",
    "schema_org_experience_requirements",
]
SOURCE_BACKED_SKILL_FIELDS = {
    "raw_skills",
    "schema_org_skills",
    "raw_keywords",
    "raw_requirements",
}
EXCLUDED_LEGACY_FIELDS = [
    "extracted_skills",
    "normalized_skills",
    "normalized_keywords",
    "seniority",
    "years_required",
    "is_remote",
    "company_type",
    "sponsorship_signal",
    "structured_signal_confidence",
]
BOUNDARY_NOTE = (
    "Phase 2.16B builds an offline hard-skill sidecar only. It does not modify "
    "Phase 1 ingestion or snapshot generation, does not change ranking, does "
    "not use legacy semantic columns as model inputs or labels, and keeps soft "
    "skills out of ranking."
)

try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(10**8)


@dataclass(frozen=True)
class FieldInput:
    row_index: int
    job_id: str
    source_field: str
    text: str


@dataclass(frozen=True)
class RawEntity:
    text: str
    start: int
    end: int
    confidence: float
    model_label: str


class EntityExtractor(Protocol):
    backend_name: str
    model_name: str
    model_revision: str
    chunking: dict[str, Any]

    def extract_batch(self, fields: list[FieldInput]) -> list[list[RawEntity]]:
        ...


class DictionaryHardSkillExtractor:
    """Local no-model baseline used for smoke tests and audit fallback."""

    backend_name = "dictionary"
    model_name = "jobpilot_static_hard_skill_dictionary"
    model_revision = PARSER_VERSION
    chunking = {
        "strategy": "regex_dictionary_no_transformer",
        "max_tokens": None,
        "stride": None,
    }

    def __init__(self) -> None:
        self.terms = dictionary_terms()
        self.patterns = [
            (
                term,
                re.compile(r"(?<![a-z0-9+#.])" + re.escape(term) + r"(?![a-z0-9+#.])", re.IGNORECASE),
            )
            for term in self.terms
            if term.strip()
        ]

    def extract_batch(self, fields: list[FieldInput]) -> list[list[RawEntity]]:
        return [self._extract_field(field) for field in fields]

    def _extract_field(self, field: FieldInput) -> list[RawEntity]:
        entities: list[RawEntity] = []
        seen: set[tuple[int, int, str]] = set()
        text = field.text

        if field.source_field in {"raw_skills", "schema_org_skills"}:
            for match in re.finditer(r"[^|,;]+", text):
                surface = clean_text(match.group(0))
                normalized = normalize_hard_skill(surface)
                if not normalized.normalized_text:
                    continue
                key = (match.start(), match.end(), normalized.normalized_text)
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        RawEntity(
                            text=surface,
                            start=match.start(),
                            end=match.end(),
                            confidence=1.0,
                            model_label="SOURCE_STRUCTURED",
                        )
                    )

        for term, pattern in self.patterns:
            for match in pattern.finditer(text):
                surface = text[match.start() : match.end()]
                normalized = normalize_hard_skill(surface)
                if not normalized.accepted:
                    continue
                key = (match.start(), match.end(), normalized.normalized_text)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    RawEntity(
                        text=surface,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                        model_label="DICTIONARY",
                    )
                )

        return sorted(entities, key=lambda item: (item.start, item.end, item.text.lower()))


class HuggingFaceKnowledgeExtractor:
    """Hugging Face token-classification adapter for ESCOXLM-R/JobBERT."""

    backend_name = "hf"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        allow_download: bool = False,
        device: str = "auto",
        batch_size: int = 4,
        stride: int = 64,
        aggregation_strategy: str = "simple",
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision or ""
        self.batch_size = batch_size
        self.stride = stride
        self.aggregation_strategy = aggregation_strategy
        self.chunking = {
            "strategy": "huggingface_token_classification_pipeline",
            "max_tokens": 512,
            "stride": stride,
            "pipeline_aggregation_strategy": "none",
            "span_aggregation_strategy": "bio_offsets",
            "field_level_offsets": True,
        }

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Phase 2.16B Hugging Face backend requires local torch and transformers. "
                "Use --backend dictionary for smoke tests or install the optional offline dependencies."
            ) from exc

        local_files_only = not allow_download
        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if self.model_revision:
            kwargs["revision"] = self.model_revision

        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = AutoModelForTokenClassification.from_pretrained(model_name, **kwargs)

        if device == "auto":
            device_id = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            device_id = -1
        else:
            device_id = int(device)

        self.pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="none",
            ignore_labels=["O"],
            device=device_id,
            batch_size=batch_size,
        )

    def extract_batch(self, fields: list[FieldInput]) -> list[list[RawEntity]]:
        if not fields:
            return []
        texts = [field.text for field in fields]
        try:
            outputs = self.pipe(texts, stride=self.stride, truncation=True, batch_size=self.batch_size)
        except TypeError:
            outputs = self.pipe(texts, batch_size=self.batch_size)
        if len(fields) == 1 and outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        return [self._convert_entities(field, output) for field, output in zip(fields, outputs)]

    def _convert_entities(self, field: FieldInput, output: list[dict[str, Any]]) -> list[RawEntity]:
        tokens: list[dict[str, Any]] = []
        for entity in output:
            text = clean_text(entity.get("word"))
            start = entity.get("start")
            end = entity.get("end")
            if start is None or end is None:
                start, end = _find_surface_offset(field.text, text)
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            if end_i <= start_i:
                continue
            label = clean_text(entity.get("entity") or entity.get("entity_group") or "")
            if label == "O":
                continue
            tokens.append(
                {
                    "start": start_i,
                    "end": end_i,
                    "label": label,
                    "score": float(entity.get("score", 0.0) or 0.0),
                }
            )

        tokens.sort(key=lambda item: (item["start"], item["end"], item["label"]))
        entities: list[RawEntity] = []
        current: dict[str, Any] | None = None

        def flush_current() -> None:
            nonlocal current
            if not current:
                return
            start_i = int(current["start"])
            end_i = int(current["end"])
            if end_i <= start_i:
                current = None
                return
            evidence = field.text[start_i:end_i]
            scores = current["scores"]
            labels = current["labels"]
            entities.append(
                RawEntity(
                    text=clean_text(evidence),
                    start=start_i,
                    end=end_i,
                    confidence=sum(scores) / len(scores) if scores else 0.0,
                    model_label="/".join(labels),
                )
            )
            current = None

        for token in tokens:
            label = clean_text(token["label"])
            start_i = int(token["start"])
            end_i = int(token["end"])
            if current and start_i < int(current["end"]) and end_i <= int(current["end"]):
                # Duplicate token from an overlapping stride window.
                continue
            gap = field.text[int(current["end"]) : start_i] if current is not None else ""
            should_continue = (
                current is not None
                and start_i >= int(current["end"])
                and (label == "I" or start_i == int(current["end"]))
                and clean_text(gap) == ""
            )
            if current is None:
                current = {"start": start_i, "end": end_i, "scores": [token["score"]], "labels": [label]}
            elif not should_continue:
                flush_current()
                current = {"start": start_i, "end": end_i, "scores": [token["score"]], "labels": [label]}
            else:
                current["end"] = end_i
                current["scores"].append(token["score"])
                current["labels"].append(label)
        flush_current()
        return entities


def _find_surface_offset(text: str, surface: str) -> tuple[int, int]:
    if not surface:
        return 0, 0
    match = re.search(re.escape(surface), text, re.IGNORECASE)
    if not match:
        return 0, len(surface)
    return match.start(), match.end()


def _iter_snapshot_rows(path: Path, *, limit: int | None = None, offset: int = 0) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        yielded = 0
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if limit is not None and yielded >= limit:
                break
            yielded += 1
            yield dict(row)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_field_text(value: Any) -> str:
    return clean_text(value)


def _row_input_hash(row: dict[str, Any], input_fields: list[str], context_fields: list[str]) -> str:
    payload = {
        field: _clean_field_text(row.get(field))
        for field in input_fields + context_fields
        if _clean_field_text(row.get(field))
    }
    return stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _coverage_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _field_inputs_for_row(
    row: dict[str, Any],
    *,
    row_index: int,
    input_fields: list[str],
    description_policy: str,
) -> list[FieldInput]:
    job_id = clean_text(row.get("job_id")) or f"row_{row_index}"
    fields: list[FieldInput] = []
    has_source_backed_skill_text = any(_clean_field_text(row.get(field)) for field in SOURCE_BACKED_SKILL_FIELDS)
    for source_field in input_fields:
        if source_field == "description_text":
            if description_policy == "never":
                continue
            if description_policy == "when_no_source_skills" and has_source_backed_skill_text:
                continue
        text = _clean_field_text(row.get(source_field))
        if text:
            fields.append(FieldInput(row_index=row_index, job_id=job_id, source_field=source_field, text=text))
    return fields


def _span_payload(
    entity: RawEntity,
    *,
    field: FieldInput,
    min_confidence: float,
    accept_surface_only: bool,
) -> tuple[dict[str, Any], NormalizationResult, bool]:
    normalized = normalize_hard_skill(entity.text)
    low_confidence = entity.confidence < min_confidence
    surface_only = normalized.normalization_status == "surface_only"
    accepted = normalized.accepted and not low_confidence and (accept_surface_only or not surface_only)
    drop_reason = (
        normalized.drop_reason
        or ("low_confidence" if low_confidence else "")
        or ("not_in_canonical_dictionary" if surface_only and not accept_surface_only else "")
    )
    payload = {
        "text": entity.text,
        "normalized_text": normalized.normalized_text,
        "label": "knowledge",
        "model_label": entity.model_label,
        "confidence": round(entity.confidence, 6),
        "start": entity.start,
        "end": entity.end,
        "source_field": field.source_field,
        "evidence": field.text[entity.start : entity.end]
        if 0 <= entity.start < entity.end <= len(field.text)
        else entity.text,
        "accepted": accepted,
        "drop_reason": drop_reason,
        "normalization_status": normalized.normalization_status,
    }
    return payload, normalized, accepted


def _dedupe_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_exact: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for span in spans:
        key = (
            clean_text(span.get("source_field")),
            int(span.get("start", 0) or 0),
            int(span.get("end", 0) or 0),
            clean_text(span.get("normalized_text") or span.get("text")).lower(),
        )
        current = by_exact.get(key)
        if not current or float(span.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            by_exact[key] = span

    candidates = sorted(
        by_exact.values(),
        key=lambda item: (
            clean_text(item.get("source_field")),
            int(item.get("start", 0) or 0),
            -(int(item.get("end", 0) or 0) - int(item.get("start", 0) or 0)),
            -float(item.get("confidence", 0.0) or 0.0),
        ),
    )
    kept: list[dict[str, Any]] = []
    for span in candidates:
        field = clean_text(span.get("source_field"))
        start = int(span.get("start", 0) or 0)
        end = int(span.get("end", 0) or 0)
        normalized = clean_text(span.get("normalized_text"))
        overlap = False
        for existing in kept:
            if clean_text(existing.get("source_field")) != field:
                continue
            if clean_text(existing.get("normalized_text")) != normalized:
                continue
            existing_start = int(existing.get("start", 0) or 0)
            existing_end = int(existing.get("end", 0) or 0)
            if max(start, existing_start) < min(end, existing_end):
                overlap = True
                break
        if not overlap:
            kept.append(span)
    return sorted(kept, key=lambda item: (clean_text(item.get("source_field")), int(item.get("start", 0) or 0)))


def _sidecar_payload_for_row(
    row: dict[str, Any],
    *,
    row_index: int,
    fields: list[FieldInput],
    extracted: list[list[RawEntity]],
    extractor: EntityExtractor,
    input_fields: list[str],
    context_fields: list[str],
    min_confidence: float,
    accept_surface_only: bool,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Counter[str] | int]]:
    raw_spans: list[dict[str, Any]] = []
    dropped_terms: list[dict[str, Any]] = []
    normalized_skills: list[str] = []
    normalized_seen: set[str] = set()
    audit_flags: set[str] = {"legacy_semantic_fields_excluded", "ranking_weight_zero_until_future_phase"}
    stats: dict[str, Counter[str] | int] = {
        "source_fields": Counter(),
        "drop_reasons": Counter(),
        "normalization_statuses": Counter(),
        "accepted": 0,
        "spans": 0,
    }

    for field, entities in zip(fields, extracted):
        for entity in entities:
            span, normalized, accepted = _span_payload(
                entity,
                field=field,
                min_confidence=min_confidence,
                accept_surface_only=accept_surface_only,
            )
            raw_spans.append(span)
            stats["source_fields"][field.source_field] += 1  # type: ignore[index]
            stats["normalization_statuses"][normalized.normalization_status] += 1  # type: ignore[index]
            if accepted and normalized.normalized_text not in normalized_seen:
                normalized_seen.add(normalized.normalized_text)
                normalized_skills.append(normalized.normalized_text)
                stats["accepted"] = int(stats["accepted"]) + 1
            elif not accepted:
                reason = clean_text(span.get("drop_reason")) or "not_accepted"
                stats["drop_reasons"][reason] += 1  # type: ignore[index]
                dropped_terms.append(
                    {
                        "text": span["text"],
                        "normalized_text": span["normalized_text"],
                        "reason": reason,
                        "source_field": span["source_field"],
                        "start": span["start"],
                        "end": span["end"],
                    }
                )

    hard_skill_spans = _dedupe_spans(raw_spans)
    stats["spans"] = len(hard_skill_spans)
    if not hard_skill_spans:
        audit_flags.add("no_model_spans")
    if hard_skill_spans and not normalized_skills:
        audit_flags.add("all_spans_dropped")
    if not normalized_skills:
        audit_flags.add("no_accepted_hard_skills")
    if dropped_terms:
        audit_flags.add("some_terms_dropped")

    confidences = [float(span["confidence"]) for span in hard_skill_spans]
    summary = {
        "span_count": len(hard_skill_spans),
        "accepted_count": len(normalized_skills),
        "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
        "min_confidence": round(min(confidences), 6) if confidences else 0.0,
        "max_confidence": round(max(confidences), 6) if confidences else 0.0,
        "source_field_counts": dict(sorted(stats["source_fields"].items())),  # type: ignore[union-attr]
    }

    actual_input_fields = [field.source_field for field in fields]
    context_available = [field for field in context_fields if _clean_field_text(row.get(field))]
    payload = {
        "job_id": clean_text(row.get("job_id")) or f"row_{row_index}",
        "parser_version": PARSER_VERSION,
        "model_name": extractor.model_name,
        "model_revision": extractor.model_revision,
        "backend": extractor.backend_name,
        "input_fields_used": actual_input_fields,
        "context_fields_available": context_available,
        "excluded_legacy_fields": EXCLUDED_LEGACY_FIELDS,
        "input_hash": _row_input_hash(row, input_fields, context_fields),
        "chunking": extractor.chunking,
        "hard_skill_spans": hard_skill_spans,
        "normalized_hard_skills": normalized_skills,
        "dropped_generic_terms": [
            item for item in dropped_terms if item["reason"] in {"too_generic", "soft_skill", "low_confidence"}
        ],
        "alias_mapping_source": "jobpilot_static_skill_aliases_v0",
        "surface_only_terms_accepted": accept_surface_only,
        "extraction_confidence_summary": summary,
        "audit_flags": sorted(audit_flags),
        "soft_skills_ranking_weight": 0,
        "generated_at": generated_at,
    }
    return payload, stats


def build_hard_skill_sidecar(
    *,
    snapshot_path: Path = OFFLINE_SNAPSHOT_CSV,
    output_path: Path = HARD_SKILL_SIDECAR_JSONL,
    manifest_path: Path = HARD_SKILL_SIDECAR_MANIFEST_JSON,
    report_path: Path = HARD_SKILL_SIDECAR_REPORT_MD,
    extractor: EntityExtractor | None = None,
    input_fields: list[str] | None = None,
    context_fields: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    row_batch_size: int = 16,
    min_confidence: float = 0.5,
    accept_surface_only: bool = False,
    description_policy: str = "always",
    progress_every: int = 0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the Phase 2.16B sidecar JSONL, manifest, and report."""

    if description_policy not in {"always", "when_no_source_skills", "never"}:
        raise ValueError("description_policy must be always, when_no_source_skills, or never")

    extractor = extractor or DictionaryHardSkillExtractor()
    input_fields = list(input_fields or DEFAULT_INPUT_FIELDS)
    context_fields = list(context_fields or DEFAULT_CONTEXT_FIELDS)
    generated_at = generated_at or _utc_now()

    total_rows = 0
    rows_with_accepted = 0
    spans_total = 0
    accepted_total = 0
    normalized_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()
    drop_reason_counter: Counter[str] = Counter()
    source_field_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    field_coverage: Counter[str] = Counter()
    audit_flag_counter: Counter[str] = Counter()
    next_progress = progress_every if progress_every > 0 else 0

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as output:
        pending_rows: list[tuple[int, dict[str, str]]] = []

        def flush() -> None:
            nonlocal total_rows, rows_with_accepted, spans_total, accepted_total, next_progress
            if not pending_rows:
                return
            fields: list[FieldInput] = []
            row_field_counts: dict[int, int] = {}
            for row_index, row in pending_rows:
                row_fields = _field_inputs_for_row(
                    row,
                    row_index=row_index,
                    input_fields=input_fields,
                    description_policy=description_policy,
                )
                row_field_counts[row_index] = len(row_fields)
                fields.extend(row_fields)

            extracted_all = extractor.extract_batch(fields) if fields else []
            cursor = 0
            for row_index, row in pending_rows:
                count = row_field_counts[row_index]
                row_fields = fields[cursor : cursor + count]
                row_extracted = extracted_all[cursor : cursor + count]
                cursor += count
                payload, stats = _sidecar_payload_for_row(
                    row,
                    row_index=row_index,
                    fields=row_fields,
                    extracted=row_extracted,
                    extractor=extractor,
                    input_fields=input_fields,
                    context_fields=context_fields,
                    min_confidence=min_confidence,
                    accept_surface_only=accept_surface_only,
                    generated_at=generated_at,
                )
                output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                total_rows += 1
                spans_total += int(stats["spans"])
                accepted_count = len(payload["normalized_hard_skills"])
                accepted_total += accepted_count
                rows_with_accepted += int(accepted_count > 0)
                source_counter[clean_text(row.get("source")) or "unknown"] += 1
                for field in input_fields:
                    field_coverage[field] += int(bool(_clean_field_text(row.get(field))))
                source_field_counter.update(payload["extraction_confidence_summary"]["source_field_counts"])
                normalized_counter.update(payload["normalized_hard_skills"])
                drop_reason_counter.update(stats["drop_reasons"])  # type: ignore[arg-type]
                for item in payload["dropped_generic_terms"]:
                    dropped_counter[item["normalized_text"] or item["text"]] += 1
                audit_flag_counter.update(payload["audit_flags"])
                if progress_every > 0 and total_rows >= next_progress:
                    print(
                        f"Phase 2.16B progress: processed {total_rows:,} rows, "
                        f"accepted rows {rows_with_accepted:,}, spans {spans_total:,}",
                        flush=True,
                    )
                    next_progress += progress_every

            pending_rows.clear()

        for row_index, row in enumerate(_iter_snapshot_rows(snapshot_path, limit=limit, offset=offset), start=offset):
            pending_rows.append((row_index, row))
            if len(pending_rows) >= row_batch_size:
                flush()
        flush()

    manifest: dict[str, Any] = {
        "generated_at": generated_at,
        "parser_version": PARSER_VERSION,
        "snapshot_path": str(snapshot_path),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "rows_written": total_rows,
        "limit": limit,
        "offset": offset,
        "backend": extractor.backend_name,
        "model_name": extractor.model_name,
        "model_revision": extractor.model_revision,
        "input_fields": input_fields,
        "context_fields": context_fields,
        "excluded_legacy_fields": EXCLUDED_LEGACY_FIELDS,
        "chunking": extractor.chunking,
        "min_confidence": min_confidence,
        "surface_only_terms_accepted": accept_surface_only,
        "description_policy": description_policy,
        "source_backed_skill_fields": sorted(SOURCE_BACKED_SKILL_FIELDS),
        "progress_every": progress_every,
        "row_batch_size": row_batch_size,
        "rows_with_normalized_hard_skills": rows_with_accepted,
        "rows_with_normalized_hard_skills_rate": _coverage_rate(rows_with_accepted, total_rows),
        "hard_skill_span_count": spans_total,
        "normalized_hard_skill_count": accepted_total,
        "mean_normalized_hard_skills_per_row": round(accepted_total / total_rows, 4) if total_rows else 0.0,
        "input_field_coverage": {
            field: {
                "rows_present": field_coverage[field],
                "rate": _coverage_rate(field_coverage[field], total_rows),
            }
            for field in input_fields
        },
        "source_counts": dict(source_counter.most_common()),
        "span_source_field_counts": dict(source_field_counter.most_common()),
        "top_normalized_hard_skills": dict(normalized_counter.most_common(50)),
        "top_dropped_terms": dict(dropped_counter.most_common(30)),
        "drop_reason_counts": dict(drop_reason_counter.most_common()),
        "audit_flag_counts": dict(audit_flag_counter.most_common()),
        "phase1_snapshot_modified": False,
        "phase1_ingestion_modified": False,
        "ranking_behavior_changed": False,
        "cloud_run_online_inference_enabled": False,
        "paid_apis_or_live_scraping_used": False,
        "legacy_semantic_fields_used_as_inputs": False,
        "extracted_skills_used_as_gold_labels": False,
        "soft_skills_used_in_ranking": False,
        "boundary_note": BOUNDARY_NOTE,
    }
    write_json(manifest_path, manifest)
    write_hard_skill_report(report_path, manifest)
    return manifest


def write_hard_skill_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2.16B Hard-Skill Sidecar Report",
        "",
        BOUNDARY_NOTE,
        "",
        "## Summary",
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Backend: `{manifest['backend']}`",
        f"- Model: `{manifest['model_name']}`",
        f"- Model revision: `{manifest['model_revision']}`",
        f"- Rows written: {manifest['rows_written']:,}",
        f"- Rows with normalized hard skills: {manifest['rows_with_normalized_hard_skills']:,} "
        f"({manifest['rows_with_normalized_hard_skills_rate']:.1%})",
        f"- Hard-skill spans emitted: {manifest['hard_skill_span_count']:,}",
        f"- Normalized hard-skill assignments: {manifest['normalized_hard_skill_count']:,}",
    ]
    if manifest.get("description_policy"):
        lines.append(f"- Description policy: `{manifest['description_policy']}`")
    runtime_chunking = manifest.get("runtime_chunking") or {}
    if runtime_chunking:
        lines.append(f"- Runtime chunking: `{runtime_chunking.get('strategy', 'unknown')}`")

    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Phase 1 ingestion and snapshot generation are not modified.",
            "- Ranking behavior is not modified.",
            "- Cloud Run does not run ESCOXLM-R in the online request path by default.",
            "- No paid APIs, live job scraping, or Phase 1 legacy semantic labels are required.",
            "- Soft skills are not ranking features and have ranking weight 0.",
            "- Phase 1 `extracted_skills` is excluded from model inputs and is not a gold label.",
            "",
            "## Input Fields",
            "",
            "| Field | Rows present | Rate |",
            "|---|---:|---:|",
        ]
    )
    for field, payload in manifest["input_field_coverage"].items():
        lines.append(f"| `{field}` | {payload['rows_present']:,} | {payload['rate']:.1%} |")

    lines.extend(["", "## Top Normalized Hard Skills", "", "| Skill | Count |", "|---|---:|"])
    for skill, count in manifest["top_normalized_hard_skills"].items():
        lines.append(f"| `{skill}` | {count:,} |")

    lines.extend(["", "## Dropped Terms", "", "| Term | Count |", "|---|---:|"])
    for term, count in manifest["top_dropped_terms"].items():
        lines.append(f"| `{term}` | {count:,} |")

    lines.extend(["", "## Audit Flags", "", "| Flag | Rows |", "|---|---:|"])
    for flag, count in manifest["audit_flag_counts"].items():
        lines.append(f"| `{flag}` | {count:,} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
