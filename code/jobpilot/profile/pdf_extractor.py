"""Resume PDF text extraction with dependency fallbacks."""

from __future__ import annotations

from pathlib import Path


class PDFExtractionError(RuntimeError):
    """Raised when no available PDF parser can extract usable text."""


def _extract_with_pymupdf(path: Path) -> str:
    import fitz  # type: ignore

    parts: list[str] = []
    with fitz.open(path) as document:
        for page in document:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()


def _extract_with_pdfplumber(path: Path) -> str:
    import pdfplumber  # type: ignore

    parts: list[str] = []
    with pdfplumber.open(path) as document:
        for page in document.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_pdf_text(pdf_path: str | Path, *, min_chars: int = 100) -> str:
    """Extract text from a resume PDF.

    PyMuPDF is tried first, then pdfplumber. Both are optional dependencies.
    Callers should catch PDFExtractionError and fall back to structured input.
    """

    path = Path(pdf_path)
    if not path.exists():
        raise PDFExtractionError(f"PDF not found: {path}")

    errors: list[str] = []
    for extractor in (_extract_with_pymupdf, _extract_with_pdfplumber):
        try:
            text = extractor(path)
        except Exception as exc:  # pragma: no cover - depends on local optional libs
            errors.append(f"{extractor.__name__}: {type(exc).__name__}: {exc}")
            continue
        if len(text.strip()) >= min_chars:
            return text
        errors.append(f"{extractor.__name__}: extracted fewer than {min_chars} chars")

    raise PDFExtractionError("PDF extraction failed; use structured profile fallback. " + " | ".join(errors))
