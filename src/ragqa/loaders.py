"""Step 1 of the pipeline: turn files into plain text with provenance.

Every piece of text keeps (source file, page number) so that, later, an answer
can point back to exactly where it came from. That traceability is what makes
RAG answers checkable instead of "trust me".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

_INLINE_SPACE = re.compile(r"[ \t ]+")


def clean_text(text: str) -> str:
    """Normalise whitespace. PDF extraction often puts tabs or odd spaces between words, which
    would otherwise leak into chunks and embeddings. Line breaks are kept (they mark structure)."""
    lines = (_INLINE_SPACE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


@dataclass(frozen=True)
class Page:
    """A block of text from one document. `page` is 1-based (always 1 for text files)."""

    source: str
    page: int
    text: str


def load_file(path: str | Path) -> list[Page]:
    """Read one file and return its pages. Raises ValueError for unsupported types."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}")
    if not p.is_file():
        raise FileNotFoundError(p)

    if suffix == ".pdf":
        return _load_pdf(p)
    text = clean_text(p.read_text(encoding="utf-8", errors="replace"))
    return [Page(p.name, 1, text)] if text else []


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader  # imported lazily so text-only use needs no PDF stack

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:  # scanned/image-only pages have no text layer and are skipped
            pages.append(Page(path.name, number, text))
    return pages


def load_paths(paths: list[str | Path]) -> list[Page]:
    """Load files and/or folders (folders are searched recursively)."""
    pages: list[Page] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES)
            for f in files:
                pages.extend(load_file(f))
        else:
            pages.extend(load_file(p))
    return pages
