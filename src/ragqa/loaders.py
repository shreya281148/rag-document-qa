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


_COLUMN_GAP = re.compile(r" {6,}")


def _looks_multi_column(layout_text: str) -> bool:
    """True if a page laid out by position looks like genuine side-by-side text columns.

    In "layout" mode a two-column article comes out with a left line and a right line glued
    together on every row, which scrambles the sentences. A CV or an invoice also has wide gaps
    (a right-aligned date, table cells), but there the text segments are short. So a row only
    counts as "multi-column" when it has TWO OR MORE long text segments separated by wide gaps,
    and a page is multi-column when many of its rows are like that.
    """
    rows = [r.replace("\t", " ").strip() for r in layout_text.splitlines() if r.strip()]
    if len(rows) < 6:
        return False
    two_sided = 0
    for row in rows:
        segments = [seg for seg in _COLUMN_GAP.split(row) if len(seg.strip()) >= 25]
        if len(segments) >= 2:
            two_sided += 1
    return two_sided / len(rows) >= 0.3


def _page_text(page) -> str:
    """Extract one PDF page's text in the best reading order.

    pypdf's default mode returns text in the order the PDF file stores it. For CVs and tables
    that can put a heading in one place and its bullet points somewhere else, so a question about
    a job never finds its details in the same passage. "layout" mode sorts text by its position on
    the page, which keeps rows and headings together. But it is wrong for real multi-column pages,
    where the default mode reads one column at a time. So: use layout mode unless the page looks
    multi-column, and fall back to the default mode if layout extraction fails or finds nothing.
    """
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
        if layout.strip() and not _looks_multi_column(layout):
            return layout
    except Exception:  # layout mode is best-effort; never lose a page because of it
        pass
    return page.extract_text() or ""


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader  # imported lazily so text-only use needs no PDF stack

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(_page_text(page))
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
