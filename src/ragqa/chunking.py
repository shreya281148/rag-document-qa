"""Step 2: split pages into overlapping chunks.

Why chunk at all? An embedding represents the *meaning of one piece of text*.
If a chunk is a whole 30-page report, its embedding is a blurry average and
retrieval is poor. If a chunk is one sentence, it loses context. So we aim for a
few paragraphs (`chunk_size` characters) and let neighbouring chunks overlap
(`chunk_overlap`) so an answer sitting on a boundary is not cut in half.

This is a hand-written "recursive" splitter: it prefers to cut at paragraph
breaks, then line breaks, then sentence ends, then spaces, and only as a last
resort in the middle of a word.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .loaders import Page

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    id: str  # deterministic: re-indexing the same text never creates duplicates
    source: str
    page: int
    index: int  # position of the chunk within its page
    text: str


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into pieces of at most `chunk_size` chars with `chunk_overlap` overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _recursive_split(text, chunk_size, _SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces that each fit in chunk_size, cutting at the best boundary."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:  # no boundary left: hard cut
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _recursive_split(text, chunk_size, rest)

    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Put the separator back (except after the last part) so no characters are lost.
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_recursive_split(piece, chunk_size, rest))
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack small pieces into chunks, starting each new chunk with the previous tail."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the next chunk at a word boundary so we don't begin mid-word
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]
            current = tail + piece
            if len(current) > chunk_size:  # overlap + piece too big: drop the overlap
                current = piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every page, attaching source/page metadata and a stable id."""
    out: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, chunk_size, chunk_overlap)):
            digest = hashlib.sha1(f"{page.source}|{page.page}|{i}|{text}".encode()).hexdigest()[:16]
            out.append(Chunk(id=digest, source=page.source, page=page.page, index=i, text=text))
    return out
