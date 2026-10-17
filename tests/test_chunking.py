import pytest

from ragqa.chunking import chunk_pages, split_text
from ragqa.loaders import Page

PARAGRAPHS = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(12))


def test_short_text_is_one_chunk():
    assert split_text("Just a short note.", 500, 50) == ["Just a short note."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ", 500, 50) == []


def test_chunks_never_exceed_size():
    for chunk in split_text(PARAGRAPHS, 300, 40):
        assert len(chunk) <= 300


UNIQUE_WORDS = " ".join(f"w{i:04d}" for i in range(400))  # every word appears exactly once


def _shared_words(a: str, b: str) -> int:
    """How many of the words that start chunk b were already at the end of chunk a."""
    tail = a.split()
    return sum(1 for w in b.split()[:15] if w in tail)


def test_neighbouring_chunks_overlap():
    chunks = split_text(UNIQUE_WORDS, 200, 50)
    assert len(chunks) > 5
    for a, b in zip(chunks, chunks[1:]):
        shared = _shared_words(a, b)
        assert 3 <= shared <= 10  # roughly 50 chars of 6-char words, trimmed to a word boundary


def test_zero_overlap_means_no_repeated_words():
    chunks = split_text(UNIQUE_WORDS, 200, 0)
    for a, b in zip(chunks, chunks[1:]):
        assert _shared_words(a, b) == 0


def test_no_content_is_lost():
    chunks = split_text(PARAGRAPHS, 300, 40)
    joined = " ".join(chunks)
    for i in range(12):
        assert f"Paragraph {i}." in joined


def test_prefers_paragraph_boundaries():
    text = ("A" * 90 + "\n\n") + ("B" * 90 + "\n\n") + ("C" * 90)
    chunks = split_text(text, 100, 0)
    assert chunks == ["A" * 90, "B" * 90, "C" * 90]  # cut between paragraphs, never mid-paragraph


def test_hard_cut_when_no_separator_exists():
    chunks = split_text("x" * 250, 100, 0)
    assert [len(c) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("hello world", size, overlap)


def test_chunk_pages_attach_metadata_and_stable_ids():
    pages = [Page("a.txt", 1, "hello world"), Page("b.pdf", 7, PARAGRAPHS)]
    first = chunk_pages(pages, 300, 40)
    second = chunk_pages(pages, 300, 40)
    assert [c.id for c in first] == [c.id for c in second]  # deterministic
    assert len({c.id for c in first}) == len(first)  # unique
    assert first[0].source == "a.txt" and first[0].page == 1
    assert all(c.source == "b.pdf" and c.page == 7 for c in first[1:])
    assert [c.index for c in first[1:]] == list(range(len(first) - 1))
