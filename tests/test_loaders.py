import pytest

from ragqa.loaders import clean_text, load_file, load_paths


def test_clean_text_collapses_tabs_and_keeps_paragraphs():
    assert clean_text("Hello\tworld    again\r\n\r\n\r\n\r\nNext   para") == "Hello world again\n\nNext para"


def test_load_text_and_markdown(tmp_path):
    (tmp_path / "a.txt").write_text("plain text here")
    (tmp_path / "b.md").write_text("# Title\n\nbody")
    assert load_file(tmp_path / "a.txt")[0].text == "plain text here"
    assert load_file(tmp_path / "b.md")[0].source == "b.md"


def test_empty_file_gives_no_pages(tmp_path):
    (tmp_path / "e.txt").write_text("   \n ")
    assert load_file(tmp_path / "e.txt") == []


def test_pdf_pages_keep_numbers_and_clean_text(sample_docs):
    pages = load_file(sample_docs / "local_llm_guide.pdf")
    assert [p.page for p in pages] == [1, 2, 3]
    assert all("\t" not in p.text for p in pages)
    assert "Retrieval-Augmented Generation" in pages[1].text


def test_unsupported_and_missing_files(tmp_path):
    (tmp_path / "x.docx").write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(tmp_path / "x.docx")
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "missing.txt")


def test_folder_is_searched_recursively_and_filtered(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub" / "b.md").write_text("beta")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    assert sorted(p.source for p in load_paths([tmp_path])) == ["a.txt", "b.md"]


class _FakePage:
    """Mimics a pypdf page: default mode returns scrambled order, layout mode reading order."""

    def __init__(self, layout=None, layout_error=False):
        self.layout, self.layout_error = layout, layout_error

    def extract_text(self, extraction_mode=None):
        if extraction_mode == "layout":
            if self.layout_error:
                raise RuntimeError("layout failed")
            return self.layout
        return "Wipro\nJob heading\n...elsewhere...\nBest Trainee award"


def test_pdf_text_uses_layout_mode_to_keep_headings_with_their_bullets():
    from ragqa.loaders import _page_text

    page = _FakePage(layout="Job heading - Wipro\n- Best Trainee award")
    assert _page_text(page) == "Job heading - Wipro\n- Best Trainee award"


def test_pdf_text_falls_back_to_default_mode_if_layout_fails_or_is_empty():
    from ragqa.loaders import _page_text

    assert "elsewhere" in _page_text(_FakePage(layout_error=True))
    assert "elsewhere" in _page_text(_FakePage(layout="   \n "))


def test_multi_column_detection_distinguishes_columns_from_tables_and_cvs():
    from ragqa.loaders import _looks_multi_column

    gap = " " * 10
    two_columns = "\n".join(
        f"left column sentence number {i} is here{gap}right column sentence number {i} is here" for i in range(8)
    )
    cv_with_dates = "\n".join(f"Job title {i} at some company{gap}Jul 2025 - Dec 2025\n- a bullet point" for i in range(4))
    table = "\n".join(f"Consulting hours{gap}12{gap}95.00 EUR{gap}1,140.00 EUR" for _ in range(8))
    assert _looks_multi_column(two_columns) is True
    assert _looks_multi_column(cv_with_dates) is False  # long text on the left only
    assert _looks_multi_column(table) is False  # many short cells
    assert _looks_multi_column("short\npage") is False


def test_page_text_uses_default_mode_for_multi_column_pages():
    from ragqa.loaders import _page_text

    gap = " " * 10
    columns = "\n".join(
        f"left column sentence number {i} is here{gap}right column sentence number {i} is here" for i in range(8)
    )
    page = _FakePage(layout=columns)
    assert "elsewhere" in _page_text(page)  # the default-mode text, not the interleaved layout text
