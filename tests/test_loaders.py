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
