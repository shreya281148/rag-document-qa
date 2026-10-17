"""Headless test of the Streamlit app using Streamlit's own AppTest framework."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml: defaults + env overrides
    monkeypatch.setenv("RAGQA_EMBEDDINGS_PROVIDER", "hash")
    monkeypatch.setenv("RAGQA_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAGQA_STORAGE_PERSIST_DIR", str(tmp_path / "db"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def click(at, label):
    next(b for b in at.sidebar.button if b.label == label).click().run()


def test_starts_empty_and_blocks_chat(app):
    assert any("No documents indexed" in i.value for i in app.sidebar.info)
    assert app.chat_input[0].disabled is True


def test_load_samples_then_ask_shows_answer_and_sources(app):
    click(app, "Load samples")
    assert any("Indexed 3 sample file(s)" in s.value for s in app.sidebar.success)
    assert app.chat_input[0].disabled is False

    app.chat_input[0].set_value("How many days of annual leave do full-time employees get?").run()
    assert not app.exception
    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    assistant_text = " ".join(m.value for m in app.chat_message[1].markdown)
    assert "28" in assistant_text
    labels = [e.label for e in app.chat_message[1].expander]
    assert any("Sources (4; 1 cited in the answer)" in label for label in labels)


def test_unrelated_question_is_refused_in_the_ui(app):
    click(app, "Load samples")
    app.chat_input[0].set_value("zzzz qqqq xxxx").run()  # shares no words with any document
    assistant_text = " ".join(m.value for m in app.chat_message[1].markdown)
    assert "could not find" in assistant_text.lower()


def test_clear_index_resets_everything(app):
    click(app, "Load samples")
    app.chat_input[0].set_value("How often do passwords expire?").run()
    click(app, "Clear index")
    assert any("No documents indexed" in i.value for i in app.sidebar.info)
    assert len(app.chat_message) == 0


def test_invalid_chunk_settings_are_rejected(app):
    sliders = {s.label: s for s in app.sidebar.slider}
    sliders["Chunk size (characters)"].set_value(200)
    sliders["Chunk overlap (characters)"].set_value(300).run()
    assert any("smaller than the chunk size" in e.value for e in app.sidebar.error)
