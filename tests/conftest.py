from pathlib import Path

import pytest

from fake_ollama import FakeOllama
from ragqa.config import Config

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "sample_docs"


@pytest.fixture
def sample_docs() -> Path:
    return SAMPLE_DOCS


@pytest.fixture
def offline_cfg(tmp_path) -> Config:
    """Config that needs no model server: hash embedder + extractive answerer, temp index."""
    cfg = Config()
    cfg.embeddings.provider = "hash"
    cfg.llm.provider = "extractive"
    cfg.storage.persist_dir = str(tmp_path / "chroma")
    return cfg


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    yield server
    server.close()
