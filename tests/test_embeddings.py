import math

import pytest

from ragqa.config import EmbeddingConfig
from ragqa.embeddings import HashEmbedder, OllamaEmbedder
from ragqa.errors import BackendUnavailable, ModelNotFound


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder()
    v1, v2 = e.embed_query("Refunds take 14 days"), e.embed_query("Refunds take 14 days")
    assert v1 == v2
    assert len(v1) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)


def test_hash_embedder_ranks_related_text_higher():
    e = HashEmbedder()
    q = e.embed_query("how many vacation days per year")
    related = e.embed_documents(["Employees get 28 vacation days per year."])[0]
    unrelated = e.embed_documents(["The server room temperature must stay below 24 degrees."])[0]
    assert cosine(q, related) > cosine(q, unrelated)


def test_hash_embedder_handles_empty_text():
    assert set(HashEmbedder().embed_query("")) == {0.0}


def test_ollama_embedder_sends_prefixes_and_model(fake_ollama):
    cfg = EmbeddingConfig(base_url=fake_ollama.base_url, model="nomic-embed-text")
    e = OllamaEmbedder(cfg)
    e.embed_documents(["alpha beta"])
    e.embed_query("what is alpha")
    (path1, body1), (path2, body2) = fake_ollama.requests
    assert path1 == path2 == "/api/embed"
    assert body1["model"] == "nomic-embed-text"
    assert body1["input"] == ["search_document: alpha beta"]
    assert body2["input"] == ["search_query: what is alpha"]


def test_ollama_embedder_batches_requests(fake_ollama):
    e = OllamaEmbedder(EmbeddingConfig(base_url=fake_ollama.base_url))
    vectors = e.embed_documents([f"text {i}" for i in range(70)])
    assert len(vectors) == 70
    assert len(fake_ollama.requests) == 3  # 32 + 32 + 6


def test_ollama_embedder_missing_model_gives_helpful_error(fake_ollama):
    e = OllamaEmbedder(EmbeddingConfig(base_url=fake_ollama.base_url, model="missing"))
    with pytest.raises(ModelNotFound, match="ollama pull missing"):
        e.embed_query("hi")


def test_ollama_embedder_server_error_and_unreachable(fake_ollama):
    with pytest.raises(BackendUnavailable, match="500"):
        OllamaEmbedder(EmbeddingConfig(base_url=fake_ollama.base_url, model="boom")).embed_query("hi")
    with pytest.raises(BackendUnavailable, match="ollama serve"):
        OllamaEmbedder(EmbeddingConfig(base_url="http://127.0.0.1:9", timeout_s=2)).embed_query("hi")
