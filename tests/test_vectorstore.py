import pytest

from ragqa.chunking import Chunk
from ragqa.embeddings import HashEmbedder
from ragqa.vectorstore import ChromaStore


def chunk(i, text, source="a.txt", page=1):
    return Chunk(id=f"id{i}", source=source, page=page, index=i, text=text)


@pytest.fixture
def store(tmp_path):
    return ChromaStore(str(tmp_path / "db"), HashEmbedder().id)


def fill(store):
    emb = HashEmbedder()
    chunks = [
        chunk(1, "Employees receive 28 days of annual leave."),
        chunk(2, "Passwords must be 14 characters long and expire every 180 days.", source="it.txt", page=2),
        chunk(3, "Business flights are booked in economy class."),
    ]
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    return emb


def test_query_returns_nearest_first_with_metadata(store):
    emb = fill(store)
    hits = store.query(emb.embed_query("how many days of annual leave"), top_k=3)
    assert hits[0].source == "a.txt" and "28 days" in hits[0].text
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert {h.page for h in hits} == {1, 2}


def test_top_k_is_capped_by_collection_size(store):
    emb = fill(store)
    assert len(store.query(emb.embed_query("anything"), top_k=50)) == 3


def test_empty_store_returns_no_hits(store):
    assert store.query(HashEmbedder().embed_query("anything"), top_k=4) == []


def test_upsert_is_idempotent(store):
    fill(store)
    fill(store)
    assert store.count() == 3


def test_sources_delete_and_reset(store):
    fill(store)
    assert store.sources() == {"a.txt": 2, "it.txt": 1}
    store.delete_source("it.txt")
    assert store.sources() == {"a.txt": 2}
    store.reset()
    assert store.count() == 0


def test_length_mismatch_is_rejected(store):
    with pytest.raises(ValueError):
        store.upsert([chunk(1, "x")], [])


def test_different_embedders_never_share_a_collection(tmp_path):
    a = ChromaStore(str(tmp_path / "db"), "embedder-A")
    b = ChromaStore(str(tmp_path / "db"), "embedder-B")
    fill(a)
    assert a.count() == 3 and b.count() == 0
