"""Step 4: store vectors and find the nearest ones (the "vector database").

We use ChromaDB, running embedded inside this program and saving to a local
folder. Nothing is sent anywhere, and telemetry is switched off.

Similarity: we ask Chroma for cosine distance and convert it to a score where
1.0 means "identical direction" and 0.0 means "unrelated".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk plus how similar it was to the question."""

    id: str
    source: str
    page: int
    index: int
    text: str
    score: float


class ChromaStore:
    def __init__(self, persist_dir: str, embedder_id: str, telemetry: bool = False):
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=telemetry))
        # One collection per embedder: vectors from different models are not comparable
        # (and may even have different lengths), so they must never share a collection.
        name = "ragqa_" + hashlib.sha1(embedder_id.encode()).hexdigest()[:12]
        self._col = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    # -- writing
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._col.upsert(  # upsert = insert-or-update, so re-indexing a file is safe
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page, "index": c.index} for c in chunks],
        )

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})

    def reset(self) -> None:
        got = self._col.get()
        if got["ids"]:
            self._col.delete(ids=got["ids"])

    # -- reading
    def query(self, embedding: list[float], top_k: int) -> list[Hit]:
        total = self._col.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                Hit(id=id_, source=meta["source"], page=int(meta["page"]), index=int(meta["index"]),
                    text=doc, score=1.0 - float(dist))
            )
        return hits  # Chroma returns nearest first

    def count(self) -> int:
        return self._col.count()

    def sources(self) -> dict[str, int]:
        """{filename: number_of_chunks} for everything currently indexed."""
        counts: dict[str, int] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"]:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return dict(sorted(counts.items()))
