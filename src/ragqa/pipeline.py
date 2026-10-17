"""The RAG pipeline: ties every step together.

    ingest:  files -> pages -> chunks -> embeddings -> vector store
    ask:     question -> embedding -> nearest chunks -> prompt -> LLM -> cited answer

Read this file top to bottom to understand the whole system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .chunking import chunk_pages
from .config import Config
from .embeddings import Embedder, make_embedder
from .llm import LLM, make_llm
from .loaders import Page, load_paths
from .prompts import NOT_FOUND, build_messages
from .vectorstore import ChromaStore, Hit


@dataclass
class IngestReport:
    files: int
    pages: int
    chunks: int


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit] = field(default_factory=list)  # the passages shown to the model, numbered 1..n
    grounded: bool = True  # False when nothing relevant was found and the LLM was not called


class RAGPipeline:
    def __init__(
        self,
        config: Config,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        store: ChromaStore | None = None,
    ):
        self.config = config
        self.embedder = embedder or make_embedder(config.embeddings)
        self.llm = llm or make_llm(config.llm)
        self.store = store or ChromaStore(config.storage.persist_dir, self.embedder.id, config.storage.telemetry)

    # ------------------------------------------------------------------ ingest
    def ingest(self, paths: list[str | Path]) -> IngestReport:
        """Index files and/or folders (PDF, TXT, MD)."""
        return self.ingest_pages(load_paths(paths))

    def ingest_pages(self, pages: list[Page]) -> IngestReport:
        cc = self.config.chunking
        chunks = chunk_pages(pages, cc.chunk_size, cc.chunk_overlap)
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.upsert(chunks, vectors)
        return IngestReport(files=len({p.source for p in pages}), pages=len(pages), chunks=len(chunks))

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, question: str) -> list[Hit]:
        """Return the most relevant chunks, dropping any below the similarity floor."""
        rc = self.config.retrieval
        floor = rc.threshold(self.embedder.default_min_score)
        hits = self.store.query(self.embedder.embed_query(question), rc.top_k)
        return [h for h in hits if h.score >= floor]

    def retrieve_scored(self, question: str) -> list[Hit]:
        """Like retrieve() but WITHOUT the similarity floor: useful to tune min_score."""
        return self.store.query(self.embedder.embed_query(question), self.config.retrieval.top_k)

    # ------------------------------------------------------------------ answer
    def answer(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:  # nothing relevant: refuse instead of letting the model guess
            return Answer(question, NOT_FOUND, [], grounded=False)
        text = self.llm.generate(build_messages(question, hits))
        return Answer(question, text, hits)

    def answer_stream(self, question: str) -> tuple[list[Hit], Iterator[str]]:
        """Like answer(), but yields the text piece by piece (for the chat UI)."""
        hits = self.retrieve(question)
        if not hits:
            return [], iter([NOT_FOUND])
        return hits, self.llm.stream(build_messages(question, hits))

    # ------------------------------------------------------------------ housekeeping
    def indexed_sources(self) -> dict[str, int]:
        return self.store.sources()

    def remove_source(self, source: str) -> None:
        self.store.delete_source(source)

    def reset(self) -> None:
        self.store.reset()
