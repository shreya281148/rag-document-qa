"""Step 3: turn text into vectors ("embeddings").

An embedding model maps text to a list of numbers so that texts with similar
MEANING end up close together. Retrieval is then just "find the chunks whose
vectors are closest to the question's vector".

Two implementations share one tiny interface, so they are interchangeable:

* OllamaEmbedder - a real neural embedding model served locally (default).
* HashEmbedder   - a small offline "bag of words" embedder with no model to
                   download. It matches on shared words rather than meaning, so
                   it is weaker; it exists so tests and quick demos run anywhere.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import requests

from .config import EmbeddingConfig
from .errors import BackendUnavailable, ModelNotFound


class Embedder(Protocol):
    id: str  # identifies model+settings; vectors from different embedders must never be mixed
    default_min_score: float  # similarity floor used when config says min_score: auto

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- Ollama
class OllamaEmbedder:
    BATCH = 32
    # Deliberately low: a floor that only removes clearly unrelated text. Tune it for your own
    # documents (see `ragqa ask --show-scores` and the evaluation script).
    default_min_score = 0.25

    def __init__(self, cfg: EmbeddingConfig):
        self.model = cfg.model
        self.base_url = cfg.base_url.rstrip("/")
        self.doc_prefix = cfg.doc_prefix
        self.query_prefix = cfg.query_prefix
        self.timeout = cfg.timeout_s
        self.id = f"ollama:{cfg.model}:{cfg.doc_prefix}:{cfg.query_prefix}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = [self.doc_prefix + t for t in texts[i : i + self.BATCH]]
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BackendUnavailable(
                f"Could not reach Ollama at {self.base_url} ({exc.__class__.__name__}). "
                "Start it with `ollama serve` (or open the Ollama app)."
            ) from exc
        if resp.status_code == 404:
            raise ModelNotFound(
                f"Embedding model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )
        if not resp.ok:
            raise BackendUnavailable(f"Ollama embedding request failed ({resp.status_code}): {resp.text[:200]}")
        embeddings = resp.json().get("embeddings")
        if not embeddings or len(embeddings) != len(inputs):
            raise BackendUnavailable("Ollama returned an unexpected embedding response.")
        return embeddings


# --------------------------------------------------------------------------- Hash
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who how when where why do does did can could should would not no yes you your we our i".split()
)


class HashEmbedder:
    """Feature-hashing bag-of-words embedder (deterministic, offline, no model)."""

    # Word-overlap similarity is much lower than neural similarity (measured: relevant 0.07-0.21,
    # unrelated < 0), so its floor is lower too.
    default_min_score = 0.05

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.id = f"hash:{dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]  # words + word pairs
        vec = [0.0] * self.dim
        for feat in features:
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()  # stable across runs
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg)
    if cfg.provider == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embeddings.provider: {cfg.provider!r}")
