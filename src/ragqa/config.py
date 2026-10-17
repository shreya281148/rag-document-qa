"""Configuration: one YAML file, optionally overridden by environment variables.

Why this exists: every knob that affects answer quality (model, chunk size,
top-k, similarity floor) lives in ONE place, so experiments are a one-line change
and results are reproducible.

Environment overrides use the pattern RAGQA_<SECTION>_<KEY>, e.g.
    RAGQA_LLM_MODEL=qwen2.5:7b
    RAGQA_RETRIEVAL_TOP_K=6
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    provider: str = "ollama"  # ollama | openai_compat | extractive
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 4096
    timeout_s: int = 120


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"  # ollama | hash
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    timeout_s: int = 60


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class RetrievalConfig:
    top_k: int = 4
    # Minimum cosine similarity for a chunk to be used. "auto" picks a sensible default for the
    # chosen embedder (similarity scales differ between embedders), or set a number such as 0.3.
    min_score: float | str = "auto"

    def threshold(self, embedder_default: float) -> float:
        return embedder_default if str(self.min_score).strip().lower() == "auto" else float(self.min_score)


@dataclass
class StorageConfig:
    persist_dir: str = ".chroma"
    telemetry: bool = False


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def validate(self) -> None:
        if self.chunking.chunk_size <= 0:
            raise ValueError("chunking.chunk_size must be > 0")
        if not 0 <= self.chunking.chunk_overlap < self.chunking.chunk_size:
            raise ValueError("chunking.chunk_overlap must be >= 0 and < chunk_size")
        if self.retrieval.top_k <= 0:
            raise ValueError("retrieval.top_k must be > 0")
        try:
            floor = self.retrieval.threshold(0.0)
        except ValueError:
            raise ValueError("retrieval.min_score must be 'auto' or a number between -1 and 1") from None
        if not -1.0 <= floor <= 1.0:
            raise ValueError("retrieval.min_score must be between -1 and 1")
        if self.llm.provider not in {"ollama", "openai_compat", "extractive"}:
            raise ValueError(f"unknown llm.provider: {self.llm.provider!r}")
        if self.embeddings.provider not in {"ollama", "hash"}:
            raise ValueError(f"unknown embeddings.provider: {self.embeddings.provider!r}")


def _coerce(value: str, template: Any) -> Any:
    """Convert an env-var string to the type of the existing default."""
    if isinstance(template, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def _apply_dict(section: Any, values: dict[str, Any]) -> None:
    known = {f.name for f in fields(section)}
    for key, val in values.items():
        if key not in known:
            raise ValueError(f"Unknown config key {key!r} in section {type(section).__name__}")
        setattr(section, key, val)


def _apply_env(cfg: Config, env: dict[str, str]) -> None:
    for sec_field in fields(cfg):
        section = getattr(cfg, sec_field.name)
        if not is_dataclass(section):
            continue
        for f in fields(section):
            name = f"RAGQA_{sec_field.name}_{f.name}".upper()
            if name in env:
                setattr(section, f.name, _coerce(env[name], getattr(section, f.name)))


def load_config(path: str | os.PathLike | None = None, env: dict[str, str] | None = None) -> Config:
    """Load config from YAML (if it exists), then apply environment overrides."""
    cfg = Config()
    candidate = Path(path) if path else Path("config.yaml")
    if candidate.exists():
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        for sec_name, values in raw.items():
            if not hasattr(cfg, sec_name):
                raise ValueError(f"Unknown config section {sec_name!r}")
            _apply_dict(getattr(cfg, sec_name), values or {})
    elif path:
        raise FileNotFoundError(f"Config file not found: {path}")
    _apply_env(cfg, dict(os.environ if env is None else env))
    cfg.validate()
    return cfg
