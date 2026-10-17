"""rag-document-qa: local, privacy-first document Q&A using RAG."""

from .config import Config, load_config
from .pipeline import Answer, RAGPipeline

__all__ = ["Config", "load_config", "RAGPipeline", "Answer"]
__version__ = "0.1.0"
