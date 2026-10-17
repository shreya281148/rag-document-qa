"""Errors that carry a human-readable 'what to do next' message."""


class RagError(Exception):
    """Base class for all errors raised by this project."""


class BackendUnavailable(RagError):
    """A local model server (Ollama / vLLM) could not be reached or rejected the request."""


class ModelNotFound(BackendUnavailable):
    """The server is running but the requested model has not been downloaded."""
