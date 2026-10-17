"""Step 5: the language model that writes the final answer.

THIS IS THE "PLUG-IN" LAYER. Every backend implements the same two methods:

    generate(messages) -> str            full answer at once
    stream(messages)   -> Iterator[str]  answer piece by piece (typing effect)

so the rest of the program never knows (or cares) which model is behind it.
Switching from Qwen to another model is a one-line change in config.yaml.

Backends
--------
* OllamaLLM       - runs models such as Qwen 2.5 locally through Ollama (default).
* OpenAICompatLLM - talks to any OpenAI-compatible server YOU host, e.g. vLLM.
                    (Still local/private if the server runs on your own machine.)
* ExtractiveLLM   - no AI model at all: quotes the most relevant sentences.
                    Used for tests and as a graceful fallback.
"""

from __future__ import annotations

import json
import re
from typing import Iterator, Protocol

import requests

from .config import LLMConfig
from .embeddings import _STOPWORDS, _TOKEN
from .errors import BackendUnavailable, ModelNotFound
from .prompts import NOT_FOUND, parse_user_message

Messages = list[dict[str, str]]


class LLM(Protocol):
    def generate(self, messages: Messages) -> str: ...

    def stream(self, messages: Messages) -> Iterator[str]: ...


# --------------------------------------------------------------------------- Ollama
class OllamaLLM:
    def __init__(self, cfg: LLMConfig):
        self.model = cfg.model
        self.base_url = cfg.base_url.rstrip("/")
        self.options = {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx}
        self.timeout = cfg.timeout_s

    def _payload(self, messages: Messages, stream: bool) -> dict:
        return {"model": self.model, "messages": messages, "stream": stream, "options": self.options}

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=stream)
        except requests.RequestException as exc:
            raise BackendUnavailable(
                f"Could not reach Ollama at {self.base_url} ({exc.__class__.__name__}). "
                "Start it with `ollama serve` (or open the Ollama app)."
            ) from exc
        if resp.status_code == 404:
            raise ModelNotFound(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if not resp.ok:
            raise BackendUnavailable(f"Ollama request failed ({resp.status_code}): {resp.text[:200]}")
        return resp

    def generate(self, messages: Messages) -> str:
        data = self._post(self._payload(messages, stream=False), stream=False).json()
        return data.get("message", {}).get("content", "").strip()

    def stream(self, messages: Messages) -> Iterator[str]:
        resp = self._post(self._payload(messages, stream=True), stream=True)
        for line in resp.iter_lines():  # Ollama streams one JSON object per line
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                raise BackendUnavailable(f"Ollama error: {obj['error']}")
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                break


# --------------------------------------------------------------------------- OpenAI-compatible (vLLM etc.)
class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig):
        self.model = cfg.model
        self.base_url = cfg.base_url.rstrip("/")
        self.temperature = cfg.temperature
        self.timeout = cfg.timeout_s

    def generate(self, messages: Messages) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={"model": self.model, "messages": messages, "temperature": self.temperature},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BackendUnavailable(
                f"Could not reach the model server at {self.base_url} ({exc.__class__.__name__})."
            ) from exc
        if not resp.ok:
            raise BackendUnavailable(f"Model server request failed ({resp.status_code}): {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"].strip()

    def stream(self, messages: Messages) -> Iterator[str]:
        yield self.generate(messages)  # simple: no token streaming for this backend


# --------------------------------------------------------------------------- Extractive (no model)
class ExtractiveLLM:
    """Answers by quoting the context sentences that best overlap with the question."""

    MAX_SENTENCES = 3

    def generate(self, messages: Messages) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        question, passages = parse_user_message(user)
        q_terms = {t for t in _TOKEN.findall(question.lower()) if t not in _STOPWORDS}

        scored: list[tuple[float, int, int, str]] = []  # (score, passage_no, order, sentence)
        order = 0
        for number, text in passages:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                terms = set(_TOKEN.findall(sentence.lower()))
                overlap = len(q_terms & terms)
                if overlap:
                    scored.append((overlap / (1 + len(terms) ** 0.5), number, order, sentence))
                order += 1
        if not scored:
            return NOT_FOUND
        best = sorted(scored, key=lambda s: -s[0])[: self.MAX_SENTENCES]
        best.sort(key=lambda s: s[2])  # restore reading order
        return " ".join(f"{sent} [{num}]" for _, num, _, sent in best)

    def stream(self, messages: Messages) -> Iterator[str]:
        for word in self.generate(messages).split(" "):
            yield word + " "


def make_llm(cfg: LLMConfig) -> LLM:
    if cfg.provider == "ollama":
        return OllamaLLM(cfg)
    if cfg.provider == "openai_compat":
        return OpenAICompatLLM(cfg)
    if cfg.provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown llm.provider: {cfg.provider!r}")
