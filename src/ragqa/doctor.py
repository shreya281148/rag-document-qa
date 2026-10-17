"""Setup check: is the model server running and are the configured models installed?

Run `ragqa doctor` before the first start. It prints exactly which command fixes each problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _installed(model: str, names: set[str]) -> bool:
    # Ollama lists "nomic-embed-text:latest" even if you configured "nomic-embed-text".
    return model in names or (":" not in model and f"{model}:latest" in names)


def _ollama_models(base_url: str, timeout: float) -> tuple[set[str] | None, str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}, ""
    except (requests.RequestException, ValueError, KeyError) as exc:
        return None, exc.__class__.__name__


def check_setup(cfg: Config, timeout: float = 3.0) -> list[Check]:
    checks: list[Check] = []
    needs: list[tuple[str, str, str]] = []  # (role, base_url, model) for Ollama-served models
    if cfg.llm.provider == "ollama":
        needs.append(("Language model", cfg.llm.base_url, cfg.llm.model))
    if cfg.embeddings.provider == "ollama":
        needs.append(("Embedding model", cfg.embeddings.base_url, cfg.embeddings.model))

    cache: dict[str, tuple[set[str] | None, str]] = {}
    for role, base_url, model in needs:
        if base_url not in cache:
            cache[base_url] = _ollama_models(base_url, timeout)
            names, err = cache[base_url]
            checks.append(
                Check("Ollama server", True, f"reachable at {base_url}") if names is not None
                else Check("Ollama server", False, f"not reachable at {base_url} ({err}). Start it: `ollama serve`")
            )
        names, _ = cache[base_url]
        if names is not None:
            ok = _installed(model, names)
            checks.append(Check(f"{role} '{model}'", ok, "installed" if ok else f"missing. Install it: `ollama pull {model}`"))

    if cfg.llm.provider == "openai_compat":
        try:
            ok = requests.get(f"{cfg.llm.base_url.rstrip('/')}/v1/models", timeout=timeout).ok
            checks.append(Check("OpenAI-compatible server", ok, "reachable" if ok else "responded with an error"))
        except requests.RequestException as exc:
            checks.append(Check("OpenAI-compatible server", False, f"not reachable ({exc.__class__.__name__})"))

    if cfg.llm.provider == "extractive" and cfg.embeddings.provider == "hash":
        checks.append(Check("Offline mode", True, "no model server needed (answers are quoted sentences)"))
    return checks
