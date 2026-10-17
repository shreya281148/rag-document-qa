import pytest

from ragqa.config import Config, load_config


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    cfg = load_config(env={})
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "qwen2.5:3b"
    assert cfg.chunking.chunk_size == 800


def test_yaml_overrides_defaults(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("llm:\n  model: qwen2.5:7b\nretrieval:\n  top_k: 6\n")
    cfg = load_config(f, env={})
    assert cfg.llm.model == "qwen2.5:7b"
    assert cfg.retrieval.top_k == 6
    assert cfg.llm.provider == "ollama"  # untouched values keep their default


def test_environment_overrides_yaml_with_correct_types(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("llm:\n  model: qwen2.5:3b\n")
    env = {"RAGQA_LLM_MODEL": "llama3.2:3b", "RAGQA_RETRIEVAL_TOP_K": "9", "RAGQA_LLM_TEMPERATURE": "0.5",
           "RAGQA_STORAGE_TELEMETRY": "true"}
    cfg = load_config(f, env=env)
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.retrieval.top_k == 9 and isinstance(cfg.retrieval.top_k, int)
    assert cfg.llm.temperature == 0.5
    assert cfg.storage.telemetry is True


def test_unknown_key_and_section_are_rejected(tmp_path):
    bad_key = tmp_path / "a.yaml"
    bad_key.write_text("llm:\n  modle: typo\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(bad_key, env={})
    bad_section = tmp_path / "b.yaml"
    bad_section.write_text("llmm:\n  model: x\n")
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(bad_section, env={})


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", env={})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.chunking, "chunk_overlap", c.chunking.chunk_size),
        lambda c: setattr(c.chunking, "chunk_size", 0),
        lambda c: setattr(c.retrieval, "top_k", 0),
        lambda c: setattr(c.retrieval, "min_score", "banana"),
        lambda c: setattr(c.retrieval, "min_score", 3.0),
        lambda c: setattr(c.llm, "provider", "magic"),
        lambda c: setattr(c.embeddings, "provider", "magic"),
    ],
)
def test_validation_rejects_bad_values(mutate):
    cfg = Config()
    mutate(cfg)
    with pytest.raises(ValueError):
        cfg.validate()


def test_min_score_auto_uses_embedder_default_and_number_overrides():
    cfg = Config()
    assert cfg.retrieval.threshold(0.25) == 0.25
    cfg.retrieval.min_score = "0.4"
    assert cfg.retrieval.threshold(0.25) == 0.4
    cfg.retrieval.min_score = 0.1
    assert cfg.retrieval.threshold(0.25) == 0.1
