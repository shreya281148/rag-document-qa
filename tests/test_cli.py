import pytest

from ragqa.cli import main
from ragqa.config import Config
from ragqa.doctor import check_setup


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml: defaults apply
    monkeypatch.setenv("RAGQA_STORAGE_PERSIST_DIR", str(tmp_path / "db"))


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_ingest_sources_ask_reset(offline_env, capsys, sample_docs):
    code, out, _ = run(capsys, "--offline", "ingest", str(sample_docs))
    assert code == 0 and "3 file(s), 5 page(s)" in out

    _, out, _ = run(capsys, "--offline", "sources")
    assert "employee_handbook.md" in out and "local_llm_guide.pdf" in out

    _, out, _ = run(capsys, "--offline", "ask", "How many days of annual leave do full-time employees get?", "--no-stream")
    assert "28" in out and "Sources:" in out and "employee_handbook.md" in out

    _, out, _ = run(capsys, "--offline", "ask", "How often do passwords expire?", "--show-scores")
    assert "similarity floor" in out and "used" in out and "180 days" in out

    _, out, _ = run(capsys, "--offline", "reset")
    assert "cleared" in out
    _, out, _ = run(capsys, "--offline", "sources")
    assert "Nothing indexed" in out


def test_cli_streams_the_answer(offline_env, capsys, sample_docs):
    run(capsys, "--offline", "ingest", str(sample_docs))
    _, out, _ = run(capsys, "--offline", "ask", "How often do passwords expire?")
    assert "180 days" in out


def test_cli_reports_backend_problems_cleanly(offline_env, capsys, monkeypatch):
    monkeypatch.setenv("RAGQA_EMBEDDINGS_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("RAGQA_EMBEDDINGS_TIMEOUT_S", "2")
    code, _, err = run(capsys, "ask", "anything")  # default providers = ollama, which is not running
    assert code == 2 and "Error:" in err and "ollama serve" in err


# ----------------------------------------------------------------- doctor
def _cfg(base_url, llm="qwen2.5:3b", emb="nomic-embed-text"):
    cfg = Config()
    cfg.llm.base_url = cfg.embeddings.base_url = base_url
    cfg.llm.model, cfg.embeddings.model = llm, emb
    return cfg


def test_doctor_all_good(fake_ollama):
    checks = check_setup(_cfg(fake_ollama.base_url))
    assert all(c.ok for c in checks)
    assert [c.name for c in checks].count("Ollama server") == 1  # same server is only checked once


def test_doctor_names_the_missing_model_and_the_fix(fake_ollama):
    checks = check_setup(_cfg(fake_ollama.base_url, llm="qwen2.5:7b"))
    bad = [c for c in checks if not c.ok]
    assert len(bad) == 1 and "qwen2.5:7b" in bad[0].name and "ollama pull qwen2.5:7b" in bad[0].detail


def test_doctor_server_down():
    checks = check_setup(_cfg("http://127.0.0.1:9"), timeout=1)
    assert checks[0].ok is False and "ollama serve" in checks[0].detail


def test_doctor_offline_and_openai_compat(fake_ollama):
    cfg = Config()
    cfg.llm.provider, cfg.embeddings.provider = "extractive", "hash"
    assert check_setup(cfg)[0].ok
    cfg = Config()
    cfg.llm.provider, cfg.embeddings.provider = "openai_compat", "hash"
    cfg.llm.base_url = fake_ollama.base_url
    assert check_setup(cfg)[0].ok


def test_cli_doctor_exit_codes(offline_env, capsys, fake_ollama, monkeypatch):
    monkeypatch.setenv("RAGQA_LLM_BASE_URL", fake_ollama.base_url)
    monkeypatch.setenv("RAGQA_EMBEDDINGS_BASE_URL", fake_ollama.base_url)
    code, out, _ = run(capsys, "doctor")
    assert code == 0 and "[OK]" in out
    monkeypatch.setenv("RAGQA_LLM_MODEL", "not-installed")
    code, out, _ = run(capsys, "doctor")
    assert code == 1 and "[FAIL]" in out and "ollama pull not-installed" in out
