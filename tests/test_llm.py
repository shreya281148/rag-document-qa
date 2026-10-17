import pytest

from ragqa.config import LLMConfig
from ragqa.errors import BackendUnavailable, ModelNotFound
from ragqa.llm import ExtractiveLLM, OllamaLLM, OpenAICompatLLM, make_llm
from ragqa.prompts import NOT_FOUND, build_messages
from ragqa.vectorstore import Hit


def hit(n, text, source="doc.txt", page=1):
    return Hit(id=str(n), source=source, page=page, index=0, text=text, score=0.9)


def messages(question, *texts):
    return build_messages(question, [hit(i, t) for i, t in enumerate(texts, 1)])


# ----------------------------------------------------------------- extractive
def test_extractive_quotes_best_sentence_with_citation():
    msgs = messages("How long do refunds take?",
                    "The office opens at nine. Refunds are processed within 14 days of the request.",
                    "Parking is available for visitors behind the building entrance.")
    answer = ExtractiveLLM().generate(msgs)
    assert "Refunds are processed within 14 days" in answer
    assert answer.endswith("[1]")


def test_extractive_says_not_found_without_overlap():
    msgs = messages("What is the capital of France?", "Parking is available for visitors behind the building.")
    assert ExtractiveLLM().generate(msgs) == NOT_FOUND


def test_extractive_stream_matches_generate():
    msgs = messages("How long do refunds take?", "Refunds are processed within 14 days of the request.")
    llm = ExtractiveLLM()
    assert "".join(llm.stream(msgs)).strip() == llm.generate(msgs)


# ----------------------------------------------------------------- ollama (against the fake server)
def test_ollama_generate_sends_model_messages_and_options(fake_ollama):
    cfg = LLMConfig(base_url=fake_ollama.base_url, model="qwen2.5:3b", temperature=0.1, num_ctx=4096)
    out = OllamaLLM(cfg).generate(messages("Q?", "Some context here."))
    assert out == "Fake answer from qwen2.5:3b. [1]"
    path, body = fake_ollama.requests[-1]
    assert path == "/api/chat"
    assert body["model"] == "qwen2.5:3b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.1, "num_ctx": 4096}
    assert body["messages"][0]["role"] == "system"
    assert "Some context here." in body["messages"][1]["content"]


def test_ollama_stream_yields_pieces_that_join_to_the_full_answer(fake_ollama):
    llm = OllamaLLM(LLMConfig(base_url=fake_ollama.base_url, model="qwen2.5:3b"))
    pieces = list(llm.stream(messages("Q?", "ctx")))
    assert len(pieces) > 1  # really streamed in parts
    assert "".join(pieces).strip() == "Fake answer from qwen2.5:3b. [1]"
    assert fake_ollama.requests[-1][1]["stream"] is True


def test_changing_the_model_is_only_a_config_change(fake_ollama):
    for model in ("qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b"):
        llm = make_llm(LLMConfig(base_url=fake_ollama.base_url, model=model))
        assert f"from {model}" in llm.generate(messages("Q?", "ctx"))


def test_ollama_errors_are_actionable(fake_ollama):
    with pytest.raises(ModelNotFound, match="ollama pull missing"):
        OllamaLLM(LLMConfig(base_url=fake_ollama.base_url, model="missing")).generate(messages("Q?", "c"))
    with pytest.raises(BackendUnavailable, match="500"):
        OllamaLLM(LLMConfig(base_url=fake_ollama.base_url, model="boom")).generate(messages("Q?", "c"))
    with pytest.raises(BackendUnavailable, match="ollama serve"):
        OllamaLLM(LLMConfig(base_url="http://127.0.0.1:9", timeout_s=2)).generate(messages("Q?", "c"))


# ----------------------------------------------------------------- openai-compatible (vLLM etc.)
def test_openai_compatible_backend(fake_ollama):
    llm = make_llm(LLMConfig(provider="openai_compat", base_url=fake_ollama.base_url, model="Qwen/Qwen2.5-7B-Instruct"))
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.generate(messages("Q?", "ctx")) == "OpenAI-style answer from Qwen/Qwen2.5-7B-Instruct"
    assert fake_ollama.requests[-1][0] == "/v1/chat/completions"
    assert "".join(llm.stream(messages("Q?", "ctx"))).startswith("OpenAI-style answer")


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        make_llm(LLMConfig(provider="magic"))
