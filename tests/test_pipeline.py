import pytest

from ragqa.config import Config
from ragqa.loaders import Page
from ragqa.pipeline import RAGPipeline
from ragqa.prompts import NOT_FOUND

SMALL_CORPUS = [
    Page("facts.txt", 1, "The office is open from 9 to 5 on weekdays. Visitors must sign in at the front desk.\n\n"
                         "Refunds are processed within 14 days of the request. Contact billing for details."),
]


class SpyLLM:
    """Records the prompt it receives; fails loudly if it is called when it should not be."""

    def __init__(self, reply="spy reply [1]"):
        self.calls, self.reply = [], reply

    def generate(self, messages):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.reply.split(" ")


def test_ingest_report_and_sources(offline_cfg, sample_docs):
    pipe = RAGPipeline(offline_cfg)
    report = pipe.ingest([sample_docs])
    assert (report.files, report.pages) == (3, 5)  # handbook(1) + policy(1) + pdf(3 pages)
    assert report.chunks >= 3
    assert set(pipe.indexed_sources()) == {"employee_handbook.md", "it_security_policy.txt", "local_llm_guide.pdf"}


def test_answer_is_grounded_and_cites_its_source(offline_cfg, sample_docs):
    pipe = RAGPipeline(offline_cfg)
    pipe.ingest([sample_docs])
    ans = pipe.answer("How many days of annual leave do full-time employees get?")
    assert ans.grounded
    assert "28" in ans.text and "[" in ans.text
    assert ans.sources[0].source == "employee_handbook.md"


def test_pdf_answers_carry_page_numbers(offline_cfg, sample_docs):
    pipe = RAGPipeline(offline_cfg)
    pipe.ingest([sample_docs])
    ans = pipe.answer("How much memory does a 7 billion parameter model need with 4-bit quantisation?")
    top = ans.sources[0]
    assert top.source == "local_llm_guide.pdf" and top.page == 1


def test_irrelevant_question_is_refused_without_calling_the_llm(offline_cfg):
    spy = SpyLLM()
    pipe = RAGPipeline(offline_cfg, llm=spy)
    pipe.ingest_pages(SMALL_CORPUS)
    ans = pipe.answer("What is the capital of France?")
    assert (ans.text, ans.grounded, ans.sources) == (NOT_FOUND, False, [])
    assert spy.calls == []  # no model call = no chance to hallucinate


def test_empty_index_is_refused(offline_cfg):
    spy = SpyLLM()
    ans = RAGPipeline(offline_cfg, llm=spy).answer("Anything at all?")
    assert ans.grounded is False and spy.calls == []


def test_prompt_contains_numbered_context_and_question(offline_cfg):
    spy = SpyLLM()
    pipe = RAGPipeline(offline_cfg, llm=spy)
    pipe.ingest_pages(SMALL_CORPUS)
    pipe.answer("How long do refunds take?")
    system, user = spy.calls[0]
    assert system["role"] == "system" and user["role"] == "user"
    assert "[1] (facts.txt, p.1)" in user["content"]
    assert "Refunds are processed within 14 days" in user["content"]
    assert user["content"].rstrip().endswith("QUESTION: How long do refunds take?")


def test_min_score_number_overrides_auto(offline_cfg):
    pipe = RAGPipeline(offline_cfg, llm=SpyLLM())
    pipe.ingest_pages(SMALL_CORPUS)
    assert pipe.retrieve("How long do refunds take?")
    pipe.config.retrieval.min_score = 0.99  # impossibly strict
    assert pipe.retrieve("How long do refunds take?") == []
    assert pipe.retrieve_scored("How long do refunds take?")  # unfiltered view still available


def test_reingesting_is_idempotent(offline_cfg, sample_docs):
    pipe = RAGPipeline(offline_cfg)
    pipe.ingest([sample_docs])
    first = pipe.store.count()
    pipe.ingest([sample_docs])
    assert pipe.store.count() == first


def test_index_survives_restart(offline_cfg, sample_docs):
    RAGPipeline(offline_cfg).ingest([sample_docs])
    reopened = RAGPipeline(offline_cfg)  # new object, same folder
    assert reopened.indexed_sources()
    assert "28" in reopened.answer("How many days of annual leave do full-time employees get?").text


def test_remove_source_and_reset(offline_cfg, sample_docs):
    pipe = RAGPipeline(offline_cfg)
    pipe.ingest([sample_docs])
    pipe.remove_source("it_security_policy.txt")
    assert "it_security_policy.txt" not in pipe.indexed_sources()
    pipe.reset()
    assert pipe.indexed_sources() == {}


def test_ingesting_nothing_is_a_noop(offline_cfg):
    report = RAGPipeline(offline_cfg).ingest_pages([])
    assert (report.files, report.pages, report.chunks) == (0, 0, 0)


def test_answer_stream(offline_cfg):
    pipe = RAGPipeline(offline_cfg, llm=SpyLLM("Fourteen days [1]"))
    pipe.ingest_pages(SMALL_CORPUS)
    hits, stream = pipe.answer_stream("How long do refunds take?")
    assert hits and "".join(stream).replace(" ", "") == "Fourteendays[1]"
    hits, stream = pipe.answer_stream("What is the capital of France?")
    assert hits == [] and list(stream) == [NOT_FOUND]


def test_full_pipeline_through_the_real_ollama_code_path(tmp_path, fake_ollama, sample_docs):
    """Ollama embeddings + Ollama chat (Qwen model name), served by the fake HTTP server."""
    cfg = Config()
    cfg.llm.base_url = cfg.embeddings.base_url = fake_ollama.base_url
    cfg.llm.model, cfg.embeddings.model = "qwen2.5:3b", "nomic-embed-text"
    cfg.storage.persist_dir = str(tmp_path / "db")
    pipe = RAGPipeline(cfg)

    pipe.ingest([sample_docs])
    embed_calls = [b for p, b in fake_ollama.requests if p == "/api/embed"]
    assert embed_calls and all(t.startswith("search_document: ") for b in embed_calls for t in b["input"])

    ans = pipe.answer("How many days of annual leave do full-time employees get?")
    assert ans.text == "Fake answer from qwen2.5:3b. [1]"
    assert ans.sources[0].source == "employee_handbook.md"  # retrieval really worked through /api/embed
    path, chat = fake_ollama.requests[-1]
    assert path == "/api/chat" and chat["model"] == "qwen2.5:3b"
    assert "28 days" in chat["messages"][1]["content"]  # the retrieved evidence reached the model
    assert fake_ollama.requests[-2][1]["input"][0].startswith("search_query: ")

    hits, stream = pipe.answer_stream("How many days of annual leave do full-time employees get?")
    assert "".join(stream).strip() == "Fake answer from qwen2.5:3b. [1]"
