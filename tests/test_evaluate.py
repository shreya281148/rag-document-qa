from pathlib import Path

from ragqa.evaluate import EvalItem, evaluate, format_report, load_items, main
from ragqa.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent


def test_question_set_is_well_formed():
    items = load_items(ROOT / "eval" / "qa_set.jsonl")
    answerable = [i for i in items if i.answerable]
    assert len(answerable) >= 12 and len(items) - len(answerable) >= 3
    assert all(i.expect for i in answerable)


def test_offline_retrieval_quality_floor(offline_cfg, sample_docs):
    """Guards against regressions in chunking/embedding/search. Thresholds are set below the
    measured values (hit/evidence 100%, MRR 0.97) to leave headroom, not to hide failures."""
    pipe = RAGPipeline(offline_cfg)
    pipe.ingest([sample_docs])
    report = evaluate(pipe, load_items(ROOT / "eval" / "qa_set.jsonl"), with_answers=True)
    assert report["hit_at_k"] >= 0.9
    assert report["evidence_at_k"] >= 0.9
    assert report["mrr"] >= 0.8
    assert report["answer_contains_phrase"] >= 0.8
    assert report["not_wrongly_refused"] >= 0.9


def test_metrics_are_computed_correctly_on_a_tiny_case(offline_cfg):
    from ragqa.loaders import Page

    pipe = RAGPipeline(offline_cfg)
    pipe.ingest_pages([Page("one.txt", 1, "Refunds are processed within 14 days of the request."),
                       Page("two.txt", 1, "The cafeteria serves lunch between noon and two.")])
    items = [EvalItem("How long do refunds take?", "one.txt", "14 days"),
             EvalItem("What is the capital of France?", None, None)]
    report = evaluate(pipe, items)
    assert report["hit_at_k"] == 1.0 and report["mrr"] == 1.0
    assert report["correctly_refused"] == 1.0
    assert "hit@" in format_report(report)


def test_report_flags_overlapping_score_ranges(offline_cfg, sample_docs):
    pipe = RAGPipeline(offline_cfg)
    pipe.ingest([sample_docs])
    report = evaluate(pipe, load_items(ROOT / "eval" / "qa_set.jsonl"))
    assert report["scores_overlap"] is True  # a known, documented limitation of the offline embedder
    assert "overlap" in format_report(report)


def test_cli_runs_offline(capsys):
    assert main(["--docs", str(ROOT / "sample_docs"), "--data", str(ROOT / "eval" / "qa_set.jsonl"), "--offline"]) == 0
    assert "Retrieval evaluation" in capsys.readouterr().out
