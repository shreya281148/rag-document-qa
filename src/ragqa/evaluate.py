"""Measure how good retrieval is, BEFORE involving a language model.

Why evaluate retrieval separately? If the right passage never reaches the model,
no model can answer correctly. So we first check the search step on a set of
questions where we know the right source and a phrase that must be in the evidence.

Metrics (for answerable questions)
  hit@k        share of questions whose expected SOURCE FILE is in the top-k results
  evidence@k   share where the expected PHRASE appears in the text of a top-k result
  MRR          mean reciprocal rank of the first result containing the expected phrase
               (1.0 = always first; 0.5 = typically second; 0 = never found)
  not refused  share the system did NOT wrongly say "not found" for (floor applied)
For unanswerable questions
  refused      share where NOTHING passed the similarity floor (the app says "not found")

Optionally (--answers) it also generates answers and checks that the expected phrase
(or a case-insensitive part of it) appears - a rough, cheap check, not a full judge.

Run:  python -m ragqa.evaluate --docs sample_docs --data eval/qa_set.jsonl --offline
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .pipeline import RAGPipeline


@dataclass
class EvalItem:
    question: str
    source: str | None
    expect: str | None

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_items(path: str | Path) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(EvalItem(d["question"], d.get("source"), d.get("expect")))
    return items


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def evaluate(pipeline: RAGPipeline, items: list[EvalItem], with_answers: bool = False) -> dict:
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]

    hit = evidence = not_refused = answer_ok = 0
    rr_sum = 0.0
    details = []
    for item in answerable:
        ranked = pipeline.retrieve_scored(item.question)  # no floor: measure pure ranking
        kept = pipeline.retrieve(item.question)  # with floor: what the app would really use
        sources = [h.source for h in ranked]
        phrase = _norm(item.expect or "")
        phrase_ranks = [r for r, h in enumerate(ranked, 1) if phrase in _norm(h.text)]

        hit += item.source in sources
        evidence += bool(phrase_ranks)
        rr_sum += 1.0 / phrase_ranks[0] if phrase_ranks else 0.0
        not_refused += bool(kept)
        row = {"question": item.question, "source_hit": item.source in sources,
               "first_evidence_rank": phrase_ranks[0] if phrase_ranks else None,
               "top_score": round(ranked[0].score, 3) if ranked else None}
        if with_answers:
            ans = pipeline.answer(item.question).text
            ok = phrase in _norm(ans)
            answer_ok += ok
            row["answer_contains_phrase"] = ok
        details.append(row)

    refused = sum(1 for i in unanswerable if not pipeline.retrieve(i.question))
    unans_scores = [s[0].score for s in (pipeline.retrieve_scored(i.question) for i in unanswerable) if s]
    ans_scores = [d["top_score"] for d in details if d["top_score"] is not None]
    n, u = len(answerable), len(unanswerable)
    result = {
        "k": pipeline.config.retrieval.top_k,
        "answerable": n,
        "hit_at_k": hit / n if n else None,
        "evidence_at_k": evidence / n if n else None,
        "mrr": rr_sum / n if n else None,
        "not_wrongly_refused": not_refused / n if n else None,
        "unanswerable": u,
        "correctly_refused": refused / u if u else None,
        # Best-match scores per group. If these ranges overlap, NO single min_score can cleanly
        # separate answerable from unanswerable questions: a useful thing to know when tuning.
        "answerable_top_scores": (min(ans_scores), max(ans_scores)) if ans_scores else None,
        "unanswerable_top_scores": (min(unans_scores), max(unans_scores)) if unans_scores else None,
        "scores_overlap": bool(ans_scores and unans_scores and max(unans_scores) >= min(ans_scores)),
        "details": details,
    }
    if with_answers and n:
        result["answer_contains_phrase"] = answer_ok / n
    return result


def format_report(r: dict) -> str:
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    lines = [
        f"Retrieval evaluation (top_k={r['k']})",
        f"  answerable questions : {r['answerable']}",
        f"  hit@{r['k']} (right file)   : {pct(r['hit_at_k'])}",
        f"  evidence@{r['k']} (phrase)  : {pct(r['evidence_at_k'])}",
        f"  MRR                  : {r['mrr']:.2f}" if r["mrr"] is not None else "  MRR                  : n/a",
        f"  not wrongly refused  : {pct(r['not_wrongly_refused'])}",
        f"  unanswerable         : {r['unanswerable']}  -> correctly refused: {pct(r['correctly_refused'])}",
    ]
    if r.get("answerable_top_scores") and r.get("unanswerable_top_scores"):
        a, b = r["answerable_top_scores"], r["unanswerable_top_scores"]
        lines.append(f"  best-match score, answerable   : {a[0]:.2f} .. {a[1]:.2f}")
        lines.append(f"  best-match score, unanswerable : {b[0]:.2f} .. {b[1]:.2f}")
        if r["scores_overlap"]:
            lines.append("  NOTE: the score ranges overlap, so no single min_score separates them cleanly.")
            lines.append("        The prompt's 'say you could not find it' rule is the second line of defence.")
    if "answer_contains_phrase" in r:
        lines.append(f"  answers containing expected phrase: {pct(r['answer_contains_phrase'])}")
    misses = [d for d in r["details"] if d["first_evidence_rank"] is None]
    if misses:
        lines.append("  questions where the evidence was NOT retrieved:")
        lines += [f"    - {d['question']} (top score {d['top_score']})" for d in misses]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    ap.add_argument("--docs", default="sample_docs", help="folder of documents to index")
    ap.add_argument("--data", default="eval/qa_set.jsonl", help="JSONL question set")
    ap.add_argument("--config", default=None, help="config YAML (default: ./config.yaml)")
    ap.add_argument("--offline", action="store_true", help="use the no-model hash embedder + extractive answerer")
    ap.add_argument("--answers", action="store_true", help="also generate answers and check the expected phrase")
    ap.add_argument("-k", type=int, default=None, help="override retrieval.top_k")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.offline:
        cfg.embeddings.provider = "hash"
        cfg.llm.provider = "extractive"
    if args.k:
        cfg.retrieval.top_k = args.k
    with tempfile.TemporaryDirectory() as tmp:  # evaluate on a throw-away index
        cfg.storage.persist_dir = tmp
        pipe = RAGPipeline(cfg)
        pipe.ingest([args.docs])
        report = evaluate(pipe, load_items(args.data), with_answers=args.answers)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
