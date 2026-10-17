"""Command-line interface: the quickest way to try the system and to see what happens inside.

    ragqa doctor                       check that Ollama and the models are ready
    ragqa ingest sample_docs           index a folder (or files)
    ragqa ask "How many leave days?"   ask a question (add --show-scores to see similarity scores)
    ragqa sources                      list what is indexed
    ragqa reset                        clear the index

Add --offline before the command to run with no model server at all.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .doctor import check_setup
from .errors import RagError
from .pipeline import RAGPipeline


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ragqa", description="Local, privacy-first document Q&A (RAG).")
    ap.add_argument("--config", default=None, help="path to a config YAML (default: ./config.yaml)")
    ap.add_argument("--offline", action="store_true", help="no model server: hash embeddings + extractive answers")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the model server and installed models")
    p = sub.add_parser("ingest", help="index files or folders (PDF, TXT, MD)")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("ask", help="ask a question about the indexed documents")
    p.add_argument("question")
    p.add_argument("--show-scores", action="store_true", help="print similarity scores for the top results")
    p.add_argument("--no-stream", action="store_true", help="print the answer at once instead of streaming")
    sub.add_parser("sources", help="list indexed documents")
    sub.add_parser("reset", help="delete everything from the index")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.offline:
        cfg.embeddings.provider, cfg.llm.provider = "hash", "extractive"

    if args.command == "doctor":
        checks = check_setup(cfg)
        for c in checks:
            print(f"[{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
        return 0 if all(c.ok for c in checks) else 1

    try:
        pipe = RAGPipeline(cfg)
        if args.command == "ingest":
            r = pipe.ingest(args.paths)
            print(f"Indexed {r.files} file(s), {r.pages} page(s) -> {r.chunks} chunk(s).")
        elif args.command == "sources":
            sources = pipe.indexed_sources()
            if not sources:
                print("Nothing indexed yet. Run: ragqa ingest <folder>")
            for name, n in sources.items():
                print(f"{name}: {n} chunk(s)")
        elif args.command == "reset":
            pipe.reset()
            print("Index cleared.")
        elif args.command == "ask":
            _ask(pipe, args)
    except RagError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def _ask(pipe: RAGPipeline, args: argparse.Namespace) -> None:
    if args.show_scores:
        floor = pipe.config.retrieval.threshold(pipe.embedder.default_min_score)
        print(f"Top results (similarity floor = {floor:.2f}; results below it are ignored):")
        for h in pipe.retrieve_scored(args.question):
            mark = "used   " if h.score >= floor else "ignored"
            print(f"  {h.score:6.3f} {mark} {h.source} p.{h.page}: {h.text[:70]!r}")
        print()
    hits, stream = pipe.answer_stream(args.question)
    if args.no_stream:
        print("".join(stream).strip())
    else:
        for piece in stream:
            print(piece, end="", flush=True)
        print()
    if hits:
        print("\nSources:")
        for n, h in enumerate(hits, start=1):
            print(f"  [{n}] {h.source}, p.{h.page}  (similarity {h.score:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
