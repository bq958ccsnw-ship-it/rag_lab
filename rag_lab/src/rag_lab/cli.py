from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_documents
from .corpus import load_documents
from .embeddings import HashingEmbedder, TfidfEmbedder
from .evaluation import evaluate, load_questions
from .generation import context_only_answer, extractive_answer
from .retrieval import Retriever

ROOT = Path(__file__).resolve().parents[2]


def _retriever(args: argparse.Namespace) -> Retriever:
    documents = load_documents(ROOT / "documents")
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    embedder = TfidfEmbedder() if args.backend == "tfidf" else HashingEmbedder()
    return Retriever(chunks, embedder)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("tfidf", "hashing"), default="tfidf")
    parser.add_argument("--chunk-size", type=int, default=120, help="maximum words per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="words repeated between chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="VirtHelp educational RAG laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve = subparsers.add_parser("retrieve", help="show top chunks for one query")
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=5)
    _common(retrieve)

    answer = subparsers.add_parser("answer", help="show context or build a local extractive answer")
    answer.add_argument("query")
    answer.add_argument("--top-k", type=int, default=5)
    answer.add_argument("--generator", choices=("context", "extractive"), default="extractive")
    _common(answer)

    evaluation = subparsers.add_parser("evaluate", help="evaluate retrieval on labelled questions")
    evaluation.add_argument("--questions", type=Path, default=ROOT / "data" / "dev_questions.json")
    evaluation.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    evaluation.add_argument("--output", type=Path)
    _common(evaluation)

    args = parser.parse_args()
    retriever = _retriever(args)
    if args.command == "retrieve":
        results = retriever.retrieve(args.query, top_k=args.top_k)
        for result in results:
            print(
                f"{result.rank}. {result.chunk.document_id} chunk={result.chunk.seq_no} "
                f"score={result.score:.4f} | {result.chunk.document_title}"
            )
            print(f"   {result.chunk.text[:300]}")
        return
    if args.command == "answer":
        results = retriever.retrieve(args.query, top_k=args.top_k)
        if args.generator == "context":
            print(context_only_answer(results))
        else:
            print(extractive_answer(args.query, results))
        return
    questions = load_questions(args.questions)
    report = evaluate(retriever, questions, ks=sorted(set(args.k)))
    report["configuration"] = {
        "backend": args.backend,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "questions": str(args.questions),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"report written to {args.output}")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
