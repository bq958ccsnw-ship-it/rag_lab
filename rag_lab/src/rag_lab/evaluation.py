from __future__ import annotations

import json
from pathlib import Path

from .retrieval import Retriever
from .metrics import retrieval_metrics


def load_questions(path: Path, *, require_gold: bool = True) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("questions file must contain a non-empty JSON array")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("question"), str):
            raise ValueError("each question requires string id and question")
        if item["id"] in seen:
            raise ValueError(f"duplicate question id: {item['id']}")
        seen.add(item["id"])
        if require_gold and not item.get("relevant_doc_ids"):
            raise ValueError(f"question {item['id']} has no gold labels")
    return value


def evaluate(
    retriever: Retriever, questions: list[dict[str, object]], *, ks: list[int]
) -> dict[str, object]:
    max_k = max(ks)
    rows: list[dict[str, object]] = []
    sums = {k: {name: 0.0 for name in ("precision", "recall", "hit", "reciprocal_rank")} for k in ks}
    for item in questions:
        results = retriever.retrieve(str(item["question"]), top_k=max_k)
        ranked_docs = _deduplicate([result.chunk.document_id for result in results])
        relevant = {str(value) for value in item["relevant_doc_ids"]}
        per_k: dict[str, object] = {}
        for k in ks:
            metrics = retrieval_metrics(ranked_docs, relevant, k=k)
            per_k[str(k)] = metrics
            for name, value in metrics.items():
                sums[k][name] += value
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "relevant_doc_ids": sorted(relevant),
                "retrieved": [
                    {
                        "rank": result.rank,
                        "document_id": result.chunk.document_id,
                        "chunk_id": result.chunk.chunk_id,
                        "score": round(result.score, 6),
                    }
                    for result in results
                ],
                "metrics": per_k,
            }
        )
    count = len(questions)
    aggregate = {
        str(k): {name: round(value / count, 6) for name, value in values.items()}
        for k, values in sums.items()
    }
    return {"question_count": count, "aggregate": aggregate, "questions": rows}


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    return [value for value in values if not (value in seen or seen.add(value))]

