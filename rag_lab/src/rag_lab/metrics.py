from __future__ import annotations


def retrieval_metrics(
    ranked_document_ids: list[str], relevant_document_ids: set[str], *, k: int
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_document_ids:
        raise ValueError("at least one relevant document is required")
    selected = ranked_document_ids[:k]
    relevant_retrieved = len(set(selected) & relevant_document_ids)
    first_rank = next(
        (rank for rank, document_id in enumerate(selected, start=1) if document_id in relevant_document_ids),
        None,
    )
    return {
        "precision": relevant_retrieved / k,
        "recall": relevant_retrieved / len(relevant_document_ids),
        "hit": float(first_rank is not None),
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
    }

