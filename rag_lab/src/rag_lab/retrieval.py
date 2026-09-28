from __future__ import annotations

from .embeddings import Embedder, cosine_similarity
from .models import Chunk, SearchResult


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("retriever requires at least one chunk")
        self.chunks = chunks
        self.embedder = embedder
        self._vectors = embedder.embed_many([chunk.text for chunk in chunks])
        if len(self._vectors) != len(chunks):
            raise RuntimeError("embedder returned an unexpected vector count")

    def retrieve(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed_many([query])[0]
        scored = [
            (cosine_similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self.chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].document_id, item[1].seq_no))
        return [
            SearchResult(rank=rank, score=score, chunk=chunk)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

