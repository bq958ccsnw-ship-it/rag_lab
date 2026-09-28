from __future__ import annotations

import re

from .models import SearchResult

_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def context_only_answer(results: list[SearchResult]) -> str:
    if not results:
        return "В доступных источниках нет контекста для ответа."
    lines = ["Найденный контекст:"]
    for result in results:
        lines.append(
            f"[{result.chunk.document_id}, chunk {result.chunk.seq_no}, score={result.score:.3f}] "
            f"{result.chunk.text}"
        )
    return "\n\n".join(lines)


def extractive_answer(query: str, results: list[SearchResult], *, sentence_count: int = 3) -> str:
    """Return the highest-overlap source sentences without any external model."""
    query_tokens = set(_TOKEN.findall(query.lower()))
    candidates: list[tuple[float, int, str, str]] = []
    for result in results:
        for sentence_index, sentence in enumerate(_SENTENCE.split(result.chunk.text)):
            sentence = sentence.strip()
            sentence_tokens = set(_TOKEN.findall(sentence.lower()))
            if not sentence_tokens:
                continue
            overlap = len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
            score = overlap + 0.05 * result.score
            candidates.append((score, -sentence_index, sentence, result.chunk.document_id))
    candidates.sort(reverse=True)
    selected = candidates[:sentence_count]
    if not selected or selected[0][0] <= 0:
        return "В доступных источниках нет достаточной информации для ответа."
    return " ".join(f"{sentence} [{document_id}]" for _, _, sentence, document_id in selected)
