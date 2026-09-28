from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    title: str
    text: str
    path: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    seq_no: int
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    score: float
    chunk: Chunk

