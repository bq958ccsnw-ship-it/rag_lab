from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def chunk_documents(
    documents: list[Document], *, chunk_size: int = 120, overlap: int = 20
) -> list[Chunk]:
    if chunk_size < 20:
        raise ValueError("chunk_size must be at least 20 words")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks


def chunk_document(document: Document, *, chunk_size: int, overlap: int) -> list[Chunk]:
    paragraphs = [paragraph.strip() for paragraph in _PARAGRAPH_BREAK.split(document.text) if paragraph.strip()]
    windows: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        windows.append(" ".join(current))
        current = current[-overlap:] if overlap else []

    for paragraph in paragraphs:
        words = paragraph.split()
        while words:
            available = chunk_size - len(current)
            if available <= 0:
                flush()
                available = chunk_size - len(current)
            current.extend(words[:available])
            words = words[available:]
            if len(current) >= chunk_size:
                flush()
    if current and (not windows or " ".join(current) != windows[-1]):
        windows.append(" ".join(current))

    result: list[Chunk] = []
    for seq_no, text in enumerate(windows):
        identity = f"{document.document_id}:{seq_no}:{text}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:16]
        result.append(
            Chunk(
                chunk_id=f"{document.document_id}:{seq_no}:{digest}",
                document_id=document.document_id,
                document_title=document.title,
                seq_no=seq_no,
                text=text,
                token_count=len(text.split()),
            )
        )
    return result

