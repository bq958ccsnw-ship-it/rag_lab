from __future__ import annotations

import re
from pathlib import Path

from .models import Document

_DOCUMENT_ID = re.compile(r"^(doc_[0-9]{2})_")


def load_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        match = _DOCUMENT_ID.match(path.name)
        if match is None:
            raise ValueError(f"document filename must start with doc_NN_: {path.name}")
        text = path.read_text(encoding="utf-8").strip()
        title = next(
            (line.removeprefix("# ").strip() for line in text.splitlines() if line.startswith("# ")),
            path.stem,
        )
        documents.append(
            Document(
                document_id=match.group(1),
                title=title,
                text=text,
                path=str(path),
            )
        )
    if not documents:
        raise ValueError(f"no Markdown documents found in {directory}")
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("document ids must be unique")
    return documents

