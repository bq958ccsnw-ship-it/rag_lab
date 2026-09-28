from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


class Embedder(Protocol):
    name: str

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class HashingEmbedder:
    """Dependency-free bag-of-words hashing baseline, useful offline."""

    dimensions: int = 1024
    name: str = "hashing-bow.v1"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@dataclass(slots=True)
class TfidfEmbedder:
    """Small fitted TF-IDF vectorizer implemented with the Python standard library."""

    name: str = "tfidf.v1"
    _vocabulary: dict[str, int] | None = None
    _idf: list[float] | None = None

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self._vocabulary is None:
            self._fit(texts)
        return [self._transform(text) for text in texts]

    def _fit(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("TF-IDF requires a non-empty corpus")
        tokenized = [_tokens(text) for text in texts]
        vocabulary = sorted({token for tokens in tokenized for token in tokens})
        self._vocabulary = {token: index for index, token in enumerate(vocabulary)}
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        count = len(texts)
        self._idf = [
            math.log((1 + count) / (1 + document_frequency[token])) + 1.0
            for token in vocabulary
        ]

    def _transform(self, text: str) -> list[float]:
        assert self._vocabulary is not None and self._idf is not None
        counts = Counter(_tokens(text))
        vector = [0.0] * len(self._vocabulary)
        total = sum(counts.values())
        if total == 0:
            return vector
        for token, count in counts.items():
            index = self._vocabulary.get(token)
            if index is not None:
                vector[index] = (count / total) * self._idf[index]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensionality")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
