from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .models import Chunk, RetrievalHit

_WORD = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(raw[index : index + 2] for index in range(max(1, len(raw) - 1)))
        else:
            tokens.append(raw)
    return tokens


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class HashingEmbedder:
    """Dependency-free semantic-ish embedding for the runnable demo.

    The interface deliberately mirrors a real embedding model, so production users can
    swap in BGE/E5 without touching authorization or evaluation code.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class HybridRetriever:
    def __init__(self, semantic_weight: float = 0.45) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.semantic_weight = semantic_weight
        self.embedder = HashingEmbedder()

    def search(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievalHit]:
        if not chunks:
            return []
        tokenized = [tokenize(chunk.text + " " + chunk.title) for chunk in chunks]
        query_tokens = tokenize(query)
        document_frequency = Counter(
            token for tokens in tokenized for token in set(tokens)
        )
        average_length = sum(map(len, tokenized)) / len(tokenized)
        query_vector = self.embedder.encode(query)
        raw: list[tuple[Chunk, float, float]] = []
        for chunk, tokens in zip(chunks, tokenized, strict=True):
            counts = Counter(tokens)
            lexical = 0.0
            for token in query_tokens:
                df = document_frequency.get(token, 0)
                idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
                frequency = counts.get(token, 0)
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(tokens) / max(average_length, 1)
                )
                lexical += idf * (frequency * 2.5 / denominator if denominator else 0)
            semantic = max(0.0, _cosine(query_vector, self.embedder.encode(chunk.text)))
            raw.append((chunk, lexical, semantic))
        lexical_max = max((item[1] for item in raw), default=1.0) or 1.0
        hits = [
            RetrievalHit(
                chunk=chunk,
                lexical_score=lexical / lexical_max,
                semantic_score=semantic,
                score=(1 - self.semantic_weight) * lexical / lexical_max
                + self.semantic_weight * semantic,
            )
            for chunk, lexical, semantic in raw
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
