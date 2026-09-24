"""Hybrid BM25 + embedding retrieval over whatever chunks the caller is allowed to see.

Collection statistics (document frequencies, average length) are computed over
the chunks passed in, i.e. over the caller's authorized view. With statistics
from the whole shared index, a user's scores would depend on other tenants'
text, which is a side channel.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .models import Chunk, RetrievalHit

_WORD = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if re.fullmatch(r"[一-鿿]+", raw):
            tokens.extend(raw[index : index + 2] for index in range(max(1, len(raw) - 1)))
        else:
            tokens.append(raw)
    return tokens


class HashingEmbedder:
    """Dependency-free stand-in with the interface of a real embedding model (BGE, E5)."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class HybridRetriever:
    def __init__(self, semantic_weight: float = 0.45, embedder: HashingEmbedder | None = None) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.semantic_weight = semantic_weight
        self.embedder = embedder or HashingEmbedder()
        self._features: dict[tuple[str, int, str], tuple[Counter, int, list[float]]] = {}
        self._stats: dict[frozenset[str], tuple[Counter, float]] = {}

    def _feature(self, chunk: Chunk) -> tuple[Counter, int, list[float]]:
        key = (chunk.id, chunk.version, chunk.text)
        if key not in self._features:
            tokens = tokenize(chunk.text + " " + chunk.title)
            self._features[key] = (Counter(tokens), len(tokens), self.embedder.encode(chunk.text))
        return self._features[key]

    def _collection(self, chunks: list[Chunk]) -> tuple[Counter, float]:
        key = frozenset((c.id, c.version) for c in chunks)
        if key not in self._stats:
            if len(self._stats) > 4096:
                self._stats.clear()
            df: Counter = Counter()
            total = 0
            for chunk in chunks:
                counts, length, _ = self._feature(chunk)
                df.update(counts.keys())
                total += length
            self._stats[key] = (df, total / len(chunks))
        return self._stats[key]

    def search(
        self, query: str, chunks: list[Chunk], top_k: int = 5, statistics_from: list[Chunk] | None = None
    ) -> list[RetrievalHit]:
        if not chunks:
            return []
        stats_chunks = statistics_from if statistics_from is not None else chunks
        df, average_length = self._collection(stats_chunks)
        n = len(stats_chunks)
        query_tokens = tokenize(query)
        query_vector = self.embedder.encode(query)
        idf = {t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in set(query_tokens)}
        raw = []
        for chunk in chunks:
            counts, length, vector = self._feature(chunk)
            lexical = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if frequency:
                    norm = 1.5 * (0.25 + 0.75 * length / average_length)
                    lexical += idf[token] * frequency * 2.5 / (frequency + norm)
            semantic = max(0.0, sum(a * b for a, b in zip(query_vector, vector, strict=True)))
            raw.append((chunk, lexical, semantic))
        lexical_max = max(item[1] for item in raw) or 1.0
        hits = [
            RetrievalHit(
                chunk=chunk,
                lexical_score=lexical / lexical_max,
                semantic_score=semantic,
                score=(1 - self.semantic_weight) * lexical / lexical_max + self.semantic_weight * semantic,
            )
            for chunk, lexical, semantic in raw
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]
