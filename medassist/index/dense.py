"""Dense retrieval: cosine similarity over a normalized matrix."""

from __future__ import annotations

import numpy as np

from medassist.core.ids import ChunkId
from medassist.core.models import Chunk
from medassist.index.embed import EmbeddingPort


class DenseIndex:
    def __init__(self, embedding: EmbeddingPort) -> None:
        self.embedding = embedding
        self.chunk_ids: list[ChunkId] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> DenseIndex:
        self.chunk_ids = [c.id for c in chunks]
        if not chunks:
            self._matrix = None
            return self
        # Prefixing the section name is a cheap, real gain: it lets a query
        # about dosing align with chunks whose text never says "dosage" but
        # whose section does.
        payloads = [
            f"{c.section.replace('_', ' ')}. {c.text}" if c.section else c.text for c in chunks
        ]
        self._matrix = self.embedding.encode(payloads)
        return self

    def search(self, query: str, k: int = 20) -> list[tuple[ChunkId, float]]:
        if self._matrix is None or not self.chunk_ids:
            return []
        vector = self.embedding.encode([query])[0]
        scores = self._matrix @ vector
        k = min(k, len(self.chunk_ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[int(i)], float(scores[int(i)])) for i in top]
