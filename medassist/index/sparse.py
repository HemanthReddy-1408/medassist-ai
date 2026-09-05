"""Okapi BM25.

A lexical index is not a fallback for a semantic one; on this corpus it is
often the better retriever. Clinical queries hinge on exact strings - a drug
name, ``HbA1c``, ``2.5 mg`` - and a dense encoder trained on general web text
happily rates ``metoprolol`` and ``metformin`` as similar because they look
alike. BM25 does not make that mistake, which is exactly why both are fused
rather than one being chosen.
"""

from __future__ import annotations

import math
from collections import Counter

from medassist.core.ids import ChunkId
from medassist.core.models import Chunk
from medassist.index.embed import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[ChunkId] = []
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._avg_len: float = 0.0

    def build(self, chunks: list[Chunk]) -> BM25Index:
        self.chunk_ids = [c.id for c in chunks]
        self._tf = []
        self._lengths = []
        self._df = Counter()
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._lengths.append(len(tokens))
            self._df.update(counts.keys())

        n = max(len(chunks), 1)
        self._avg_len = (sum(self._lengths) / n) or 1.0
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at a small
        # positive value: the raw form goes negative for terms in more than half
        # the documents, which would let a common word subtract from a score.
        self._idf = {
            term: max(math.log((n - df + 0.5) / (df + 0.5) + 1.0), 1e-6)
            for term, df in self._df.items()
        }
        return self

    def search(self, query: str, k: int = 20) -> list[tuple[ChunkId, float]]:
        terms = tokenize(query)
        if not terms or not self.chunk_ids:
            return []
        scores: dict[int, float] = {}
        for term in set(terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_index, counts in enumerate(self._tf):
                freq = counts.get(term)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self._lengths[doc_index] / self._avg_len)
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * (
                    freq * (self.k1 + 1) / (freq + self.k1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.chunk_ids[i], score) for i, score in ranked]
