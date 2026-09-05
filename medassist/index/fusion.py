"""Reciprocal Rank Fusion.

RRF combines rankings by position rather than by score, which is the property
that matters here: BM25 scores are unbounded sums of idf terms and cosine
scores live in [-1, 1]. Normalizing them onto a common scale requires assuming
a distribution for each, and that assumption is wrong often enough to reorder
results. Rank is comparable without any such assumption.

``k`` damps the head of each list. At k=60 the gap between rank 1 and rank 2 is
small, so a chunk both retrievers rank highly beats one that a single retriever
ranks first - which is the behaviour you want from an ensemble.
"""

from __future__ import annotations

from medassist.core.ids import ChunkId


def reciprocal_rank_fusion(
    rankings: list[list[tuple[ChunkId, float]]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[ChunkId, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")

    fused: dict[ChunkId, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
    # Tie-break on id so the ordering is total and reproducible; without it,
    # two chunks with identical fused scores can swap between runs and make a
    # deterministic evaluation look flaky.
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
