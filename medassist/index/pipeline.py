"""The retrieval pipeline, and the trace it emits.

The trace is the reason this file exists. Retrieval either produced the right
context or it did not, and when it did not there are four different bugs it
could be - a gap in the corpus, a weak encoder, an over-eager reranker, or a
context window too small. Recording what survived each stage turns that
question from a debugging session into a lookup.

Optional multi-query expansion (RAG-Fusion) issues the paraphrases as separate
retrievals and fuses all of them. It helps most where a lay phrasing shares no
vocabulary with clinical text - "what can I eat" against "dietary
modification" - and the harness measures whether it earned its extra calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from medassist.core.enums import RetrievalStage
from medassist.core.ids import ChunkId
from medassist.core.models import Chunk, Document, RetrievalTrace, StageRecord
from medassist.index.chunking import ChunkSpec, chunk_corpus
from medassist.index.dense import DenseIndex
from medassist.index.embed import EmbeddingPort, default_embedding
from medassist.index.fusion import reciprocal_rank_fusion
from medassist.index.rerank import (
    RerankSpec,
    format_context,
    rerank,
    sections_for_intent,
    select_context,
)
from medassist.index.sparse import BM25Index


@dataclass(frozen=True)
class RetrievalSpec:
    first_pass_k: int = 30
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    rerank_spec: RerankSpec = field(default_factory=RerankSpec)
    max_context_chars: int = 9000
    max_context_chunks: int = 8
    max_per_doc: int = 3
    use_mmr_embedding: bool = False  # embedding-based MMR: better, slower


@dataclass
class RetrievalResult:
    chunk_ids: list[ChunkId]
    trace: RetrievalTrace
    context: str

    @property
    def empty(self) -> bool:
        return not self.chunk_ids


class Retriever:
    def __init__(
        self,
        *,
        embedding: EmbeddingPort | None = None,
        spec: RetrievalSpec | None = None,
        chunk_spec: ChunkSpec | None = None,
    ) -> None:
        self.embedding = embedding or default_embedding()
        self.spec = spec or RetrievalSpec()
        self.chunk_spec = chunk_spec or ChunkSpec()
        self.chunks: dict[ChunkId, Chunk] = {}
        self.documents: dict[str, Document] = {}
        self._dense = DenseIndex(self.embedding)
        self._sparse = BM25Index()
        self.snapshot_id = ""

    # -- construction ------------------------------------------------------

    def build(self, documents: list[Document], *, snapshot_id: str = "") -> Retriever:
        chunks = chunk_corpus(documents, self.chunk_spec)
        self.chunks = {c.id: c for c in chunks}
        self.documents = {d.id: d for d in documents}
        self.snapshot_id = snapshot_id
        self._dense.build(chunks)
        self._sparse.build(chunks)
        return self

    @property
    def indexed_ids(self) -> set[ChunkId]:
        return set(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    # -- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        intent: str = "",
        expansions: list[str] | None = None,
    ) -> RetrievalResult:
        stages: list[StageRecord] = []
        queries = [query, *(expansions or [])]

        started = time.perf_counter()
        dense_runs = [self._dense.search(q, self.spec.first_pass_k) for q in queries]
        dense_ms = (time.perf_counter() - started) * 1000
        dense_merged = _merge(dense_runs)
        stages.append(
            StageRecord(
                stage=RetrievalStage.DENSE,
                chunk_ids=[cid for cid, _ in dense_merged],
                scores={str(cid): s for cid, s in dense_merged[:20]},
                took_ms=dense_ms,
            )
        )

        started = time.perf_counter()
        sparse_runs = [self._sparse.search(q, self.spec.first_pass_k) for q in queries]
        sparse_ms = (time.perf_counter() - started) * 1000
        sparse_merged = _merge(sparse_runs)
        stages.append(
            StageRecord(
                stage=RetrievalStage.SPARSE,
                chunk_ids=[cid for cid, _ in sparse_merged],
                scores={str(cid): s for cid, s in sparse_merged[:20]},
                took_ms=sparse_ms,
            )
        )

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [*dense_runs, *sparse_runs],
            k=self.spec.rrf_k,
            weights=[self.spec.dense_weight] * len(dense_runs)
            + [self.spec.sparse_weight] * len(sparse_runs),
        )
        stages.append(
            StageRecord(
                stage=RetrievalStage.FUSED,
                chunk_ids=[cid for cid, _ in fused],
                took_ms=(time.perf_counter() - started) * 1000,
            )
        )

        started = time.perf_counter()
        reranked = rerank(
            query,
            fused,
            self.chunks,
            spec=self.spec.rerank_spec,
            embedding=self.embedding if self.spec.use_mmr_embedding else None,
            intent_sections=sections_for_intent(intent),
        )
        stages.append(
            StageRecord(
                stage=RetrievalStage.RERANKED,
                chunk_ids=[cid for cid, _ in reranked],
                scores={str(cid): s for cid, s in reranked[:20]},
                took_ms=(time.perf_counter() - started) * 1000,
            )
        )

        selected, dropped = select_context(
            reranked,
            self.chunks,
            max_chars=self.spec.max_context_chars,
            max_chunks=self.spec.max_context_chunks,
            max_per_doc=self.spec.max_per_doc,
        )
        stages.append(StageRecord(stage=RetrievalStage.SELECTED, chunk_ids=selected))

        trace = RetrievalTrace(
            query=query, stages=stages, selected=selected, truncated_by_budget=dropped
        )
        return RetrievalResult(
            chunk_ids=selected, trace=trace, context=format_context(selected, self.chunks)
        )

    def get(self, chunk_id: ChunkId) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def handles(self, chunk_ids: list[ChunkId]) -> dict[str, ChunkId]:
        """Map ``C1``…``Cn`` back to real chunk ids."""
        return {f"C{i}": cid for i, cid in enumerate(chunk_ids, start=1)}


def _merge(runs: list[list[tuple[ChunkId, float]]]) -> list[tuple[ChunkId, float]]:
    """Best score per chunk across query variants, highest first."""
    best: dict[ChunkId, float] = {}
    for run in runs:
        for chunk_id, score in run:
            if score > best.get(chunk_id, float("-inf")):
                best[chunk_id] = score
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
