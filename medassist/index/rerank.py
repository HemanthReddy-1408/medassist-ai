"""Reranking and context selection.

Three problems separate a ranked list from a usable context window.

**Redundancy.** The top 8 fused hits are frequently the same paragraph of the
same label, because overlapping chunks are near-duplicates by construction. A
context of 8 restatements of one sentence has the token cost of 8 chunks and
the information of one. MMR fixes this by trading relevance against novelty.

**Authority.** For a dosing question an FDA label section outranks a review
abstract. That is a property of the source, invisible to any similarity score.

**Budget.** The context window is finite, and *which* chunks get cut is a
measurable failure mode - ``LOST_IN_SELECTION`` exists precisely to name it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medassist.core.enums import SourceKind
from medassist.core.ids import ChunkId
from medassist.core.models import Chunk
from medassist.index.embed import EmbeddingPort, tokenize

# Sections that carry decision-grade content for a given question shape.
_DOSAGE_SECTIONS = {"dosage_and_administration", "dosage_forms_and_strengths"}
_SAFETY_SECTIONS = {
    "boxed_warning", "contraindications", "warnings_and_cautions",
    "warnings", "drug_interactions", "adverse_reactions",
}


@dataclass(frozen=True)
class RerankSpec:
    mmr_lambda: float = 0.72       # 1.0 = pure relevance, 0.0 = pure diversity
    authority_weight: float = 0.06
    section_weight: float = 0.08
    top_k: int = 12


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rerank(
    query: str,
    ranked: list[tuple[ChunkId, float]],
    chunks: dict[ChunkId, Chunk],
    *,
    spec: RerankSpec | None = None,
    embedding: EmbeddingPort | None = None,
    intent_sections: set[str] | None = None,
) -> list[tuple[ChunkId, float]]:
    """MMR over the fused list, with authority and section priors."""
    spec = spec or RerankSpec()
    if not ranked:
        return []

    candidates = [(cid, score) for cid, score in ranked if cid in chunks]
    if not candidates:
        return []

    # Fused scores are tiny (~1/60) and not comparable across queries; rescale
    # to [0,1] so the prior weights below mean the same thing every time.
    scores = np.array([s for _, s in candidates], dtype=np.float64)
    span = float(scores.max() - scores.min())
    base = (scores - scores.min()) / span if span > 1e-12 else np.ones_like(scores)

    priors = np.zeros_like(base)
    for i, (cid, _) in enumerate(candidates):
        chunk = chunks[cid]
        authority = float(chunk.meta.get("authority", 0)) / 3.0
        priors[i] += spec.authority_weight * authority
        if intent_sections and chunk.section in intent_sections:
            priors[i] += spec.section_weight
    relevance = base + priors

    if embedding is not None:
        vectors = embedding.encode([chunks[cid].text for cid, _ in candidates])
        similarity = vectors @ vectors.T
    else:
        token_sets = [set(tokenize(chunks[cid].text)) for cid, _ in candidates]
        similarity = np.array(
            [[_jaccard(a, b) for b in token_sets] for a in token_sets], dtype=np.float64
        )

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < spec.top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: spec.mmr_lambda * relevance[i]
                - (1 - spec.mmr_lambda) * max(similarity[i][j] for j in selected),
            )
        selected.append(best)
        remaining.discard(best)

    return [(candidates[i][0], float(relevance[i])) for i in selected]


def sections_for_intent(intent: str) -> set[str]:
    from medassist.core.enums import Intent

    return {
        Intent.DRUG_INFO.value: _DOSAGE_SECTIONS | {"indications_and_usage"},
        Intent.DRUG_INTERACTION.value: _SAFETY_SECTIONS,
        Intent.SYMPTOM_TRIAGE.value: {"overview", "warnings"},
        Intent.CONDITION_OVERVIEW.value: {"overview", "indications_and_usage"},
        Intent.REPORT_INTERPRETATION.value: {"overview"},
        Intent.EVIDENCE_LOOKUP.value: {"conclusions", "results", "abstract"},
    }.get(intent, set())


def select_context(
    ranked: list[tuple[ChunkId, float]],
    chunks: dict[ChunkId, Chunk],
    *,
    max_chars: int = 9000,
    max_chunks: int = 8,
    max_per_doc: int = 3,
) -> tuple[list[ChunkId], list[ChunkId]]:
    """Fill the context window in rank order. Returns ``(selected, dropped)``.

    ``max_per_doc`` stops one exhaustive FDA label from consuming the whole
    window and starving every other source - which on this corpus it otherwise
    reliably does, because labels are long and repetitive.
    """
    selected: list[ChunkId] = []
    dropped: list[ChunkId] = []
    used = 0
    per_doc: dict[str, int] = {}

    for chunk_id, _score in ranked:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        doc_count = per_doc.get(chunk.doc_id, 0)
        if (
            len(selected) >= max_chunks
            or used + len(chunk.text) > max_chars
            or doc_count >= max_per_doc
        ):
            dropped.append(chunk_id)
            continue
        selected.append(chunk_id)
        per_doc[chunk.doc_id] = doc_count + 1
        used += len(chunk.text)

    return selected, dropped


def format_context(chunk_ids: list[ChunkId], chunks: dict[ChunkId, Chunk]) -> str:
    """Render context with stable citation handles.

    Chunks are labelled ``[C1]``…``[Cn]`` rather than by their ULID. Models cite
    short handles far more reliably than 26-character identifiers, and the
    mapping back to real ids is done in code where it cannot be hallucinated.
    """
    blocks: list[str] = []
    for i, chunk_id in enumerate(chunk_ids, start=1):
        chunk = chunks[chunk_id]
        source = chunk.source.value if isinstance(chunk.source, SourceKind) else str(chunk.source)
        header = f"[C{i}] source={source} title={chunk.meta.get('doc_title', '')!r}"
        if chunk.section:
            header += f" section={chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)
