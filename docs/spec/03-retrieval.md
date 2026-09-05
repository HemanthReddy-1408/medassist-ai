# 03 — Retrieval  **BUILT**

`medassist/index/` — 776 lines. The Evidence Engine's retrieval half.

## 3.1 Pipeline

```
query (+ optional expansions)
   │
   ├──► DENSE     cosine over normalized embeddings, k=30
   ├──► SPARSE    Okapi BM25, k=30
   │
   ▼
 FUSED           reciprocal rank fusion, k=60
   │
   ▼
 RERANKED        MMR (λ=0.72) + authority prior + section prior, top 12
   │
   ▼
 SELECTED        context budget: 8 chunks / 9000 chars / max 3 per doc
   │
   ▼
 RetrievalTrace  every stage recorded, in rank order
```

## 3.2 Chunking

Section-aware, sentence-aligned, 900 chars target with 150 overlap.

Fixed-width chunking cuts an FDA label mid-sentence and produces a span
beginning `"...mg twice daily in patients with"` — unusable as a citation and
actively dangerous as a dosage source. So cuts happen inside section boundaries
and on sentence boundaries.

The sentence splitter refuses to break on `2.5` (decimal), `e.g.`, `i.e.`,
`vs`, `Dr.`, or `No.` Splitting `2.5 mg` into two sentences destroys exactly
the dosage terms that matter most. *Verified against the live metformin label:
101 chunks, all offsets resolving exactly.*

A sentence longer than the target becomes its own chunk rather than being
force-split; enumerated dosage sentences are meaningful whole and meaningless
in halves.

Overlap exists because a fact straddling a boundary is otherwise retrievable
from neither side. The cost is duplicated text, which MMR removes downstream.

## 3.3 Why both a dense and a lexical index

A lexical index is not a fallback here; on this corpus it is frequently the
better retriever. Clinical queries hinge on exact strings — a drug name,
`HbA1c`, `2.5 mg`. A dense encoder trained on general web text will rate
`metoprolol` and `metformin` as similar because they *look* alike. BM25 will
not make that mistake.

Conversely BM25 has no signal at all on `"what can I eat"` against `"dietary
modification"`. Neither retriever dominates, so both run and are fused.

The tokenizer keeps intra-word hyphens and decimals, so `type-2`, `hba1c` and
`2.5` survive as single tokens.

BM25 uses Robertson/Sparck-Jones idf **floored at a small positive value**. The
raw form goes negative for terms appearing in more than half the corpus, which
would let a common word *subtract* from a document's score.

## 3.4 Fusion: rank, not score

RRF combines by position rather than score, and that is the point. BM25 scores
are unbounded sums of idf terms; cosine scores live in [-1, 1]. Mapping them
onto a common scale requires assuming a distribution for each, and that
assumption is wrong often enough to reorder results. Rank is comparable with no
such assumption.

Ties break on chunk id so the ordering is total. Without it, two equally-fused
chunks can swap between runs and make a deterministic evaluation look flaky.

## 3.5 Reranking

Three problems separate a ranked list from a usable context window:

**Redundancy.** The top 8 fused hits are frequently the same paragraph, because
overlapping chunks are near-duplicates by construction. Eight restatements of
one sentence cost eight chunks of budget and carry one chunk of information.
MMR trades relevance against novelty (λ=0.72).

**Authority.** For a dosing question an FDA label section outranks a review
abstract — a property of the source, invisible to any similarity score.

**Section priors.** A `drug_info` query boosts `dosage_and_administration`; a
`drug_interaction` query boosts the safety sections. Fused scores are rescaled
to [0,1] before the priors are added, so a fixed prior weight means the same
thing on every query.

## 3.6 Selection and the context budget

`max_per_doc=3` stops one exhaustive FDA label from consuming the whole window
and starving every other source — which on this corpus it otherwise reliably
does, since labels are long and repetitive.

Chunks that survive ranking but do not fit are recorded in
`truncated_by_budget`. That list is what makes `LOST_IN_SELECTION` a
distinguishable diagnosis rather than a guess.

Context is rendered with stable handles `[C1]…[Cn]` rather than ULIDs. Models
cite short handles far more reliably, and the mapping back to real chunk ids is
done in code where it cannot be hallucinated. **This is a hallucination
countermeasure, not a formatting choice** — a model asked to reproduce
`chk_01M1RF1J98K007M79A3XMWMK49` will sometimes emit a plausible-looking
identifier that indexes nothing.

## 3.7 Multi-query expansion (RAG-Fusion)

Optional. Paraphrases are retrieved separately and all runs are fused. It helps
most where lay phrasing shares no vocabulary with clinical text. It also costs
a model call and multiplies retrieval latency — so §08 measures whether it
earns that, per intent, rather than enabling it by assumption.

## 3.8 Embedding backends, and why there are two

| Backend | Basis | Role |
|---|---|---|
| `HashingEmbedding` | Hashed words + char trigrams, sublinear tf, blake2b | Deterministic baseline, CI default, zero download |
| `SentenceTransformerEmbedding` | MiniLM or a biomedical encoder | Real semantics |

Char trigrams give the hashing backend morphological signal
(`nephropathy`/`nephropathies`) that plain word hashing lacks. blake2b rather
than Python's `hash()` because the latter is salted per process — a corpus
embedded in one process would not match a query embedded in another.

Having both is not indecision. `make eval-embeddings` runs both arms over the
same gold set and reports whether the difference survives a significance test
(§08.4). The choice becomes a measurement instead of a preference.

Encodings are memoized on disk keyed by backend name and text hash;
re-embedding an unchanged corpus per evaluation arm is otherwise the single
largest avoidable cost in the harness.
