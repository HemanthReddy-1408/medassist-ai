# ADR-0002 — Record every retrieval stage, attribute every miss

**Status:** accepted · **Wave:** 0 · **BUILT**

## Context
`recall@10 = 0.71` says retrieval is imperfect. It does not say whether to fix
the scraper, the encoder, the reranker, or the context budget.

## Decision
`RetrievalTrace` records what survived each stage in rank order.
`attribute(gold, indexed)` walks them in order and names the first stage at
which no gold chunk survived: `NOT_INDEXED` → `NOT_RETRIEVED` →
`LOST_IN_RERANK` → `LOST_IN_SELECTION`.

## Consequences
Every miss carries a diagnosis. Ordering is the method — a chunk never indexed
cannot be blamed on the reranker. Cost: traces are large; the corpus-wide
indexed set is passed at attribution time rather than stored per query, which
would otherwise make traces larger than the corpus.

## Rejected
*Log the final context only.* Cheapest to store and useless for diagnosis,
which is the entire reason the trace exists.
