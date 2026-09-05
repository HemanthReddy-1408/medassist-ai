# ADR-0003 — Corpus snapshots are content-addressed and verified on read

**Status:** accepted · **Wave:** 0 · **BUILT**

## Context
A metric is meaningless without the corpus it was measured against. "Recall
rose from 0.61 to 0.74" may record only that someone re-scraped PubMed.

## Decision
`sha256` over sorted `(source_uid, text)` pairs, truncated to 16 hex chars.
Document ids are excluded — they are fresh ULIDs per scrape and would make
every rebuild a new snapshot. `read()` re-hashes and **refuses a corrupt
snapshot**.

## Consequences
Every `RunRecord` carries its snapshot; comparisons across snapshots are
flagged. Re-labelling gold chunk ids on rebuild is real cost, accepted as the
price of the retrieval metrics meaning anything.
