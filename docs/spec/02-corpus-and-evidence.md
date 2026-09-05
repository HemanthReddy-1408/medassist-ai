# 02 — Corpus, Provenance, and Temporal Validity

## 2.1 Sources  **BUILT**

Three, all public, all verified live.

| Source | API | Yields | Authority |
|---|---|---|---:|
| openFDA drug labels | `api.fda.gov/drug/label.json` | Structured Product Labels, 16 clinical sections | 3 |
| PubMed | NCBI E-utilities `esearch` + `efetch` | Abstracts with structured section labels | 2 |
| MedlinePlus | `wsearch.nlm.nih.gov` | NLM consumer health topics | 2 |

**Why these three and not a web crawler.** Each is a stable, citable, versioned
public API with a licence permitting this use. A general web scrape yields
documents with no authority ranking, no publication date, and no stable
identifier — which makes conflict resolution (§2.4) and temporal reasoning
(§2.5) impossible, and citations unverifiable.

MedlinePlus earns its place because a large share of real questions arrive in
lay language, and a corpus of only abstracts and regulatory labels answers them
in a register the asker cannot use. Retrieval quality is measured per source,
so the effect of that choice is visible rather than assumed.

### Politeness  **BUILT**

NCBI permits 3 requests/second unauthenticated and will block above it. The
fetcher enforces a per-host minimum interval (thread-safe), retries only
429/5xx with exponential backoff, and caches every raw response on disk. A
corpus rebuild does not re-hammer a service offering the data for free.

## 2.2 Normalization and sections  **BUILT**

Sections are carried as explicit `(name, start, end)` character offsets, not
inferred later from markdown headings.

This is load-bearing for safety. The dosage guard (§07.2) requires that a
numeric dose in an answer cite a chunk from an FDA label's
`dosage_and_administration` section. Recovering that by re-parsing prose would
be guesswork on precisely the claim where guessing is least acceptable.

FDA sections retained, ordered by clinical weight rather than SPL order:
`boxed_warning`, `indications_and_usage`, `dosage_and_administration`,
`dosage_forms_and_strengths`, `contraindications`, `warnings_and_cautions`,
`warnings`, `drug_interactions`, `adverse_reactions`,
`use_in_specific_populations`, `pregnancy`, `pediatric_use`, `geriatric_use`,
`overdosage`, `mechanism_of_action`, `information_for_patients`.
`how_supplied` and packaging panels are dropped — noise that dilutes retrieval.

Text normalization applies NFKC before tokenization. Clinical text from SPL
documents and PDFs is full of ligatures and full-width forms; unfolded, they
break lexical matching on exactly the drug names that matter.

*Verified:* every emitted section offset resolves to its own heading, and
`doc.text[chunk.start:chunk.end] == chunk.text` for every chunk.

## 2.3 Snapshots  **BUILT**

> An evaluation number is meaningless without the corpus it was measured
> against.

"Recall@10 rose from 0.61 to 0.74" is only a result if the corpus was
identical. Otherwise it may record that someone re-scraped PubMed and got three
more papers.

A snapshot is therefore content-addressed: `sha256` over `(source_uid, text)`
pairs in sorted order, truncated to 16 hex characters. Document ids are freshly
minted ULIDs on every scrape, so hashing them would make every rebuild a new
snapshot even when nothing changed — defeating the purpose.

`SnapshotStore.read()` **re-hashes on load and refuses a corrupt snapshot.** A
silently-edited corpus would invalidate every number measured against it, and
the failure would otherwise be invisible. *Verified: a one-character edit is
detected and raises.*

Every `RunRecord` carries `corpus_snapshot`. §08 refuses to compare two runs
from different snapshots without saying so.

## 2.4 Evidence graph and conflict detection  **SPECIFIED**

Nodes are `Claim`, `Chunk`, `Document`. Edges are typed:

```
supports · contradicts · updates · supersedes · derived_from · cites
```

Conflict detection runs when two retrieved chunks assert incompatible things
about the same subject — a genuine occurrence on this corpus, since a 2019
review and a 2024 label routinely disagree on first-line therapy.

**Resolution is by ordered rule, not by majority vote.** Majority vote across
retrieved chunks measures how many times something was written down, which
correlates with how prolific a topic is, not with what is true.

Ordered resolution factors:

1. **Authority** — FDA label > NLM consumer topic ≈ peer-reviewed abstract.
2. **Temporal validity** — a superseding document wins (§2.5).
3. **Study design** — systematic review / RCT > cohort > case report, read from
   PubMed `PublicationType`, which is curated metadata rather than inference.
4. **Population match** — a paediatric finding does not resolve an adult query.

When the rules do **not** produce a winner, the conflict is surfaced in
`Answer.conflicts` and the confidence penalty in §06.5 applies. Unresolved
disagreement is reported, never averaged into a single confident sentence.
This is a stopping condition (§05.6), not an error.

## 2.5 Temporal validity  **SPECIFIED**

Every document carries `published`, `fetched_at`, and derived `valid_from` /
`superseded_by`. Medical guidance changes; a system that cites a 2016
recommendation with no indication that a 2023 revision exists is confidently
wrong in the most damaging way.

Rules:

- A retrieved chunk older than a configurable horizon (default 8 years) for an
  intent that depends on current practice is **annotated**, not dropped.
  Dropping it loses the mechanism and history a question may actually be about.
- Where two documents cover the same subject and one supersedes the other, the
  superseded chunk is demoted in ranking and its use in a claim forces a
  temporal caveat into the answer.
- `effective_time` on an FDA label is the authoritative recency signal for
  dosing; it is a regulatory field, not a scrape artefact.
