# MedAssist X

**A safety-critical, evidence-grounded multi-agent platform — with clinical
question answering as its first environment.**

Not a medical chatbot. The contribution is the machinery that decides whether
an answer is allowed to be released, and the harness that measures how often it
gets that right.

> **This system is not clinically reliable and is not intended to be.** It
> cannot replace a clinician, and no number in this repository should be read
> as evidence that it could. What it *is*: a platform whose reliability is
> continuously measured, adversarially challenged, statistically calibrated,
> and gated in CI.

---

## The two questions this is built to answer

Most RAG systems can tell you an answer looked wrong. This one is built to tell
you **which part broke**, because those have different fixes.

### 1. Was the retrieval wrong?

Every retrieval records what survived each stage. A miss is attributed to the
first stage that lost the gold chunk:

```
NOT_INDEXED        the corpus never had it          → fix the scraper
NOT_RETRIEVED      indexed, neither retriever found it → fix the encoder
LOST_IN_RERANK     retrieved, then discarded         → fix reranker weights
LOST_IN_SELECTION  ranked well, cut by the budget    → fix the context window
```

`recall@10 = 0.71` tells you to work harder. `recall@10 = 0.71, of which 62%
NOT_INDEXED` tells you that touching the reranker would be wasted effort.

### 2. Did the model hallucinate?

**An answer is not a string.** It is a list of atomic claims, each carrying
citations to spans of retrieved text — so each one is verified independently by
three verifiers, and the runtime (not the model) decides what is released:

```
SUPPORTED     → retain
UNSUPPORTED   → qualify or remove
CONTRADICTED  → remove, record a safety event
UNVERIFIABLE  → retain   (hedges aren't factual assertions)
```

One of the three verifiers is deterministic: **every number in a claim must
appear in a cited span.** An LLM judge will accept "2000 mg" as approximately
consistent with "2550 mg". In dosing, approximately consistent is wrong.

---

## Status — what is actually built

This README distinguishes working code from design. The repository it replaced
advertised "Multi-source RAG" while `pubmed_node` returned the string
`"Clinical data lookup not implemented yet."` That is the pattern this project
is organised against.

| Subsystem | Lines | Status |
|---|---:|---|
| Domain model — 9 closed enums, typed ULIDs, model-level invariants | 654 | **BUILT**, 30 tests |
| Model gateway — retries, disk cache, cost, structured output + repair | 302 | **BUILT**, 14 tests |
| Corpus — 3 live scrapers, section offsets, content-addressed snapshots | 411 | **BUILT**, 22 tests |
| Retrieval — chunking, BM25 + dense, RRF, MMR rerank, full trace | 776 | **BUILT**, 35 tests |
| Capabilities, orchestration, agents | — | **SPECIFIED** — [§04](docs/spec/04-capabilities-and-contracts.md), [§05](docs/spec/05-orchestration.md) |
| Verification, confidence calibration | — | **SPECIFIED** — [§06](docs/spec/06-verification-and-confidence.md) |
| Guards, policy engine, decision gate | — | **SPECIFIED** — [§07](docs/spec/07-safety-and-policy.md) |
| Evaluation harness, statistics, CI gates | — | **SPECIFIED** — [§08](docs/spec/08-evaluation.md) |
| Red team, regression loop | — | **SPECIFIED** — [§09](docs/spec/09-redteam.md) |

**101 tests, 0.24s, no network required.** Live-provider tests are marked and
excluded by default — a suite needing an API key is a suite that gets skipped.

---

## Corpus

Three public sources, all verified live. Not a general web crawl: each is
stable, citable, dated, and licensed for this use — without which conflict
resolution and temporal reasoning are impossible and citations unverifiable.

| Source | Yields | Authority |
|---|---|---:|
| openFDA drug labels | Structured Product Labels, 16 clinical sections | 3 |
| PubMed (E-utilities) | Abstracts with structured section labels | 2 |
| MedlinePlus | NLM consumer health topics | 2 |

Sections are carried as **character offsets**, not inferred later. This is
load-bearing: the dosage guard requires a numeric dose to cite a chunk from an
FDA label's `dosage_and_administration` section, and recovering that by
re-parsing prose would be guesswork on the claim where guessing is least
acceptable.

Snapshots are **content-addressed and re-verified on read.** A metric is
meaningless without the corpus it was measured against — "recall rose from 0.61
to 0.74" may only record that someone re-scraped PubMed.

---

## The guard that motivates the architecture

```
"I'm on warfarin. What should I eat?"
      ↓
"Leafy greens are an excellent source of vitamins."
```

That claim is **perfectly grounded** — leafy greens are healthy, and the corpus
says so. It is also a vitamin-K interaction with warfarin.

No groundedness metric catches this. No faithfulness score catches this.
Correctness here is *relational* — the answer is wrong for **this person** —
and only an explicit cross-check against the patient's own medication list
sees it. That is why `lifestyle_guidance` is a MEDIUM-risk capability with a
dedicated guard, and why guards are deterministic code rather than prompt
instructions.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add a Groq (or OpenAI-compatible) key
make test                     # 101 tests, offline
make demo                     # live: scrape → index → retrieve, with the trace
```

## Documentation

[`docs/spec/`](docs/spec/) — 13 documents, 1,500 lines. Start with
[§00 Overview](docs/spec/00-overview.md) for scope and **non-goals**, then
[§01 Domain Model](docs/spec/01-domain-model.md) →
[§03 Retrieval](docs/spec/03-retrieval.md) →
[§06 Verification](docs/spec/06-verification-and-confidence.md) for the spine.

[`docs/adr/`](docs/adr/) — 8 decision records, each with what was rejected and
why.

## Honesty rules for every number this project publishes

1. State the **sample size** and a **confidence interval**.
2. Name the **corpus snapshot hash** it was measured against.
3. If a judge produced it, state its **agreement with human labels** (Cohen's κ)
   — or brand it `UNCALIBRATED`. This one is enforced by a model validator, not
   by convention.
4. Publish the metrics that came out badly alongside the ones that did not.
