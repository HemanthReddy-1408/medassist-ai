# MedAssist X

**Serving-time admission control for clinical answers.**

A multi-agent system that produces answers, and a **release gate** that decides,
per request and in-band, whether each one may reach the person who asked.

> **This system is not clinically reliable and is not intended to be.** It
> cannot replace a clinician. What it *is*: a demonstration that an LLM answer
> can be held to a decision procedure rather than trusted.

---

## The constraint everything follows from

> **At serving time there is no gold label, and there is no second chance.**

An evaluation harness compares an answer against a known-correct one, offline,
with unlimited time. The gate has neither. It must decide whether *this* answer
may reach *this* person using only the answer, the retrieved context, and the
patient's own record — before anything is shown.

So every check is **reference-free** (none may require knowing the right
answer) and **affordable** (all of them run before the user sees a token).

## What the gate actually does

Four checks, cheapest first. A claim removed by check 2 never reaches the model
call, so the expensive stage runs on a shrinking set.

```
  claims
    ├─ 1. citation resolution   ~0.02 ms   handle indexes nothing  → REMOVE
    ├─ 2. numeric grounding     ~0.30 ms   digits not in a citation → REMOVE
    ├─ 3. relational safety     ~3.00 ms   wrong for this patient  → QUALIFY / REMOVE
    └─ 4. entailment          ~1400 ms   cited source disagrees   → REMOVE + BLOCK
  ↓
  RELEASE · RELEASE_WITH_CAVEAT · ABSTAIN · BLOCK · ESCALATE
```

Measured on a live run against a real corpus — see `make demo-gate`.

### Check 2 beats the expensive one

```
claim   "The recommended starting dose is 800 mg twice daily."
cites   [C1] → "The recommended starting dose is 500 mg twice daily."
verdict REMOVE — numeric_grounding: 800 mg appears in no cited span   (0.33 ms)
```

An LLM judge asked "is this consistent?" frequently accepts that — the sentence
is otherwise identical and both numbers are the same kind of thing. Arithmetic
does not, costs nothing, and cannot be argued with. Meanwhile `2.5 g` and
`2500 mg` *do* match, because units are normalized before comparison.

### Check 3 is the one nothing else catches

```
claim    "Increase potassium intake."          ← true, well-cited, grounded
patient  medications: [warfarin, lisinopril]
verdict  REMOVE — hyperkalaemia risk with an ACE inhibitor   (3.03 ms)
```

Every groundedness and faithfulness metric passes this claim, because every one
of them stops at the corpus. Correctness here is **relational** — it holds
between the answer and the patient's record. That is why the gate takes the
patient record as an input rather than treating personalization as a prompt
detail, and why the interaction table is curated, cited data rather than model
knowledge.

### Everything fails closed

| Failure | Behaviour |
|---|---|
| Judge unreachable, or skips a claim | `ABSTAIN` — unverified is not verified |
| Retrieval found nothing | `ABSTAIN`, naming the gap |
| Gate latency budget exhausted | `ABSTAIN` |
| Output truncated by a reasoning preamble | `ABSTAIN`, diagnosed as budget exhaustion |

A safety component that disables itself under load is not a safety component.

---

## What this is *not*

Not an evaluation platform, and the distinction is deliberate.

| | An evaluation platform | **MedAssist X** |
|---|---|---|
| Question | *Did this change help?* | *May this answer be released?* |
| When | Offline, batch | **In-band, per request** |
| Gold labels | Always available | **Never available** |
| Method | Comparative inference across arms | Reference-free verification + gating |
| Verdict about | A **change** | A specific **answer, for a specific person** |

Paired significance tests, multiple-comparison correction, cross-arm regression
gates and experiment tracking are **out of scope on purpose** — they answer
questions about *changes*, offline, with labels. This repository exports
decision records for such a platform to consume ([§10.3](docs/spec/10-decision-audit.md))
and stops there.

---

## Status — what is actually built


This README distinguishes working code from design. The repository it replaced
advertised "Multi-source RAG" while `pubmed_node` returned the string
`"Clinical data lookup not implemented yet."` That is the pattern this project
is organised against.

| Subsystem | Lines | Status |
|---|---:|---|
| Domain model — 12 closed enums, typed ULIDs, model-level invariants | 848 | **BUILT**, 26 tests |
| Model gateway — retries, disk cache, cost, JSON-mode fallback, adaptive budget | 354 | **BUILT**, 21 tests |
| Corpus — 3 live scrapers, section offsets, content-addressed snapshots | 576 | **BUILT**, 22 tests |
| Retrieval — chunking, BM25 + dense, RRF, MMR rerank, full trace | 866 | **BUILT**, 29 tests |
| **Release gate** — 4 checks, cascade, fail-closed, per-check attribution | 1,041 | **BUILT**, 45 tests |
| Claim-structured generation, handle resolution | 134 | **BUILT** |
| Capabilities, orchestration, agents | — | **SPECIFIED** — [§04](docs/spec/04-capabilities-and-contracts.md), [§05](docs/spec/05-orchestration.md) |
| Confidence calibration, risk–coverage | — | **SPECIFIED** — [§06](docs/spec/06-verification-and-confidence.md) |
| Red-flag triage, policy engine | — | **SPECIFIED** — [§07](docs/spec/07-safety-and-policy.md) |
| Adversarial cases, decision records | — | **SPECIFIED** — [§09](docs/spec/09-redteam.md), [§10](docs/spec/10-decision-audit.md) |

**152 tests, 1.6s, no network or API key required.** Live-provider tests are marked and
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
make test                     # 152 tests, offline
make demo                     # live: scrape → index → retrieve, with the trace
make demo-gate                # live: generate claims → run the gate
```

## Documentation

[`docs/spec/`](docs/spec/) — 13 documents. Start with
[§08 The Release Gate](docs/spec/08-release-gate.md), the centrepiece, then
[§00 Overview](docs/spec/00-overview.md) for scope and **non-goals**, and
[§03 Retrieval](docs/spec/03-retrieval.md) for what feeds the gate.

[`docs/adr/`](docs/adr/) — 8 decision records, each with what was rejected and
why.

## Honesty rules for every number this project publishes

1. State the **sample size** and a **confidence interval**.
2. Name the **corpus snapshot hash** it was measured against.
3. If a judge produced it, state its **agreement with human labels** (Cohen's κ)
   — or brand it `UNCALIBRATED`. Enforced by a model validator, not by convention.
4. Publish the metrics that came out badly alongside the ones that did not.
