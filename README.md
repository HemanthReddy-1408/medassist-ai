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

Rules are keyed on drug *classes*, so one grapefruit rule covers every CYP3A4
substrate — and deliberately does not cover rosuvastatin, which is not one. A
rule naming simvastatin protects nobody on lovastatin though the mechanism is
identical, which is the kind of gap that looks like coverage until someone is
harmed by it.

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

Every row below is implemented and tested. The repository this replaced
advertised "Multi-source RAG" while `pubmed_node` returned the string
`"Clinical data lookup not implemented yet."` That is the pattern this project
is organised against.

| Subsystem | Lines | Tests |
|---|---:|---:|
| Domain model — 15 closed enums, monotonic ULIDs, model-level invariants | 858 | 26 |
| Model gateway — retries, disk cache, cost, JSON fallback, adaptive budget | 354 | 21 |
| Corpus — 3 live scrapers, section offsets, content-addressed snapshots | 576 | 22 |
| Retrieval — chunking, BM25 + dense, RRF, MMR rerank, full trace | 866 | 29 |
| **Release gate** — 6 checks, cascade, fail-closed, per-check attribution | 1,319 | 45 |
| Patient model — drug classes, brand resolution, interaction rules | 183 | 25 |
| Clinical guards — triage, injection, PII, deny-by-default policy | 376 | 29 |
| Capabilities + orchestration — registry, validated DAG, parallel executor | 687 | 30 |
| Reports — parsing, reference intervals, findings, trends | 404 | 31 |
| Confidence — six factors, ECE/Brier, risk–coverage, threshold selection | 381 | 36 |
| **Longitudinal memory** — shelf life, continuity, confirmations | 446 | 27 |
| Red team — 15 attacks, 10 classes, scored with intervals | 475 | 29 |
| Decision records + serving — pipeline, API, CLI, UI | 848 | 22 |

**8,274 lines of implementation · 3,367 of tests · 1,751 of specification.**
**448 tests, 2.4s, no network or API key required.**

---

## Memory across visits

> A profile uploads a report showing haemoglobin low at 9.1 g/dL. Five months
> later, the same profile uploads a lipid panel with no haemoglobin on it.

Forgetting the finding loses something the person is still living with.
Assuming it still holds means reasoning from a five-month-old value. So:

```
Hemoglobin: 9.1 (2026-04-08) -> not measured   [unchecked]
HbA1c:      7.4 (2026-04-08) -> 8.2            [worsening]

Before I use your history, please confirm:
  - The most recent Hemoglobin I have is 9.1 g/dL (low) as of 2026-04-08,
    which is now 150 days old. Should I still treat that as current?
```

Shelf life is a clinical property, not a storage policy — INR expires in 30
days, a chronic diagnosis in two years. Questions are capped at three and
ordered by severity, because a system that opens with nine gets none answered.
Details in [§13](docs/spec/13-longitudinal-memory.md).

---

## Adversarial results

15 attacks across 10 classes, each with a machine-checkable expected behaviour.
The judge is **scripted to say "supported" for every attack**, so any defence
that holds was held by deterministic code with no help from the model.

```
defended 15/15 (100%, 95% CI 80%-100%)
over-refusals: 0
```

The interval is doing honest work: 15 cases do not establish much, and a point
estimate would imply precision it does not have. Two attacks failed on first
run and both were real bugs — a 10,000-character "claim" released under a
caveat, and an authority-spoofing payload in a retrieved chunk that produced a
diagnosis. Both fixed, both now regression tests.

`make redteam` reproduces it.

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

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add a Groq (or OpenAI-compatible) key
make test                     # 448 tests, offline, no API key
make redteam                  # the adversarial suite
make demo                     # live: scrape → index → retrieve, with the trace
make demo-gate                # live: generate claims → run the gate
make serve                    # HTTP API on :8000
make ui                       # Streamlit, with the evidence pane
```

## Documentation

[`docs/spec/`](docs/spec/) — 14 documents. Start with
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
