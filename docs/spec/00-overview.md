# 00 — Overview, Scope, and Non-Goals

**MedAssist X** — a safety-critical, evidence-grounded multi-agent platform, with
clinical question answering as its first environment.

Status vocabulary used throughout this spec. It is load-bearing: a spec that
does not distinguish what exists from what is planned is marketing.

| Marker | Meaning |
|---|---|
| **BUILT** | Implemented, exercised by tests, verified against live services |
| **PARTIAL** | Implemented, with a named gap stated inline |
| **SPECIFIED** | Designed here in enough detail to implement; no code yet |
| **DEFERRED** | Deliberately not being built, with the reason recorded |

---

## 1. The principle everything derives from

> **The runtime owns control flow. The model only proposes.**

Every enforcement point in this system exists to keep that sentence true. An
LLM chooses *what to suggest*; it never chooses what is permitted, what is
executed, what is cited, or what is released. The moment those collapse into
one another, no amount of measurement is meaningful, because the thing being
measured also decides whether it passed.

This has a direct consequence for the architecture: **an answer is not a
string**. It is a list of atomic claims, each carrying citations to spans of
retrieved text. A blob of prose can only be assessed with a vibe. A list of
claims can be attributed one at a time, mechanically, and reported as "7 of 9
supported, 1 unsupported, 1 contradicted by the cited source."

## 2. Seven laws

Each is enforced somewhere specific, named here so the claim is checkable.

| # | Law | Enforced by |
|---|---|---|
| 1 | No answer without evidence where evidence is required | Verification engine (§06), `Claim.citations` |
| 2 | No action without authorization | Policy engine (§07), capability contracts (§04) |
| 3 | No high-risk output without independent verification | Decision gate (§07.4) |
| 4 | No confidence number without calibration | `JudgeReliability` invariant (§01.4), ECE/Brier (§06.5) |
| 5 | No agent autonomy without a bounded budget | `Budget.subdivide` (§05.4) |
| 6 | No release without regression evaluation | CI gates (§08.6) |
| 7 | Every discovered failure becomes a permanent test | Red-team → regression loop (§09.4) |

Law 4 is enforced in the type system rather than by convention:
`ClaimAssessment` refuses construction if `reliability=CALIBRATED` and `kappa is
None`. A judge score either carries the agreement coefficient it was measured
at, or it is branded `UNCALIBRATED`. There is no third option. **BUILT**

## 3. Honest framing

This system is not clinically reliable and will not be. It cannot replace a
clinician, and no measurement in this repository should be read as evidence
that it could. What it is:

> A research-grade platform whose reliability is **continuously measured,
> adversarially challenged, statistically calibrated, and gated in CI.**

That is a defensible engineering claim. "Most reliable medical AI" is not, and
publishing it would discredit every real number in the repository.

Concretely, the honesty rules for any number this project publishes:

1. Every metric states its **sample size** and a **confidence interval**.
2. Every metric names the **corpus snapshot hash** it was measured against.
3. A judge-produced metric states its **agreement with human labels**.
4. Metrics that came out badly are published alongside those that did not.

## 4. Scope — six capabilities, not thirty

The capability set is deliberately small. Each one exercises different
machinery, so the set demonstrates breadth without the architecture becoming a
list of stubs.

| Capability | Exercises | Status |
|---|---|---|
| `evidence_qa` | Hybrid retrieval, claim verification, citation binding | **SPECIFIED** |
| `interaction_check` | Structured cross-referencing, high-risk gating | **SPECIFIED** |
| `report_interpretation` | Document parsing, analyte extraction, reference intervals | **SPECIFIED** |
| `lifestyle_guidance` | Personalization, drug–food contraindication cross-check | **SPECIFIED** |
| `treatment_comparison` | Multi-source conflict detection, evidence grading | **SPECIFIED** |
| `literature_synthesis` | Fan-out retrieval, deduplication, contradiction analysis | **SPECIFIED** |

`lifestyle_guidance` is the one to look at closely. "What should I eat?" is a
lay question with a genuinely hard safety property behind it: the answer must
be cross-checked against the asker's own medication list. Advising leafy greens
to a patient on warfarin is a vitamin-K interaction; advising grapefruit to a
patient on simvastatin is a CYP3A4 interaction. Both are *correct general
advice* and *wrong for that person*, which is a failure mode no
groundedness metric catches — the claim is perfectly supported by the corpus.
It requires a distinct guard (§07.3).

## 5. Non-goals

Recorded because an unstated non-goal reads as an oversight.

| Not building | Why |
|---|---|
| Synthetic patient simulation at scale | If a model writes both the case and the gold answer, the evaluation measures the generator's self-consistency. 60 hand-labelled cases on a real corpus are worth more than 10,000 synthetic ones. |
| Arbitrary code-execution sandbox | A correct sandbox (cgroups, seccomp, network namespaces) is its own project. A half-correct one is a vulnerability presented as a feature. |
| General coding assistant capability | Dilutes the domain story and shares no machinery with the rest. |
| Video/audio modality | No corpus, no gold labels, therefore no way to measure it. |
| Separate graph database | The evidence graph is thousands of nodes. Postgres/in-memory adjacency is correct at this scale; Neo4j would be resume decoration. |
| Replacing clinical judgement | See §3. |

## 6. What exists today

2,517 lines, verified against live PubMed, openFDA and MedlinePlus.

| Subsystem | Lines | Status |
|---|---:|---|
| Domain model, ids, enums, config, errors | 654 | **BUILT** |
| Model gateway — retries, disk cache, cost, structured output w/ repair | 302 | **BUILT** |
| Corpus — three live scrapers, normalization, content-addressed snapshots | 411 | **BUILT** |
| Retrieval — chunking, dense+sparse, RRF, MMR rerank, traced pipeline | 776 | **BUILT** |
| Capability registry, orchestration, agents | — | **SPECIFIED** (§04–§05) |
| Verification, confidence calibration | — | **SPECIFIED** (§06) |
| Safety, policy engine, decision gate | — | **SPECIFIED** (§07) |
| Evaluation harness, statistics, CI gates | — | **SPECIFIED** (§08) |
| Red team, regression loop | — | **SPECIFIED** (§09) |
| Observability, replay, experiments | — | **SPECIFIED** (§10) |

## 7. Reading order

`01` domain model → `03` retrieval → `06` verification are the spine. Read
those three and the system's actual contribution is clear. `04`–`05` are the
agent infrastructure, `07`–`09` the reliability plane, `10`–`11` the surfaces.
