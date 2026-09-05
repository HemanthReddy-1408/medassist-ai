# MedAssist X — Technical Reference

A micro-level description of what exists, how a request moves through it, and
why each non-obvious decision was made. Written against the code as it stands.
Every number here was measured on this machine against the committed corpus
snapshot `9dceed0a7ffb3338`; anything unmeasured is labelled.

- **Specification:** `docs/spec/00`–`13` + 8 ADRs. This document is the
  *implementation* companion; the spec is the contract.
- **Scale:** 8,274 lines of implementation, 3,367 of tests, 1,751 of spec.
- **Tests:** 448, 2.4 s, no network and no API key.
- **Operational guide:** [RUNNING.md](RUNNING.md).

This document is organised around **the path of a single request**, because
this is a serving system and its behaviour is a sequence, not a diagram. Module
reference follows in §7.

---

## 1. The one sentence

> **At serving time there is no gold label, and there is no second chance.**

Everything below follows. Each check must be *reference-free* — none may need
to know the right answer, because nothing does — and *affordable*, because all
of them run before the user sees a token.

---

## 2. The request path

```
  ask("can I eat leafy greens?", profile=…)
        │
   [1]  ├─ triage                       6 µs      BLOCK on hit, no retrieval
   [2]  ├─ PII redaction               40 µs      nothing unredacted leaves the box
   [3]  ├─ retrieval                    8 ms      dense ∥ sparse → RRF → MMR → select
   [4]  ├─ claim generation          ~900 ms      1 model call, JSON, handles not ULIDs
   [5]  ├─ THE GATE
        │   ├─ structural              1.2 µs     atomic-claim bound
        │   ├─ citation resolution     2.0 µs     handle → real chunk, in-window
        │   ├─ context integrity      26 µs       is the *source* compromised?
        │   ├─ numeric grounding      26 µs       digits present in a cited span
        │   ├─ dosage provenance       1.9 µs     label §dosage or nothing
        │   ├─ relational safety     249 µs       wrong for *this* patient?
        │   └─ entailment           ~1400 ms      1 batched model call, survivors only
   [6]  ├─ decision record            80 µs       per-check verdicts, salted digest
   [7]  └─ confidence                 15 µs       six factors, product in log space
        ↓
   RELEASE · RELEASE_WITH_CAVEAT · ABSTAIN · BLOCK · ESCALATE
```

**Ordering is the specification, not an optimisation.** Three consequences are
tested directly:

1. Triage pre-empts everything. On a red flag the run returns in **0 ms at
   $0.00000** — no retrieval, no generation, no model call to soften the
   referral. `test_red_flag_blocks_before_any_model_call` asserts the generator
   was called zero times.
2. Deterministic checks run before the expensive one, so entailment sees a
   *shrinking* set. `test_deterministic_removal_skips_the_model_call` asserts a
   claim removed by numeric grounding never reaches the judge.
3. Redaction precedes generation. `test_pii_never_reaches_the_generator`
   inspects the prompt the stub subject actually received.

### 2.1 Why the cheap checks earn their place

The 26 µs numeric check catches what the 1,400 ms entailment check does not:

```
claim   "The recommended starting dose is 800 mg twice daily."
cites   [C1] → "The recommended starting dose is 500 mg twice daily."
```

An entailment judge frequently accepts this. The sentence is otherwise
identical and both numbers are the same *kind* of thing. Arithmetic does not,
costs 26 µs, and cannot be argued with. Meanwhile `2.5 g` and `2500 mg` *do*
match, because units normalise to a base before comparison
(`gate/quantities.py`).

### 2.2 The check nothing else catches

```
claim    "Increase potassium intake."      ← true, well-cited, entailed
patient  medications: [warfarin, lisinopril]
verdict  REMOVE — hyperkalaemia with an ACE inhibitor
```

Every groundedness metric passes that claim, because every one of them stops at
the corpus. Correctness here is **relational** — it holds between the answer and
the patient's record. It is also the slowest deterministic check by 10× (249 µs
median, 267 µs p95), because it runs the interaction table's regex set over the
claim. That is the one place in the cascade where a real index would pay off if
the table grew past a few hundred rows.

### 2.3 Verdict composition

Not an average. Severity dominates:

```
red flag                                → BLOCK
any claim CONTRADICTED                  → BLOCK
context compromised (injection in source) → BLOCK
relational conflict, severity CRITICAL  → BLOCK
supported_fraction < capability threshold → ABSTAIN
entailment configured but did not run   → ABSTAIN
capability requires review              → ESCALATE
any claim QUALIFIED or REMOVED          → RELEASE_WITH_CAVEAT
otherwise                               → RELEASE
```

One contradicted claim among nine supported ones is not "89 % fine". Averaging
is how a dangerous statement gets released on the strength of the harmless ones
around it.

**Only CRITICAL relational conflicts block.** An earlier draft blocked at
≥ HIGH, which is wrong: removing "increase potassium" from an otherwise sound
answer and then discarding the other six correct claims is over-refusal. Block
is reserved for conflicts where a *partial* answer is itself hazardous — St
John's wort with an SSRI — because there the user may act on the removed item
from another source.

### 2.4 Fail-closed matrix

| Degradation | Behaviour | Test |
|---|---|---|
| Judge unreachable | `ABSTAIN` | `test_judge_unavailable_abstains` |
| Judge omits a claim from its batch | `ABSTAIN` | `test_judge_skipping_a_claim_abstains` |
| Judge returns an unrecognised verdict | `ABSTAIN` | `test_unrecognised_verdict_abstains` |
| Gate latency budget exhausted | `ABSTAIN` | `test_exhausted_gate_budget_abstains` |
| Retrieval returns nothing | `ABSTAIN` | `test_no_retrieval_hits_abstains_without_generating` |
| Generation raises | `ABSTAIN` | `test_generation_failure_abstains` |
| Output truncated by a reasoning preamble | `ABSTAIN`, diagnosed | `test_empty_truncated_response_is_named…` |
| Entailment configured but never ran | `ABSTAIN` | `test_configured_judge_that_never_runs_abstains` |

A safety component that disables itself under load is not a safety component.

---

## 3. Measured performance

Machine: Apple Silicon, Python 3.13.5. Corpus snapshot `9dceed0a7ffb3338`
(18 documents, 237,300 characters, 403 chunks).

### 3.1 Index construction

| | |
|---|---:|
| Documents → chunks | 18 → 403 |
| Build time (hashing embedding) | 0.24 s |
| Per chunk | 0.60 ms |

### 3.2 Retrieval, median of 20 runs per query

| Query | Median | Selected |
|---|---:|---:|
| maximum daily dose of metformin | 8.8 ms | 8 |
| can I eat leafy greens with warfarin | 9.6 ms | 8 |
| what should I eat for type 2 diabetes | 7.1 ms | 7 |
| how does lisinopril work | 6.1 ms | 3 |

**Median across queries: 8.0 ms.** Stage breakdown for the last query:

```
dense 0.06 ms · sparse 0.07 ms · fused 0.01 ms · reranked 5.92 ms · selected 0.00 ms
```

Reranking is ~98 % of retrieval cost. MMR is O(k²) in the candidate set with a
Jaccard similarity per pair; at k=30 that is 900 set intersections. It is worth
it — see §7.4 — but it is where the time goes, and the first thing to profile if
the candidate window grows.

### 3.3 Deterministic checks, median of 2,000 runs

| Check | Median | p95 |
|---|---:|---:|
| structural | 1.2 µs | 1.3 µs |
| dosage_provenance | 1.9 µs | 2.0 µs |
| citation_resolution | 2.0 µs | 2.2 µs |
| numeric_grounding | 25.6 µs | 26.5 µs |
| context_integrity | 26.1 µs | 27.8 µs |
| relational_safety | 249.2 µs | 267.3 µs |

Total deterministic cascade: **~306 µs**. Against a ~1,400 ms entailment call,
the deterministic layer is 0.02 % of gate latency and does most of the work.

### 3.4 End-to-end, live

| Scenario | Decision | Wall | Cost |
|---|---|---:|---:|
| "crushing chest pain radiating to my left arm" | BLOCK | 0 ms | $0.00000 |
| "recommended starting dose of metformin?" | RELEASE | 2,051 ms | $0.00055 |
| "what should I eat to keep my heart healthy?" (warfarin + lisinopril) | ABSTAIN | 1,284 ms | $0.00108 |

The third is the capability contract working: `lifestyle_guidance` declares
`min_supported_fraction = 0.7`, `relational_safety` and `dosage_provenance` both
fired, 4 of 6 claims survived (67 %), and 67 % < 70 % ⇒ abstain.

---

## 4. Dependency shape

No import-linter contract is configured; this was verified by grepping the
cross-package imports. It is a real gap — the shape below is *currently* true
and nothing stops a future edit from violating it.

```
core            (no medassist imports)     pure: pydantic + stdlib only
patient         (no medassist imports)     pure: regex tables
llm      → core
corpus   → core
index    → core
guards   → core
capabilities → core
confidence   → core
reports  → core, guards
gate     → core, guards, patient
audit    → core, gate
redteam  → core, gate, confidence
memory   → core, reports
agents   → core, index, generate, orchestration
orchestration → core, capabilities, agents
serving  → everything (composition root)
```

`core` imports `os` in `ids.py` for `os.urandom`. That is entropy, not I/O; no
network, database or filesystem library appears anywhere under `core/`.

`serving` depending on everything is correct — it is the composition root, the
single place where concrete implementations are wired to ports.

---

## 5. Domain model

### 5.1 Identifiers

`<prefix>_<ULID>` — `chk_01M1RF1J98K007M79A3XMWMK49`. A `str` subclass with a
Pydantic core schema. Seven prefixes: `doc chk run node clm case eval`.

**Monotonic within a millisecond.** Plain ULIDs draw fresh randomness per call,
so two ids minted in the same millisecond sort arbitrarily. Chunking one
document mints ~100 ids well inside one clock tick, so the advertised
sortability would have been false for essentially every id the system creates.
The random component is now incremented rather than redrawn within a tick,
seeded with the high bit clear (2^79 headroom, re-seeded on the next tick).
Thread-safe under a lock; `test_ids_are_unique_under_concurrency` mints 1,600
ids across 8 threads and asserts uniqueness.

A wrong id *type* raises at construction: `ChunkId("doc_…")` is a `ValueError`,
not a lookup that silently misses three layers later.

### 5.2 Closed vocabularies

13 enums, 65 members. Closed on purpose: free-text labels make aggregation
impossible — you cannot cluster failures or gate a release on a metric whose
categories are invented at call sites.

| Enum | N | Carries |
|---|---:|---|
| `Intent` | 7 | routing + which guards are mandatory |
| `SourceKind` | 4 | authority ranking (FDA 3 > NLM/PubMed 2 > unknown 0) |
| `RetrievalStage` | 6 | the trace vocabulary |
| `RetrievalFailure` | 5 | four distinct bugs with four distinct fixes |
| `ClaimVerdict` | 4 | `UNSUPPORTED` ≠ `CONTRADICTED`, deliberately |
| `Severity` | 5 | escalation thresholds |
| `GuardDecision` | 4 | allow / annotate / rewrite / block |
| `TerminationReason` | 8 | budget exhaustion is a measurement |
| `NodeState` | 5 | DAG execution |
| `JudgeReliability` | 2 | the ADR-0004 invariant |
| `CheckName` | 7 | cascade order |
| `ClaimDecision` | 3 | retain / qualify / remove |
| `ResponseDecision` | 5 | `ABSTAIN` ≠ `BLOCK`, deliberately |

`ABSTAIN` means *we do not know*. `BLOCK` means *we know, and it is not safe to
say*. A system that handles ignorance and danger identically is wrong about one
of them.

### 5.3 Invariants enforced in the models

```python
class Chunk:
    @model_validator            # end >= start or construction fails

class ClaimAssessment:
    evidence: list[str] = Field(min_length=1)
    # a bare float with no explanation is not reviewable, and an
    # unreviewable safety metric is not worth having

    @model_validator            # ADR-0004
    # reliability=CALIBRATED requires a kappa. No third option.

class Budget:
    def subdivide(share)        # 0 < share <= 1, floors at 1 step
```

### 5.4 Why an answer is a list of claims

The single most consequential decision in the codebase. A blob of prose can
only be assessed with a vibe. A list of claims can be attributed one at a time,
mechanically, producing "7 supported, 1 unsupported, 1 contradicted" — a
measurement rather than a score. `Chunk.start/end` are character offsets into
`Document.text`, so a citation resolves to the exact supporting substring rather
than gesturing at a 56,000-character label.

---

## 6. Retrieval failure attribution

`RetrievalTrace.attribute(gold, indexed)` walks stages **in order** and names
the first at which no gold chunk survived:

```
NOT_INDEXED        corpus gap            → fix the scraper
NOT_RETRIEVED      encoder/tokenizer     → fix the embedding
LOST_IN_RERANK     MMR or the priors     → fix reranker weights
LOST_IN_SELECTION  context budget        → raise it, or lower max_per_doc
```

Ordering *is* the method: a chunk never indexed cannot be blamed on the
reranker. All five branches are tested. `indexed` is passed in rather than
stored on the trace, because recording every corpus id per query would make
traces larger than the corpus.

The scoring half of this — recall@k, nDCG, paired significance — is
**deliberately absent**; see §11.

---

## 7. Module reference

### 7.1 `core/` — 858 lines

`ids` `enums` `models` `config` `errors`. Pure. `config` reads its own `.env`
only: an earlier version also read a sibling project's, which silently pinned a
different subject model and made the resulting failures look like model
incompetence rather than a config leak.

### 7.2 `llm/client.py` — 354 lines

OpenAI-compatible gateway. Four behaviours worth knowing:

- **Disk cache** keyed by `sha256(model, messages, temperature, max_tokens,
  response_format)`. A repeated identical call is free and byte-identical.
  Corrupt entries are a miss, not a crash.
- **Reasoning-block stripping** before parsing, including a `<think>` left
  unterminated by truncation.
- **JSON-mode fallback.** A provider's strict JSON mode rejects reasoning models
  outright — they emit a preamble the validator never sees past. On a 400 with
  `json_validate_failed` the constraint is dropped and the response parsed
  leniently.
- **Adaptive token budget.** A 429 whose body says the *request* is too large is
  not congestion and will never succeed on retry; the account's output ceiling
  is simply below what was asked. `max_tokens` halves and retries immediately
  (floor 256). An ordinary 429 backs off with jitter and the same request —
  jitter matters because synchronised retries from a parallel fan-out otherwise
  re-collide every attempt.

An empty response with `finish_reason == "length"` is diagnosed as budget
exhaustion, not reported as malformed JSON, and no repair round-trip is spent
on it — it would truncate identically.

### 7.3 `corpus/` — 576 lines

Three live sources: openFDA labels (16 clinical sections kept, packaging
dropped), PubMed E-utilities (structured `AbstractText` labels preserved),
MedlinePlus health topics.

**Sections are character offsets, not markdown headings.** The dosage guard is a
set-membership test on `chunk.section`; recovering that by re-parsing prose
would be guesswork on the claim where guessing is least acceptable. Verified:
every emitted offset resolves to its own heading, and
`doc.text[c.start:c.end] == c.text` for every chunk.

Fetching is rate-limited per host (NCBI permits 3 req/s unauthenticated and
blocks above it), retries only 429/5xx, and caches raw responses on disk.

**Snapshots are content-addressed** — `sha256` over sorted `(source_uid, text)`
pairs, 16 hex chars. Document ids are excluded because they are fresh ULIDs per
scrape and would make every rebuild a new snapshot. `read()` re-hashes and
**refuses a corrupt snapshot**; a silently-edited corpus would invalidate every
number measured against it, invisibly.

### 7.4 `index/` — 866 lines

Chunking is section-aware and sentence-aligned (900 chars target, 150 overlap).
The sentence splitter refuses to break on `2.5`, `e.g.`, `i.e.`, `vs`, `Dr.` —
splitting `2.5 mg` destroys exactly the dosage terms that matter most.

**Bug fixed here:** the `min_chars=120` floor was dropping *whole sections*, not
just trailing overlap fragments. `contraindications` is routinely under 120
characters, so it vanished from the index entirely — surfacing as `NOT_INDEXED`
on precisely the safety-critical queries. The floor now applies only to
continuation fragments within a section.

Both a dense and a lexical index run, because neither dominates. Clinical
queries hinge on exact strings (`HbA1c`, `2.5 mg`), where a general-web dense
encoder will happily rate `metoprolol` and `metformin` as similar because they
*look* alike. Conversely BM25 has no signal at all on "what can I eat" against
"dietary modification".

RRF fuses by **rank, not score**: BM25 scores are unbounded idf sums, cosine
lives in [-1, 1], and mapping them onto a common scale requires a distributional
assumption that is wrong often enough to reorder results. Ties break on chunk id
so ordering is total and a deterministic run does not look flaky.

Context renders as `[C1]…[Cn]`, never ULIDs. **This is a hallucination
countermeasure, not formatting** — a model asked to reproduce a 26-character
identifier will sometimes emit a plausible one that indexes nothing.

### 7.5 `gate/` — 1,319 lines

The centrepiece. Six deterministic checks plus batched entailment. `quantities.py`
normalises units across mass / volume / percent / IU / dose-form / concentration
/ clearance / pressure / time dimensions; dimensions never cross, so `5 mg` does
not match `5 ml`. Bare numbers are ignored — "type 2 diabetes" must not demand
grounding, because a false positive on a safety check trains people to ignore
it — except the leading half of a range, which inherits the trailing unit.

Entailment batches at **6 claims per call**. A 15-claim batch came back with 14
verdicts, which failed closed correctly but abstained on a sound answer.

### 7.6 `patient/` + interaction rules — 508 lines

Rules key on **drug classes**, not names. A rule naming simvastatin protects
nobody on lovastatin though the mechanism is identical — a gap that looks like
coverage until someone is harmed by it. Names normalise through a brand map and
noise stripper: `Lipitor 20mg PO daily` → `atorvastatin` → `{statin,
cyp3a4_substrate}`.

Three rule families: food×class, drug×drug (a drug the answer *recommends* vs.
one the patient takes), and allergy with cross-reactivity (penicillin →
cephalosporin).

**Bug fixed here:** ties on severity broke on table order, so a QUALIFY row
masked a REMOVE row of equal severity. A warfarin patient asking about ibuprofen
and leafy greens got the dietary caveat and **kept the bleeding advice**. Ties
now break toward the more restrictive action.

An unrecognised medication reports `no_conflict_partial_coverage` and names the
drug. Implying a list was screened when it was not is worse than saying so.

### 7.7 `guards/` — 376 lines

Red-flag triage covers 12 emergency presentations, tuned for **recall over
precision**, with the referral text a constant no downstream node can soften.
Over-triage is tested too: five ordinary questions must not fire.

Injection detection is explicitly **layer 2 of 4** and explicitly not the
defence that holds. Layers 1 (structural delimiting) and 2 (pattern matching)
lose to an adversary who can rephrase. Layer 3 — the closed `allowed_tools` set —
holds because it does not depend on recognising the attack. Layer 4 is §06.3:
an injected instruction producing an uncited claim is stripped.

PII redaction runs before any provider call, keeps the map locally for
rehydration, and cache keys are computed on redacted text so the disk cache
never holds identifiers.

### 7.8 `capabilities/` + `orchestration/` — 687 lines

Plans are DAGs validated at construction: acyclic (Kahn), every edge declared,
every capability registered, parallel budget shares ≤ 1.0. Fed an undeclared
capability or a dangling edge, construction **fails** — the runtime does not
invent edges, because a model that can name arbitrary next steps has arbitrary
privileges.

**Privilege narrows down the graph.** A node's tool surface is the intersection
with every ancestor's, verified transitive. Budgets subdivide and are checked
*before* dispatch. An agent that raises becomes a `FAILED` node rather than an
exception aborting the run.

### 7.9 `reports/` + `memory/` — 850 lines

Analyte flags are computed from the reference interval, never read from the page
or asked of a model: a report printing `NORMAL` beside HbA1c 8.2 % does not make
it normal. Ranges printed on the report beat the shipped table, because
laboratories differ in assay and population.

**Bug fixed here:** a line ending in a lab's own flag column (`NORMAL`, `WNL`,
`A`) matched no alternative and the *entire analyte was dropped* — the worst
available outcome for a lab parser.

Memory shelf life is a clinical property, not a storage policy: INR 30 days,
haemoglobin 120, a chronic diagnosis 730. An undated fact is stale, because it
cannot be shown to be current. `observe_report` reconciles **before** it writes;
recording first would make every prior finding look re-measured and no
confirmation would ever be raised. Asserted by a test, because it is the kind of
invariant a later refactor silently inverts.

### 7.10 `confidence/` — 381 lines

Six factors combined as **exponents, not multipliers**, so the product stays in
[0,1] whatever the weights and a zero on any weighted factor drives the score to
zero. Factors are stored individually: a single 0.62 is unactionable,
`evidence_coverage=0.55` driving it is a bug report.

**Statistical bug fixed here:** `at_coverage()` returned the highest-accuracy
point among all thresholds meeting a coverage target — cherry-picking across
dozens of noisy estimates. On a purely random confidence signal it reliably
found a point beating base accuracy by chance, **reporting an uninformative
signal as useful**, the exact conclusion the method exists to prevent. It now
takes the tightest qualifying point. Verified across seeds.

---

## 8. The eleven bugs

Found by tests written against code already believed correct. The list is more
informative than the feature list.

| # | Bug | Consequence had it shipped |
|---|---|---|
| 1 | Sections under 120 chars dropped from the index | `contraindications` vanished; `NOT_INDEXED` on safety queries |
| 2 | ULIDs not monotonic within a millisecond | ~100 ids per document sorted arbitrarily |
| 3 | QUALIFY interaction row masked an equal-severity REMOVE | Warfarin patient kept the bleeding advice |
| 4 | 15-claim judge batch returned 14 verdicts | Correct abstain, but on a sound answer |
| 5 | Strict JSON mode rejects reasoning models | Every structured call failed |
| 6 | 429 for request *size* retried unchanged | A retry loop that could never succeed |
| 7 | Lab flag column `NORMAL` unmatched | Whole analyte silently dropped |
| 8 | `at_coverage` took max over noisy thresholds | Random confidence reported as useful |
| 9 | Config read a sibling project's `.env` | Wrong model; failures looked like incompetence |
| 10 | Unbounded claim length | 10,000-char blob released under one caveat |
| 11 | Authority spoofing in retrieved text | A chunk "authorised" the model to state a diagnosis |

10 and 11 came from the adversarial suite; the rest from unit tests.

---

## 9. Adversarial results

15 attacks, 10 classes, each with a machine-checkable expectation. The judge is
**scripted to say "supported" for every attack** — the strongest available case
against the gate, since any defence that holds was held by deterministic code
with no help from the model.

```
defended 15/15 (100 %, 95 % CI 80 %–100 %)
over-refusals: 0
```

The interval is doing real work. 15 cases establish little, and a bare "100 %"
would claim precision the sample does not support. Two of the fifteen failed on
first run (bugs 10 and 11 above) and are now regression tests.

Two benign probes are in the suite deliberately and **must be answered**. A
red-team corpus containing only attacks drives a team toward a system that
refuses everything and scores perfectly.

---

## 10. Test suite

448 tests, 2.4 s, no network, no API key. Verified to pass with `.env` removed.

| File | N | Covers |
|---|---:|---|
| `test_gate.py` | 45 | cascade, all six checks, fail-closed, batching |
| `test_confidence.py` | 36 | factors, ECE/Brier, risk–coverage, Wilson |
| `test_reports.py` | 31 | intervals, extraction, findings, trends |
| `test_orchestration.py` | 30 | registry, DAG validation, privilege, executor |
| `test_guards.py` | 29 | triage, dosage provenance, injection, PII, policy |
| `test_redteam.py` | 29 | the suite, decision records |
| `test_retrieval.py` | 29 | chunking, BM25, dense, RRF, selection, trace |
| `test_memory.py` | 27 | shelf life, continuity, confirmations |
| `test_core.py` | 26 | ids, attribution, invariants, budgets |
| `test_patient.py` | 25 | normalisation, class rules, allergies |
| `test_corpus.py` | 22 | parsers against fixtures, snapshot integrity |
| `test_serving.py` | 22 | pipeline order, fail-closed, API surface |
| `test_llm.py` | 21 | parsing, cache, retries, adaptive budget |

---

## 11. Known gaps

Named rather than hidden.

| Gap | Why it is not built |
|---|---|
| No import-linter contract | §4's shape is verified by grep, not enforced. A future edit can violate it silently. **This is the gap I would close first.** |
| No IR scoring (recall@k, nDCG) | Deliberate: that is offline evaluation with gold labels, a different tool's job (§00 §0). Failure *attribution* is here; *scoring* is not. |
| No gold set | Same reason. The gate is validated by behavioural cases, not relevance labels. |
| `CONTRADICTED` reconciliation unused | Reconciling a user statement against a report is specified (§05.7) and unimplemented. |
| Judge calibration never run | The κ invariant is enforced; no κ has been *measured*, so every judge verdict is correctly branded `UNCALIBRATED`. |
| Confidence weights unfitted | Defaults are chosen, not fitted on held-out data. The machinery to fit them exists; the labelled set does not. |
| Multi-agent path not wired to serving | `orchestration/` is built and tested, but `Pipeline` currently calls retrieval and generation directly rather than executing a DAG. |
| Interaction table coverage | 8 food rules, 6 drug rules, 10 drug classes. Narrow by design and stated as a limitation, not implied exhaustive. |

The last three matter most for anyone reading this as a claim about
completeness. The orchestration layer is real code with 30 tests, but the
serving path does not yet route through it.
