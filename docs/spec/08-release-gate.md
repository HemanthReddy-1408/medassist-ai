# 08 — The Release Gate  **CORE**

The central artifact. Everything else in this system exists to feed it.

## 8.1 The constraint that defines this project

> **At serving time there is no gold label, and there is no second chance.**

An evaluation harness compares a produced answer against a known-correct one,
offline, with as much time as it wants. The gate has neither luxury. It must
decide whether *this* answer may reach *this* person, using only the answer,
the retrieved context, and the patient's own record — in-band, inside a latency
budget, before anything is shown.

That is a different engineering problem from measurement, and it is the one
this repository is about. Two consequences run through the whole design:

1. **Every check must be reference-free.** No check may require knowing the
   right answer, because nothing does.
2. **Every check must be affordable.** A verifier costing 4 seconds and a model
   call per claim cannot run on every request, so the gate is a *cascade*
   (§8.4), not a checklist.

## 8.2 Decision surface

The gate emits one decision per claim and one for the response.

```
per claim      RETAIN · QUALIFY · REMOVE
per response   RELEASE · RELEASE_WITH_CAVEAT · ABSTAIN · BLOCK · ESCALATE
```

`ABSTAIN` and `BLOCK` are distinct. Abstain means *we do not know* — the
evidence was insufficient, and the honest output is to say so. Block means *we
know, and it is not safe to say* — a red flag fired, or a claim was
contradicted by its own citation. Collapsing them produces a system that
handles ignorance and danger identically.

## 8.3 The four checks

| # | Check | Cost | Reference-free because |
|---|---|---|---|
| 1 | **Citation resolution** | ~0 | Membership test against the context window |
| 2 | **Numeric grounding** | ~0 | The digits are either in a cited span or they are not |
| 3 | **Relational safety** | ~0 | Lookup against the patient's own record |
| 4 | **Entailment** | model call | Judge sees claim + span only, never a gold answer |

The first three are deterministic code over structured data. They need no
model, run in microseconds, and cannot be talked out of their verdict.

### Check 2 is the one to look at

Every number in a claim must appear in a span that claim cites, unit-normalized.

```
claim  "The maximum recommended daily dose is 2000 mg."
cites  [C2] → "...The maximum recommended daily dose is 2550 mg."
                                                    ^^^^
verdict  REMOVE — numeric_ungrounded
```

An LLM judge asked "is this consistent?" will frequently accept that, because
2000 and 2550 are the same *kind* of thing and the sentence is otherwise
identical. A regex will not. In dosing, approximately consistent is wrong, and
this is precisely where the cheapest check outperforms the most expensive one.

### Check 3 is the one nothing else catches

```
claim    "Leafy green vegetables are an excellent source of vitamins."
verdict  SUPPORTED — the corpus says exactly this
patient  medications: [warfarin]
gate     QUALIFY — vitamin-K antagonism; consistency caveat required
```

The claim is true, well-cited, and perfectly grounded. It is wrong **for this
person**. Correctness here is *relational*: it holds between the answer and the
patient's record, not between the answer and the corpus. No groundedness or
faithfulness metric can see it, because every one of them stops at the corpus.

This is why the gate takes the patient record as an input rather than treating
personalization as a prompt detail.

## 8.4 The cascade, and its latency budget

Checks run cheapest-first, and each stage may terminate the whole gate.

```
  claims
    │
    ├─ 1. citation resolution      ~0.1 ms   unresolvable → REMOVE
    ├─ 2. numeric grounding        ~0.3 ms   ungrounded   → REMOVE
    ├─ 3. relational safety        ~0.5 ms   conflict     → QUALIFY / REMOVE / BLOCK
    │
    │    ── everything surviving here is cheap-clean ──
    │
    └─ 4. entailment (batched)     ~250 ms   contradicted → REMOVE + safety event
              │
              └─ only claims that survived 1–3, batched into one call
    ↓
  response decision
```

Ordering is the specification, not an optimisation. A claim removed by check 2
never reaches the model call, so the expensive stage runs on a shrinking set.
On a typical 8-claim answer the deterministic stages eliminate 1–3 claims
before any model is invoked.

**The gate has its own budget** (`gate_latency_ms`, `gate_cost_usd`), separate
from the generation budget. On exhaustion the gate **fails closed**: the
response degrades to `ABSTAIN`, never to an ungated release. A safety component
that disables itself under load is not a safety component.

## 8.5 Verdict composition

The response decision is a fold over claim verdicts, and it is deliberately
**not** an average:

```
red flag fired                         → BLOCK + emergency referral
any claim CONTRADICTED                 → BLOCK
relational conflict, severity CRITICAL → BLOCK
supported_fraction < threshold         → ABSTAIN
entailment configured but did not run  → ABSTAIN
capability requires human review       → ESCALATE
any claim QUALIFIED or REMOVED         → RELEASE_WITH_CAVEAT
otherwise                              → RELEASE
```

Severity dominates. One contradicted claim among nine supported ones is not
"89% fine" — averaging is how a dangerous statement gets released on the
strength of the harmless statements around it.

**Only CRITICAL relational conflicts block.** An earlier draft blocked at
severity >= HIGH, which is wrong. When the check removes "increase potassium
intake" from an otherwise sound heart-health answer, blocking the whole
response discards six correct claims in order to suppress one - the unsafe
claim is already gone, and the rest is useful. Blocking is reserved for
conflicts where a *partial* answer is itself hazardous (St John's wort with an
SSRI, tyramine with an MAOI), because there the user may act on the removed
item from another source. This is the over-refusal trade-off made explicitly
rather than by default.

## 8.6 Fail-closed defaults

Every failure mode resolves toward silence:

| Failure | Behaviour |
|---|---|
| Judge unreachable / times out | claims stay unverified → `ABSTAIN` |
| Context empty (retrieval found nothing) | `ABSTAIN`, naming the gap |
| Structured output unparseable after repair | `ABSTAIN` |
| Patient record unavailable, capability needs it | `ABSTAIN`, naming the missing field |
| Gate budget exhausted | `ABSTAIN` |

## 8.7 Validation — and what is deliberately not here

The gate is validated by **behavioural cases**: an input, and the decision the
gate is required to reach. Not IR relevance labels.

```python
class GateCase:
    id: CaseId
    claims: list[Claim]              # some deliberately fabricated
    context: list[Chunk]
    profile: PatientProfile | None
    expected_response: ResponseDecision
    expected_claim_verdicts: dict[ClaimId, ClaimDecision]
    reason: str                      # which check must fire, and why
```

Three families: **fabrication** (invented citations, ungrounded numbers),
**relational** (correct-but-wrong-for-this-patient), **degradation** (judge
down, empty context, budget exhausted — must fail closed).

> **Out of scope, on purpose: comparative evaluation.**
>
> "Did gate v17 beat v16, and is the difference significant?" is a question
> about a *change*, answered offline with gold labels, paired tests and
> multiple-comparison correction. That is a different tool's job, and building
> a second copy of it here would be duplicated effort producing a worse
> version. This repository exports its decision records in a form such a
> platform can consume (§10), and stops there.

What *is* measured here is the gate's own operating characteristic — the
risk–coverage curve of §06.6 — because that is what sets its thresholds, and a
threshold cannot be chosen offline by someone else.
