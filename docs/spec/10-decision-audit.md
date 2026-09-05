# 10 — Decision Audit  **SPECIFIED**

## 10.1 What is recorded, and why it is not a trace log

Every release decision is recorded as a **DecisionRecord** — not for debugging,
but because a gate that cannot explain a refusal is indistinguishable from one
that is broken.

```
DecisionRecord
├── run_id · question · profile_digest (hashed, never raw PII)
├── retrieval: selected chunk ids + per-stage survivor counts
├── claims: text, citations, and the verdict of each of the four checks
├── the check that fired, per claim               ← the load-bearing field
├── relational findings: which medication, which interaction, which severity
├── response decision + the rule that produced it
├── gate latency and cost, per stage
└── fail-closed reason, if any
```

**Per-check attribution is the point.** "Claim 3 was removed" is not
actionable. "Claim 3 was removed by `numeric_grounding`: `2000 mg` appears in
no cited span" is a bug report against the generator, and "removed by
`relational_safety`: vitamin K × warfarin" is correct behaviour that should
never be tuned away.

## 10.2 User-facing explanation

The same record renders to the user. "Why do you believe this?" is answerable
by showing the claim, its cited span, and which checks it passed — and a system
that cannot show that is asking to be trusted rather than earning it.

Refusals are explained in the same terms: *which* check refused, and what would
change the outcome ("no FDA label in the index covers this drug").

## 10.3 Export

Decision records serialize to line-delimited JSON keyed by `run_id`, carrying
the corpus snapshot hash and model identifiers. That is the boundary: an
offline evaluation platform can ingest them to compare gate versions across a
labelled set (§08.7), and this repository does not attempt that itself.

## 10.4 Privacy

`profile_digest` is a salted hash. Raw age, medications and conditions never
enter a record, and PII is redacted before any text reaches a provider, with
the redaction map held locally. Cache keys are computed on redacted text, so
the disk cache never holds identifiers.
