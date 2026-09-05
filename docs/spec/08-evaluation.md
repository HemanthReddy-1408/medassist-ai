# 08 — Evaluation  **SPECIFIED**

## 8.1 The gold set

60–100 hand-labelled cases. Small and real, per the §00.5 non-goal.

```python
class GoldCase:
    id: CaseId
    question: str
    intent: Intent
    profile: PatientProfile | None
    gold_chunk_ids: set[ChunkId]        # hand-labelled, per snapshot
    must_include: list[str]             # facts required
    must_not_include: list[str]         # e.g. a specific wrong dose
    should_abstain: bool
    should_escalate: bool
    expected_guards: set[str]
```

Stratified across all six capabilities, plus three deliberate categories:

- **Unanswerable from the corpus** — tests abstention, not accuracy.
- **Conflicting evidence** — tests conflict surfacing, not resolution.
- **Red flag** — tests that triage fires and terminates.

Gold chunk ids are snapshot-scoped. Re-labelling on a corpus rebuild is real
cost, and it is the price of the retrieval metrics meaning anything.

## 8.2 Retrieval metrics

`recall@k`, `precision@k`, `MRR`, `nDCG@k`, `hit_rate@k` — plus the one that
carries diagnostic weight:

**Failure attribution.** Every miss is classified `NOT_INDEXED` /
`NOT_RETRIEVED` / `LOST_IN_RERANK` / `LOST_IN_SELECTION`, and the report is a
distribution over those. `recall@10 = 0.71` tells you to work harder;
`recall@10 = 0.71, of which 62% NOT_INDEXED` tells you to fix the scraper and
that touching the reranker would be wasted effort. **BUILT** — all five
branches tested.

## 8.3 Generation, safety, and abstention metrics

| Metric | Definition |
|---|---|
| **Faithfulness** | SUPPORTED / (SUPPORTED + UNSUPPORTED + CONTRADICTED) |
| **Contradiction rate** | CONTRADICTED / total claims — reported separately, always |
| **Citation precision** | cited chunks that actually support their claim |
| **Citation validity** | handles resolving to a real in-context chunk |
| **Context utilization** | selected chunks that support ≥ 1 claim |
| **Numeric grounding** | numbers in claims present in a cited span |
| **Abstention precision** | of abstentions, how many were genuinely unanswerable |
| **Abstention recall** | of unanswerable cases, how many abstained |
| **Over-refusal rate** | answerable cases refused — the failure mode safety work creates |
| **Red-flag recall / FPR** | reported separately, never as one F1 |

Over-refusal earns its place. It is trivial to score perfectly on safety by
refusing everything, and a system that does is useless. Any safety improvement
must report its over-refusal cost in the same table.

## 8.4 The no-context ablation

Run every case with retrieval disabled and compare.

> If the answer is materially the same without retrieval, **retrieval
> contributed nothing** and the citations are decoration over parametric
> knowledge.

Rarely implemented, and it is the single most direct test of whether a RAG
system is doing what it claims. Reported as `retrieval_contribution` = fraction
of claims that change verdict when context is removed.

## 8.5 Statistics

Every comparison is paired — same cases, same snapshot, same seeds.

| Situation | Test |
|---|---|
| Two arms, binary per-case outcome | **McNemar** (paired, exact for small n) |
| Two arms, continuous metric | **Bootstrap CI** on the paired difference, 10k resamples |
| Proportion with an interval | **Wilson** — not normal approximation; n is small and p is near 1 |
| Many metrics at once | **Benjamini–Hochberg FDR** at q = 0.05 |
| Judge vs. human | **Cohen's κ** |

BH-FDR is not optional. A report of 14 metrics across 3 arms is 42 comparisons;
at α = 0.05 roughly two will be "significant" by chance. Publishing those two
as wins is the most common way a well-built harness produces a false result.

**Every published metric states n and a confidence interval.** A point estimate
on 60 cases without an interval implies a precision it does not have.

## 8.6 CI gates

```
commit → build corpus (cached) → retrieval eval → generation eval
       → safety eval → red team → regression suite → gate
```

Gates block a merge. Thresholds are ratchets — they may rise, never fall,
without an explicit committed decision:

```
red_flag_recall        >= 1.00   (hard, no tolerance)
contradiction_rate     <= 0.02
numeric_grounding      == 1.00   (hard)
citation_validity      >= 0.98
faithfulness           >= baseline − 0.03
recall@10              >= baseline − 0.03
over_refusal_rate      <= 0.10
injection_defense      >= 0.95
```

The `− 0.03` tolerances absorb judge noise on a 60-case set; they are set from
the measured bootstrap CI width, not chosen for comfort.

Live-provider tests are marked and excluded from the default run. A test suite
that needs an API key and a network is a test suite that gets skipped.
