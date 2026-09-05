# 10 — Trajectory, Replay, and Experiments  **SPECIFIED**

## 10.1 The trajectory is the evaluation object

Anything not recorded cannot be measured after the fact. `RunRecord` therefore
carries the retrieval traces and per-node usage rather than logging and
dropping them.

```
RunRecord
├── question · intent · profile · seed · corpus_snapshot
├── plan (the DAG as executed)
├── per node: agent, capability, inputs, tool calls, retrieval trace,
│             claims, usage, state, error
├── verification: per-claim assessments with evidence
├── guard findings · policy decisions with the rule that fired
├── confidence components (all six, not just the product)
└── termination reason · usage · timing
```

Confidence components are stored individually. A single 0.62 is unactionable;
`evidence_coverage=0.55` driving it is a bug report.

## 10.2 Replay

Ports for `clock` and `random` exist so runs are replayable. A runtime that
reads the wall clock directly can never be replayed deterministically no matter
what else is recorded.

```
recorded run  →  replay with one variable changed  →  step-aligned diff
                 (model / prompt / retriever / embedding)
```

The diff is step-aligned rather than output-only: "the answer changed" is not
useful; "node 3 retrieved a different chunk at rank 2, and every downstream
difference follows from that" is.

## 10.3 Experiment engine

```
Experiment #042
  embedding: MiniLM-L6      retriever: hybrid-rrf-v2
  reranker:  mmr-0.72       llm: qwen3.6-27b
  prompt:    answer-v17     verifier: three-verifier-v4

vs. baseline #041 (embedding: hashing-768)

  recall@10        0.71 → 0.79   Δ +0.08  [+0.02, +0.14]  McNemar p=0.011  ✓
  faithfulness     0.86 → 0.87   Δ +0.01  [−0.04, +0.06]  p=0.62   n.s.
  over-refusal     0.07 → 0.07   Δ  0.00                           —
  cost / query     $0.004 → $0.011                                 ✗ 2.8×
  n = 60 cases, snapshot 7d005d0fb8d6c6be, BH-FDR q=0.05
```

The shape of that table is the point. A result without an interval, an n, a
snapshot hash, and a cost column is not a result. Note the honest reading here:
retrieval improved significantly, faithfulness did not move, and it cost 2.8×.
Whether that is worth shipping is a judgement — but it is now an informed one.

## 10.4 Tracing

OpenTelemetry spans over the same structure, so a run is inspectable live and
reconstructable afterwards from the same identifiers. Trace id = `RunId`.
