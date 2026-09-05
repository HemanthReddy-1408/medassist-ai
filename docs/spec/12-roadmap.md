# 12 — Build Order

Sequenced so that **every stage ends with something measurable**. The
alternative — building all the infrastructure and evaluating at the end — means
the first honest number arrives after the last commit, and every design
decision before it was a guess.

| Wave | Delivers | Ends with |
|---|---|---|
| **0** ✅ | Domain model, model gateway, 3 live scrapers, snapshots, hybrid retrieval + trace | Retrieval runs end-to-end on a real corpus |
| **1** | Corpus build CLI, gold set (60 cases), retrieval metrics + **failure attribution** | *First real number:* recall@k with a diagnosis of every miss |
| **2** | Claim-based generation, citation binding, three verifiers, numeric grounding | Faithfulness and citation validity, with a no-context ablation |
| **3** | Capability registry, contracts, supervisor, DAG executor, budgets, parallel fan-out | Multi-agent runs with enforced budgets |
| **4** | Guards: red-flag, dosage, drug–food, injection, PII. Policy engine + decision gate | Safety metrics incl. **over-refusal** |
| **5** | Report parsing, analyte extraction, reference intervals, lifestyle guidance | The two capabilities that need patient context |
| **6** | Judge calibration (κ), computed confidence, ECE/Brier/selective accuracy | Confidence that means something |
| **7** | Red-team suite, regression loop, CI gates | A ratchet |
| **8** | Replay, experiment engine, reliability report, UI evidence pane | The dashboard, populated with measured numbers |

## Wave 1 is the important one

It is tempting to build agents next — they are the visible part. But without
the gold set and retrieval metrics, every subsequent decision (which embedding,
which reranker weights, whether query expansion earns its cost) is settled by
taste. Wave 1 costs a day of hand-labelling and makes the following seven waves
empirical.

## What "done" means

Not "all boxes ticked". Done is:

- Every claim in the README is backed by a number in `artifacts/`.
- Every number states n, a CI, and its corpus snapshot.
- The metrics that came out badly are in the report too.
