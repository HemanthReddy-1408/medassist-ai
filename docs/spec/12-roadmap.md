# 12 — Build Order

Sequenced so the **release gate exists early and everything after it is
measured against a working decision**, rather than arriving last.

| Wave | Delivers | Ends with |
|---|---|---|
| **0** ✅ | Domain model, model gateway, 3 live scrapers, snapshots, hybrid retrieval + trace | Retrieval runs end-to-end on a real corpus |
| **1** | Claim extraction, the four checks, the cascade, fail-closed degradation | **A working gate**: fabricated citations and ungrounded doses are caught |
| **2** | Relational safety: interaction table, patient record, drug–food/drug–drug | The check nothing else catches (§08.3) |
| **3** | Capability registry, contracts, supervisor, DAG executor, budgets | Multi-agent runs with enforced budgets |
| **4** | Red-flag triage, dosage grounding, policy engine, escalation | Clinical guards, all deterministic |
| **5** | Report parsing, analytes, reference intervals, lifestyle guidance | The capabilities needing patient context |
| **6** | Risk–coverage curve, threshold selection, computed confidence | Thresholds chosen from the operating characteristic, not by taste |
| **7** | Adversarial cases, fail-closed suite, decision records | A gate that is explainable and degrades safely |
| **8** | API, CLI, UI evidence pane | The surfaces |

## Why Wave 1 is the gate, not a labelled dataset

The obvious first move is a gold set and retrieval metrics. It is the wrong one
here for two reasons.

**It answers the wrong question.** Recall@10 tells you whether retrieval found
the right chunks. The gate has to decide, with no gold labels at all, whether
an answer is safe to release. Those are different problems, and only the second
is this project's.

**It is the boundary with the other tool.** IR metrics, stage attribution and
paired significance testing belong to an offline evaluation platform. Building
them here produces a second, worse copy.

So Wave 1 builds the thing that cannot be borrowed: a gate that catches a
fabricated citation and an ungrounded dose, in-band, in milliseconds.

## What "done" means

- Every claim in the README is backed by a decision record in `artifacts/`.
- The gate fails closed under every degradation in §08.6, with a test each.
- Refusals name the check that produced them.
