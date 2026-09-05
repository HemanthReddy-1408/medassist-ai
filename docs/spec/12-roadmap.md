# 12 — Build Status

All nine waves are implemented. Sequenced so the release gate existed early and
everything after it was measured against a working decision, rather than the
first honest signal arriving after the last commit.

| Wave | Delivers | Status |
|---|---|---|
| **0** | Domain model, model gateway, 3 live scrapers, snapshots, hybrid retrieval + trace | ✅ |
| **1** | Claim extraction, the checks, the cascade, fail-closed degradation | ✅ |
| **2** | Relational safety: drug classes, food/drug/allergy rules, patient record | ✅ |
| **3** | Capability registry, validated plan DAG, parallel executor, budgets | ✅ |
| **4** | Red-flag triage, dosage provenance, injection detection, policy engine | ✅ |
| **5** | Report parsing, reference intervals, findings, trends | ✅ |
| **6** | Computed confidence, ECE/Brier, risk–coverage, threshold selection | ✅ |
| **7** | Adversarial suite, decision records, per-check attribution | ✅ |
| **8** | End-to-end pipeline, HTTP API, CLI, Streamlit evidence pane | ✅ |
| **9** | Longitudinal memory, shelf life, continuity, confirmations | ✅ |

## What the waves actually found

Nine bugs were found by tests written against code already believed correct.
Recorded because the list is more informative than the feature list:

| Bug | Consequence had it shipped |
|---|---|
| Sections under 120 chars dropped from the index | `contraindications` routinely vanished — `NOT_INDEXED` on safety queries |
| ULIDs not monotonic within a millisecond | ~100 chunk ids per document sorted arbitrarily |
| QUALIFY interaction row masked a REMOVE of equal severity | Warfarin patient kept the bleeding advice, got the dietary caveat |
| 15-claim judge batch returned 14 verdicts | Correct abstain, but on a sound answer |
| Strict JSON mode rejects reasoning models | Every structured call failed |
| 429 for request *size* retried unchanged | Retry loop that could never succeed |
| Lab flag column `NORMAL` unmatched | The whole analyte silently dropped |
| `at_coverage` took max over noisy thresholds | Random confidence reported as useful |
| Config read a sibling project's `.env` | Wrong model, failures looked like incompetence |

Two more came from the adversarial suite rather than unit tests: an
unbounded-length claim released under a caveat, and an authority-spoofing
payload in a retrieved chunk producing a diagnosis.

## Deliberately not built

Recorded in §00.5 with reasons, and unchanged: comparative evaluation and
significance testing (a different tool's job, §00 §0), synthetic patient
simulation at scale, an arbitrary code-execution sandbox, a graph database,
video/audio modality.

One gap inside a built subsystem is named rather than hidden: `CONTRADICTED`
reconciliation between a user statement and a report (§13.7) is specified and
unimplemented.

## What "done" means here

- Every claim in the README is backed by output the repository can reproduce.
- Every published rate carries n and a Wilson interval.
- The gate fails closed under every degradation in §08.6, with a test each.
- Refusals name the check that produced them.
