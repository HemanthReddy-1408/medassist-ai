# ADR-0004 — A judge score carries its kappa, or is branded uncalibrated

**Status:** accepted · **Wave:** 0 (invariant) / 6 (calibration) · **BUILT**

## Context
An LLM judge never compared to a human is an opinion with a decimal point.
Faithfulness numbers produced by such judges are widely published anyway.

## Decision
`ClaimAssessment` refuses construction when `reliability=CALIBRATED` and
`kappa is None`. Calibration requires ≥100 stratified pairs, two independent
human annotators, inter-annotator agreement reported *first*, and Cohen's κ
≥ 0.6.

## Consequences
Uncalibrated numbers remain publishable but are permanently labelled. If the
humans disagree, the rubric is underspecified and gets fixed — the judge is not
calibrated against an incoherent target.

## Rejected
*Trust a strong judge model.* Unfalsifiable, and the failure is invisible.
