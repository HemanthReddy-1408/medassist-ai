# 06 — Verification and Calibrated Confidence  **SPECIFIED**

This is the answer to *"did the model hallucinate?"*, and the part of the
system that most distinguishes it from a RAG chatbot.

## 6.1 Claim extraction

The generator is prompted to emit claims with citation handles directly, rather
than prose that a second pass decomposes:

```json
{"claims": [
  {"text": "Metformin is first-line pharmacotherapy for type 2 diabetes.",
   "cites": ["C1", "C4"]},
  {"text": "The maximum recommended daily dose is 2550 mg.",
   "cites": ["C2"], "is_dosage": true}
]}
```

Prose is then *rendered from* the claims. Generating prose first and decomposing
after introduces an extraction step that itself hallucinates, and produces
claims the answer never made.

Handles are resolved to real `ChunkId`s **in code**. A handle naming a chunk not
in the context window is dropped and recorded as `citation_invalid` — the model
does not get to define the mapping.

## 6.2 Attribution: three verifiers, disagreement preserved

Each claim is checked independently by three verifiers:

| Verifier | Method | Catches |
|---|---|---|
| **Citation** | Is the cited chunk in context? Does it lexically overlap the claim? | Fabricated and mis-attached citations |
| **Entailment** | NLI-style judge: does the cited span entail, contradict, or neither? | Unsupported and contradicted claims |
| **Numeric** | Every number in the claim must appear in a cited span, unit-normalized | Correct-sounding wrong doses |

The numeric verifier is deterministic and non-negotiable. `"2550 mg"` in a
claim must appear in a cited chunk. An LLM judge will accept `"2000 mg"` as
"approximately consistent"; a regex will not. In dosing, approximately
consistent is wrong.

Verdicts land in `ClaimAssessment` with mandatory `evidence`. Where verifiers
disagree, **the most severe verdict wins and the disagreement is recorded**.
Averaging three verifiers into a score discards the fact that one of them found
a contradiction.

## 6.3 The generator does not decide what is released

```
SUPPORTED     → retain
UNSUPPORTED   → qualify ("not established in the sources consulted") or remove
CONTRADICTED  → remove, and record a safety event
UNVERIFIABLE  → retain (hedges and referral advice are not factual assertions)
```

Applied by the runtime, in code. The alternative — telling the model "only say
supported things" — is a request, and this system does not implement safety
properties as requests.

## 6.4 Judge calibration, or the number is branded

An LLM judge that has never been compared to a human is an opinion with a
decimal point.

Procedure:
1. Sample ≥ 100 claim/context pairs stratified across intents and verdicts.
2. Two human annotators label independently; inter-annotator agreement is
   reported first. **If the humans do not agree, the task is underspecified and
   the judge cannot be calibrated against it** — fix the rubric, not the judge.
3. Compute Cohen's κ between judge and adjudicated human labels.
4. κ ≥ 0.6 → `CALIBRATED`, and the κ travels with every score.
   Otherwise → `UNCALIBRATED`, and every downstream report says so.

Enforced by the `ClaimAssessment` model validator. **BUILT.**

Judge and subject are always different models, and the judge sees the context
and the claim but never the subject model's identity or reasoning trace.

## 6.5 Confidence is computed, not asked

Asking a model for its confidence measures its fluency at producing confident
prose. Instead:

```
confidence = w1·evidence_quality      (source authority × study design)
           × w2·evidence_coverage     (fraction of claims SUPPORTED)
           × w3·source_agreement      (1 − unresolved conflict rate)
           × w4·retrieval_relevance   (rerank score of selected context)
           × w5·model_consistency     (agreement across k=3 samples at T>0)
           × w6·temporal_validity     (recency vs. intent horizon)
```

Weights are **fitted on a held-out set**, not chosen by taste, and the fit is
reported with its residuals.

`model_consistency` is the self-consistency signal: sample the same question
three times at temperature 0.7 and measure claim-level agreement. A model that
answers differently each time is uncertain regardless of how it phrases itself.
It costs 3× on the sampled fraction, so it runs on high-risk capabilities only.

## 6.6 Calibration is itself measured

A confidence number is only useful if 0.8 means *right about 80% of the time*.

| Metric | Question |
|---|---|
| **ECE** (15 bins) | Do predicted confidences match observed accuracy? |
| **Brier score** | Overall probabilistic accuracy |
| **Reliability curve** | Where does it break — over- or under-confident, and in which band? |
| **Selective accuracy** | Accuracy vs. coverage as the abstention threshold moves |
| **AURC** | Area under the risk–coverage curve |

Selective accuracy is the one that matters operationally: it answers *"if we
abstain on the least-confident 20%, how much does accuracy on the remaining 80%
improve?"* If the answer is "not at all", the confidence signal is worthless no
matter how good its ECE looks.
