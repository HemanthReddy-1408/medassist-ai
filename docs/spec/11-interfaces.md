# 11 — Interfaces  **SPECIFIED**

## 11.1 HTTP API

```
POST /v1/ask            question, profile?, capability? → Answer + run_id + confidence
POST /v1/reports        multipart upload → LabReport (parsed, redacted)
GET  /v1/runs/{id}      full RunRecord (auth-scoped)
GET  /v1/runs/{id}/trace  retrieval traces + verification detail
POST /v1/feedback       user correction → episodic memory + candidate gold case
GET  /v1/reliability    current reliability report, with n and CIs
GET  /v1/healthz        liveness; snapshot id and index size
```

`/v1/runs/{id}/trace` is exposed to the user, not just to operators. "Why do
you believe this?" is answerable by showing the evidence chain, and a system
that cannot show it is asking to be trusted rather than earning it.

Streaming: claims stream as they are verified, never before. Streaming
unverified text and retracting it is worse than a slower first token.

## 11.2 CLI

```
medassist corpus build --spec datasets/spec.yaml     # scrape → snapshot
medassist corpus list                                # snapshots with hashes
medassist ask "..." --profile p.json --explain
medassist report parse report.pdf
medassist eval run --suite gold --arm baseline
medassist eval compare baseline candidate            # McNemar + BH-FDR
medassist redteam run
medassist replay <run_id> --model <other>
```

## 11.3 UI

Streamlit, three panes: conversation, **evidence** (retrieved chunks with the
spans that support each claim highlighted), and **reliability** (per-claim
verdicts, confidence components, guard findings).

The evidence pane is the product. A chat window that hides its evidence is the
thing this project exists to argue against.
