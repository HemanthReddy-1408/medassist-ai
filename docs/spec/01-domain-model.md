# 01 — Domain Model  **BUILT**

`medassist/core/` — 654 lines, no I/O, no network imports. This layer is pure so
that every type above it can be constructed in a test without a database, a
model provider, or a network.

## 1.1 Identifiers

`<prefix>_<ULID>` — `chk_01M1RF1J98K007M79A3XMWMK49`. A `str` subclass with a
Pydantic core schema, implemented once in `core/ids.py`.

Three properties earn the complexity:

1. ULIDs sort lexicographically by creation time, so ordering by id is free.
2. The prefix makes an id self-describing in a log line or a trace.
3. A wrong id *type* raises at construction. `ChunkId("doc_...")` is a
   `ValueError`, not a lookup that silently returns nothing three layers later.

Prefixes: `doc` `chk` `run` `node` `clm` `case` `eval`.

## 1.2 Closed enumerations

Nine closed vocabularies. Closed on purpose: free-text labels make aggregation
impossible — you cannot cluster failures, compute drift, or gate a release on a
metric whose categories are invented at call sites.

| Enum | Members | Why closed |
|---|---:|---|
| `Intent` | 7 | Routing and mandatory-guard selection both key off it |
| `SourceKind` | 4 | Determines authority ranking in conflict resolution |
| `RetrievalStage` | 6 | The trace vocabulary; failure attribution walks it in order |
| `RetrievalFailure` | 5 | Four distinct bugs with four distinct fixes |
| `ClaimVerdict` | 4 | `UNSUPPORTED` and `CONTRADICTED` must not be averaged |
| `Severity` | 5 | Escalation thresholds are defined on it |
| `GuardDecision` | 4 | allow / annotate / rewrite / block |
| `TerminationReason` | 8 | Budget exhaustion is a measurement, not merely an error |
| `NodeState` | 5 | DAG execution states |
| `JudgeReliability` | 2 | calibrated / uncalibrated — the ADR-0004 invariant |

### `RetrievalFailure` is the most important enum here

```
NONE · NOT_INDEXED · NOT_RETRIEVED · LOST_IN_RERANK · LOST_IN_SELECTION
```

These are four different bugs:

- `NOT_INDEXED` → the corpus is missing the document. Fix the scraper.
- `NOT_RETRIEVED` → indexed but neither retriever surfaced it. Fix the encoder
  or the tokenizer.
- `LOST_IN_RERANK` → retrieved, then discarded by MMR or the priors. Fix the
  reranker weights.
- `LOST_IN_SELECTION` → survived ranking, cut by the context budget. Raise the
  budget or tighten `max_per_doc`.

Collapsing them into "recall@10 is low" throws away the entire diagnosis. This
is the mechanism behind the user-facing question *"is the retrieval wrong, or
did the model hallucinate?"* — §01.5 answers the second half.

### `ClaimVerdict`: why `UNSUPPORTED ≠ CONTRADICTED`

An **unsupported** claim may still be true — the retrieved context simply did
not cover it. A **contradicted** claim is the cited source saying otherwise.
In a clinical setting the second is far worse. A single "faithfulness: 0.78"
that averages them hides the only distinction that matters for triage.

`UNVERIFIABLE` is the fourth member and exists so the metric is not gamed by
hedging: "consult your doctor" is not a factual assertion, and counting it as
unsupported would penalise exactly the behaviour we want.

## 1.3 Core types

```
Document   id source source_uid title text url published meta fetched_at
             .authority -> int   (FDA 3 > MedlinePlus/PubMed 2 > unknown 0)

Chunk      id doc_id text ordinal section source start end meta
             start/end are character offsets into Document.text

StageRecord      stage chunk_ids scores took_ms
RetrievalTrace   query stages selected truncated_by_budget
                   .attribute(gold, indexed) -> RetrievalFailure

Citation   chunk_id doc_id quote source
Claim      id text citations is_dosage is_safety_critical
Answer     run_id intent claims prose context_chunks guards disclaimer blocked conflicts

Budget     max_steps max_tokens max_cost_usd max_wall_time_s max_agent_errors
             .subdivide(share) -> Budget
Usage      steps prompt_tokens completion_tokens cost_usd wall_time_s agent_errors cache_hits
AgentResult / RunRecord

PatientProfile  age sex weight_kg conditions medications allergies pregnant smoker
LabAnalyte      name value unit ref_low ref_high raw
                  .flag -> low|normal|high|unknown   (computed, never asked of a model)
LabReport       analytes report_type collected narrative unparsed source_name
```

`Chunk.start`/`end` are not decoration. Without exact offsets a citation is a
document-level gesture — "it's somewhere in this 56,000-character FDA label" —
and the verification engine cannot quote the span that supports a claim.

`LabAnalyte.flag` is computed from the reference interval, never asked of a
model. A model can be argued out of "high"; `value > ref_high` cannot.

## 1.4 Invariants enforced in the models themselves

Three rules that would otherwise be conventions the codebase drifts away from:

```python
class Chunk:
    @model_validator          # end >= start, or construction fails

class ClaimAssessment:
    evidence: list[str] = Field(min_length=1)
    # a bare float with no explanation is not reviewable, and an
    # unreviewable safety metric is not worth having

    @model_validator          # ADR-0004
    # reliability=CALIBRATED requires a kappa. No third option.
```

## 1.5 The two questions this model is shaped to answer

The whole domain model exists to make these mechanically answerable rather than
matters of opinion:

**"Was the retrieval wrong?"** → `RetrievalTrace.attribute(gold, indexed)`
walks the stages in order and names the first one at which no gold chunk
survived. Ordering is the method: a chunk that was never indexed cannot be
blamed on the reranker. **BUILT**, all five branches tested.

**"Did the model hallucinate?"** → the answer is a list of `Claim`s, each
independently attributed to a retrieved span, producing a `ClaimAssessment`
with a `ClaimVerdict` and mandatory evidence. **SPECIFIED** — see §06.

Note that these are genuinely independent. An answer can be perfectly grounded
in context that was retrieved for the wrong question, and an answer can be
correct while citing nothing. Reporting one number for both would make each
failure invisible in the presence of the other.
