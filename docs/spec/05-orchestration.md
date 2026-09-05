# 05 — Orchestration  **SPECIFIED**

## 5.1 The plan is a DAG, and the runtime owns it

The supervisor emits a plan; the runtime executes it. The supervisor does not
execute anything, and the plan is validated before a single node runs.

```python
@dataclass(frozen=True)
class PlanNode:
    id: NodeId
    agent: str
    capability: str
    inputs: dict[str, Any]
    depends_on: tuple[NodeId, ...]
    budget_share: float

@dataclass(frozen=True)
class Plan:
    nodes: tuple[PlanNode, ...]
    # validated at construction: acyclic, every dependency exists,
    # every agent registered, every capability declared,
    # parallel budget shares sum to <= 1.0
```

**A router selects among declared edges only.** Fed a node naming an
unregistered agent, or an edge to a node that does not exist, plan construction
fails with `plan_invalid` — the runtime does not invent an edge and does not
"do its best". This is the single most important property of the orchestrator:
a model that can name arbitrary next steps is a model with arbitrary
privileges.

## 5.2 Worked example

`"I'm on warfarin and just got diagnosed with type 2 diabetes — what should I
eat, and is metformin safe with my current meds?"`

```
                    triage (red-flag scan)
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      pharmacology     consumer_health   evidence
      (metformin ×      (T2D diet,        (recent
       warfarin)         MedlinePlus)      literature)
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                     interaction_check          ← HIGH risk
                             │
                             ▼
                   lifestyle_guidance
                (diet × warfarin: vitamin K)    ← drug–food guard
                             │
                             ▼
                       synthesizer
                             │
                             ▼
                         critic
                             │
                             ▼
                     decision gate
```

Three nodes run in parallel. The interaction check gates on both drug nodes.
Lifestyle guidance cannot run before the interaction check, because its output
must be cross-checked against the medication list.

## 5.3 Parallel execution

Independent nodes execute concurrently on a bounded thread pool. Concurrency is
capped by the provider's rate limit, not by the DAG's width — a plan fanning
out to 12 nodes against a 30 RPM limit produces 12 simultaneous 429s and a
retry storm.

Backoff is exponential **with jitter**, because synchronised retries from a
parallel fan-out re-collide on every attempt. **BUILT** in the model gateway.

## 5.4 Budgets subdivide, never reset

```python
child = parent.subdivide(share)   # BUILT
```

If each node received a fresh allowance, a graph of N nodes would have N times
the stated budget, and the stated budget would be a lie.

Parallel fan-out shares must sum to ≤ 1.0. **Router branches are exempt**:
exactly one branch runs, so requiring a 4-way route's shares to sum to 1.0
would force every branch down to 0.25 of the budget for no reason.

Every exhaustion is a `TerminationReason` and a recorded measurement. An agent
that reliably needs 9 of 10 steps is differently healthy from one that needs 3,
and only one of them will survive a corpus that grows.

## 5.5 Self-correction, bounded

```
PLAN → EXECUTE → OBSERVE → EVALUATE
                              │
                    ┌─────────┴─────────┐
                  pass                 fail
                    │                   │
                 VERIFY             DIAGNOSE
                                        │
                            ┌───────────┼───────────┐
                            ▼           ▼           ▼
                          retry      replan      fallback
```

`DIAGNOSE` is not an LLM asked "what went wrong". It is a lookup on typed
signals the runtime already holds:

| Signal | Diagnosis | Action |
|---|---|---|
| `RetrievalFailure.NOT_RETRIEVED` | query/document vocabulary mismatch | replan with query expansion |
| `RetrievalFailure.LOST_IN_SELECTION` | context budget too tight | retry with a larger window |
| `RetrievalFailure.NOT_INDEXED` | corpus gap | **abstain** — no retry can fix it |
| claims unsupported > threshold | generation ungrounded | retry at temperature 0 with claim-only prompt |
| `StructuredOutputError` | shape failure | one repair round-trip (**BUILT**) |
| conflicting evidence unresolved | genuine disagreement | surface conflict, do not retry |

`NOT_INDEXED → abstain` matters. Retrying a query against a corpus that does
not contain the answer burns budget to arrive at the same place, and the third
attempt is the one where a model starts inventing.

**Revision is capped at one round per node.** Uncapped critic loops oscillate:
the critic objects, the agent over-corrects, the critic objects to the
correction. The cap is enforced by the runtime, not requested in a prompt.

## 5.6 Stopping is a feature

Hard stops, checked **before** each model call, never after:

```
max_steps · max_tokens · max_cost_usd · max_wall_time_s · max_agent_errors
```

Semantic stops:

```
confidence < threshold                    → abstain, state why
evidence insufficient for capability      → abstain, name the missing source
conflicting evidence unresolved (§02.4)   → surface both, do not adjudicate
authorization denied                      → stop, do not retry with less scope
safety policy triggered                   → block, escalate
```

Ordering is the specification. A budget checked *after* the call has already
spent the tokens it exists to prevent.

**A denial returns to the loop rather than terminating the run.** Whether an
agent recovers from a denial is one of the more interesting things to measure;
killing it on first denial destroys that signal.

## 5.7 Memory  **SPECIFIED**

| Tier | Holds | Lifetime |
|---|---|---|
| Working | current plan, blackboard, retrieved context | one run |
| Episodic | prior runs, feedback, corrections | per user |
| Semantic | stable facts about the user (conditions, meds, allergies) | until contradicted |

Every memory entry carries `provenance`, `confidence`, `expires_at`, and
`asserted_at`.

**Conflict, never silent overwrite.** If semantic memory holds *"not on
anticoagulants"* and today's uploaded report shows warfarin, the system raises
a conflict and asks. Overwriting loses the fact that the record was
contradicted — and in this domain, the contradiction is often the clinically
interesting event.

Memory is retrieved, not injected wholesale. Everything the model sees passes
the same injection guard as retrieved documents (§07.5); episodic memory
contains user-supplied text and is therefore an untrusted channel.
