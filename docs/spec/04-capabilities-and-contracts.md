# 04 — Capability Registry and Agent Contracts  **SPECIFIED**

## 4.1 Why a registry rather than twenty agents

Twenty independently-written agents share no invariants. Every one re-decides
what it may call, how long it may run, and what counts as done — and those
decisions live in prose inside twenty prompts, where nothing can enforce them.

Instead: a capability is a **declared, typed, authorizable unit of work**, and
agents are the interchangeable machinery that fulfils one.

```python
@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    risk: RiskLevel
    allowed_tools: frozenset[str]        # closed set, not a suggestion
    evidence_required: bool
    min_independent_sources: int
    verification: frozenset[str]         # which verifiers must pass
    requires_human_review: bool | Callable[[Context], bool]
    budget: Budget
    fallback: str                        # capability name, or "abstain"
```

The registry is the single place that answers "may this run, with what, for how
long, and what must be true before its output is released."

## 4.2 The six capabilities

| Capability | Risk | Evidence | Min sources | Human review |
|---|---|---|---:|---|
| `evidence_qa` | LOW | required | 1 | no |
| `literature_synthesis` | LOW | required | 3 | no |
| `treatment_comparison` | MEDIUM | required | 2 | no |
| `report_interpretation` | MEDIUM | required | 1 | if any critical-range analyte |
| `lifestyle_guidance` | MEDIUM | required | 1 | if medication list non-empty |
| `interaction_check` | **HIGH** | required | 2 | **always, on a positive finding** |

`interaction_check` is HIGH because a false negative is silent. An answer that
fails to mention that two drugs interact looks exactly like a correct answer.
No groundedness metric detects it, because everything the answer *does* say is
well supported. Only an explicit structured cross-reference against both
labels' `drug_interactions` sections catches it — which is why this capability
is a structured lookup with an LLM narration step, never an LLM lookup.

## 4.3 Agent contracts

Every agent declares, and the runtime enforces:

```
purpose · inputs · outputs · allowed_tools · forbidden_actions
risk_level · evidence_requirement · confidence_requirement
max_iterations · max_cost · max_latency · fallback · escalation_policy
```

Worked example:

```yaml
agent: EvidenceAgent
purpose: answer a clinical question from retrieved literature
allowed_tools: [retrieve, expand_query]
forbidden_actions: [emit_dosage_without_label_citation, cite_uncited_chunk]
max_iterations: 6
evidence_requirement: ">=1 independent source per claim"
on_insufficient_evidence: abstain      # never: guess
escalation: SupervisorAgent
```

`on_insufficient_evidence: abstain` is the important line. The default
behaviour of every LLM agent is to answer anyway. Abstention has to be an
explicit, rewarded, *measured* outcome (§08.3) or it never happens.

## 4.4 Agent roster

| Agent | Capability served | Notes |
|---|---|---|
| `SupervisorAgent` | all | Decomposes, delegates, verifies, recovers. Never calls a tool directly. |
| `TriageAgent` | any with symptoms | Runs first, always. Red-flag detection (§07.1). |
| `EvidenceAgent` | `evidence_qa`, `literature_synthesis` | Literature retrieval and synthesis |
| `PharmacologyAgent` | `interaction_check`, `evidence_qa` | FDA-label-first; the only agent permitted to emit dosage claims |
| `ConsumerHealthAgent` | `evidence_qa` | Lay-register answers from MedlinePlus |
| `ReportAgent` | `report_interpretation` | Analyte extraction, reference intervals, trends |
| `LifestyleAgent` | `lifestyle_guidance` | Diet/activity; output always passes the drug–food guard |
| `ComparisonAgent` | `treatment_comparison` | Builds the comparison matrix, surfaces conflicts |
| `SynthesizerAgent` | all | Merges claims, resolves or surfaces conflicts |
| `CriticAgent` | all | Independent verification pass; may send work back once |

`SupervisorAgent` calls no tools. It plans and delegates only. An orchestrator
that can also act will, under pressure from a hard task, quietly do the work
itself and bypass every per-agent restriction in this table.

`CriticAgent` uses a **different model** from the agent it reviews. A model
reviewing its own output agrees with itself far more often than it should; the
review then measures nothing.
