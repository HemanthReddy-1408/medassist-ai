"""Domain models.

The single most consequential decision in this file: **an answer is not a
string**. It is a list of atomic claims, each carrying its own citations.

Everything downstream depends on that. If the answer were a blob of prose you
could only ask "does this look grounded?" and answer it with a vibe. Because it
is a list of claims, you can attribute each one to a retrieved span
independently, and report that 7 of 9 claims are supported, 1 is unsupported
and 1 is contradicted - which is an actionable measurement rather than a score.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medassist.core.enums import (
    ClaimVerdict,
    GuardDecision,
    Intent,
    JudgeReliability,
    NodeState,
    RetrievalFailure,
    RetrievalStage,
    Severity,
    SourceKind,
    TerminationReason,
)
from medassist.core.ids import ChunkId, ClaimId, DocumentId, NodeId, RunId


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class Document(Frozen):
    """A normalized source document, whatever API it came from."""

    id: DocumentId
    source: SourceKind
    source_uid: str = Field(description="Stable id at origin: PMID, FDA set_id, MedlinePlus URL")
    title: str
    text: str
    url: str = ""
    published: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=_utcnow)

    @property
    def authority(self) -> int:
        """Higher wins when two sources disagree.

        An FDA label outranks a PubMed abstract on dosing not because it is
        more scientific but because it is the legally-controlled document; a
        single abstract may describe an off-label study.
        """
        return {
            SourceKind.FDA_LABEL: 3,
            SourceKind.MEDLINEPLUS: 2,
            SourceKind.PUBMED_ABSTRACT: 2,
            SourceKind.UNKNOWN: 0,
        }[self.source]


class Chunk(Frozen):
    """A retrievable span of a document.

    ``start``/``end`` are character offsets into ``Document.text`` so a citation
    can be resolved back to the exact substring that supports it. Without those
    offsets a citation is only a document-level gesture.
    """

    id: ChunkId
    doc_id: DocumentId
    text: str
    ordinal: int
    section: str = ""
    source: SourceKind = SourceKind.UNKNOWN
    start: int = 0
    end: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _offsets_sane(self) -> Chunk:
        if self.end < self.start:
            raise ValueError(f"chunk {self.id}: end {self.end} precedes start {self.start}")
        return self


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class StageRecord(Frozen):
    """What survived one retrieval stage, in rank order.

    Recorded for *every* stage, not just the last one. This is what makes
    failure attribution mechanical: to learn where a gold chunk was lost you
    scan the stages in order and report the first one that does not contain it.
    """

    stage: RetrievalStage
    chunk_ids: list[ChunkId]
    scores: dict[str, float] = Field(default_factory=dict)
    took_ms: float = 0.0


class RetrievalTrace(Frozen):
    query: str
    stages: list[StageRecord]
    selected: list[ChunkId]
    truncated_by_budget: list[ChunkId] = Field(default_factory=list)

    def ids_at(self, stage: RetrievalStage) -> list[ChunkId]:
        for rec in self.stages:
            if rec.stage == stage:
                return rec.chunk_ids
        return []

    def attribute(
        self, gold: set[ChunkId], indexed: set[ChunkId] | None = None
    ) -> RetrievalFailure:
        """Report the first stage at which *no* gold chunk survived.

        Ordering matters and is the whole method: a chunk that was never
        indexed cannot be blamed on the reranker.

        ``indexed`` is passed in rather than stored on the trace. Recording
        every chunk id in the corpus on every query would make traces larger
        than the corpus, and the index membership is the same for all queries
        in a run anyway.
        """
        if not gold:
            return RetrievalFailure.NONE
        if set(self.selected) & gold:
            return RetrievalFailure.NONE

        indexed = indexed if indexed is not None else set(self.ids_at(RetrievalStage.INDEXED))
        if indexed and not (indexed & gold):
            return RetrievalFailure.NOT_INDEXED

        first_pass = set(self.ids_at(RetrievalStage.DENSE)) | set(
            self.ids_at(RetrievalStage.SPARSE)
        )
        if not (first_pass & gold):
            return RetrievalFailure.NOT_RETRIEVED

        reranked = set(self.ids_at(RetrievalStage.RERANKED))
        if reranked and not (reranked & gold):
            return RetrievalFailure.LOST_IN_RERANK

        # It survived ranking but did not make the context window.
        return RetrievalFailure.LOST_IN_SELECTION


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


class Citation(Frozen):
    chunk_id: ChunkId
    doc_id: DocumentId
    quote: str = Field(default="", description="Verbatim span the claim rests on")
    source: SourceKind = SourceKind.UNKNOWN


class Claim(Frozen):
    """One atomic factual assertion extracted from an answer."""

    id: ClaimId = Field(default_factory=ClaimId.new)
    text: str
    citations: list[Citation] = Field(default_factory=list)
    is_dosage: bool = False
    is_safety_critical: bool = False


class ClaimAssessment(Frozen):
    """The verdict on one claim, with the evidence that produced it.

    ``evidence`` is required and non-empty. A bare float with no explanation is
    not reviewable, and an unreviewable safety metric is not worth having.
    """

    claim_id: ClaimId
    verdict: ClaimVerdict
    evidence: list[str] = Field(min_length=1)
    supporting_chunks: list[ChunkId] = Field(default_factory=list)
    judge_model: str = ""
    reliability: JudgeReliability = JudgeReliability.UNCALIBRATED
    kappa: float | None = None

    @model_validator(mode="after")
    def _calibrated_scores_carry_kappa(self) -> ClaimAssessment:
        if self.reliability is JudgeReliability.CALIBRATED and self.kappa is None:
            raise ValueError(
                "a CALIBRATED verdict must carry the kappa it was calibrated at; "
                "otherwise brand it UNCALIBRATED"
            )
        return self


class GuardFinding(Frozen):
    guard: str
    decision: GuardDecision
    severity: Severity
    message: str
    spans: list[str] = Field(default_factory=list)


class Answer(Frozen):
    """The system's response. Prose is *derived* from claims, never the reverse."""

    run_id: RunId
    intent: Intent
    claims: list[Claim]
    prose: str
    context_chunks: list[ChunkId] = Field(default_factory=list)
    guards: list[GuardFinding] = Field(default_factory=list)
    disclaimer: str = ""
    blocked: bool = False
    conflicts: list[str] = Field(default_factory=list)

    @property
    def cited_chunk_ids(self) -> set[ChunkId]:
        return {c.chunk_id for claim in self.claims for c in claim.citations}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class Budget(Mutable):
    """Consumption limits. Every field is also a measurement when exhausted."""

    max_steps: int = 24
    max_tokens: int = 120_000
    max_cost_usd: float = 0.50
    max_wall_time_s: float = 180.0
    max_agent_errors: int = 3

    def subdivide(self, share: float) -> Budget:
        """Carve a child budget out of this one.

        Children may never reset a budget, only take a slice of what remains.
        The alternative - each node getting a fresh allowance - means a graph
        of N nodes has N times the stated budget, which makes the stated budget
        a lie.
        """
        if not 0 < share <= 1:
            raise ValueError(f"share must be in (0, 1], got {share}")
        return Budget(
            max_steps=max(1, int(self.max_steps * share)),
            max_tokens=max(1, int(self.max_tokens * share)),
            max_cost_usd=self.max_cost_usd * share,
            max_wall_time_s=self.max_wall_time_s * share,
            max_agent_errors=self.max_agent_errors,
        )


class Usage(Mutable):
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    wall_time_s: float = 0.0
    agent_errors: int = 0
    cache_hits: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def absorb(self, other: Usage) -> None:
        self.steps += other.steps
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        self.agent_errors += other.agent_errors
        self.cache_hits += other.cache_hits
        self.wall_time_s = max(self.wall_time_s, other.wall_time_s)


class AgentResult(Frozen):
    """What one specialist agent produced."""

    agent: str
    node_id: NodeId
    claims: list[Claim] = Field(default_factory=list)
    notes: str = ""
    retrieval: RetrievalTrace | None = None
    usage: Usage = Field(default_factory=Usage)
    state: NodeState = NodeState.SUCCEEDED
    error: str = ""


class RunRecord(Frozen):
    """The complete, replayable record of one question.

    This is the evaluation object. Anything not on here cannot be measured
    after the fact, which is why the retrieval traces and per-node usage are
    carried rather than logged and dropped.
    """

    run_id: RunId
    question: str
    intent: Intent
    answer: Answer
    node_results: list[AgentResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    termination: TerminationReason = TerminationReason.COMPLETED
    corpus_snapshot: str = ""
    seed: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    took_s: float = 0.0


# --------------------------------------------------------------------------
# Patient context and lab reports
# --------------------------------------------------------------------------


class PatientProfile(Frozen):
    """What the system knows about the person asking.

    Everything is optional, and the absence of a field is itself information:
    dietary guidance that would be unsafe without knowing kidney function is
    withheld rather than guessed, and the answer says which fact was missing.
    """

    age: int | None = None
    sex: str = ""
    weight_kg: float | None = None
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    pregnant: bool | None = None
    smoker: bool | None = None

    def describe(self) -> str:
        bits: list[str] = []
        if self.age is not None:
            bits.append(f"age {self.age}")
        if self.sex:
            bits.append(self.sex)
        if self.pregnant:
            bits.append("pregnant")
        if self.conditions:
            bits.append("conditions: " + ", ".join(self.conditions))
        if self.medications:
            bits.append("medications: " + ", ".join(self.medications))
        if self.allergies:
            bits.append("allergies: " + ", ".join(self.allergies))
        return "; ".join(bits) or "no profile provided"


class LabAnalyte(Frozen):
    """One measured value from a lab report, with its reference interval.

    ``flag`` is computed from the interval rather than asked of the model. A
    model can be talked out of "high"; ``value > ref_high`` cannot.
    """

    name: str
    value: float
    unit: str = ""
    ref_low: float | None = None
    ref_high: float | None = None
    raw: str = ""

    @property
    def flag(self) -> str:
        if self.ref_low is not None and self.value < self.ref_low:
            return "low"
        if self.ref_high is not None and self.value > self.ref_high:
            return "high"
        if self.ref_low is None and self.ref_high is None:
            return "unknown"
        return "normal"

    @property
    def abnormal(self) -> bool:
        return self.flag in ("low", "high")

    def describe(self) -> str:
        interval = ""
        if self.ref_low is not None or self.ref_high is not None:
            low = "" if self.ref_low is None else f"{self.ref_low:g}"
            high = "" if self.ref_high is None else f"{self.ref_high:g}"
            interval = f" (ref {low}-{high})"
        return f"{self.name}: {self.value:g} {self.unit}{interval} [{self.flag}]".strip()


class LabReport(Frozen):
    """A parsed report. ``unparsed`` is kept deliberately.

    Text the extractor could not turn into an analyte is retained verbatim so
    that a downstream agent can still read it, and so the extraction recall is
    measurable instead of silently zero.
    """

    analytes: list[LabAnalyte] = Field(default_factory=list)
    report_type: str = ""
    collected: str = ""
    narrative: str = ""
    unparsed: str = ""
    source_name: str = ""

    @property
    def abnormal(self) -> list[LabAnalyte]:
        return [a for a in self.analytes if a.abnormal]

    def summarize(self) -> str:
        if not self.analytes:
            return self.narrative[:500] or "no structured results extracted"
        lines = [a.describe() for a in self.analytes]
        return "\n".join(lines)
