"""Closed vocabularies.

Every enum here is closed on purpose. Free-text labels make aggregation
impossible: you cannot cluster failures, compute drift, or gate a release on a
metric whose categories are invented at call sites.
"""

from __future__ import annotations

from enum import StrEnum


class Intent(StrEnum):
    """What the user is actually asking for.

    Routing depends on this, and so does which guards are mandatory:
    SYMPTOM_TRIAGE always runs the red-flag guard, DRUG_INFO always runs the
    dosage-grounding guard.
    """

    DRUG_INFO = "drug_info"
    DRUG_INTERACTION = "drug_interaction"
    SYMPTOM_TRIAGE = "symptom_triage"
    CONDITION_OVERVIEW = "condition_overview"
    EVIDENCE_LOOKUP = "evidence_lookup"
    REPORT_INTERPRETATION = "report_interpretation"
    OUT_OF_SCOPE = "out_of_scope"


class SourceKind(StrEnum):
    """Provenance of a document. Determines authority ranking in synthesis."""

    FDA_LABEL = "fda_label"          # regulatory, authoritative for dosing
    PUBMED_ABSTRACT = "pubmed_abstract"  # primary literature
    MEDLINEPLUS = "medlineplus"      # consumer-facing, NLM-curated
    UNKNOWN = "unknown"


class RetrievalStage(StrEnum):
    """Stages a candidate chunk passes through.

    The whole point of naming these is that when retrieval fails we can say
    *where* it failed rather than that it failed.
    """

    INDEXED = "indexed"
    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"
    RERANKED = "reranked"
    SELECTED = "selected"


class RetrievalFailure(StrEnum):
    """Where a gold chunk was lost. Mutually exclusive and exhaustive.

    NOT_INDEXED is a corpus bug, NOT_RETRIEVED is an embedding/lexical bug,
    LOST_IN_RERANK is a reranker bug, LOST_IN_SELECTION is a budget bug. Those
    have four different fixes, which is why collapsing them into "bad recall"
    is a mistake.
    """

    NONE = "none"
    NOT_INDEXED = "not_indexed"
    NOT_RETRIEVED = "not_retrieved"
    LOST_IN_RERANK = "lost_in_rerank"
    LOST_IN_SELECTION = "lost_in_selection"


class ClaimVerdict(StrEnum):
    """Result of attributing one atomic claim to the retrieved context.

    UNSUPPORTED and CONTRADICTED are deliberately distinct. An unsupported
    claim may still be true - the context just did not cover it. A contradicted
    claim is the context saying otherwise. In a clinical setting the second is
    far worse, and averaging them into one "faithfulness" number hides that.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"  # not a factual assertion (hedge, advice to see a doctor)


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardDecision(StrEnum):
    ALLOW = "allow"
    ANNOTATE = "annotate"   # answer proceeds, but a warning is attached
    REWRITE = "rewrite"     # offending span must be removed before release
    BLOCK = "block"         # answer is replaced by a safe fallback


class TerminationReason(StrEnum):
    """Why an orchestration run stopped. Budget exhaustion is a measurement,
    not merely an error: an agent that reliably needs 9 of 10 steps is
    differently healthy from one that needs 3."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MAX_WALL_TIME = "max_wall_time"
    NO_PROGRESS = "no_progress"
    GUARD_BLOCKED = "guard_blocked"
    AGENT_ERROR = "agent_error"


class NodeState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class JudgeReliability(StrEnum):
    """A judge score is only as trustworthy as its agreement with humans.

    A score carries a measured kappa, or it is branded UNCALIBRATED. There is
    no third option, and the model validator enforces that.
    """

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"


class CheckName(StrEnum):
    """The four gate checks, in cascade order (cheapest first).

    Ordering is the specification, not an optimisation: a claim removed by a
    deterministic check never reaches the model call, so the expensive stage
    runs on a shrinking set.
    """

    STRUCTURAL = "structural"
    CONTEXT_INTEGRITY = "context_integrity"
    CITATION_RESOLUTION = "citation_resolution"
    NUMERIC_GROUNDING = "numeric_grounding"
    DOSAGE_PROVENANCE = "dosage_provenance"
    RELATIONAL_SAFETY = "relational_safety"
    ENTAILMENT = "entailment"


class ClaimDecision(StrEnum):
    RETAIN = "retain"
    QUALIFY = "qualify"   # released, with a caveat attached
    REMOVE = "remove"


class ResponseDecision(StrEnum):
    """ABSTAIN and BLOCK are deliberately distinct.

    Abstain means *we do not know* - the evidence was insufficient, and the
    honest output is to say so. Block means *we know, and it is not safe to
    say*. A system that handles ignorance and danger identically is wrong about
    one of them.
    """

    RELEASE = "release"
    RELEASE_WITH_CAVEAT = "release_with_caveat"
    ABSTAIN = "abstain"
    BLOCK = "block"
    ESCALATE = "escalate"
