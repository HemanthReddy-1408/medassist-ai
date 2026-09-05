"""Decision records produced by the gate.

A gate that cannot explain a refusal is indistinguishable from one that is
broken, so every verdict carries the check that produced it and a
machine-readable reason. "Claim 3 was removed" is not actionable; "removed by
numeric_grounding: 2000 mg appears in no cited span" is a bug report.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from medassist.core.enums import CheckName, ClaimDecision, ResponseDecision, Severity
from medassist.core.ids import ChunkId, ClaimId


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CheckOutcome(Frozen):
    """One check's verdict on one claim."""

    check: CheckName
    decision: ClaimDecision
    reason: str = Field(description="Machine-readable slug, e.g. 'numeric_ungrounded'")
    detail: str = Field(default="", description="Human-readable explanation")
    severity: Severity = Severity.INFO
    supporting_chunks: list[ChunkId] = Field(default_factory=list)
    took_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.decision is ClaimDecision.RETAIN


class ClaimJudgement(Frozen):
    """The fold of every check that ran on one claim."""

    claim_id: ClaimId
    decision: ClaimDecision
    outcomes: list[CheckOutcome] = Field(default_factory=list)
    caveat: str = ""

    @property
    def firing_check(self) -> CheckName | None:
        """The check responsible for a non-RETAIN decision.

        The first failing check in cascade order, since later checks may not
        have run at all once an earlier one removed the claim.
        """
        for outcome in self.outcomes:
            if outcome.decision is self.decision and not outcome.passed:
                return outcome.check
        return None

    @property
    def explanation(self) -> str:
        outcome = next((o for o in self.outcomes if not o.passed), None)
        if outcome is None:
            return "all checks passed"
        return f"{outcome.check.value}: {outcome.detail or outcome.reason}"


def fold_claim(claim_id: ClaimId, outcomes: list[CheckOutcome]) -> ClaimJudgement:
    """Combine check outcomes into one claim decision.

    Severity dominates: REMOVE beats QUALIFY beats RETAIN. A claim is not
    "mostly fine" because three of four checks liked it.
    """
    order = {ClaimDecision.RETAIN: 0, ClaimDecision.QUALIFY: 1, ClaimDecision.REMOVE: 2}
    decision = max((o.decision for o in outcomes), key=lambda d: order[d], default=ClaimDecision.RETAIN)
    caveats = [o.detail for o in outcomes if o.decision is ClaimDecision.QUALIFY and o.detail]
    return ClaimJudgement(
        claim_id=claim_id, decision=decision, outcomes=outcomes, caveat=" ".join(caveats)
    )


class GateReason(Frozen):
    """Why the response as a whole was decided that way."""

    rule: str
    detail: str = ""
    severity: Severity = Severity.INFO


__all__ = [
    "CheckOutcome",
    "ClaimJudgement",
    "GateReason",
    "ResponseDecision",
    "fold_claim",
]
