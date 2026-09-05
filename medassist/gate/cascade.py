"""The release gate.

Runs the checks cheapest-first, folds their verdicts into one decision per
claim and one for the response, and fails closed on every degradation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from medassist.core.enums import (
    CheckName,
    ClaimDecision,
    ResponseDecision,
    Severity,
)
from medassist.core.ids import ClaimId
from medassist.core.models import Claim, Usage
from medassist.gate.checks import DETERMINISTIC_CHECKS, Check, GateContext
from medassist.gate.decisions import CheckOutcome, ClaimJudgement, GateReason, fold_claim
from medassist.gate.entailment import EntailmentCheck, EntailmentUnavailable

_SEVERITY_RANK = {
    Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
    Severity.HIGH: 3, Severity.CRITICAL: 4,
}


@dataclass(frozen=True)
class GateSpec:
    #: The gate's own budget, separate from generation. On exhaustion the gate
    #: fails closed - a safety component that disables itself under load is not
    #: a safety component.
    latency_budget_ms: float = 6000.0
    cost_budget_usd: float = 0.02
    #: Below this fraction of claims surviving, the answer is too thin to release.
    min_supported_fraction: float = 0.6
    entailment_enabled: bool = True
    unsupported_action: ClaimDecision = ClaimDecision.QUALIFY
    requires_human_review: bool = False


@dataclass
class GateOutcome:
    decision: ResponseDecision
    reason: GateReason
    judgements: list[ClaimJudgement] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    entailment_ran: bool = False

    def _with(self, decision: ClaimDecision) -> list[ClaimId]:
        return [j.claim_id for j in self.judgements if j.decision is decision]

    @property
    def retained(self) -> list[ClaimId]:
        return self._with(ClaimDecision.RETAIN)

    @property
    def qualified(self) -> list[ClaimId]:
        return self._with(ClaimDecision.QUALIFY)

    @property
    def removed(self) -> list[ClaimId]:
        return self._with(ClaimDecision.REMOVE)

    @property
    def released(self) -> bool:
        return self.decision in (ResponseDecision.RELEASE, ResponseDecision.RELEASE_WITH_CAVEAT)

    @property
    def caveats(self) -> list[str]:
        return [j.caveat for j in self.judgements if j.caveat]

    def explain(self) -> str:
        lines = [f"{self.decision.value.upper()} - {self.reason.rule}: {self.reason.detail}"]
        for judgement in self.judgements:
            lines.append(f"  {judgement.decision.value:8} {judgement.explanation}")
        return "\n".join(lines)


class ReleaseGate:
    def __init__(
        self,
        *,
        entailment: EntailmentCheck | None = None,
        checks: tuple[Check, ...] = DETERMINISTIC_CHECKS,
        spec: GateSpec | None = None,
    ) -> None:
        self.checks = checks
        self.entailment = entailment
        self.spec = spec or GateSpec()

    def evaluate(
        self,
        claims: list[Claim],
        ctx: GateContext,
        *,
        red_flag: str = "",
    ) -> GateOutcome:
        started = time.perf_counter()
        stage_latency: dict[str, float] = {}
        usage = Usage()

        # Triage pre-empts everything: no retrieval, no verification, no
        # softening. The referral text is a constant, not a generation.
        if red_flag:
            return GateOutcome(
                decision=ResponseDecision.BLOCK,
                reason=GateReason(rule="red_flag", detail=red_flag, severity=Severity.CRITICAL),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if not claims:
            return GateOutcome(
                decision=ResponseDecision.ABSTAIN,
                reason=GateReason(
                    rule="no_claims", detail="the generator produced no verifiable claims"
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # -- stage 1-3: deterministic, microseconds ------------------------
        outcomes: dict[ClaimId, list[CheckOutcome]] = {c.id: [] for c in claims}
        for check in self.checks:
            stage_started = time.perf_counter()
            for claim in claims:
                if any(o.decision is ClaimDecision.REMOVE for o in outcomes[claim.id]):
                    continue  # already removed; later checks add nothing
                outcomes[claim.id].append(check.run(claim, ctx))
            stage_latency[check.name.value] = (time.perf_counter() - stage_started) * 1000

        survivors = [
            c for c in claims
            if not any(o.decision is ClaimDecision.REMOVE for o in outcomes[c.id])
        ]

        # -- stage 4: entailment, on the survivors only --------------------
        entailment_ran = False
        elapsed_ms = (time.perf_counter() - started) * 1000
        if self.spec.entailment_enabled and self.entailment is not None and survivors:
            if elapsed_ms >= self.spec.latency_budget_ms:
                return self._fail_closed(
                    "gate_budget_exhausted",
                    f"{elapsed_ms:.0f}ms spent before entailment could run",
                    claims, outcomes, started, stage_latency, usage,
                )
            stage_started = time.perf_counter()
            try:
                judged, judge_usage = self.entailment.run_batch(survivors, ctx)
            except EntailmentUnavailable as exc:
                return self._fail_closed(
                    "entailment_unavailable", str(exc),
                    claims, outcomes, started, stage_latency, usage,
                )
            usage.absorb(judge_usage)
            for claim_id, outcome in judged.items():
                outcomes[claim_id].append(outcome)
            stage_latency[CheckName.ENTAILMENT.value] = (time.perf_counter() - stage_started) * 1000
            entailment_ran = True

        judgements = [fold_claim(c.id, outcomes[c.id]) for c in claims]
        decision, reason = self._decide(judgements, entailment_ran)
        return GateOutcome(
            decision=decision, reason=reason, judgements=judgements, usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            stage_latency_ms=stage_latency, entailment_ran=entailment_ran,
        )

    def _fail_closed(
        self, rule: str, detail: str, claims, outcomes, started, stage_latency, usage
    ) -> GateOutcome:
        """Any degradation resolves toward silence, never toward release."""
        return GateOutcome(
            decision=ResponseDecision.ABSTAIN,
            reason=GateReason(rule=rule, detail=detail, severity=Severity.HIGH),
            judgements=[fold_claim(c.id, outcomes[c.id]) for c in claims],
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            stage_latency_ms=stage_latency,
        )

    def _decide(
        self, judgements: list[ClaimJudgement], entailment_ran: bool
    ) -> tuple[ResponseDecision, GateReason]:
        """Fold claim verdicts into a response decision.

        Severity dominates; this is deliberately not an average. One
        contradicted claim among nine supported ones is not "89% fine" -
        averaging is how a dangerous statement rides out on the strength of the
        harmless ones around it.
        """
        worst: CheckOutcome | None = None
        for judgement in judgements:
            for outcome in judgement.outcomes:
                if outcome.passed:
                    continue
                if worst is None or _SEVERITY_RANK[outcome.severity] > _SEVERITY_RANK[worst.severity]:
                    worst = outcome

        if worst is not None and worst.severity is Severity.CRITICAL:
            if worst.reason.startswith("entailment:contradicted"):
                return ResponseDecision.BLOCK, GateReason(
                    rule="claim_contradicted",
                    detail=f"a cited source contradicts its own claim: {worst.detail}",
                    severity=Severity.CRITICAL,
                )
            if worst.check is CheckName.RELATIONAL_SAFETY:
                return ResponseDecision.BLOCK, GateReason(
                    rule="relational_critical", detail=worst.detail, severity=Severity.CRITICAL,
                )
            if worst.check is CheckName.CONTEXT_INTEGRITY:
                # The retrieved corpus is an untrusted channel. If a cited
                # source carries an instruction payload, the surrounding
                # answer was assembled from compromised context and none of
                # it should be released on the strength of the rest.
                return ResponseDecision.BLOCK, GateReason(
                    rule="context_compromised", detail=worst.detail, severity=Severity.CRITICAL,
                )

        total = len(judgements)
        kept = sum(1 for j in judgements if j.decision is not ClaimDecision.REMOVE)
        fraction = kept / total if total else 0.0

        if fraction < self.spec.min_supported_fraction:
            return ResponseDecision.ABSTAIN, GateReason(
                rule="insufficient_support",
                detail=f"only {kept} of {total} claims survived verification "
                       f"({fraction:.0%} < {self.spec.min_supported_fraction:.0%})",
                severity=Severity.MEDIUM,
            )

        if self.spec.entailment_enabled and self.entailment is not None and not entailment_ran:
            return ResponseDecision.ABSTAIN, GateReason(
                rule="unverified", detail="entailment did not run; claims are unverified",
                severity=Severity.HIGH,
            )

        if self.spec.requires_human_review:
            return ResponseDecision.ESCALATE, GateReason(
                rule="capability_requires_review",
                detail="this capability escalates on any positive finding",
            )

        qualified = [j for j in judgements if j.decision is ClaimDecision.QUALIFY]
        removed = total - kept
        if qualified or removed:
            return ResponseDecision.RELEASE_WITH_CAVEAT, GateReason(
                rule="qualified",
                detail=f"{len(qualified)} claim(s) qualified, {removed} removed",
                severity=worst.severity if worst else Severity.LOW,
            )

        return ResponseDecision.RELEASE, GateReason(
            rule="all_checks_passed", detail=f"{total} claim(s) verified"
        )
