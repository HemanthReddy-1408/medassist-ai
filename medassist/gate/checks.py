"""The three deterministic checks.

None of them needs a model, a network call, or a known-correct answer. That is
what makes them usable at serving time, and it is also why they run first: a
claim removed here never reaches the expensive stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from medassist.core.enums import CheckName, ClaimDecision, Severity
from medassist.core.ids import ChunkId
from medassist.core.models import Chunk, Claim, PatientProfile
from medassist.gate.decisions import CheckOutcome
from medassist.gate.interactions import TABLE_VERSION, find_conflicts
from medassist.patient.normalize import normalize
from medassist.gate.quantities import find_unmatched


@dataclass
class GateContext:
    """Everything a check may look at. Notably absent: a gold answer."""

    chunks: dict[ChunkId, Chunk]
    context_ids: list[ChunkId] = field(default_factory=list)
    profile: PatientProfile | None = None
    evidence_required: bool = True

    def resolved(self, claim: Claim) -> list[ChunkId]:
        """Cited chunks that are actually in the context window."""
        window = set(self.context_ids)
        return [c.chunk_id for c in claim.citations if c.chunk_id in window]

    def spans(self, claim: Claim) -> list[str]:
        return [self.chunks[cid].text for cid in self.resolved(claim) if cid in self.chunks]


class Check(Protocol):
    name: CheckName

    def run(self, claim: Claim, ctx: GateContext) -> CheckOutcome: ...


def _timed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class CitationCheck:
    """Does the claim cite anything, and does what it cites exist?

    A model asked to cite ``[C1]``-``[C8]`` will occasionally emit ``[C11]``,
    or cite a chunk that was dropped from the window. Both produce a citation
    that looks valid to a reader and indexes nothing.
    """

    name = CheckName.CITATION_RESOLUTION

    def run(self, claim: Claim, ctx: GateContext) -> CheckOutcome:
        started = time.perf_counter()

        if not claim.citations:
            decision = ClaimDecision.REMOVE if ctx.evidence_required else ClaimDecision.RETAIN
            return CheckOutcome(
                check=self.name,
                decision=decision,
                reason="uncited" if ctx.evidence_required else "uncited_permitted",
                detail=("the claim cites no source, and this capability requires evidence"
                        if ctx.evidence_required else ""),
                severity=Severity.HIGH if ctx.evidence_required else Severity.INFO,
                took_ms=_timed(started),
            )

        resolved = ctx.resolved(claim)
        if not resolved:
            dangling = [str(c.chunk_id) for c in claim.citations]
            return CheckOutcome(
                check=self.name,
                decision=ClaimDecision.REMOVE,
                reason="citation_unresolvable",
                detail=f"cites {len(dangling)} chunk(s), none present in the context window",
                severity=Severity.HIGH,
                took_ms=_timed(started),
            )

        partial = len(resolved) < len(claim.citations)
        return CheckOutcome(
            check=self.name,
            decision=ClaimDecision.RETAIN,
            reason="citations_partially_resolved" if partial else "resolved",
            detail=(f"{len(claim.citations) - len(resolved)} of {len(claim.citations)} "
                    "citations did not resolve" if partial else ""),
            severity=Severity.LOW if partial else Severity.INFO,
            supporting_chunks=resolved,
            took_ms=_timed(started),
        )


class NumericCheck:
    """Every quantity in the claim must appear in a span the claim cites.

    This is the cheapest check and, on dosing claims, the most valuable. See
    ``quantities.py`` for why arithmetic beats a judge here.
    """

    name = CheckName.NUMERIC_GROUNDING

    def run(self, claim: Claim, ctx: GateContext) -> CheckOutcome:
        started = time.perf_counter()
        spans = ctx.spans(claim)
        if not spans:
            return CheckOutcome(
                check=self.name, decision=ClaimDecision.RETAIN, reason="no_spans_to_check",
                detail="", took_ms=_timed(started),
            )

        unmatched = find_unmatched(claim.text, spans)
        if not unmatched:
            return CheckOutcome(
                check=self.name, decision=ClaimDecision.RETAIN, reason="grounded",
                supporting_chunks=ctx.resolved(claim), took_ms=_timed(started),
            )

        rendered = ", ".join(str(q) for q in unmatched)
        return CheckOutcome(
            check=self.name,
            decision=ClaimDecision.REMOVE,
            reason="numeric_ungrounded",
            detail=f"{rendered} appears in no cited span",
            # A wrong dose is the most consequential thing this system can emit.
            severity=Severity.CRITICAL if claim.is_dosage else Severity.HIGH,
            took_ms=_timed(started),
        )


class RelationalCheck:
    """Is this claim, true and well-cited as it may be, wrong for *this person*?

    The check nothing else catches. Groundedness stops at the corpus; this one
    holds the claim against the patient's own record.
    """

    name = CheckName.RELATIONAL_SAFETY

    def run(self, claim: Claim, ctx: GateContext) -> CheckOutcome:
        started = time.perf_counter()
        profile = ctx.profile
        if profile is None or not (profile.medications or profile.conditions or profile.allergies):
            return CheckOutcome(
                check=self.name, decision=ClaimDecision.RETAIN,
                reason="no_patient_record", took_ms=_timed(started),
            )

        medications = normalize(profile.medications)
        conflicts = find_conflicts(
            claim.text, medications, profile.conditions, profile.allergies
        )
        if not conflicts:
            unknown = [m.raw for m in medications if not m.known]
            return CheckOutcome(
                check=self.name,
                decision=ClaimDecision.RETAIN,
                reason="no_conflict" if not unknown else "no_conflict_partial_coverage",
                # An unrecognised medication means this check could not reason
                # about it. Saying so beats implying the list was fully screened.
                detail=("" if not unknown
                        else f"not screened against: {', '.join(unknown)}"),
                severity=Severity.INFO if not unknown else Severity.LOW,
                took_ms=_timed(started),
            )

        # Most severe finding wins, and at equal severity the more restrictive
        # action wins. Ordering on severity alone let a QUALIFY row that
        # happened to sit earlier in the table mask a REMOVE row of the same
        # severity - a patient on warfarin asking about ibuprofen and leafy
        # greens got the dietary caveat and kept the bleeding advice.
        severity_rank = {
            Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
            Severity.HIGH: 3, Severity.CRITICAL: 4,
        }
        action_rank = {
            ClaimDecision.RETAIN: 0, ClaimDecision.QUALIFY: 1, ClaimDecision.REMOVE: 2,
        }
        finding = max(
            conflicts, key=lambda f: (severity_rank[f.severity], action_rank[f.action])
        )
        return CheckOutcome(
            check=self.name,
            decision=finding.action,
            reason=f"{finding.kind}_interaction:{finding.trigger}",
            detail=finding.caveat,
            severity=finding.severity,
            took_ms=_timed(started),
        )

    @property
    def table_version(self) -> str:
        return TABLE_VERSION


DETERMINISTIC_CHECKS: tuple[Check, ...] = (CitationCheck(), NumericCheck(), RelationalCheck())
