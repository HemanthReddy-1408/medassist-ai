"""Reconciling a new report against what we already knew.

This is the longitudinal half of the system. A single abnormal value and a
value that has been abnormal at every draw since March are different clinical
facts, and only the second is visible across time.

The behaviour that matters: when a prior abnormal finding is **not re-measured**
in a new report, the system neither forgets it nor assumes it still holds. It
asks. Silently carrying a stale abnormality forward would let the model reason
from a value that may be a year out of date; silently dropping it would lose a
finding the person is still living with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from medassist.core.enums import Severity
from medassist.core.models import LabReport
from medassist.memory.store import MemoryEntry, MemoryKind, MemoryStore


class Continuity(StrEnum):
    RESOLVED = "resolved"        # was abnormal, now in range
    PERSISTING = "persisting"    # abnormal then, abnormal now
    IMPROVING = "improving"      # abnormal both times, moving toward the interval
    WORSENING = "worsening"      # abnormal both times, moving away
    NEW = "new"                  # abnormal now, no prior record
    UNCHECKED = "unchecked"      # abnormal before, absent from this report


class ConfirmReason(StrEnum):
    NOT_REMEASURED = "not_remeasured"
    STALE = "stale"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Continuation:
    key: str
    continuity: Continuity
    previous: MemoryEntry | None
    current_value: float | None
    current_flag: str
    unit: str = ""

    def describe(self) -> str:
        if self.previous is None:
            return f"{self.key}: {self.current_value:g} {self.unit} ({self.continuity.value})"
        prior = f"{self.previous.value:g}" if self.previous.value is not None else "?"
        now = f"{self.current_value:g}" if self.current_value is not None else "not measured"
        return (
            f"{self.key}: {prior} ({self.previous.asserted_on}) -> {now} "
            f"[{self.continuity.value}]"
        )


@dataclass(frozen=True)
class ConfirmationRequest:
    """A question to put to the person before their history is used."""

    key: str
    reason: ConfirmReason
    severity: Severity
    prior: MemoryEntry
    question: str

    def describe(self) -> str:
        return f"[{self.reason.value}] {self.question}"


@dataclass
class Reconciliation:
    continuations: list[Continuation] = field(default_factory=list)
    confirmations: list[ConfirmationRequest] = field(default_factory=list)
    new_entries: list[MemoryEntry] = field(default_factory=list)

    @property
    def needs_user_input(self) -> bool:
        return bool(self.confirmations)

    def by_continuity(self, continuity: Continuity) -> list[Continuation]:
        return [c for c in self.continuations if c.continuity is continuity]

    def summary(self) -> str:
        lines = [c.describe() for c in self.continuations]
        if self.confirmations:
            lines.append("")
            lines.append("Before I use your history, please confirm:")
            lines.extend(f"  - {c.question}" for c in self.confirmations)
        return "\n".join(lines)


def _severity_of(entry: MemoryEntry) -> Severity:
    from medassist.core.models import LabAnalyte
    from medassist.reports.analytes import is_critical

    if entry.value is None:
        return Severity.LOW
    analyte = LabAnalyte(name=entry.key, value=entry.value, unit=entry.unit)
    if is_critical(analyte):
        return Severity.CRITICAL
    return Severity.MEDIUM if entry.abnormal else Severity.INFO


def _direction(previous: MemoryEntry, value: float, flag: str) -> Continuity:
    if flag == "normal":
        return Continuity.RESOLVED
    if previous.value is None:
        return Continuity.PERSISTING
    # Moving toward the reference interval is improvement, whichever side it
    # started on.
    if previous.flag == "low":
        return Continuity.IMPROVING if value > previous.value else Continuity.WORSENING
    if previous.flag == "high":
        return Continuity.IMPROVING if value < previous.value else Continuity.WORSENING
    return Continuity.PERSISTING


def _question(entry: MemoryEntry, reason: ConfirmReason) -> str:
    described = entry.describe()
    if reason is ConfirmReason.NOT_REMEASURED:
        return (
            f"Your earlier record shows {described}, and this report does not "
            f"include {entry.key}. Is that still the case, has it been rechecked "
            "since, or was it treated?"
        )
    if reason is ConfirmReason.STALE:
        return (
            f"The most recent {entry.key} I have is {described}, which is now "
            f"{entry.age_days()} days old. Should I still treat that as current?"
        )
    return f"My record says {described}, which conflicts with what you have told me. Which is right?"


def reconcile(
    prior: list[MemoryEntry],
    report: LabReport,
    *,
    digest: str = "",
    today: date | None = None,
    max_questions: int = 3,
    provenance: str = "",
) -> Reconciliation:
    """Compare a new report against memory and decide what to ask.

    ``max_questions`` is a real constraint, not a nicety. A system that opens
    with nine questions gets none of them answered, so only the most severe
    unresolved findings are raised.
    """
    from medassist.memory.store import entries_from_report

    latest_prior: dict[str, MemoryEntry] = {}
    for entry in prior:
        if entry.kind is MemoryKind.LAB_FINDING:
            latest_prior[entry.key] = entry

    current = {a.name: a for a in report.analytes}
    continuations: list[Continuation] = []
    candidates: list[ConfirmationRequest] = []

    for key, entry in latest_prior.items():
        analyte = current.get(key)
        if analyte is not None:
            continuity = (
                _direction(entry, analyte.value, analyte.flag)
                if entry.abnormal
                else (Continuity.NEW if analyte.abnormal else Continuity.RESOLVED)
            )
            continuations.append(
                Continuation(
                    key=key, continuity=continuity, previous=entry,
                    current_value=analyte.value, current_flag=analyte.flag,
                    unit=analyte.unit or entry.unit,
                )
            )
            continue

        # Not re-measured. Only abnormal history is worth raising: confirming
        # that a normal value is still normal has no effect on any answer.
        if not entry.abnormal:
            continue
        continuations.append(
            Continuation(
                key=key, continuity=Continuity.UNCHECKED, previous=entry,
                current_value=None, current_flag="", unit=entry.unit,
            )
        )
        reason = (
            ConfirmReason.STALE if entry.is_stale(today) else ConfirmReason.NOT_REMEASURED
        )
        candidates.append(
            ConfirmationRequest(
                key=key, reason=reason, severity=_severity_of(entry),
                prior=entry, question=_question(entry, reason),
            )
        )

    for name, analyte in current.items():
        if name not in latest_prior and analyte.abnormal:
            continuations.append(
                Continuation(
                    key=name, continuity=Continuity.NEW, previous=None,
                    current_value=analyte.value, current_flag=analyte.flag,
                    unit=analyte.unit,
                )
            )

    rank = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
    candidates.sort(key=lambda c: (rank[c.severity], c.key))

    return Reconciliation(
        continuations=continuations,
        confirmations=candidates[:max_questions],
        new_entries=entries_from_report(digest, report, provenance=provenance),
    )


def observe_report(
    store: MemoryStore,
    digest: str,
    report: LabReport,
    *,
    today: date | None = None,
    provenance: str = "",
) -> Reconciliation:
    """Reconcile, then record. Reading before writing is the whole point.

    Recording first would make every prior finding look re-measured, and no
    confirmation would ever be raised.
    """
    prior = store.for_profile(digest)
    result = reconcile(
        prior, report, digest=digest, today=today, provenance=provenance
    )
    store.extend(result.new_entries)
    return result
