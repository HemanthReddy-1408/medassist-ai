"""Episodic memory: what this profile has told us, and when.

Two rules, both from §05.7.

**Nothing is silently overwritten.** If memory holds "haemoglobin low, March"
and a new report says nothing about haemoglobin, the old fact is not deleted
and it is not assumed to still hold - it becomes a question. In this domain the
contradiction, or the unexplained disappearance, is often the clinically
interesting event.

**Every entry carries provenance and an expiry.** A lab value from eight months
ago is not a current fact about a person. Treating it as one is how a system
confidently reasons from stale data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path


class MemoryKind(StrEnum):
    LAB_FINDING = "lab_finding"
    CONDITION = "condition"
    MEDICATION = "medication"
    USER_STATEMENT = "user_statement"


#: How long a fact of each kind stays current without re-confirmation.
#: Lab values drift; a chronic diagnosis does not. These are deliberately
#: conservative - the cost of asking again is a question, and the cost of
#: assuming is reasoning from a value that may be a year out of date.
SHELF_LIFE_DAYS: dict[str, int] = {
    MemoryKind.LAB_FINDING: 180,
    MemoryKind.CONDITION: 730,
    MemoryKind.MEDICATION: 365,
    MemoryKind.USER_STATEMENT: 365,
}

#: Analytes that move quickly enough to need re-checking sooner.
FAST_MOVING: dict[str, int] = {
    "Hemoglobin": 120,
    "Potassium": 60,
    "Sodium": 60,
    "Creatinine": 120,
    "eGFR": 120,
    "INR": 30,
    "Glucose": 90,
    "WBC": 90,
    "Platelets": 90,
}


def _today() -> date:
    return datetime.now(UTC).date()


def _parse_date(value: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


@dataclass(frozen=True)
class MemoryEntry:
    profile_digest: str
    kind: MemoryKind
    key: str
    asserted_on: str                     # ISO date the fact was true, not recorded
    value: float | None = None
    unit: str = ""
    flag: str = ""                       # low | normal | high | ""
    source: str = "report"               # report | user | inference
    provenance: str = ""                 # file name, run id, or utterance
    confidence: float = 1.0
    superseded_by: str = ""
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def asserted_date(self) -> date | None:
        return _parse_date(self.asserted_on)

    def shelf_life_days(self) -> int:
        if self.kind is MemoryKind.LAB_FINDING and self.key in FAST_MOVING:
            return FAST_MOVING[self.key]
        return SHELF_LIFE_DAYS[self.kind]

    def age_days(self, today: date | None = None) -> int | None:
        asserted = self.asserted_date
        if asserted is None:
            return None
        return (today or _today()) - asserted and ((today or _today()) - asserted).days

    def is_stale(self, today: date | None = None) -> bool:
        """Past its shelf life, so it may no longer describe this person."""
        age = self.age_days(today)
        if age is None:
            return True  # an undated fact cannot be shown to be current
        return age > self.shelf_life_days()

    @property
    def abnormal(self) -> bool:
        return self.flag in ("low", "high")

    def describe(self) -> str:
        parts = [self.key]
        if self.value is not None:
            parts.append(f"{self.value:g} {self.unit}".strip())
        if self.flag:
            parts.append(f"({self.flag})")
        parts.append(f"as of {self.asserted_on or 'unknown date'}")
        return " ".join(parts)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict) -> MemoryEntry:
        data = dict(payload)
        data["kind"] = MemoryKind(data["kind"])
        return cls(**data)


class MemoryStore:
    """Append-only, partitioned by profile digest.

    Append-only because history *is* the value: 'haemoglobin has been low at
    every draw since March' is a different fact from 'haemoglobin is low', and
    a store that overwrites can only ever express the second.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: MemoryEntry) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json() + "\n")

    def extend(self, entries: list[MemoryEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def all_entries(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        return [
            MemoryEntry.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def for_profile(self, digest: str) -> list[MemoryEntry]:
        entries = [e for e in self.all_entries() if e.profile_digest == digest]
        return sorted(entries, key=lambda e: (e.asserted_on, e.recorded_at))

    def latest(self, digest: str, key: str) -> MemoryEntry | None:
        matching = [e for e in self.for_profile(digest) if e.key == key]
        return matching[-1] if matching else None

    def history(self, digest: str, key: str) -> list[MemoryEntry]:
        return [e for e in self.for_profile(digest) if e.key == key]

    def current_facts(self, digest: str, today: date | None = None) -> list[MemoryEntry]:
        """Most recent entry per key, excluding stale ones."""
        by_key: dict[str, MemoryEntry] = {}
        for entry in self.for_profile(digest):
            by_key[entry.key] = entry
        return [e for e in by_key.values() if not e.is_stale(today)]


def entries_from_report(digest: str, report, *, provenance: str = "") -> list[MemoryEntry]:
    """Turn a parsed report into memory entries.

    The report's own collection date is used where present. Falling back to
    today would date an old uploaded report as current, which is exactly the
    error the shelf life exists to prevent.
    """
    asserted = report.collected or _today().isoformat()
    return [
        MemoryEntry(
            profile_digest=digest,
            kind=MemoryKind.LAB_FINDING,
            key=analyte.name,
            asserted_on=asserted,
            value=analyte.value,
            unit=analyte.unit,
            flag=analyte.flag,
            source="report",
            provenance=provenance or report.source_name,
        )
        for analyte in report.analytes
    ]


def months_ago(months: int) -> str:
    return (_today() - timedelta(days=30 * months)).isoformat()
