"""Loading reports from text or PDF, with PII removed before anything else."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from medassist.core.errors import MedAssistError
from medassist.core.models import LabReport
from medassist.guards.pii import Redaction, redact
from medassist.reports.analytes import extract_analytes

_COLLECTED = re.compile(
    r"\b(?:collected|drawn|specimen date|date of (?:service|collection))\b[:\s]*"
    r"(?P<date>\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)
_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("lipid", "lipid panel"),
    ("metabolic", "metabolic panel"),
    ("complete blood count", "complete blood count"),
    (" cbc", "complete blood count"),
    ("thyroid", "thyroid panel"),
    ("liver", "liver function"),
    ("coagulation", "coagulation"),
    ("laboratory report", "laboratory report"),
)


class ReportParseError(MedAssistError):
    """The file could not be read as a report."""


@dataclass
class ParsedReport:
    report: LabReport
    redaction: Redaction

    @property
    def contained_pii(self) -> bool:
        return self.redaction.found


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ReportParseError(f"cannot read {path}: {exc}") from exc


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ReportParseError(
            "PDF support needs pypdf: pip install pypdf"
        ) from exc
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ReportParseError(f"cannot parse {path} as PDF: {exc}") from exc
    # Bound the work: a 500-page document handed to a lab-report parser is a
    # malformed input, not a report, and should fail predictably.
    pages = reader.pages[:40]
    return "\n".join((page.extract_text() or "") for page in pages)


def detect_type(text: str) -> str:
    lowered = text.lower()
    for needle, label in _TYPE_HINTS:
        if needle in lowered:
            return label
    return ""


def detect_collected(text: str) -> str:
    match = _COLLECTED.search(text)
    return match.group("date") if match else ""


def parse_report(text: str, *, source_name: str = "") -> ParsedReport:
    """Redact, then structure. Order matters: nothing unredacted is retained."""
    redaction = redact(text)
    body = redaction.text
    analytes, unparsed = extract_analytes(body)
    return ParsedReport(
        report=LabReport(
            analytes=analytes,
            report_type=detect_type(body),
            collected=detect_collected(body),
            narrative=" ".join(unparsed)[:2000],
            unparsed="\n".join(unparsed),
            source_name=source_name,
        ),
        redaction=redaction,
    )


def load_report(path: str | Path) -> ParsedReport:
    path = Path(path)
    if not path.exists():
        raise ReportParseError(f"no such report: {path}")
    return parse_report(read_text(path), source_name=path.name)
