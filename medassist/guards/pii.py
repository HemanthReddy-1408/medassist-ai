"""PII detection and redaction.

Uploaded reports carry names, medical record numbers, dates of birth. Redaction
runs **before** any text reaches a provider, and the map stays local so the
answer can be rehydrated for the user. Cache keys are computed on redacted
text, so the on-disk cache never holds identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: tuple[tuple[str, str], ...] = (
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("MRN", r"\b(?:MRN|Medical Record(?: Number)?)[:\s#]*([A-Z0-9-]{5,})\b"),
    ("NHS", r"\b\d{3}\s?\d{3}\s?\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    ("PHONE", r"\b(?:\+?\d{1,2}[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    ("DOB", r"\b(?:DOB|Date of Birth|Born)[:\s]*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b"),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("NAME", r"\b(?:Patient(?:\s+Name)?|Name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z'.-]+){1,2})\b"),
    # Conversational self-identification. The labelled form above only catches
    # a report header; a person typing "my name is Jane Doe" into the chat box
    # was reaching the provider unredacted. Two capitalised tokens are required
    # so that "I am Diabetic" at the start of a sentence does not match.
    # The prefix is matched case-insensitively via a scoped flag; the captured
    # name is not, because requiring two capitalised tokens is what keeps
    # "I am diabetic" from being treated as a name.
    ("NAME", r"\b(?i:my name is|name'?s|i am|i'm|this is)\s+([A-Z][a-z]+\s+[A-Z][a-z'.-]+)\b"),
    ("ADDRESS", r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr)\b"),
)

_COMPILED = tuple((label, re.compile(p)) for label, p in PATTERNS)


@dataclass
class Redaction:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.mapping)

    @property
    def kinds(self) -> list[str]:
        return sorted({token.strip("[]").rsplit("_", 1)[0] for token in self.mapping})

    def rehydrate(self, text: str) -> str:
        """Put the originals back, for display to the user only."""
        for token, original in self.mapping.items():
            text = text.replace(token, original)
        return text


def redact(text: str) -> Redaction:
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    out = text

    for label, pattern in _COMPILED:
        for match in list(pattern.finditer(out)):
            # Prefer the capture group where one exists, so "MRN: 12345"
            # redacts the number and keeps the label readable.
            original = match.group(1) if match.groups() else match.group(0)
            if not original or original in mapping.values():
                continue
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            mapping[token] = original
            out = out.replace(original, token)

    return Redaction(text=out, mapping=mapping)


def contains_pii(text: str) -> bool:
    return redact(text).found
