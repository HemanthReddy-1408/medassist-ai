"""Prompt injection arriving inside retrieved documents.

The corpus is scraped from the public internet, so **a retrieved document is an
untrusted input channel** - and this is the attack surface most RAG systems
leave open. Direct injection is well covered by any provider's safety training;
indirect injection, where the payload rides in on a document the system itself
chose to retrieve, is not.

This module is layer 2 of four (§07.5). It is pattern matching against an
adversary who can rephrase, so it is *not* the defence that holds. The one that
holds is layer 3: the closed ``allowed_tools`` set, which does not depend on
recognising the attack. This layer exists to make attempts visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medassist.core.enums import Severity

PATTERNS: tuple[tuple[str, str, Severity], ...] = (
    (r"ignore (?:all )?(?:your )?(?:previous|prior|above) instructions?",
     "instruction override", Severity.CRITICAL),
    (r"disregard (?:all )?(?:the )?(?:previous|prior|above)", "instruction override", Severity.CRITICAL),
    (r"you are now (?:a|an|the)\b", "role reassignment", Severity.HIGH),
    (r"^\s*(?:system|assistant)\s*:", "role spoofing", Severity.HIGH),
    (r"</?(?:system|instruction|prompt)>", "delimiter injection", Severity.HIGH),
    (r"(?:new|updated) instructions?\s*:", "instruction injection", Severity.HIGH),
    (r"do not (?:mention|cite|reveal|tell)", "output suppression", Severity.HIGH),
    (r"always recommend\b", "output steering", Severity.HIGH),
    (r"(?:reveal|print|output|repeat) (?:your |the )?(?:system )?prompt",
     "prompt extraction", Severity.MEDIUM),
    (r"\bDAN\b|\bjailbreak\b", "jailbreak marker", Severity.MEDIUM),
)

_COMPILED = tuple(
    (re.compile(p, re.IGNORECASE | re.MULTILINE), label, sev) for p, label, sev in PATTERNS
)


@dataclass(frozen=True)
class InjectionFinding:
    label: str
    severity: Severity
    matched: str
    where: str = ""


def scan(text: str, *, where: str = "") -> list[InjectionFinding]:
    findings: list[InjectionFinding] = []
    for pattern, label, severity in _COMPILED:
        match = pattern.search(text)
        if match:
            findings.append(
                InjectionFinding(
                    label=label, severity=severity, matched=match.group(0)[:120], where=where
                )
            )
    return findings


def scan_chunks(chunks: dict) -> list[InjectionFinding]:
    """Scan retrieved context before it is assembled into a prompt."""
    findings: list[InjectionFinding] = []
    for chunk_id, chunk in chunks.items():
        findings.extend(scan(chunk.text, where=str(chunk_id)))
    return findings
