"""Claim-structured generation.

The generator emits claims with citation handles; prose is rendered *from* the
claims. Generating prose first and decomposing it afterwards adds an extraction
step that hallucinates too, and produces claims the answer never actually made
- attributing a failure to the generator that belongs to the extractor.

Handles (``C1``…``Cn``) are resolved to real chunk ids **in code**. A model
asked to echo a 26-character ULID will occasionally emit a plausible-looking
identifier that indexes nothing, so the mapping is never the model's to define.
"""

from __future__ import annotations

import re

from medassist.core.ids import ChunkId
from medassist.core.models import Chunk, Citation, Claim, PatientProfile, Usage
from medassist.gate.quantities import extract_quantities

_HANDLE = re.compile(r"C(\d+)", re.IGNORECASE)
_MASS_DIMENSIONS = {"mass", "mass_per_kg", "dose_form", "iu"}

SYSTEM = """You answer clinical questions strictly from the passages provided.

Break your answer into short, atomic factual claims. Each claim must be one
assertion, and must cite the passage handles that support it.

Rules:
- Use ONLY the passages given. If they do not answer the question, say so with
  an empty claims list rather than filling the gap from memory.
- Every claim cites at least one handle, e.g. "C1".
- Copy numbers and doses EXACTLY as they appear in the passage you cite.
- Mark a claim with "is_dosage": true when it states a dose, strength or amount.
- A claim that is advice rather than a fact (for example, to consult a doctor)
  should still be listed, cited to the most relevant passage.

Reply with JSON only:
{"claims":[{"text":"...","cites":["C1"],"is_dosage":false}],
 "insufficient":false}"""


def build_prompt(
    question: str, context: str, profile: PatientProfile | None = None
) -> list[dict[str, str]]:
    parts = [f"QUESTION: {question}"]
    if profile is not None:
        # The profile is given so the answer can be relevant, but relational
        # safety is NOT delegated to the model - it is checked in code (§08.3).
        parts.append(f"PATIENT CONTEXT: {profile.describe()}")
    parts.append(f"PASSAGES:\n{context}")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def resolve_handles(raw: object, handles: dict[str, ChunkId], chunks: dict[ChunkId, Chunk]) -> list[Citation]:
    """Turn ``["C1", "C4"]`` into citations, dropping anything that misses.

    A handle that names no chunk in the window is silently discarded here and
    caught by the gate's citation check, which reports it as
    ``citation_unresolvable`` rather than letting it look valid.
    """
    if not isinstance(raw, list):
        return []
    citations: list[Citation] = []
    seen: set[ChunkId] = set()
    for item in raw:
        match = _HANDLE.search(str(item))
        if match is None:
            continue
        chunk_id = handles.get(f"C{int(match.group(1))}")
        if chunk_id is None or chunk_id in seen:
            continue
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        seen.add(chunk_id)
        citations.append(
            Citation(chunk_id=chunk_id, doc_id=chunk.doc_id, source=chunk.source)
        )
    return citations


def looks_like_dosage(text: str) -> bool:
    return any(q.dimension in _MASS_DIMENSIONS for q in extract_quantities(text))


def generate_claims(
    client,
    question: str,
    context: str,
    handles: dict[str, ChunkId],
    chunks: dict[ChunkId, Chunk],
    *,
    profile: PatientProfile | None = None,
    max_tokens: int = 900,
) -> tuple[list[Claim], Usage, bool]:
    """Returns ``(claims, usage, insufficient)``.

    ``insufficient`` is the model declaring the passages do not answer the
    question. It is honoured rather than second-guessed - an abstention the
    generator volunteers is cheaper and more reliable than one recovered from a
    low support fraction downstream.
    """
    payload, usage = client.structured(
        build_prompt(question, context, profile), temperature=0.0, max_tokens=max_tokens
    )
    if not isinstance(payload, dict):
        return [], usage, True

    insufficient = bool(payload.get("insufficient"))
    claims: list[Claim] = []
    for item in payload.get("claims") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        citations = resolve_handles(item.get("cites"), handles, chunks)
        claims.append(
            Claim(
                text=text,
                citations=citations,
                is_dosage=bool(item.get("is_dosage")) or looks_like_dosage(text),
            )
        )
    return claims, usage, insufficient


def render_prose(claims: list[Claim], kept: set[ChunkId] | None = None) -> str:
    """Prose is derived from the surviving claims, never the other way round."""
    return " ".join(c.text for c in claims)
