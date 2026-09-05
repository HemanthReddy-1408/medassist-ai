"""Section-aware chunking.

Fixed-width chunking cuts an FDA label mid-sentence and hands the retriever a
span that begins "…mg twice daily in patients with" - text that is unusable as
a citation and actively dangerous as a dosage source. So chunks here are cut
inside section boundaries and on sentence boundaries, and every chunk keeps the
character offsets that let a citation resolve back to the exact substring.

Overlap exists because the alternative is worse: a fact that straddles a
boundary is retrievable from neither side. The cost is duplicated text in the
index, which the fusion step deduplicates by document position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medassist.core.ids import ChunkId
from medassist.core.models import Chunk, Document

# Sentence terminator not preceded by a common abbreviation or a digit (so
# "2.5 mg" and "e.g." do not become sentence breaks - both are everywhere in
# clinical text and both produce garbage chunks when treated as boundaries).
_ABBREV = r"(?<!\b[A-Za-z])(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bDr)(?<!\bNo)(?<!\bapprox)"
_SENTENCE = re.compile(rf"{_ABBREV}(?<![0-9])([.!?])\s+(?=[A-Z(])")


@dataclass(frozen=True)
class ChunkSpec:
    target_chars: int = 900
    overlap_chars: int = 150
    min_chars: int = 120
    respect_sections: bool = True

    def __post_init__(self) -> None:
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap must be smaller than the target size, else chunking loops")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for each sentence."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE.finditer(text):
        end = match.end(1)
        if end > cursor:
            spans.append((cursor, end))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


def _section_spans(doc: Document, spec: ChunkSpec) -> list[tuple[str, int, int]]:
    raw = doc.meta.get("sections") if spec.respect_sections else None
    if not raw:
        return [("", 0, len(doc.text))]
    spans = [(str(s["name"]), int(s["start"]), int(s["end"])) for s in raw]  # type: ignore[index]
    return [s for s in spans if s[2] > s[1]] or [("", 0, len(doc.text))]


def chunk_document(doc: Document, spec: ChunkSpec | None = None) -> list[Chunk]:
    spec = spec or ChunkSpec()
    chunks: list[Chunk] = []
    ordinal = 0

    for section_name, sec_start, sec_end in _section_spans(doc, spec):
        body = doc.text[sec_start:sec_end]
        sentences = split_sentences(body)
        if not sentences:
            continue

        window: list[tuple[int, int]] = []
        size = 0
        emitted_here = 0
        for span in sentences:
            length = span[1] - span[0]
            # A single sentence longer than the target becomes its own chunk
            # rather than being force-split; long enumerated dosage sentences
            # are meaningful whole and meaningless in halves.
            if size and size + length > spec.target_chars:
                ordinal, emitted = _emit(
                    chunks, doc, window, sec_start, section_name, ordinal, spec,
                    first_in_section=emitted_here == 0,
                )
                emitted_here += emitted
                window = _carry_overlap(window, spec.overlap_chars)
                size = sum(e - s for s, e in window)
            window.append(span)
            size += length

        if window:
            ordinal, _ = _emit(
                chunks, doc, window, sec_start, section_name, ordinal, spec,
                first_in_section=emitted_here == 0,
            )

    return chunks


def _carry_overlap(window: list[tuple[int, int]], overlap: int) -> list[tuple[int, int]]:
    carried: list[tuple[int, int]] = []
    total = 0
    for span in reversed(window):
        length = span[1] - span[0]
        if total + length > overlap and carried:
            break
        carried.insert(0, span)
        total += length
    return carried


def _emit(
    chunks: list[Chunk],
    doc: Document,
    window: list[tuple[int, int]],
    offset: int,
    section: str,
    ordinal: int,
    spec: ChunkSpec,
    *,
    first_in_section: bool,
) -> tuple[int, int]:
    """Append one chunk. Returns ``(next_ordinal, emitted_count)``."""
    if not window:
        return ordinal, 0
    start, end = window[0][0], window[-1][1]
    text = doc.text[offset + start : offset + end].strip()
    if not text:
        return ordinal, 0
    # The floor discards trailing overlap fragments, NOT whole sections. A
    # short section is still a section: `contraindications` is frequently
    # under 120 characters and is the last thing that should silently vanish
    # from the index. Dropping it would show up as NOT_INDEXED on exactly the
    # safety-critical queries, which is the failure this project exists to
    # catch rather than to cause.
    if len(text) < spec.min_chars and not first_in_section:
        return ordinal, 0
    chunks.append(
        Chunk(
            id=ChunkId.new(),
            doc_id=doc.id,
            text=text,
            ordinal=ordinal,
            section=section,
            source=doc.source,
            start=offset + start,
            end=offset + end,
            meta={
                "doc_title": doc.title,
                "doc_uid": doc.source_uid,
                "url": doc.url,
                "authority": doc.authority,
            },
        )
    )
    return ordinal + 1, 1


def chunk_corpus(docs: list[Document], spec: ChunkSpec | None = None) -> list[Chunk]:
    spec = spec or ChunkSpec()
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, spec))
    return out
