"""Text normalization and section assembly shared by every source.

Sections are carried as explicit ``(name, start, end)`` offsets rather than
inferred from markdown headings later. The chunker uses them to tag each chunk
with the section it came from, and the dosage guard depends on that tag: a
numeric dose is only allowed to stand if it cites a chunk from an FDA label's
``dosage_and_administration`` section. Recovering that by re-parsing prose would
be guesswork on exactly the claim where guessing is least acceptable.
"""

from __future__ import annotations

import html
import re
import unicodedata

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")
_TAGS = re.compile(r"<[^>]+>")


def clean(text: str) -> str:
    """Unescape entities, drop markup, normalize unicode and whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<\s*(br|/p|/div|/li)\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    # NFKC folds ligatures and full-width forms; medical text from PDFs and SPL
    # documents is full of them, and unfolded they break lexical matching.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def assemble_sections(parts: list[tuple[str, str]]) -> tuple[str, list[dict[str, object]]]:
    """Join ``(name, body)`` pairs into one document, recording offsets.

    Returns the document text and a list of ``{name, start, end}`` spans whose
    offsets index into that exact string.
    """
    buffer: list[str] = []
    spans: list[dict[str, object]] = []
    cursor = 0
    for name, body in parts:
        body = clean(body)
        if not body:
            continue
        heading = f"## {name.replace('_', ' ').title()}\n"
        block = heading + body
        start = cursor + len(heading)
        spans.append({"name": name, "start": start, "end": start + len(body)})
        buffer.append(block)
        cursor += len(block) + 2  # the "\n\n" join below
    return "\n\n".join(buffer), spans


def section_at(spans: list[dict[str, object]], offset: int) -> str:
    for span in spans:
        if int(span["start"]) <= offset < int(span["end"]):  # type: ignore[arg-type]
            return str(span["name"])
    return ""
