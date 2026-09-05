"""MedlinePlus health topics - NLM's consumer-facing encyclopedia.

Included because a large share of real questions are asked in lay language, and
a corpus of only journal abstracts and regulatory labels answers them in a
register the asker cannot use. Retrieval quality is measured per source, so the
effect of this choice is visible rather than assumed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from medassist.core.enums import SourceKind
from medassist.core.ids import DocumentId
from medassist.core.models import Document
from medassist.corpus.http import Fetcher
from medassist.corpus.normalize import assemble_sections, clean

BASE = "https://wsearch.nlm.nih.gov/ws/query"


def search(fetcher: Fetcher, term: str, retmax: int = 5) -> list[Document]:
    raw = fetcher.get(BASE, {"db": "healthTopics", "term": term, "retmax": retmax})
    return parse(raw)


def parse(raw: str) -> list[Document]:
    root = ET.fromstring(raw)
    docs: list[Document] = []
    for document in root.findall(".//list/document"):
        url = document.get("url", "")
        fields: dict[str, list[str]] = {}
        for content in document.findall("content"):
            name = content.get("name", "")
            fields.setdefault(name, []).append(" ".join(content.itertext()))

        title = clean(_first(fields, "title"))
        summary = _first(fields, "FullSummary")
        if not title or not summary:
            continue

        also_called = [clean(a) for a in fields.get("altTitle", [])]
        parts = [("overview", summary)]
        if also_called:
            parts.append(("also_called", "; ".join(also_called)))
        text, spans = assemble_sections(parts)

        docs.append(
            Document(
                id=DocumentId.new(),
                source=SourceKind.MEDLINEPLUS,
                source_uid=f"MEDLINEPLUS:{url.rsplit('/', 1)[-1]}",
                title=title,
                text=text,
                url=url,
                meta={
                    "also_called": also_called,
                    "groups": [clean(g) for g in fields.get("groupName", [])][:5],
                    "mesh": [clean(m) for m in fields.get("mesh", [])][:10],
                    "sections": spans,
                },
            )
        )
    return docs


def _first(fields: dict[str, list[str]], key: str) -> str:
    values = fields.get(key) or []
    return values[0] if values else ""
