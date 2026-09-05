"""PubMed via NCBI E-utilities.

``esearch`` returns PMIDs for a query; ``efetch`` returns full records as XML.
Abstracts arrive as a sequence of ``<AbstractText Label="BACKGROUND">`` elements
in structured abstracts, and those labels are preserved as section headings -
they are genuine document structure, and keeping them lets a retrieved chunk
say it came from a paper's METHODS rather than its CONCLUSIONS.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from medassist.core.config import SETTINGS
from medassist.core.enums import SourceKind
from medassist.core.ids import DocumentId
from medassist.core.models import Document
from medassist.corpus.http import Fetcher
from medassist.corpus.normalize import assemble_sections

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search(fetcher: Fetcher, term: str, retmax: int = 20) -> list[str]:
    params: dict[str, str | int] = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    if SETTINGS.ncbi_key:
        params["api_key"] = SETTINGS.ncbi_key
    import json

    payload = json.loads(fetcher.get(f"{BASE}/esearch.fcgi", params))
    return list(payload.get("esearchresult", {}).get("idlist", []))


def fetch(fetcher: Fetcher, pmids: list[str]) -> list[Document]:
    if not pmids:
        return []
    params: dict[str, str | int] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if SETTINGS.ncbi_key:
        params["api_key"] = SETTINGS.ncbi_key
    xml = fetcher.get(f"{BASE}/efetch.fcgi", params)
    return parse(xml)


def parse(xml: str) -> list[Document]:
    root = ET.fromstring(xml)
    docs: list[Document] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text.strip()
        title = _text(title_el)

        # Structured abstracts carry real section labels (BACKGROUND, METHODS,
        # CONCLUSIONS). Preserving them means a retrieved chunk can say which
        # part of a paper it came from, which matters: a claim sourced from a
        # paper's BACKGROUND is restating prior work, not reporting a finding.
        parts: list[tuple[str, str]] = []
        for abstract_text in article.findall(".//Abstract/AbstractText"):
            body = _text(abstract_text)
            if not body:
                continue
            label = (abstract_text.get("Label") or "").strip().lower() or "abstract"
            parts.append((label, body))
        if not parts:
            continue  # a citation with no abstract is not retrievable content
        text, spans = assemble_sections(parts)

        journal = _text(article.find(".//Journal/Title"))
        year = _text(article.find(".//JournalIssue/PubDate/Year"))
        pub_types = [_text(p) for p in article.findall(".//PublicationType")]
        mesh = [_text(m) for m in article.findall(".//MeshHeading/DescriptorName")]

        docs.append(
            Document(
                id=DocumentId.new(),
                source=SourceKind.PUBMED_ABSTRACT,
                source_uid=f"PMID:{pmid}",
                title=title,
                text=text,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                published=year,
                meta={
                    "pmid": pmid,
                    "journal": journal,
                    "publication_types": [p for p in pub_types if p],
                    "mesh_terms": [m for m in mesh if m][:20],
                    "sections": spans,
                },
            )
        )
    return docs


def _text(element: ET.Element | None) -> str:
    """Flatten an element including inline markup (``<i>``, ``<sup>``, ...).

    ``element.text`` alone silently truncates at the first inline tag, which in
    PubMed titles is common enough to lose half of them.
    """
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())
