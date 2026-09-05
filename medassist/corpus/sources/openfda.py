"""openFDA drug labels - the Structured Product Label for an approved drug.

These are the authoritative documents for dosing, contraindications and boxed
warnings, which is why they outrank literature in synthesis. Only the sections
that carry clinical decision content are kept; ``how_supplied`` and packaging
panels are noise for question answering and dilute retrieval.
"""

from __future__ import annotations

import json

from medassist.core.enums import SourceKind
from medassist.core.ids import DocumentId
from medassist.core.models import Document
from medassist.corpus.http import Fetcher
from medassist.corpus.normalize import assemble_sections

BASE = "https://api.fda.gov/drug/label.json"

# Ordered by clinical weight, not by their order in the SPL.
SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "drug_interactions",
    "adverse_reactions",
    "use_in_specific_populations",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "overdosage",
    "mechanism_of_action",
    "information_for_patients",
]


def fetch_drug(fetcher: Fetcher, drug: str, limit: int = 2) -> list[Document]:
    """Fetch labels for a generic drug name."""
    raw = fetcher.get(
        BASE, {"search": f'openfda.generic_name:"{drug}"', "limit": limit}
    )
    return parse(raw, drug)


def parse(raw: str, queried: str = "") -> list[Document]:
    payload = json.loads(raw)
    docs: list[Document] = []
    for result in payload.get("results", []):
        meta_fda = result.get("openfda", {})
        generic = _first(meta_fda.get("generic_name")) or queried
        brand = _first(meta_fda.get("brand_name"))
        set_id = result.get("set_id") or result.get("id") or generic

        parts = [(name, _join(result.get(name))) for name in SECTIONS if result.get(name)]
        text, spans = assemble_sections(parts)
        if not text:
            continue

        title = f"{generic.title()} ({brand})" if brand else generic.title()
        docs.append(
            Document(
                id=DocumentId.new(),
                source=SourceKind.FDA_LABEL,
                source_uid=f"FDA:{set_id}",
                title=f"FDA Label - {title}",
                text=text,
                url=f"https://labels.fda.gov/{set_id}",
                published=str(result.get("effective_time", ""))[:4],
                meta={
                    "set_id": set_id,
                    "generic_name": generic,
                    "brand_name": brand,
                    "manufacturer": _first(meta_fda.get("manufacturer_name")),
                    "route": _first(meta_fda.get("route")),
                    "pharm_class": (meta_fda.get("pharm_class_epc") or [])[:3],
                    "rxcui": (meta_fda.get("rxcui") or [])[:3],
                    "sections": spans,
                    "has_boxed_warning": bool(result.get("boxed_warning")),
                },
            )
        )
    return docs


def _first(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value) if isinstance(value, str) else ""


def _join(value: object) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(v) for v in value)
    return str(value or "")
