from __future__ import annotations

import pytest

from medassist.core.enums import SourceKind
from medassist.core.ids import DocumentId
from medassist.core.models import Document
from medassist.corpus.normalize import assemble_sections


def _doc(uid: str, source: SourceKind, title: str, parts: list[tuple[str, str]]) -> Document:
    text, spans = assemble_sections(parts)
    return Document(
        id=DocumentId.new(), source=source, source_uid=uid, title=title,
        text=text, meta={"sections": spans},
    )


@pytest.fixture
def corpus() -> list[Document]:
    """A tiny corpus with the structure of the real one: sectioned labels,
    a consumer topic, and an abstract - so section-aware behaviour is exercised."""
    return [
        _doc(
            "FDA:metformin-1", SourceKind.FDA_LABEL, "FDA Label - Metformin",
            [
                ("boxed_warning",
                 "WARNING: LACTIC ACIDOSIS. Postmarketing cases of metformin-associated "
                 "lactic acidosis have resulted in death and hypothermia."),
                ("dosage_and_administration",
                 "The recommended starting dose is 500 mg orally twice daily with meals. "
                 "Increase in increments of 500 mg weekly. The maximum recommended daily "
                 "dose is 2550 mg."),
                ("contraindications",
                 "Metformin is contraindicated in patients with severe renal impairment "
                 "with eGFR below 30 mL/min/1.73 m2."),
            ],
        ),
        _doc(
            "FDA:warfarin-1", SourceKind.FDA_LABEL, "FDA Label - Warfarin",
            [
                ("drug_interactions",
                 "Foods high in vitamin K, such as leafy green vegetables, may reduce the "
                 "anticoagulant effect of warfarin. Patients should maintain a consistent "
                 "dietary intake of vitamin K."),
                ("dosage_and_administration",
                 "Individualize the dose based on INR. A typical starting dose is 2 to 5 mg "
                 "once daily."),
            ],
        ),
        _doc(
            "MEDLINEPLUS:diabetestype2.html", SourceKind.MEDLINEPLUS, "Diabetes Type 2",
            [("overview",
              "Type 2 diabetes means your blood sugar is too high. Eating a healthy diet, "
              "staying active and losing weight can help you control it. Many people also "
              "need medicine to manage their blood glucose.")],
        ),
        _doc(
            "PMID:28770321", SourceKind.PUBMED_ABSTRACT, "Metformin: clinical use",
            [("conclusions",
              "Metformin remains the optimal first-line therapy for type 2 diabetes "
              "given its efficacy, safety profile and low cost.")],
        ),
    ]
