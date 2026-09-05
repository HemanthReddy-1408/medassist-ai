"""Corpus: parsers against recorded fixtures, and snapshot integrity."""

from __future__ import annotations

import pytest

from medassist.core.enums import SourceKind
from medassist.core.errors import CorpusError
from medassist.corpus.normalize import assemble_sections, clean, section_at
from medassist.corpus.snapshot import SnapshotStore, content_hash
from medassist.corpus.sources import medlineplus, openfda, pubmed

PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID Version="1">28770321</PMID>
<Article><Journal><Title>Diabetologia</Title>
<JournalIssue><PubDate><Year>2017</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Metformin: clinical use in <i>type 2</i> diabetes.</ArticleTitle>
<Abstract>
<AbstractText Label="BACKGROUND">Metformin is widely used.</AbstractText>
<AbstractText Label="CONCLUSIONS">It remains first-line therapy.</AbstractText>
</Abstract>
<PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">999</PMID>
<Article><ArticleTitle>No abstract here.</ArticleTitle></Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""

MEDLINEPLUS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult><list num="1" start="0">
<document rank="0" url="https://medlineplus.gov/diabetestype2.html">
<content name="title">Diabetes &lt;span&gt;Type 2&lt;/span&gt;</content>
<content name="altTitle">Adult-onset diabetes</content>
<content name="FullSummary">&lt;p&gt;Your blood sugar is too high.&lt;/p&gt;</content>
</document></list></nlmSearchResult>"""

OPENFDA_JSON = """{"results":[{"set_id":"abc-123","effective_time":"20240115",
"boxed_warning":["WARNING: LACTIC ACIDOSIS."],
"dosage_and_administration":["Start at 500 mg twice daily."],
"how_supplied":["Bottles of 100."],
"openfda":{"generic_name":["METFORMIN HYDROCHLORIDE"],"brand_name":["GLUCOPHAGE"],
"manufacturer_name":["Acme"],"route":["ORAL"]}}]}"""


class TestNormalize:
    def test_unescapes_entities_and_strips_markup(self):
        assert clean("<p>a &amp; b</p>") == "a & b"

    def test_nfkc_folds_ligatures(self):
        # SPL and PDF-derived text is full of these; unfolded they break
        # lexical matching on the drug names that matter.
        assert clean("ﬁbrillation") == "fibrillation"

    def test_collapses_runs_of_blank_lines(self):
        assert clean("a\n\n\n\n\nb") == "a\n\nb"


class TestSectionAssembly:
    def test_offsets_index_the_returned_text_exactly(self):
        text, spans = assemble_sections([("dosage", "Take one."), ("warnings", "Do not drive.")])
        for span in spans:
            assert text[span["start"] : span["end"]] == {
                "dosage": "Take one.", "warnings": "Do not drive."
            }[span["name"]]

    def test_empty_sections_are_skipped(self):
        _, spans = assemble_sections([("a", "content"), ("b", "   ")])
        assert [s["name"] for s in spans] == ["a"]

    def test_section_at_resolves_an_offset(self):
        text, spans = assemble_sections([("dosage", "Take one tablet daily.")])
        assert section_at(spans, text.index("tablet")) == "dosage"
        assert section_at(spans, 0) == ""  # the heading itself belongs to no section


class TestPubMedParser:
    def test_extracts_structured_abstract_sections(self):
        docs = pubmed.parse(PUBMED_XML)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_uid == "PMID:28770321"
        assert {s["name"] for s in doc.meta["sections"]} == {"background", "conclusions"}

    def test_title_keeps_text_inside_inline_markup(self):
        # element.text alone truncates at the first inline tag, losing half of
        # PubMed's titles.
        assert pubmed.parse(PUBMED_XML)[0].title == "Metformin: clinical use in type 2 diabetes."

    def test_citation_without_an_abstract_is_skipped(self):
        # Not retrievable content; indexing it adds a chunk that can never
        # support a claim.
        assert all(d.source_uid != "PMID:999" for d in pubmed.parse(PUBMED_XML))

    def test_metadata_captured(self):
        doc = pubmed.parse(PUBMED_XML)[0]
        assert doc.meta["journal"] == "Diabetologia"
        assert doc.published == "2017"
        assert "Review" in doc.meta["publication_types"]


class TestOpenFDAParser:
    def test_keeps_clinical_sections_and_drops_packaging(self):
        doc = openfda.parse(OPENFDA_JSON)[0]
        names = {s["name"] for s in doc.meta["sections"]}
        assert "dosage_and_administration" in names and "boxed_warning" in names
        assert "how_supplied" not in names  # noise that dilutes retrieval

    def test_authority_outranks_literature(self):
        assert openfda.parse(OPENFDA_JSON)[0].authority == 3

    def test_boxed_warning_flagged_in_metadata(self):
        assert openfda.parse(OPENFDA_JSON)[0].meta["has_boxed_warning"] is True

    def test_effective_time_becomes_the_recency_signal(self):
        assert openfda.parse(OPENFDA_JSON)[0].published == "2024"


class TestMedlinePlusParser:
    def test_strips_markup_from_title_and_summary(self):
        doc = medlineplus.parse(MEDLINEPLUS_XML)[0]
        assert doc.title == "Diabetes Type 2"
        assert "<p>" not in doc.text and "Your blood sugar is too high." in doc.text

    def test_alt_titles_retained(self):
        assert "Adult-onset diabetes" in medlineplus.parse(MEDLINEPLUS_XML)[0].meta["also_called"]


class TestSnapshots:
    def test_hash_ignores_document_ids_and_ordering(self, corpus):
        """Ids are fresh ULIDs per scrape; hashing them would make every
        rebuild a new snapshot even when nothing changed."""
        assert content_hash(corpus) == content_hash(list(reversed(corpus)))

    def test_hash_changes_when_content_changes(self, corpus):
        from medassist.core.models import Document

        edited = [*corpus[:-1], Document(**{**corpus[-1].model_dump(), "text": "different"})]
        assert content_hash(edited) != content_hash(corpus)

    def test_roundtrip(self, corpus, tmp_path):
        store = SnapshotStore(tmp_path)
        manifest = store.write(corpus, spec_name="unit")
        loaded, loaded_manifest = store.read(manifest.snapshot_id)
        assert len(loaded) == len(corpus)
        assert loaded_manifest.by_source[SourceKind.FDA_LABEL.value] == 2

    def test_corruption_is_detected_on_read(self, corpus, tmp_path):
        """A silently-edited corpus invalidates every number measured against
        it, and the failure would otherwise be invisible."""
        store = SnapshotStore(tmp_path)
        manifest = store.write(corpus)
        path = store.path(manifest.snapshot_id) / "documents.jsonl"
        path.write_text(path.read_text().replace("500 mg", "5000 mg"))
        with pytest.raises(CorpusError, match="corrupt"):
            store.read(manifest.snapshot_id)

    def test_empty_snapshot_refused(self, tmp_path):
        with pytest.raises(CorpusError, match="empty"):
            SnapshotStore(tmp_path).write([])

    def test_latest_pointer_tracks_the_last_write(self, corpus, tmp_path):
        store = SnapshotStore(tmp_path)
        manifest = store.write(corpus)
        assert store.latest() == manifest.snapshot_id
