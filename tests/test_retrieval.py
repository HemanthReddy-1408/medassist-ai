"""Retrieval: chunking fidelity, ranking behaviour, and the trace."""

from __future__ import annotations

import pytest

from medassist.core.enums import RetrievalStage, SourceKind
from medassist.index.chunking import ChunkSpec, chunk_corpus, chunk_document, split_sentences
from medassist.index.dense import DenseIndex
from medassist.index.embed import HashingEmbedding, tokenize
from medassist.index.fusion import reciprocal_rank_fusion
from medassist.index.pipeline import Retriever
from medassist.index.rerank import format_context, select_context
from medassist.index.sparse import BM25Index


class TestSentenceSplitting:
    def test_does_not_split_on_a_decimal_dose(self):
        # Splitting "2.5 mg" destroys exactly the terms that matter most.
        assert len(split_sentences("Take 2.5 mg daily with food.")) == 1

    @pytest.mark.parametrize("text", ["See Dr. Smith today.", "Use e.g. water.", "Drug A vs. B."])
    def test_does_not_split_on_common_abbreviations(self, text):
        assert len(split_sentences(text)) == 1

    def test_splits_real_sentences(self):
        assert len(split_sentences("Take one tablet. Do not exceed two. Call a doctor.")) == 3


class TestChunking:
    def test_offsets_resolve_to_the_chunk_text(self, corpus):
        """Without exact offsets a citation cannot quote its supporting span."""
        for doc in corpus:
            for chunk in chunk_document(doc):
                assert doc.text[chunk.start : chunk.end].strip() == chunk.text

    def test_chunks_carry_their_section(self, corpus):
        sections = {c.section for c in chunk_document(corpus[0])}
        assert "dosage_and_administration" in sections
        assert "boxed_warning" in sections

    def test_a_chunk_never_spans_two_sections(self, corpus):
        # The dosage guard is a set-membership test on this field; a chunk
        # straddling two sections would make it meaningless.
        doc = corpus[0]
        spans = {s["name"]: (s["start"], s["end"]) for s in doc.meta["sections"]}
        for chunk in chunk_document(doc):
            start, end = spans[chunk.section]
            assert chunk.start >= start and chunk.end <= end

    def test_overlap_must_be_smaller_than_target(self):
        with pytest.raises(ValueError, match="overlap"):
            ChunkSpec(target_chars=100, overlap_chars=100)

    def test_no_section_is_silently_dropped(self, corpus):
        """Regression: a section under min_chars vanished from the index entirely.

        `contraindications` is routinely under 120 characters, and dropping it
        surfaces as NOT_INDEXED on exactly the safety-critical queries.
        """
        for doc in corpus:
            declared = {s["name"] for s in doc.meta["sections"]}
            indexed = {c.section for c in chunk_document(doc)}
            assert declared == indexed, f"{doc.source_uid}: lost {declared - indexed}"

    def test_short_trailing_fragments_are_still_dropped(self):
        """The floor must still do its original job on continuation fragments."""
        from medassist.core.enums import SourceKind
        from tests.conftest import _doc

        long_section = " ".join(f"Sentence number {i} about dosing." for i in range(60))
        doc = _doc("FDA:x", SourceKind.FDA_LABEL, "X", [("dosage_and_administration", long_section)])
        chunks = chunk_document(doc, ChunkSpec(target_chars=300, overlap_chars=60, min_chars=120))
        assert all(len(c.text) >= 120 for c in chunks[1:])

    def test_ordinals_are_contiguous_per_document(self, corpus):
        for doc in corpus:
            chunks = chunk_document(doc)
            assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestTokenizer:
    def test_keeps_decimals_and_hyphens(self):
        tokens = tokenize("Take 2.5 mg for type-2 diabetes and check HbA1c")
        assert "2.5" in tokens and "type-2" in tokens and "hba1c" in tokens

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the patient and the dose")


class TestBM25:
    def test_finds_the_exact_drug_name(self, corpus):
        chunks = chunk_corpus(corpus)
        by_id = {c.id: c for c in chunks}
        results = BM25Index().build(chunks).search("warfarin vitamin K leafy greens", k=3)
        assert results
        assert "warfarin" in by_id[results[0][0]].meta["doc_uid"].lower()

    def test_unknown_terms_return_nothing_rather_than_noise(self, corpus):
        index = BM25Index().build(chunk_corpus(corpus))
        assert index.search("xyzzy quuxbaz", k=5) == []

    def test_idf_never_negative(self, corpus):
        # The raw Robertson idf goes negative for terms in >half the corpus,
        # letting a common word subtract from a score.
        index = BM25Index().build(chunk_corpus(corpus))
        assert all(v > 0 for v in index._idf.values())


class TestDense:
    def test_ranks_a_paraphrase_above_an_unrelated_chunk(self, corpus):
        chunks = chunk_corpus(corpus)
        by_id = {c.id: c for c in chunks}
        results = DenseIndex(HashingEmbedding()).build(chunks).search("lactic acidosis death", k=3)
        assert by_id[results[0][0]].section == "boxed_warning"

    def test_empty_index_returns_empty(self):
        assert DenseIndex(HashingEmbedding()).build([]).search("anything") == []


class TestFusion:
    def test_a_chunk_ranked_by_both_beats_one_ranked_by_either(self):
        from medassist.core.ids import ChunkId

        both, only_a, only_b = ChunkId.new(), ChunkId.new(), ChunkId.new()
        fused = reciprocal_rank_fusion(
            [[(only_a, 9.0), (both, 1.0)], [(only_b, 9.0), (both, 1.0)]]
        )
        assert fused[0][0] == both

    def test_ordering_is_total_and_reproducible(self):
        from medassist.core.ids import ChunkId

        ids = [ChunkId.new() for _ in range(4)]
        ranking = [[(i, 1.0) for i in ids]]
        assert reciprocal_rank_fusion(ranking) == reciprocal_rank_fusion(ranking)

    def test_mismatched_weights_rejected(self):
        with pytest.raises(ValueError, match="weights"):
            reciprocal_rank_fusion([[], []], weights=[1.0])


class TestSelection:
    def test_per_doc_cap_prevents_one_label_starving_the_window(self, corpus):
        chunks = chunk_corpus(corpus)
        by_id = {c.id: c for c in chunks}
        fda = [(c.id, 1.0) for c in chunks if c.meta["doc_uid"] == "FDA:metformin-1"]
        selected, dropped = select_context(fda, by_id, max_per_doc=2, max_chunks=8)
        assert len(selected) == 2
        assert dropped  # the rest are recorded, not silently lost

    def test_dropped_chunks_are_recorded_for_attribution(self, corpus):
        chunks = chunk_corpus(corpus)
        by_id = {c.id: c for c in chunks}
        ranked = [(c.id, 1.0) for c in chunks]
        selected, dropped = select_context(ranked, by_id, max_chunks=1, max_per_doc=1)
        assert len(selected) == 1
        assert len(dropped) == len(chunks) - 1


class TestContextFormatting:
    def test_uses_short_handles_not_ulids(self, corpus):
        chunks = chunk_corpus(corpus)[:3]
        by_id = {c.id: c for c in chunks}
        rendered = format_context([c.id for c in chunks], by_id)
        assert "[C1]" in rendered and "[C3]" in rendered
        # A model asked to echo a ULID will sometimes invent a plausible one.
        assert not any(str(c.id) in rendered for c in chunks)


class TestPipeline:
    def test_trace_records_every_stage_in_order(self, corpus):
        retriever = Retriever(embedding=HashingEmbedding()).build(corpus)
        result = retriever.retrieve("maximum daily dose of metformin", intent="drug_info")
        stages = [s.stage for s in result.trace.stages]
        assert stages == [
            RetrievalStage.DENSE, RetrievalStage.SPARSE, RetrievalStage.FUSED,
            RetrievalStage.RERANKED, RetrievalStage.SELECTED,
        ]

    def test_retrieves_the_dosage_section_for_a_dosing_question(self, corpus):
        retriever = Retriever(embedding=HashingEmbedding()).build(corpus)
        result = retriever.retrieve("what is the maximum daily dose of metformin", intent="drug_info")
        sections = {retriever.get(cid).section for cid in result.chunk_ids}
        assert "dosage_and_administration" in sections

    def test_authority_breaks_a_tie_without_overriding_relevance(self, corpus):
        """The prior is a tiebreak (weight 0.06), not an override.

        A far more relevant abstract must still outrank a barely-relevant
        label - otherwise the prior would degrade retrieval rather than order
        it. So this asserts the contract directly: equal base relevance,
        higher authority wins.
        """
        from medassist.index.chunking import chunk_corpus
        from medassist.index.rerank import rerank

        chunks = {c.id: c for c in chunk_corpus(corpus)}
        abstract = next(c for c in chunks.values() if c.source is SourceKind.PUBMED_ABSTRACT)
        label = next(c for c in chunks.values() if c.source is SourceKind.FDA_LABEL)

        tied = [(abstract.id, 0.5), (label.id, 0.5)]
        ranked = rerank("metformin first line therapy", tied, chunks)
        assert ranked[0][0] == label.id

    def test_relevance_still_beats_authority(self, corpus):
        """The other half of the same contract, stated as its own case."""
        retriever = Retriever(embedding=HashingEmbedding()).build(corpus)
        result = retriever.retrieve("metformin remains the optimal first-line therapy")
        top = retriever.get(result.chunk_ids[0])
        assert top.source is SourceKind.PUBMED_ABSTRACT

    def test_handles_map_back_to_real_chunk_ids(self, corpus):
        retriever = Retriever(embedding=HashingEmbedding()).build(corpus)
        result = retriever.retrieve("vitamin K", intent="drug_interaction")
        handles = retriever.handles(result.chunk_ids)
        assert handles["C1"] == result.chunk_ids[0]

    def test_attribution_works_against_the_live_index(self, corpus):
        from medassist.core.enums import RetrievalFailure

        retriever = Retriever(embedding=HashingEmbedding()).build(corpus)
        result = retriever.retrieve("maximum daily dose of metformin", intent="drug_info")
        gold = {
            c.id for c in retriever.chunks.values() if c.section == "dosage_and_administration"
        }
        assert result.trace.attribute(gold, retriever.indexed_ids) is RetrievalFailure.NONE

        from medassist.core.ids import ChunkId

        missing = {ChunkId.new()}
        verdict = result.trace.attribute(missing, retriever.indexed_ids)
        assert verdict is RetrievalFailure.NOT_INDEXED
