"""Invariants of the pure core. No network, no provider, no database."""

from __future__ import annotations

import pytest

from medassist.core.enums import ClaimVerdict, JudgeReliability, RetrievalFailure, RetrievalStage
from medassist.core.ids import ChunkId, DocumentId, PrefixedId
from medassist.core.models import (
    Budget,
    Chunk,
    ClaimAssessment,
    LabAnalyte,
    RetrievalTrace,
    StageRecord,
    Usage,
)


class TestIds:
    def test_wrong_prefix_is_a_construction_error(self):
        # A type error, not a lookup that silently misses three layers later.
        with pytest.raises(ValueError, match="must start with"):
            ChunkId("doc_01ABC")

    def test_ids_sort_by_creation_order(self):
        """Including within a single millisecond - see the monotonicity note.

        Chunking one document mints ~100 ids faster than the clock ticks, so
        plain (non-monotonic) ULIDs would leave them unordered.
        """
        minted = [ChunkId.new() for _ in range(500)]
        assert minted == sorted(minted)

    def test_ids_stay_ordered_across_a_clock_tick(self):
        import time

        first = ChunkId.new()
        time.sleep(0.002)
        assert first < ChunkId.new()

    def test_ids_are_unique_under_concurrency(self):
        import threading

        minted: list[ChunkId] = []
        lock = threading.Lock()

        def mint() -> None:
            batch = [ChunkId.new() for _ in range(200)]
            with lock:
                minted.extend(batch)

        threads = [threading.Thread(target=mint) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(minted)) == len(minted) == 1600

    def test_prefixes_are_distinct_across_types(self):
        assert ChunkId.new().startswith("chk_")
        assert DocumentId.new().startswith("doc_")

    def test_non_string_rejected(self):
        with pytest.raises(TypeError):
            ChunkId(123)  # type: ignore[arg-type]

    def test_test_flag_prevents_pytest_collection(self):
        # TestCaseId would otherwise be collected as a test class.
        assert PrefixedId.__test__ is False


class TestRetrievalAttribution:
    """All five branches. This is the 'was retrieval wrong?' mechanism."""

    @staticmethod
    def _trace(stages, selected):
        return RetrievalTrace(
            query="q",
            stages=[StageRecord(stage=s, chunk_ids=ids) for s, ids in stages],
            selected=selected,
        )

    def setup_method(self):
        self.gold = ChunkId.new()
        self.other = ChunkId.new()

    def test_success_is_none(self):
        t = self._trace([(RetrievalStage.DENSE, [self.gold])], [self.gold])
        assert t.attribute({self.gold}) is RetrievalFailure.NONE

    def test_not_indexed(self):
        t = self._trace(
            [
                (RetrievalStage.INDEXED, [self.other]),
                (RetrievalStage.DENSE, [self.other]),
                (RetrievalStage.SPARSE, [self.other]),
            ],
            [self.other],
        )
        assert t.attribute({self.gold}) is RetrievalFailure.NOT_INDEXED

    def test_not_retrieved(self):
        t = self._trace(
            [
                (RetrievalStage.INDEXED, [self.gold, self.other]),
                (RetrievalStage.DENSE, [self.other]),
                (RetrievalStage.SPARSE, [self.other]),
            ],
            [self.other],
        )
        assert t.attribute({self.gold}) is RetrievalFailure.NOT_RETRIEVED

    def test_lost_in_rerank(self):
        t = self._trace(
            [
                (RetrievalStage.INDEXED, [self.gold, self.other]),
                (RetrievalStage.DENSE, [self.gold, self.other]),
                (RetrievalStage.SPARSE, [self.other]),
                (RetrievalStage.RERANKED, [self.other]),
            ],
            [self.other],
        )
        assert t.attribute({self.gold}) is RetrievalFailure.LOST_IN_RERANK

    def test_lost_in_selection(self):
        t = self._trace(
            [
                (RetrievalStage.INDEXED, [self.gold, self.other]),
                (RetrievalStage.DENSE, [self.gold, self.other]),
                (RetrievalStage.SPARSE, [self.other]),
                (RetrievalStage.RERANKED, [self.gold, self.other]),
            ],
            [self.other],
        )
        assert t.attribute({self.gold}) is RetrievalFailure.LOST_IN_SELECTION

    def test_indexed_set_may_be_passed_instead_of_stored(self):
        # Storing every corpus id per query would make traces larger than the corpus.
        t = self._trace([(RetrievalStage.DENSE, [self.other])], [self.other])
        assert t.attribute({self.gold}, indexed={self.other}) is RetrievalFailure.NOT_INDEXED

    def test_no_gold_labels_is_not_a_failure(self):
        t = self._trace([(RetrievalStage.DENSE, [self.other])], [self.other])
        assert t.attribute(set()) is RetrievalFailure.NONE


class TestClaimAssessmentInvariants:
    """ADR-0004: a judge score carries its kappa, or is branded uncalibrated."""

    def test_calibrated_without_kappa_is_rejected(self):
        with pytest.raises(ValueError, match="kappa"):
            ClaimAssessment(
                claim_id=__import__("medassist.core.ids", fromlist=["ClaimId"]).ClaimId.new(),
                verdict=ClaimVerdict.SUPPORTED,
                evidence=["the cited span states this"],
                reliability=JudgeReliability.CALIBRATED,
            )

    def test_calibrated_with_kappa_is_accepted(self):
        from medassist.core.ids import ClaimId

        assessment = ClaimAssessment(
            claim_id=ClaimId.new(),
            verdict=ClaimVerdict.SUPPORTED,
            evidence=["the cited span states this"],
            reliability=JudgeReliability.CALIBRATED,
            kappa=0.71,
        )
        assert assessment.kappa == 0.71

    def test_evidence_may_not_be_empty(self):
        from medassist.core.ids import ClaimId

        # A bare float with no explanation is not reviewable.
        with pytest.raises(ValueError):
            ClaimAssessment(
                claim_id=ClaimId.new(), verdict=ClaimVerdict.SUPPORTED, evidence=[]
            )

    def test_uncalibrated_needs_no_kappa(self):
        from medassist.core.ids import ClaimId

        assessment = ClaimAssessment(
            claim_id=ClaimId.new(), verdict=ClaimVerdict.UNSUPPORTED, evidence=["absent"]
        )
        assert assessment.reliability is JudgeReliability.UNCALIBRATED


class TestBudget:
    def test_subdivide_never_resets(self):
        parent = Budget(max_steps=24, max_cost_usd=0.50)
        child = parent.subdivide(0.25)
        assert child.max_steps == 6
        assert child.max_cost_usd == pytest.approx(0.125)

    def test_children_cannot_exceed_the_parent(self):
        parent = Budget(max_steps=10, max_cost_usd=1.0)
        shares = [0.3, 0.3, 0.4]
        assert sum(parent.subdivide(s).max_cost_usd for s in shares) <= parent.max_cost_usd

    @pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
    def test_invalid_shares_rejected(self, share):
        with pytest.raises(ValueError):
            Budget().subdivide(share)

    def test_subdivision_floors_at_one_step(self):
        # A share small enough to round to zero would make a node unable to act.
        assert Budget(max_steps=4).subdivide(0.01).max_steps == 1


class TestUsage:
    def test_absorb_sums_cost_and_takes_max_wall_time(self):
        # Parallel nodes: cost adds, wall time does not.
        total = Usage(cost_usd=0.01, wall_time_s=2.0, prompt_tokens=100)
        total.absorb(Usage(cost_usd=0.02, wall_time_s=5.0, prompt_tokens=50))
        assert total.cost_usd == pytest.approx(0.03)
        assert total.wall_time_s == 5.0
        assert total.total_tokens == 150


class TestChunk:
    def test_reversed_offsets_rejected(self):
        with pytest.raises(ValueError, match="precedes"):
            Chunk(id=ChunkId.new(), doc_id=DocumentId.new(), text="x", ordinal=0, start=10, end=5)


class TestLabAnalyte:
    """Flags are computed from the interval, never asked of a model."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(8.2, "high"), (3.1, "low"), (5.0, "normal")],
    )
    def test_flag_from_reference_interval(self, value, expected):
        analyte = LabAnalyte(name="HbA1c", value=value, unit="%", ref_low=4.0, ref_high=5.6)
        assert analyte.flag == expected
        assert analyte.abnormal is (expected != "normal")

    def test_missing_interval_is_unknown_not_normal(self):
        # Silently calling an unreferenced value "normal" is the dangerous default.
        assert LabAnalyte(name="X", value=1.0).flag == "unknown"
