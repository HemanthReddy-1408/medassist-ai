"""Wave 6: computed confidence, calibration, and selective prediction."""

from __future__ import annotations

import random

import pytest

from medassist.confidence.calibration import (
    calibration,
    risk_coverage,
    select_threshold,
    wilson_interval,
)
from medassist.confidence.compute import (
    ConfidenceWeights,
    combine,
    evidence_coverage,
    evidence_quality,
    model_consistency,
    retrieval_relevance,
    source_agreement,
    temporal_validity,
)
from medassist.core.enums import ClaimDecision, SourceKind
from medassist.core.ids import ChunkId, DocumentId
from medassist.core.models import Chunk

DOC = DocumentId.new()


class FakeJudgement:
    def __init__(self, decision: ClaimDecision) -> None:
        self.decision = decision


def chunk(source: SourceKind, **meta) -> Chunk:
    return Chunk(id=ChunkId.new(), doc_id=DOC, text="t", ordinal=0, source=source, meta=meta)


class TestFactors:
    def test_label_outranks_an_abstract(self):
        label = evidence_quality([chunk(SourceKind.FDA_LABEL)])
        abstract = evidence_quality([chunk(SourceKind.PUBMED_ABSTRACT)])
        assert label > abstract

    def test_study_design_moves_the_score(self):
        strong = evidence_quality([chunk(SourceKind.PUBMED_ABSTRACT, publication_types=["Meta-Analysis"])])
        weak = evidence_quality([chunk(SourceKind.PUBMED_ABSTRACT, publication_types=["Case Reports"])])
        assert strong > weak

    def test_no_evidence_scores_zero(self):
        assert evidence_quality([]) == 0.0

    def test_one_strong_source_beats_many_weak_ones(self):
        strong = evidence_quality([chunk(SourceKind.FDA_LABEL)])
        many_weak = evidence_quality([chunk(SourceKind.UNKNOWN) for _ in range(5)])
        assert strong > many_weak

    def test_qualified_claims_earn_half_credit(self):
        judgements = [FakeJudgement(ClaimDecision.RETAIN)] * 3 + [FakeJudgement(ClaimDecision.QUALIFY)]
        assert evidence_coverage(judgements) == pytest.approx(0.875)

    def test_all_removed_is_zero_coverage(self):
        assert evidence_coverage([FakeJudgement(ClaimDecision.REMOVE)] * 3) == 0.0

    def test_conflicts_reduce_agreement(self):
        assert source_agreement(2, 10) == pytest.approx(0.8)
        assert source_agreement(0, 10) == 1.0

    def test_unmeasured_consistency_does_not_penalise(self):
        """One sample means the signal was not measured, not that it failed."""
        assert model_consistency([["a claim"]]) == 1.0

    def test_identical_samples_score_one_divergent_score_zero(self):
        assert model_consistency([["a", "b"], ["a", "b"]]) == 1.0
        assert model_consistency([["a", "b"], ["c", "d"]]) == 0.0

    def test_recent_evidence_beats_old(self):
        assert temporal_validity([2025]) > temporal_validity([2005])

    def test_unknown_age_is_neither_fresh_nor_stale(self):
        assert 0.4 < temporal_validity([]) < 0.8

    def test_relevance_averages_only_the_selected_chunks(self):
        a, b, unused = ChunkId.new(), ChunkId.new(), ChunkId.new()
        scores = {str(a): 0.8, str(b): 0.6, str(unused): 0.0}
        assert retrieval_relevance(scores, [a, b]) == pytest.approx(0.7)


class TestCombination:
    def test_zero_coverage_drives_confidence_to_zero(self):
        """No verified claims means no confidence, whatever else held."""
        factors = {
            "evidence_quality": 0.95, "evidence_coverage": 0.0,
            "source_agreement": 1.0, "retrieval_relevance": 0.9,
        }
        assert combine(factors).value < 0.001

    def test_stays_within_bounds_whatever_the_weights(self):
        factors = {"evidence_quality": 0.9, "evidence_coverage": 0.9}
        heavy = ConfidenceWeights(evidence_quality=5.0, evidence_coverage=5.0)
        assert 0.0 <= combine(factors, heavy).value <= 1.0

    def test_a_zero_weight_factor_drops_out(self):
        factors = {"evidence_coverage": 1.0, "temporal_validity": 0.1}
        with_it = combine(factors, ConfidenceWeights(temporal_validity=1.0)).value
        without = combine(factors, ConfidenceWeights(temporal_validity=0.0)).value
        assert without > with_it

    def test_weakest_factor_is_reported(self):
        """A single 0.62 is unactionable; the factor driving it is a bug report."""
        report = combine({"evidence_quality": 0.9, "evidence_coverage": 0.4})
        assert report.weakest == "evidence_coverage"
        assert "evidence_coverage" in report.explain()


class TestCalibrationMetrics:
    def test_a_well_ordered_signal_has_low_ece(self):
        random.seed(7)
        conf = [random.random() for _ in range(500)]
        correct = [random.random() < c for c in conf]
        assert calibration(conf, correct).ece < 0.08

    def test_overconfidence_is_detected_and_named(self):
        report = calibration([0.9] * 200, [i % 2 == 0 for i in range(200)])
        assert report.ece == pytest.approx(0.4, abs=0.01)
        populated = [b for b in report.bins if b.count]
        assert populated[0].direction == "overconfident"

    def test_underconfidence_is_detected(self):
        report = calibration([0.2] * 100, [True] * 90 + [False] * 10)
        assert [b for b in report.bins if b.count][0].direction == "underconfident"

    def test_perfect_prediction_scores_zero_brier(self):
        assert calibration([1.0, 0.0], [True, False]).brier == 0.0

    def test_confidence_of_exactly_one_is_binned(self):
        assert sum(b.count for b in calibration([1.0], [True]).bins) == 1

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            calibration([0.5], [True, False])

    def test_out_of_range_confidence_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calibration([1.5], [True])

    def test_empty_input_is_not_a_crash(self):
        assert calibration([], []).n == 0


class TestSelectivePrediction:
    def test_abstaining_on_low_confidence_raises_accuracy(self):
        random.seed(7)
        conf = [random.random() for _ in range(500)]
        correct = [random.random() < c for c in conf]
        report = risk_coverage(conf, correct)
        assert report.at_coverage(0.5).accuracy > report.base_accuracy

    def test_an_uninformative_signal_is_reported_as_not_useful(self):
        """The conclusion this module exists to protect."""
        random.seed(11)
        noise = [random.random() for _ in range(600)]
        correct = [random.random() < 0.6 for _ in range(600)]
        assert risk_coverage(noise, correct).useful is False

    def test_an_informative_signal_is_reported_as_useful(self):
        random.seed(7)
        conf = [random.random() for _ in range(500)]
        assert risk_coverage(conf, [random.random() < c for c in conf]).useful is True

    def test_at_coverage_takes_the_tightest_qualifying_point(self):
        # Not the best-scoring one: max over many noisy thresholds is
        # cherry-picking, and made random signals look informative.
        random.seed(3)
        conf = [random.random() for _ in range(200)]
        point = risk_coverage(conf, [random.random() < c for c in conf]).at_coverage(0.6)
        assert 0.6 <= point.coverage < 0.72

    def test_full_coverage_risk_equals_base_error(self):
        report = risk_coverage([0.5] * 10, [True] * 7 + [False] * 3)
        assert report.points[0].risk == pytest.approx(0.3)

    def test_threshold_selection_prefers_more_coverage_among_qualifying(self):
        """Abstaining more than necessary is a real cost, not free safety."""
        conf = [0.1, 0.2, 0.9, 0.95, 0.99]
        correct = [False, False, True, True, True]
        threshold, point = select_threshold(conf, correct, target_risk=0.0, min_coverage=0.5)
        assert threshold == pytest.approx(0.9)
        assert point.coverage == pytest.approx(0.6)

    def test_unattainable_target_fails_toward_silence(self):
        threshold, point = select_threshold(
            [0.5] * 10, [False] * 10, target_risk=0.05, min_coverage=0.5
        )
        assert threshold == 1.0
        assert point is None

    def test_aurc_is_lower_for_a_better_signal(self):
        random.seed(5)
        good = [random.random() for _ in range(300)]
        good_correct = [random.random() < c for c in good]
        bad = [random.random() for _ in range(300)]
        bad_correct = [random.random() < 0.5 for _ in range(300)]
        assert risk_coverage(good, good_correct).aurc < risk_coverage(bad, bad_correct).aurc


class TestWilsonInterval:
    def test_interval_stays_within_bounds_near_one(self):
        """The normal approximation runs past 1.0 exactly here."""
        low, high = wilson_interval(19, 20)
        assert 0.0 < low < 0.95 and high <= 1.0

    def test_perfect_score_does_not_claim_certainty(self):
        low, high = wilson_interval(20, 20)
        assert low < 1.0 and high == 1.0

    def test_wider_interval_for_smaller_samples(self):
        small = wilson_interval(9, 10)
        large = wilson_interval(90, 100)
        assert (small[1] - small[0]) > (large[1] - large[0])

    def test_no_observations_is_maximally_uncertain(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)
