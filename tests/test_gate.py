"""The release gate. Behavioural cases: an input, and the decision required.

Deliberately not IR relevance labels - the gate's job is to decide whether an
answer may be released, with no gold answer available.
"""

from __future__ import annotations

import pytest

from medassist.core.enums import (
    CheckName,
    ClaimDecision,
    ResponseDecision,
    Severity,
    SourceKind,
)
from medassist.core.ids import ChunkId, DocumentId
from medassist.core.models import Chunk, Citation, Claim, PatientProfile, Usage
from medassist.gate import GateContext, GateSpec, ReleaseGate
from medassist.gate.checks import CitationCheck, NumericCheck, RelationalCheck
from medassist.gate.entailment import EntailmentCheck, EntailmentUnavailable
from medassist.gate.quantities import extract_quantities, find_unmatched

DOC = DocumentId.new()


def chunk(text: str, section: str = "", source: SourceKind = SourceKind.FDA_LABEL) -> Chunk:
    return Chunk(
        id=ChunkId.new(), doc_id=DOC, text=text, ordinal=0, section=section, source=source
    )


def claim(text: str, *chunks: Chunk, dosage: bool = False) -> Claim:
    return Claim(
        text=text,
        citations=[Citation(chunk_id=c.id, doc_id=DOC) for c in chunks],
        is_dosage=dosage,
    )


@pytest.fixture
def dose_chunk() -> Chunk:
    return chunk("The maximum recommended daily dose is 2550 mg.", "dosage_and_administration")


@pytest.fixture
def greens_chunk() -> Chunk:
    return chunk(
        "Leafy green vegetables are an excellent source of vitamins and fibre.",
        "overview", SourceKind.MEDLINEPLUS,
    )


def context(*chunks: Chunk, profile: PatientProfile | None = None, **kw) -> GateContext:
    return GateContext(
        chunks={c.id: c for c in chunks}, context_ids=[c.id for c in chunks],
        profile=profile, **kw,
    )


class StubJudge:
    """A judge whose verdicts are scripted, so gate logic is tested in isolation.

    Batch-aware: it consumes its script in call order, so a verdict scripted
    for the tenth claim still lands on the tenth claim once batching splits the
    list across calls.
    """

    def __init__(self, verdicts: list[str] | Exception):
        self.verdicts = verdicts
        self.calls = 0
        self._cursor = 0

    def structured(self, messages, **kw):
        self.calls += 1
        if isinstance(self.verdicts, Exception):
            raise self.verdicts
        asked = messages[-1]["content"].count("CLAIM ")
        window = self.verdicts[self._cursor : self._cursor + asked]
        self._cursor += asked
        return (
            {"judgements": [
                {"id": i, "verdict": v, "evidence": "stub"}
                for i, v in enumerate(window, start=1)
            ]},
            Usage(steps=1, prompt_tokens=100, completion_tokens=20, cost_usd=0.0001),
        )


def gate_with(verdicts, **spec_kw) -> ReleaseGate:
    return ReleaseGate(
        entailment=EntailmentCheck(StubJudge(verdicts)), spec=GateSpec(**spec_kw)
    )


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------


class TestQuantities:
    def test_unit_conversion_matches(self):
        assert not find_unmatched("The dose is 2.5 g.", ["a maximum of 2500 mg daily"])

    def test_a_near_miss_dose_does_not_match(self):
        """The case an LLM judge routinely accepts and arithmetic does not."""
        unmatched = find_unmatched("The maximum is 2000 mg.", ["the maximum is 2550 mg"])
        assert [q.raw for q in unmatched] == ["2000 mg"]

    def test_bare_numbers_are_not_quantities(self):
        # "type 2 diabetes" must not demand grounding; a false positive on a
        # safety check trains people to ignore it.
        assert extract_quantities("Metformin treats type 2 diabetes.") == []

    def test_range_lower_bound_inherits_the_unit(self):
        values = {q.normalized for q in extract_quantities("Give 500 to 1000 mg daily.")}
        assert values == {500.0, 1000.0}

    def test_comma_thousands_parsed(self):
        assert extract_quantities("up to 2,550 mg")[0].normalized == 2550.0

    def test_dimensions_do_not_cross(self):
        assert find_unmatched("Take 5 mg.", ["Take 5 ml."])

    def test_compound_units_beat_their_prefixes(self):
        # "mg/dl" must not be read as "mg".
        assert extract_quantities("glucose 126 mg/dl")[0].dimension == "conc_mgdl"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestCitationCheck:
    def test_uncited_claim_removed_when_evidence_required(self):
        outcome = CitationCheck().run(claim("Metformin is safe."), context())
        assert outcome.decision is ClaimDecision.REMOVE
        assert outcome.reason == "uncited"

    def test_uncited_claim_allowed_when_evidence_not_required(self):
        outcome = CitationCheck().run(
            claim("Metformin is safe."), context(evidence_required=False)
        )
        assert outcome.decision is ClaimDecision.RETAIN

    def test_citation_to_a_chunk_outside_the_window_is_removed(self, dose_chunk):
        ghost = chunk("not in the window")
        outcome = CitationCheck().run(claim("x", ghost), context(dose_chunk))
        assert outcome.decision is ClaimDecision.REMOVE
        assert outcome.reason == "citation_unresolvable"

    def test_partial_resolution_is_retained_but_flagged(self, dose_chunk):
        ghost = chunk("not in the window")
        outcome = CitationCheck().run(claim("x", dose_chunk, ghost), context(dose_chunk))
        assert outcome.decision is ClaimDecision.RETAIN
        assert outcome.reason == "citations_partially_resolved"


class TestNumericCheck:
    def test_wrong_dose_removed(self, dose_chunk):
        outcome = NumericCheck().run(
            claim("The maximum daily dose is 2000 mg.", dose_chunk, dosage=True),
            context(dose_chunk),
        )
        assert outcome.decision is ClaimDecision.REMOVE
        assert outcome.severity is Severity.CRITICAL

    def test_correct_dose_retained(self, dose_chunk):
        outcome = NumericCheck().run(
            claim("The maximum daily dose is 2550 mg.", dose_chunk, dosage=True),
            context(dose_chunk),
        )
        assert outcome.decision is ClaimDecision.RETAIN

    def test_equivalent_units_retained(self, dose_chunk):
        outcome = NumericCheck().run(
            claim("The maximum daily dose is 2.55 g.", dose_chunk, dosage=True),
            context(dose_chunk),
        )
        assert outcome.decision is ClaimDecision.RETAIN

    def test_non_dosage_numeric_error_is_high_not_critical(self, dose_chunk):
        outcome = NumericCheck().run(
            claim("Roughly 900 mg was studied.", dose_chunk), context(dose_chunk)
        )
        assert outcome.severity is Severity.HIGH


class TestRelationalCheck:
    """Correctness that is relational, not propositional."""

    def test_true_well_cited_claim_qualified_for_this_patient(self, greens_chunk):
        outcome = RelationalCheck().run(
            claim("Leafy green vegetables are an excellent source of vitamins.", greens_chunk),
            context(greens_chunk, profile=PatientProfile(medications=["warfarin 5mg"])),
        )
        assert outcome.decision is ClaimDecision.QUALIFY
        assert "warfarin" in outcome.detail.lower()

    def test_same_claim_untouched_for_a_different_patient(self, greens_chunk):
        outcome = RelationalCheck().run(
            claim("Leafy green vegetables are an excellent source of vitamins.", greens_chunk),
            context(greens_chunk, profile=PatientProfile(medications=["metformin"])),
        )
        assert outcome.decision is ClaimDecision.RETAIN

    def test_grapefruit_with_a_statin_is_removed(self):
        c = chunk("Grapefruit juice is a healthy source of vitamin C.")
        outcome = RelationalCheck().run(
            claim("Grapefruit juice is a healthy breakfast choice.", c),
            context(c, profile=PatientProfile(medications=["simvastatin 20 mg"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE

    def test_conditions_are_searched_as_well_as_medications(self):
        c = chunk("A high-protein diet supports muscle maintenance.")
        outcome = RelationalCheck().run(
            claim("A high-protein diet is recommended.", c),
            context(c, profile=PatientProfile(conditions=["chronic kidney disease stage 4"])),
        )
        assert outcome.decision is ClaimDecision.QUALIFY

    def test_no_profile_means_no_finding(self, greens_chunk):
        outcome = RelationalCheck().run(
            claim("Leafy greens are healthy.", greens_chunk), context(greens_chunk)
        )
        assert outcome.reason == "no_patient_record"

    def test_most_severe_row_wins(self):
        # A patient on both warfarin (QUALIFY) and an anticoagulant NSAID rule
        # (REMOVE) must not have the REMOVE masked by the QUALIFY.
        c = chunk("Ibuprofen and leafy greens are both common.")
        outcome = RelationalCheck().run(
            claim("Ibuprofen is fine, and leafy greens are healthy.", c),
            context(c, profile=PatientProfile(medications=["warfarin"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------


class TestCascade:
    def test_clean_answer_is_released(self, dose_chunk):
        outcome = gate_with(["supported"]).evaluate(
            [claim("The maximum daily dose is 2550 mg.", dose_chunk, dosage=True)],
            context(dose_chunk),
        )
        assert outcome.decision is ResponseDecision.RELEASE
        assert outcome.entailment_ran

    def test_deterministic_removal_skips_the_model_call(self, dose_chunk):
        """A claim removed by check 2 must never reach the expensive stage."""
        judge = StubJudge(["supported"])
        gate = ReleaseGate(entailment=EntailmentCheck(judge), spec=GateSpec(min_supported_fraction=0.0))
        gate.evaluate(
            [claim("The maximum daily dose is 2000 mg.", dose_chunk, dosage=True)],
            context(dose_chunk),
        )
        assert judge.calls == 0  # nothing survived to be judged

    def test_contradicted_claim_blocks_the_whole_response(self, dose_chunk):
        outcome = gate_with(["contradicted"], min_supported_fraction=0.0).evaluate(
            [claim("The maximum daily dose is 2550 mg.", dose_chunk)], context(dose_chunk)
        )
        assert outcome.decision is ResponseDecision.BLOCK
        assert outcome.reason.rule == "claim_contradicted"

    def test_one_bad_claim_is_not_averaged_away(self, dose_chunk):
        """Severity dominates: 9 good claims do not carry 1 contradicted one."""
        claims = [claim(f"Fact {i}.", dose_chunk) for i in range(9)]
        claims.append(claim("A contradicted statement.", dose_chunk))
        outcome = gate_with(["supported"] * 9 + ["contradicted"]).evaluate(
            claims, context(dose_chunk)
        )
        assert outcome.decision is ResponseDecision.BLOCK

    def test_unsupported_claim_yields_a_caveat_not_a_block(self, dose_chunk):
        outcome = gate_with(["supported", "unsupported"]).evaluate(
            [claim("Fact one.", dose_chunk), claim("Fact two.", dose_chunk)],
            context(dose_chunk),
        )
        assert outcome.decision is ResponseDecision.RELEASE_WITH_CAVEAT
        assert outcome.qualified

    def test_hedges_are_not_penalised(self, dose_chunk):
        """UNVERIFIABLE exists so the metric is not gamed by hedging - and so
        that recommending a doctor is not scored as a hallucination."""
        outcome = gate_with(["supported", "unverifiable"]).evaluate(
            [claim("Fact one.", dose_chunk), claim("Talk to your doctor.", dose_chunk)],
            context(dose_chunk),
        )
        assert outcome.decision is ResponseDecision.RELEASE

    def test_relational_qualify_reaches_the_response(self, greens_chunk):
        outcome = gate_with(["supported"]).evaluate(
            [claim("Leafy green vegetables are an excellent source of vitamins.", greens_chunk)],
            context(greens_chunk, profile=PatientProfile(medications=["warfarin"])),
        )
        assert outcome.decision is ResponseDecision.RELEASE_WITH_CAVEAT
        assert any("warfarin" in c.lower() for c in outcome.caveats)

    def test_critical_relational_conflict_blocks(self):
        c = chunk("St John's wort is a popular herbal supplement.")
        outcome = gate_with(["supported"], min_supported_fraction=0.0).evaluate(
            [claim("St John's wort may help with low mood.", c)],
            context(c, profile=PatientProfile(medications=["sertraline 50mg"])),
        )
        assert outcome.decision is ResponseDecision.BLOCK
        assert outcome.reason.rule == "relational_critical"

    def test_too_few_survivors_abstains(self, dose_chunk):
        outcome = gate_with(["supported"], min_supported_fraction=0.6).evaluate(
            [
                claim("Dose is 2000 mg.", dose_chunk, dosage=True),   # removed
                claim("Dose is 3000 mg.", dose_chunk, dosage=True),   # removed
                claim("Dose is 2550 mg.", dose_chunk, dosage=True),   # kept
            ],
            context(dose_chunk),
        )
        assert outcome.decision is ResponseDecision.ABSTAIN
        assert outcome.reason.rule == "insufficient_support"

    def test_escalation_when_the_capability_demands_review(self, dose_chunk):
        outcome = gate_with(["supported"], requires_human_review=True).evaluate(
            [claim("Fact.", dose_chunk)], context(dose_chunk)
        )
        assert outcome.decision is ResponseDecision.ESCALATE

    def test_red_flag_pre_empts_everything(self, dose_chunk):
        judge = StubJudge(["supported"])
        gate = ReleaseGate(entailment=EntailmentCheck(judge))
        outcome = gate.evaluate(
            [claim("Fact.", dose_chunk)], context(dose_chunk),
            red_flag="crushing chest pain radiating to the arm",
        )
        assert outcome.decision is ResponseDecision.BLOCK
        assert judge.calls == 0  # no retrieval, no verification, no softening

    def test_per_check_attribution_is_recorded(self, dose_chunk):
        outcome = gate_with(["supported"], min_supported_fraction=0.0).evaluate(
            [claim("The dose is 2000 mg.", dose_chunk, dosage=True)], context(dose_chunk)
        )
        judgement = outcome.judgements[0]
        assert judgement.firing_check is CheckName.NUMERIC_GROUNDING
        assert "2000 mg" in judgement.explanation


class TestFailClosed:
    """Every degradation resolves toward silence, never toward release."""

    def test_judge_unavailable_abstains(self, dose_chunk):
        gate = ReleaseGate(entailment=EntailmentCheck(StubJudge(RuntimeError("503"))))
        outcome = gate.evaluate([claim("Fact.", dose_chunk)], context(dose_chunk))
        assert outcome.decision is ResponseDecision.ABSTAIN
        assert outcome.reason.rule == "entailment_unavailable"

    def test_judge_skipping_a_claim_abstains(self, dose_chunk):
        """A claim the judge did not verdict is unverified, not passed."""
        gate = ReleaseGate(entailment=EntailmentCheck(StubJudge(["supported"])))
        outcome = gate.evaluate(
            [claim("One.", dose_chunk), claim("Two.", dose_chunk)], context(dose_chunk)
        )
        assert outcome.decision is ResponseDecision.ABSTAIN

    def test_unrecognised_verdict_abstains(self, dose_chunk):
        gate = ReleaseGate(entailment=EntailmentCheck(StubJudge(["probably fine"])))
        outcome = gate.evaluate([claim("Fact.", dose_chunk)], context(dose_chunk))
        assert outcome.decision is ResponseDecision.ABSTAIN

    def test_exhausted_gate_budget_abstains(self, dose_chunk):
        gate = ReleaseGate(
            entailment=EntailmentCheck(StubJudge(["supported"])),
            spec=GateSpec(latency_budget_ms=0.0),
        )
        outcome = gate.evaluate([claim("Fact.", dose_chunk)], context(dose_chunk))
        assert outcome.decision is ResponseDecision.ABSTAIN
        assert outcome.reason.rule == "gate_budget_exhausted"

    def test_no_claims_abstains(self):
        assert gate_with([]).evaluate([], context()).decision is ResponseDecision.ABSTAIN

    def test_configured_judge_that_never_runs_abstains(self, dose_chunk):
        """If entailment is configured but every claim was removed first, the
        answer is empty rather than verified - abstain, do not release."""
        outcome = gate_with(["supported"], min_supported_fraction=0.0).evaluate(
            [claim("The dose is 2000 mg.", dose_chunk, dosage=True)], context(dose_chunk)
        )
        assert outcome.decision is ResponseDecision.ABSTAIN
        assert outcome.reason.rule == "unverified"


class TestEntailmentBatching:
    def test_all_survivors_go_in_one_call(self, dose_chunk):
        judge = StubJudge(["supported"] * 5)
        gate = ReleaseGate(entailment=EntailmentCheck(judge))
        gate.evaluate([claim(f"Fact {i}.", dose_chunk) for i in range(5)], context(dose_chunk))
        assert judge.calls == 1

    def test_usage_is_accounted(self, dose_chunk):
        outcome = gate_with(["supported"]).evaluate(
            [claim("Fact.", dose_chunk)], context(dose_chunk)
        )
        assert outcome.usage.cost_usd > 0

    def test_unavailable_is_raised_not_defaulted(self, dose_chunk):
        with pytest.raises(EntailmentUnavailable):
            EntailmentCheck(StubJudge(RuntimeError("boom"))).run_batch(
                [claim("Fact.", dose_chunk)], context(dose_chunk)
            )


class TestEntailmentBatchSize:
    """Judges drop claims from long lists; smaller batches lose fewer answers."""

    def test_claims_are_split_across_calls(self, dose_chunk):
        class CountingJudge(StubJudge):
            def structured(self, messages, **kw):
                self.calls += 1
                asked = messages[-1]["content"].count("CLAIM ")
                return (
                    {"judgements": [
                        {"id": i, "verdict": "supported", "evidence": "stub"}
                        for i in range(1, asked + 1)
                    ]},
                    Usage(steps=1, prompt_tokens=10, completion_tokens=5, cost_usd=0.00001),
                )

        judge = CountingJudge([])
        gate = ReleaseGate(entailment=EntailmentCheck(judge, batch_size=6))
        claims = [claim(f"Fact {i}.", dose_chunk) for i in range(15)]
        outcome = gate.evaluate(claims, context(dose_chunk))
        assert judge.calls == 3  # 6 + 6 + 3
        assert outcome.decision is ResponseDecision.RELEASE
        assert len(outcome.retained) == 15

    def test_usage_accumulates_across_batches(self, dose_chunk):
        class CountingJudge(StubJudge):
            def structured(self, messages, **kw):
                self.calls += 1
                asked = messages[-1]["content"].count("CLAIM ")
                return (
                    {"judgements": [
                        {"id": i, "verdict": "supported", "evidence": "s"}
                        for i in range(1, asked + 1)
                    ]},
                    Usage(steps=1, prompt_tokens=10, completion_tokens=5, cost_usd=0.001),
                )

        gate = ReleaseGate(entailment=EntailmentCheck(CountingJudge([]), batch_size=2))
        outcome = gate.evaluate(
            [claim(f"Fact {i}.", dose_chunk) for i in range(4)], context(dose_chunk)
        )
        assert outcome.usage.cost_usd == pytest.approx(0.002)  # two batches

    def test_a_dropped_claim_in_any_batch_still_fails_closed(self, dose_chunk):
        class ForgetfulJudge(StubJudge):
            def structured(self, messages, **kw):
                self.calls += 1
                asked = messages[-1]["content"].count("CLAIM ")
                return (
                    {"judgements": [
                        {"id": i, "verdict": "supported", "evidence": "s"}
                        for i in range(1, asked)  # drops the last one
                    ]},
                    Usage(steps=1),
                )

        gate = ReleaseGate(entailment=EntailmentCheck(ForgetfulJudge([]), batch_size=3))
        outcome = gate.evaluate(
            [claim(f"Fact {i}.", dose_chunk) for i in range(3)], context(dose_chunk)
        )
        assert outcome.decision is ResponseDecision.ABSTAIN
