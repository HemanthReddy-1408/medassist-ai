"""Wave 2: medication normalization and class-based relational safety."""

from __future__ import annotations

import pytest

from medassist.core.enums import ClaimDecision, Severity, SourceKind
from medassist.core.ids import ChunkId, DocumentId
from medassist.core.models import Chunk, Citation, Claim, PatientProfile
from medassist.gate.checks import GateContext, RelationalCheck
from medassist.gate.interactions import (
    DRUG_RULES,
    FOOD_RULES,
    find_allergy_conflicts,
    find_conflicts,
    find_drug_conflicts,
)
from medassist.patient.normalize import (
    classes_of,
    mentions_drug,
    normalize,
    to_ingredient,
)

DOC = DocumentId.new()


def claim(text: str) -> Claim:
    chunk_id = ChunkId.new()
    return Claim(text=text, citations=[Citation(chunk_id=chunk_id, doc_id=DOC)])


def ctx_for(profile: PatientProfile) -> GateContext:
    c = Chunk(id=ChunkId.new(), doc_id=DOC, text="source", ordinal=0, source=SourceKind.MEDLINEPLUS)
    return GateContext(chunks={c.id: c}, context_ids=[c.id], profile=profile)


class TestNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Lipitor 20mg PO daily", "atorvastatin"),
            ("atorvastatin calcium 20 mg", "atorvastatin"),
            ("Coumadin", "warfarin"),
            ("metformin HCl 500 mg twice daily", "metformin"),
            ("WARFARIN 5MG", "warfarin"),
            ("aspirin 81 mg", "aspirin"),
        ],
    )
    def test_brands_salts_and_strengths_resolve_to_an_ingredient(self, raw, expected):
        assert to_ingredient(raw) == expected

    def test_unknown_drug_keeps_a_stable_label_but_is_marked_unknown(self):
        med = normalize(["zzyzxamine 10mg"])[0]
        assert med.ingredient == "zzyzxamine"
        assert med.known is False

    def test_classes_are_unioned_across_a_list(self):
        assert classes_of(["Zestril 10mg", "Aldactone 25mg"]) >= {
            "ace_inhibitor", "potassium_sparing", "diuretic"
        }

    def test_empty_entries_are_dropped(self):
        assert normalize(["", "   ", "20 mg"]) == []

    def test_mentions_drug_resolves_brands_in_free_text(self):
        assert mentions_drug("Try Advil for the pain.") == ["ibuprofen"]

    def test_mentions_drug_is_word_bounded(self):
        # "aspirin" must not match inside another word.
        assert mentions_drug("aspirinoid compounds are unrelated") == []


class TestClassBasedRules:
    """A rule naming one drug protects nobody on its siblings."""

    @pytest.mark.parametrize("statin", ["simvastatin", "atorvastatin", "lovastatin", "Lipitor"])
    def test_grapefruit_rule_covers_every_cyp3a4_statin(self, statin):
        outcome = RelationalCheck().run(
            claim("Grapefruit juice is a healthy breakfast choice."),
            ctx_for(PatientProfile(medications=[f"{statin} 20mg"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE

    def test_grapefruit_rule_also_covers_non_statin_cyp3a4_substrates(self):
        outcome = RelationalCheck().run(
            claim("Grapefruit is a healthy choice."),
            ctx_for(PatientProfile(medications=["tacrolimus 1mg"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE

    def test_statin_without_cyp3a4_involvement_is_not_flagged(self):
        # Rosuvastatin is not a CYP3A4 substrate; flagging it would be a false
        # positive, and false positives train people to ignore the warning.
        outcome = RelationalCheck().run(
            claim("Grapefruit is a healthy choice."),
            ctx_for(PatientProfile(medications=["rosuvastatin 10mg"])),
        )
        assert outcome.decision is ClaimDecision.RETAIN

    @pytest.mark.parametrize("drug", ["lisinopril", "losartan", "spironolactone", "Aldactone"])
    def test_potassium_rule_covers_every_potassium_sparing_drug(self, drug):
        outcome = RelationalCheck().run(
            claim("Eat more bananas for potassium."),
            ctx_for(PatientProfile(medications=[f"{drug} 10mg"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE


class TestDrugDrugConflicts:
    """A drug the answer recommends, against drugs the patient already takes."""

    def test_nsaid_recommended_to_an_anticoagulated_patient(self):
        outcome = RelationalCheck().run(
            claim("You can take ibuprofen for the pain."),
            ctx_for(PatientProfile(medications=["warfarin 5mg"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE
        assert "bleeding" in outcome.detail.lower()

    def test_second_serotonergic_agent_is_critical(self):
        findings = find_drug_conflicts(
            "Fluoxetine may help.", normalize(["sertraline 50mg"])
        )
        assert findings and findings[0].severity is Severity.CRITICAL

    def test_a_patients_own_medication_is_not_flagged_against_itself(self):
        # Naming a drug the patient already takes is normal, not a conflict.
        assert find_drug_conflicts("Keep taking your warfarin.", normalize(["warfarin 5mg"])) == []

    def test_unrelated_recommendation_is_clean(self):
        assert find_drug_conflicts("Metformin is first-line.", normalize(["lisinopril"])) == []


class TestAllergyConflicts:
    def test_direct_allergy_blocks_the_drug(self):
        findings = find_allergy_conflicts("Amoxicillin is appropriate.", ["amoxicillin"])
        assert findings and findings[0].action is ClaimDecision.REMOVE

    def test_cross_reactivity_is_caught(self):
        """Penicillin allergy implies cephalosporin risk."""
        findings = find_allergy_conflicts("Cephalexin would work.", ["penicillin"])
        assert findings
        assert "cross-reactivity" in findings[0].mechanism

    def test_class_level_allergy_matches_a_member(self):
        findings = find_allergy_conflicts("Take ibuprofen.", ["nsaid"])
        assert findings

    def test_unrelated_allergy_does_not_fire(self):
        assert find_allergy_conflicts("Take ibuprofen.", ["penicillin"]) == []

    def test_no_allergies_recorded_is_not_a_finding(self):
        assert find_allergy_conflicts("Take amoxicillin.", []) == []


class TestSeverityOrdering:
    def test_most_severe_finding_wins_across_families(self):
        """A CRITICAL drug conflict must not be masked by a QUALIFY food rule."""
        outcome = RelationalCheck().run(
            claim("Drink less alcohol, and consider adding apixaban."),
            ctx_for(PatientProfile(medications=["metformin 500mg", "warfarin 5mg"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE
        assert outcome.severity is Severity.CRITICAL

    def test_equal_severity_breaks_toward_the_restrictive_action(self):
        outcome = RelationalCheck().run(
            claim("Ibuprofen is fine, and leafy greens are healthy."),
            ctx_for(PatientProfile(medications=["warfarin"])),
        )
        assert outcome.decision is ClaimDecision.REMOVE


class TestCoverageHonesty:
    def test_unrecognised_medication_is_reported_not_silently_passed(self):
        """Implying a list was screened when it was not is the worst outcome."""
        outcome = RelationalCheck().run(
            claim("Exercise regularly."),
            ctx_for(PatientProfile(medications=["zzyzxamine 10mg"])),
        )
        assert outcome.decision is ClaimDecision.RETAIN
        assert outcome.reason == "no_conflict_partial_coverage"
        assert "zzyzxamine" in outcome.detail

    def test_fully_recognised_list_reports_clean_coverage(self):
        outcome = RelationalCheck().run(
            claim("Exercise regularly."), ctx_for(PatientProfile(medications=["metformin"]))
        )
        assert outcome.reason == "no_conflict"

    def test_every_rule_carries_a_citation(self):
        # A curated table without sources is model knowledge with extra steps.
        assert all(rule.source for rule in FOOD_RULES)
        assert all(rule.source for rule in DRUG_RULES)

    def test_conditions_are_screened_alongside_medications(self):
        findings = find_conflicts(
            "A high-protein diet is recommended.", [], ["chronic kidney disease stage 4"]
        )
        assert findings and findings[0].action is ClaimDecision.QUALIFY
