"""Wave 9: episodic memory, staleness, and reconciliation across visits."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from medassist.core.enums import Severity
from medassist.memory.recall import (
    ConfirmReason,
    Continuity,
    observe_report,
    reconcile,
)
from medassist.memory.store import (
    MemoryEntry,
    MemoryKind,
    MemoryStore,
    entries_from_report,
    months_ago,
)
from medassist.reports.parse import parse_report


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.jsonl")


def entry(key: str, value: float, flag: str, months: int, kind=MemoryKind.LAB_FINDING) -> MemoryEntry:
    return MemoryEntry(
        profile_digest="p1", kind=kind, key=key, asserted_on=months_ago(months),
        value=value, unit="g/dL", flag=flag,
    )


class TestShelfLife:
    def test_fast_moving_analytes_expire_sooner(self):
        """Haemoglobin drifts; a chronic diagnosis does not."""
        blood = entry("Hemoglobin", 9.1, "low", months=5)
        condition = MemoryEntry(
            profile_digest="p1", kind=MemoryKind.CONDITION, key="type 2 diabetes",
            asserted_on=months_ago(12),
        )
        assert blood.is_stale()
        assert not condition.is_stale()

    def test_a_recent_value_is_current(self):
        assert not entry("Hemoglobin", 9.1, "low", months=1).is_stale()

    def test_an_undated_fact_cannot_be_shown_to_be_current(self):
        undated = MemoryEntry(
            profile_digest="p1", kind=MemoryKind.LAB_FINDING, key="Hemoglobin",
            asserted_on="", value=9.1, flag="low",
        )
        assert undated.is_stale()

    def test_shelf_life_differs_by_analyte(self):
        assert entry("INR", 3.0, "high", 0).shelf_life_days() < entry("HbA1c", 8.0, "high", 0).shelf_life_days()


class TestStore:
    def test_history_is_append_only(self, store):
        """'Low at every draw since March' is a different fact from 'low'."""
        store.extend([entry("Hemoglobin", 9.1, "low", 6), entry("Hemoglobin", 9.4, "low", 3)])
        assert len(store.history("p1", "Hemoglobin")) == 2

    def test_latest_returns_the_most_recent(self, store):
        store.extend([entry("Hemoglobin", 9.1, "low", 6), entry("Hemoglobin", 9.4, "low", 3)])
        assert store.latest("p1", "Hemoglobin").value == 9.4

    def test_profiles_are_isolated(self, store):
        store.append(entry("Hemoglobin", 9.1, "low", 1))
        store.append(
            MemoryEntry(profile_digest="p2", kind=MemoryKind.LAB_FINDING,
                        key="Hemoglobin", asserted_on=months_ago(1), value=14.0, flag="normal")
        )
        assert len(store.for_profile("p1")) == 1
        assert store.latest("p2", "Hemoglobin").value == 14.0

    def test_current_facts_exclude_stale_ones(self, store):
        store.extend([entry("Hemoglobin", 9.1, "low", 8), entry("HbA1c", 8.2, "high", 1)])
        assert {e.key for e in store.current_facts("p1")} == {"HbA1c"}

    def test_report_collection_date_is_used_not_today(self):
        """Dating an old uploaded report as current defeats the shelf life."""
        report = parse_report("Collected: 2025-01-15\nHemoglobin 9.1 g/dL").report
        assert entries_from_report("p1", report)[0].asserted_on == "2025-01-15"

    def test_empty_store_reads_cleanly(self, store):
        assert store.for_profile("nobody") == []


class TestReconciliation:
    """The case: a profile reports low haemoglobin, returns months later."""

    def test_a_prior_abnormal_not_remeasured_becomes_a_question(self, store):
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(5)}\nHemoglobin 9.1 g/dL\nHbA1c: 7.4 %"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nHbA1c: 8.2 %\nLDL 162 mg/dL"
        ).report)

        assert result.needs_user_input
        keys = {c.key for c in result.confirmations}
        assert "Hemoglobin" in keys
        question = next(c for c in result.confirmations if c.key == "Hemoglobin").question
        assert "Hemoglobin" in question and "9.1" in question

    def test_the_unmeasured_finding_is_neither_forgotten_nor_assumed(self, store):
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(2)}\nHemoglobin 9.1 g/dL"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nLDL 162 mg/dL"
        ).report)
        unchecked = result.by_continuity(Continuity.UNCHECKED)
        assert [c.key for c in unchecked] == ["Hemoglobin"]

    def test_a_remeasured_value_is_not_questioned(self, store):
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(5)}\nHemoglobin 9.1 g/dL"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nHemoglobin 9.4 g/dL"
        ).report)
        assert not result.needs_user_input

    def test_normal_history_is_never_raised(self, store):
        """Confirming a normal value is still normal changes no answer."""
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(5)}\nPotassium 4.2 mmol/L"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nLDL 162 mg/dL"
        ).report)
        assert result.confirmations == []

    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            (9.1, 14.0, Continuity.RESOLVED),
            (9.1, 10.5, Continuity.IMPROVING),
            (9.1, 8.0, Continuity.WORSENING),
        ],
    )
    def test_direction_of_travel_is_classified(self, store, before, after, expected):
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(5)}\nHemoglobin {before} g/dL"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nHemoglobin {after} g/dL"
        ).report)
        assert result.by_continuity(expected)

    def test_a_high_value_falling_is_improving(self, store):
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(5)}\nHbA1c: 9.0 %"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nHbA1c: 7.5 %"
        ).report)
        assert result.by_continuity(Continuity.IMPROVING)

    def test_first_time_abnormal_is_marked_new(self, store):
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nHemoglobin 9.1 g/dL"
        ).report)
        assert result.by_continuity(Continuity.NEW)
        assert not result.needs_user_input

    def test_stale_and_not_remeasured_are_distinguished(self, store):
        recent = reconcile(
            [entry("Hemoglobin", 9.1, "low", months=1)],
            parse_report("LDL 162 mg/dL").report,
        )
        old = reconcile(
            [entry("Hemoglobin", 9.1, "low", months=8)],
            parse_report("LDL 162 mg/dL").report,
        )
        assert recent.confirmations[0].reason is ConfirmReason.NOT_REMEASURED
        assert old.confirmations[0].reason is ConfirmReason.STALE

    def test_questions_are_capped_so_they_get_answered(self, store):
        """A system that opens with nine questions gets none answered."""
        prior = [
            entry("Hemoglobin", 9.1, "low", 2), entry("Potassium", 6.2, "high", 2),
            entry("Sodium", 128, "low", 2), entry("Glucose", 210, "high", 2),
            entry("WBC", 1.5, "low", 2),
        ]
        result = reconcile(prior, parse_report("LDL 162 mg/dL").report, max_questions=3)
        assert len(result.confirmations) == 3

    def test_most_severe_question_is_asked_first(self, store):
        prior = [entry("HbA1c", 7.2, "high", 2), entry("Potassium", 6.8, "high", 2)]
        result = reconcile(prior, parse_report("LDL 162 mg/dL").report, max_questions=1)
        assert result.confirmations[0].key == "Potassium"
        assert result.confirmations[0].severity is Severity.CRITICAL

    def test_reconcile_reads_before_it_writes(self, store):
        """Recording first would make every prior finding look re-measured."""
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(2)}\nHemoglobin 9.1 g/dL"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nLDL 162 mg/dL"
        ).report)
        assert result.needs_user_input
        assert len(store.history("p1", "Hemoglobin")) == 1

    def test_summary_is_readable(self, store):
        observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(5)}\nHemoglobin 9.1 g/dL"
        ).report)
        result = observe_report(store, "p1", parse_report(
            f"Collected: {months_ago(0)}\nLDL 162 mg/dL"
        ).report)
        text = result.summary()
        assert "please confirm" in text.lower()


class TestApiIntegration:
    def test_reports_endpoint_returns_confirmations(self, tmp_path, monkeypatch):
        import medassist.serving.api as api
        from medassist.core.models import PatientProfile

        memory = MemoryStore(tmp_path / "memory.jsonl")
        monkeypatch.setattr(api, "_state", lambda: {"memory": memory})
        profile = PatientProfile(age=54, conditions=["anaemia"])

        api.parse_lab_report(api.ReportRequest(
            text=f"Collected: {months_ago(4)}\nHemoglobin 9.1 g/dL", profile=profile,
        ))
        second = api.parse_lab_report(api.ReportRequest(
            text=f"Collected: {months_ago(0)}\nLDL 162 mg/dL", profile=profile,
        ))
        assert second["history"]["needs_user_input"] is True
        assert second["history"]["confirmations"][0]["key"] == "Hemoglobin"

    def test_memory_endpoint_separates_current_from_stale(self, tmp_path, monkeypatch):
        import medassist.serving.api as api
        from medassist.core.models import PatientProfile

        memory = MemoryStore(tmp_path / "memory.jsonl")
        monkeypatch.setattr(api, "_state", lambda: {"memory": memory})
        profile = PatientProfile(age=54)
        api.parse_lab_report(api.ReportRequest(
            text=f"Collected: {months_ago(8)}\nHemoglobin 9.1 g/dL", profile=profile,
        ))
        payload = api.memory_for_profile(api.MemoryQuery(profile=profile))
        assert payload["stale"] and not payload["current"]

    def test_remember_false_reconciles_without_writing(self, tmp_path, monkeypatch):
        import medassist.serving.api as api
        from medassist.core.models import PatientProfile

        memory = MemoryStore(tmp_path / "memory.jsonl")
        monkeypatch.setattr(api, "_state", lambda: {"memory": memory})
        profile = PatientProfile(age=54)
        api.parse_lab_report(api.ReportRequest(
            text="Collected: 2026-01-01\nHemoglobin 9.1 g/dL", profile=profile, remember=False,
        ))
        assert memory.all_entries() == []


class TestDates:
    def test_age_is_computed_from_the_assertion_date(self):
        recent = MemoryEntry(
            profile_digest="p", kind=MemoryKind.LAB_FINDING, key="X",
            asserted_on=(date.today() - timedelta(days=30)).isoformat(),
        )
        assert 29 <= recent.age_days() <= 31

    @pytest.mark.parametrize("value", ["2026-01-15", "15/01/2026", "01/15/2026"])
    def test_common_date_formats_parse(self, value):
        e = MemoryEntry(profile_digest="p", kind=MemoryKind.LAB_FINDING, key="X", asserted_on=value)
        assert e.asserted_date is not None
