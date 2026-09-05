"""Wave 5: report parsing, reference intervals, findings and trends."""

from __future__ import annotations

import pytest

from medassist.core.enums import Severity
from medassist.core.models import LabAnalyte, LabReport
from medassist.reports.analytes import canonical_name, extract_analytes, is_critical, lookup
from medassist.reports.interpret import findings, requires_escalation, retrieval_query, trends
from medassist.reports.parse import ReportParseError, load_report, parse_report

REPORT = """LABORATORY REPORT
Patient Name: Jane Doe
MRN: A1234567
Collected: 2026-08-14

HbA1c: 8.2 % (ref 4.0-5.6)
Glucose 168 mg/dL [70 - 99]
Potassium 6.8 mmol/L
Creatinine 1.1 mg/dL
LDL 162 mg/dL
eGFR 88 mL/min
Comment: follow up with the ordering clinician.
"""


@pytest.fixture
def parsed():
    return parse_report(REPORT, source_name="labs.txt")


class TestReferenceIntervals:
    @pytest.mark.parametrize(
        ("alias", "expected"),
        [("a1c", "HbA1c"), ("hemoglobin a1c", "HbA1c"), ("hgb", "Hemoglobin"),
         ("ldl-c", "LDL"), ("sgpt", "ALT"), ("k", "Potassium")],
    )
    def test_aliases_resolve_to_a_canonical_analyte(self, alias, expected):
        assert canonical_name(alias) == expected

    def test_unknown_analyte_is_titled_not_dropped(self):
        assert canonical_name("some novel marker") == "Some Novel Marker"

    def test_critical_thresholds_are_distinct_from_abnormal(self):
        """Abnormal and emergency are different clinical facts."""
        assert is_critical(LabAnalyte(name="Potassium", value=6.8, ref_low=3.5, ref_high=5.1))
        assert not is_critical(LabAnalyte(name="Potassium", value=5.4, ref_low=3.5, ref_high=5.1))

    def test_one_sided_intervals_are_supported(self):
        interval = lookup("LDL")
        assert interval is not None and interval.low is None and interval.high == 100


class TestExtraction:
    def test_analytes_are_extracted(self, parsed):
        names = {a.name for a in parsed.report.analytes}
        assert {"HbA1c", "Glucose", "Potassium", "Creatinine", "LDL", "eGFR"} <= names

    def test_printed_range_wins_over_the_shipped_table(self):
        """Laboratories differ in assay and population; the page is authoritative."""
        analytes, _ = extract_analytes("Potassium 5.4 mmol/L [3.0 - 6.0]")
        assert analytes[0].ref_low == 3.0 and analytes[0].ref_high == 6.0
        assert analytes[0].flag == "normal"  # normal against the printed range

    def test_table_is_used_when_no_range_is_printed(self):
        analytes, _ = extract_analytes("Potassium 6.8 mmol/L")
        assert analytes[0].ref_high == 5.1
        assert analytes[0].flag == "high"

    def test_flags_are_computed_not_read_from_the_page(self):
        # A page saying "NORMAL" next to 8.2% does not make it normal.
        analytes, _ = extract_analytes("HbA1c: 8.2 % NORMAL")
        assert analytes[0].flag == "high"

    def test_unparsed_lines_are_retained(self, parsed):
        """Extraction recall is measurable only if misses are kept."""
        assert "Comment: follow up with the ordering clinician." in parsed.report.unparsed

    def test_header_noise_is_not_read_as_an_analyte(self):
        analytes, _ = extract_analytes("Page 2\nAge 64\nHbA1c: 8.2 %")
        assert [a.name for a in analytes] == ["HbA1c"]

    def test_unknown_analyte_without_a_range_is_skipped(self):
        analytes, unparsed = extract_analytes("Mystery Marker 42 zz")
        assert analytes == []
        assert unparsed  # but the line survives for a human to read

    def test_unknown_analyte_with_a_printed_range_is_kept(self):
        analytes, _ = extract_analytes("Mystery Marker 42 zz [10 - 20]")
        assert analytes and analytes[0].flag == "high"


class TestPIIHandling:
    def test_pii_is_removed_before_structuring(self, parsed):
        assert parsed.contained_pii
        assert "Jane Doe" not in parsed.report.unparsed
        assert "A1234567" not in parsed.report.unparsed

    def test_clinical_values_survive(self, parsed):
        assert any(a.name == "HbA1c" and a.value == 8.2 for a in parsed.report.analytes)

    def test_report_type_and_date_detected(self, parsed):
        assert parsed.report.report_type == "laboratory report"
        assert parsed.report.collected == "2026-08-14"


class TestFindings:
    def test_only_abnormal_values_become_findings(self, parsed):
        names = {f.analyte for f in findings(parsed.report)}
        assert "Creatinine" not in names  # 1.1 is within 0.6-1.3
        assert "HbA1c" in names

    def test_critical_findings_sort_first(self, parsed):
        assert findings(parsed.report)[0].analyte == "Potassium"
        assert findings(parsed.report)[0].severity is Severity.CRITICAL

    def test_severity_scales_with_distance_from_the_interval(self):
        near = LabReport(analytes=[LabAnalyte(name="Sodium", value=146, unit="mmol/L", ref_low=135, ref_high=145)])
        far = LabReport(analytes=[LabAnalyte(name="Sodium", value=158, unit="mmol/L", ref_low=135, ref_high=145)])
        assert findings(near)[0].severity is Severity.MEDIUM
        assert findings(far)[0].severity is Severity.HIGH

    def test_any_critical_value_forces_escalation(self, parsed):
        assert requires_escalation(parsed.report) is True

    def test_a_normal_report_does_not_escalate(self):
        report = LabReport(analytes=[LabAnalyte(name="Sodium", value=140, ref_low=135, ref_high=145)])
        assert requires_escalation(report) is False

    def test_retrieval_query_is_built_from_the_abnormal_values(self, parsed):
        """Retrieving on the whole report drowns the abnormal in the normal."""
        query = retrieval_query(parsed.report)
        assert "Potassium" in query and "Creatinine" not in query

    def test_a_normal_report_still_yields_a_usable_query(self):
        report = LabReport(analytes=[LabAnalyte(name="Sodium", value=140, ref_low=135, ref_high=145)])
        assert retrieval_query(report)


class TestTrends:
    def setup_method(self):
        self.old = parse_report("HbA1c: 6.9 %\nLDL 130 mg/dL\nPotassium 4.2 mmol/L").report
        self.new = parse_report("HbA1c: 8.2 %\nLDL 162 mg/dL\nPotassium 6.8 mmol/L").report

    def test_direction_and_magnitude_are_computed(self):
        by_name = {t.analyte: t for t in trends(self.old, self.new)}
        assert by_name["HbA1c"].direction == "rising"
        assert by_name["Potassium"].percent == pytest.approx(61.9, abs=0.5)

    def test_largest_movement_sorts_first(self):
        assert trends(self.old, self.new)[0].analyte == "Potassium"

    def test_small_changes_read_as_stable(self):
        older = parse_report("HbA1c: 8.0 %").report
        newer = parse_report("HbA1c: 8.2 %").report
        assert trends(older, newer)[0].direction == "stable"

    def test_analytes_absent_from_the_earlier_report_are_skipped(self):
        older = parse_report("HbA1c: 6.9 %").report
        assert [t.analyte for t in trends(older, self.new)] == ["HbA1c"]

    def test_falling_values_are_labelled(self):
        assert trends(self.new, self.old)[0].direction == "falling"


class TestLoading:
    def test_missing_file_raises_a_typed_error(self, tmp_path):
        with pytest.raises(ReportParseError, match="no such report"):
            load_report(tmp_path / "absent.txt")

    def test_text_file_roundtrip(self, tmp_path):
        path = tmp_path / "labs.txt"
        path.write_text(REPORT)
        parsed = load_report(path)
        assert parsed.report.source_name == "labs.txt"
        assert len(parsed.report.analytes) >= 6


class TestLabFlagColumn:
    """Regression: an unmatched flag column silently dropped the whole analyte."""

    @pytest.mark.parametrize("flag", ["H", "L", "N", "A", "HIGH", "LOW", "NORMAL", "ABNORMAL", "WNL"])
    def test_every_flag_column_still_yields_the_analyte(self, flag):
        analytes, _ = extract_analytes(f"HbA1c: 8.2 % {flag}")
        assert len(analytes) == 1
        assert analytes[0].value == 8.2

    def test_the_lab_flag_never_overrides_the_computed_one(self):
        # A page printing NORMAL beside 8.2% does not make it normal.
        analytes, _ = extract_analytes("HbA1c: 8.2 % NORMAL")
        assert analytes[0].flag == "high"
