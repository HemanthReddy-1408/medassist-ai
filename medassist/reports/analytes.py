"""Reference intervals and analyte extraction.

Two rules govern this module.

**Ranges printed on the report win.** Laboratories differ in assay and
population, so a range on the page is authoritative over any table shipped
here. The table below is a fallback for reports that omit them.

**Flags are computed, never asked of a model.** A model can be argued out of
"high"; ``value > ref_high`` cannot. Extraction may use a model; adjudication
never does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medassist.core.models import LabAnalyte


@dataclass(frozen=True)
class ReferenceInterval:
    name: str
    unit: str
    low: float | None
    high: float | None
    #: Outside this, the value is a clinical emergency rather than merely
    #: abnormal. Used to force escalation (§04.2).
    critical_low: float | None = None
    critical_high: float | None = None
    aliases: tuple[str, ...] = ()

    def critical(self, value: float) -> bool:
        if self.critical_low is not None and value < self.critical_low:
            return True
        return self.critical_high is not None and value > self.critical_high


INTERVALS: tuple[ReferenceInterval, ...] = (
    ReferenceInterval("HbA1c", "%", 4.0, 5.6, None, 10.0, ("hemoglobin a1c", "haemoglobin a1c", "a1c", "glycated haemoglobin")),
    ReferenceInterval("Glucose", "mg/dL", 70, 99, 40, 400, ("blood glucose", "fasting glucose", "fbs")),
    ReferenceInterval("Creatinine", "mg/dL", 0.6, 1.3, None, 4.0, ("creat", "serum creatinine")),
    ReferenceInterval("eGFR", "mL/min", 90, None, 15, None, ("gfr", "estimated gfr")),
    ReferenceInterval("Potassium", "mmol/L", 3.5, 5.1, 2.5, 6.5, ("k", "serum potassium")),
    ReferenceInterval("Sodium", "mmol/L", 135, 145, 120, 160, ("na", "serum sodium")),
    ReferenceInterval("LDL", "mg/dL", None, 100, None, 250, ("ldl cholesterol", "ldl-c")),
    ReferenceInterval("HDL", "mg/dL", 40, None, 20, None, ("hdl cholesterol", "hdl-c")),
    ReferenceInterval("Triglycerides", "mg/dL", None, 150, None, 500, ("trig", "tg")),
    ReferenceInterval("Total Cholesterol", "mg/dL", None, 200, None, None, ("cholesterol", "tc")),
    ReferenceInterval("TSH", "mIU/L", 0.4, 4.0, 0.01, 20.0, ("thyroid stimulating hormone",)),
    ReferenceInterval("Hemoglobin", "g/dL", 12.0, 17.0, 7.0, 20.0, ("haemoglobin", "hb", "hgb")),
    ReferenceInterval("Platelets", "10^9/L", 150, 400, 50, 1000, ("plt", "platelet count")),
    ReferenceInterval("WBC", "10^9/L", 4.0, 11.0, 1.0, 30.0, ("white blood cells", "white cell count", "wcc")),
    ReferenceInterval("ALT", "U/L", 7, 56, None, 500, ("alanine aminotransferase", "sgpt")),
    ReferenceInterval("AST", "U/L", 10, 40, None, 500, ("aspartate aminotransferase", "sgot")),
    ReferenceInterval("INR", "", 0.8, 1.2, None, 5.0, ("international normalized ratio", "international normalised ratio")),
    ReferenceInterval("Vitamin D", "ng/mL", 30, 100, None, None, ("25-oh vitamin d", "25-hydroxyvitamin d")),
)

_BY_NAME: dict[str, ReferenceInterval] = {}
for interval in INTERVALS:
    _BY_NAME[interval.name.lower()] = interval
    for alias in interval.aliases:
        _BY_NAME[alias] = interval

# "HbA1c: 8.2 % (ref 4.0-5.6)" / "Potassium 5.9 mmol/L  [3.5 - 5.1]" / "LDL 160 mg/dL"
_LINE = re.compile(
    r"""^\s*
    (?P<name>[A-Za-z][A-Za-z0-9\s/()'.-]{1,44}?)      # analyte name
    \s*[:\t]?\s+
    (?P<value>-?\d+(?:\.\d+)?)                        # value
    \s*
    (?P<unit>%|[A-Za-z][A-Za-z/^0-9µ]{0,12})?          # optional unit
    # A laboratory's own flag column is parsed only so the line still matches.
    # It is never used to decide abnormality: a page printing "NORMAL" beside
    # 8.2% does not make it normal, and an unmatched flag previously caused the
    # whole analyte to be dropped silently - the worst outcome available.
    (?:\s*[\[(]?\s*(?:ref(?:erence)?[:\s]*)?          # optional printed range
       (?P<low>-?\d+(?:\.\d+)?)\s*[-–to]{1,2}\s*(?P<high>-?\d+(?:\.\d+)?)
       \s*[\])]?)?
    \s*(?P<flag>\b(?:ABNORMAL|NORMAL|HIGH|LOW|H|L|N|A|WNL)\b)?\s*$
    """,
    re.VERBOSE | re.MULTILINE,
)

_NOISE_NAMES = frozenset({
    "page", "date", "time", "phone", "fax", "age", "sex", "dob", "mrn",
    "collected", "reported", "ordered", "accession", "specimen",
})


def lookup(name: str) -> ReferenceInterval | None:
    return _BY_NAME.get(name.strip().lower())


def canonical_name(name: str) -> str:
    interval = lookup(name)
    return interval.name if interval else " ".join(name.split()).title()


def extract_analytes(text: str) -> tuple[list[LabAnalyte], list[str]]:
    """Parse a report into analytes. Returns ``(analytes, unparsed_lines)``.

    Unparsed lines are kept rather than dropped: extraction recall is then
    measurable instead of silently zero, and a downstream agent can still read
    what the parser could not structure.
    """
    analytes: list[LabAnalyte] = []
    parsed_lines: set[int] = set()
    lines = text.splitlines()

    for index, line in enumerate(lines):
        match = _LINE.match(line)
        if match is None:
            continue
        raw_name = " ".join(match.group("name").split())
        if raw_name.lower() in _NOISE_NAMES or len(raw_name) < 2:
            continue

        interval = lookup(raw_name)
        printed_low = match.group("low")
        printed_high = match.group("high")

        # A range printed on the report beats the shipped table: laboratories
        # differ in assay and reference population.
        if printed_low is not None:
            low: float | None = float(printed_low)
            high: float | None = float(printed_high)
        elif interval is not None:
            low, high = interval.low, interval.high
        else:
            low = high = None

        if interval is None and printed_low is None:
            continue  # an unknown name with no range is not interpretable

        analytes.append(
            LabAnalyte(
                name=canonical_name(raw_name),
                value=float(match.group("value")),
                unit=(match.group("unit") or (interval.unit if interval else "")).strip(),
                ref_low=low,
                ref_high=high,
                raw=line.strip(),
            )
        )
        parsed_lines.add(index)

    unparsed = [
        line.strip()
        for index, line in enumerate(lines)
        if index not in parsed_lines and line.strip()
    ]
    return analytes, unparsed


def is_critical(analyte: LabAnalyte) -> bool:
    interval = lookup(analyte.name)
    return bool(interval and interval.critical(analyte.value))
