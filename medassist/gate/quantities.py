"""Quantity extraction and unit normalization.

Exists so the numeric grounding check can tell that "2.5 g" and "2500 mg" are
the same dose, while "2000 mg" and "2550 mg" are not.

That asymmetry is the whole point. An LLM judge asked "are these consistent?"
will frequently accept 2000 against 2550 - the sentences are otherwise
identical and both numbers are the same *kind* of thing. It will equally often
reject 2.5 g against 2500 mg for the opposite reason. Arithmetic gets both
right, costs nothing, and cannot be argued with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (canonical dimension, factor to that dimension's base unit)
_UNITS: dict[str, tuple[str, float]] = {
    # mass - base milligram
    "mcg": ("mass", 0.001), "µg": ("mass", 0.001), "ug": ("mass", 0.001),
    "microgram": ("mass", 0.001), "micrograms": ("mass", 0.001),
    "mg": ("mass", 1.0), "milligram": ("mass", 1.0), "milligrams": ("mass", 1.0),
    "g": ("mass", 1000.0), "gram": ("mass", 1000.0), "grams": ("mass", 1000.0),
    "kg": ("mass", 1_000_000.0),
    # volume - base millilitre
    "ml": ("volume", 1.0), "millilitre": ("volume", 1.0), "milliliter": ("volume", 1.0),
    "l": ("volume", 1000.0), "litre": ("volume", 1000.0), "liter": ("volume", 1000.0),
    # dimensions with no sub-multiples worth converting; each is its own space
    "%": ("percent", 1.0), "percent": ("percent", 1.0),
    "iu": ("iu", 1.0), "unit": ("dose_unit", 1.0), "units": ("dose_unit", 1.0),
    "tablet": ("dose_form", 1.0), "tablets": ("dose_form", 1.0),
    "capsule": ("dose_form", 1.0), "capsules": ("dose_form", 1.0),
    "mmol/l": ("conc_mmol", 1.0), "mg/dl": ("conc_mgdl", 1.0),
    "mg/kg": ("mass_per_kg", 1.0), "mcg/kg": ("mass_per_kg", 0.001),
    "ml/min": ("clearance", 1.0),
    "mmhg": ("pressure", 1.0),
    "hour": ("time", 1.0), "hours": ("time", 1.0), "hr": ("time", 1.0),
    "day": ("time", 24.0), "days": ("time", 24.0),
    "week": ("time", 168.0), "weeks": ("time", 168.0),
}

# Longest-first so "mg/dl" wins over "mg", and "ml/min" over "ml".
_UNIT_PATTERN = "|".join(
    re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)
)
_QUANTITY = re.compile(
    rf"(?P<value>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_PATTERN})\b",
    re.IGNORECASE,
)
# A bare number that hands its unit to the one after it: "500 to 1000 mg".
_RANGE_LEAD = re.compile(
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:to|through|-|–|—)\s*(?=\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    dimension: str
    normalized: float
    raw: str

    def matches(self, other: Quantity, *, rel_tol: float = 1e-6) -> bool:
        if self.dimension != other.dimension:
            return False
        scale = max(abs(self.normalized), abs(other.normalized), 1e-12)
        return abs(self.normalized - other.normalized) / scale <= rel_tol

    def __str__(self) -> str:
        return self.raw


def _parse_value(text: str) -> float:
    return float(text.replace(",", ""))


def extract_quantities(text: str) -> list[Quantity]:
    """Every number carrying a recognised clinical unit.

    Bare numbers are deliberately ignored. "Type 2 diabetes" and "stage 3
    kidney disease" contain digits that are not quantities, and demanding they
    appear in a cited span would reject correct claims - a false positive on a
    safety check trains people to ignore it.

    The one exception is the leading half of a range: in "500 to 1000 mg" the
    500 inherits the trailing unit, because it *is* a dose.
    """
    quantities: list[Quantity] = []
    for match in _QUANTITY.finditer(text):
        unit_key = match.group("unit").lower()
        dimension, factor = _UNITS[unit_key]
        value = _parse_value(match.group("value"))
        quantities.append(
            Quantity(
                value=value, unit=unit_key, dimension=dimension,
                normalized=value * factor, raw=match.group(0).strip(),
            )
        )

    for match in _RANGE_LEAD.finditer(text):
        following = _QUANTITY.search(text, match.end())
        if following is None or following.start() > match.end() + 12:
            continue
        unit_key = following.group("unit").lower()
        dimension, factor = _UNITS[unit_key]
        value = _parse_value(match.group("value"))
        quantities.append(
            Quantity(
                value=value, unit=unit_key, dimension=dimension,
                normalized=value * factor,
                raw=f"{match.group('value')} {unit_key} (range lower bound)",
            )
        )

    return quantities


def find_unmatched(claim: str, spans: list[str]) -> list[Quantity]:
    """Quantities in ``claim`` that appear in none of ``spans``."""
    available = [q for span in spans for q in extract_quantities(span)]
    return [
        q for q in extract_quantities(claim)
        if not any(q.matches(other) for other in available)
    ]
