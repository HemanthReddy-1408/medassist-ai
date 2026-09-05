"""Medication name normalization and drug-class membership.

Interaction rules keyed on individual drug names do not survive contact with
real input. A patient writes "Lipitor 20mg", "atorvastatin calcium", or
"atorva 20" and a regex for `\\batorvastatin\\b` matches one of the three. Worse,
a rule written for simvastatin silently fails to protect the patient on
lovastatin, though the mechanism is identical.

So names are normalized to an ingredient, ingredients are grouped into classes,
and interaction rules are written against **classes**. One rule then covers
every member, including ones added later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Strip strength, form and frequency: "Lipitor 20 mg PO daily" -> "lipitor".
_NOISE = re.compile(
    r"""\b(
        \d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|%)   # strengths
        | (?:po|iv|im|sc|prn|bid|tid|qid|qd|qhs|od)   # routes and frequencies
        | (?:once|twice|three\s+times|four\s+times)
        | (?:daily|nightly|weekly|hourly|day|days|week)
        | (?:tablet|tablets|capsule|capsules|pill|pills|dose|doses)
        | (?:oral|extended|immediate|release|er|xr|sr|la)
        | (?:hydrochloride|hcl|sodium|calcium|potassium|sulfate|tartrate|maleate|besylate|succinate)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

#: Brand -> generic ingredient. Common enough to be worth carrying; the class
#: table below is what actually does the work.
BRANDS: dict[str, str] = {
    "coumadin": "warfarin", "jantoven": "warfarin",
    "eliquis": "apixaban", "xarelto": "rivaroxaban", "pradaxa": "dabigatran",
    "plavix": "clopidogrel",
    "lipitor": "atorvastatin", "zocor": "simvastatin", "crestor": "rosuvastatin",
    "mevacor": "lovastatin", "pravachol": "pravastatin",
    "glucophage": "metformin", "fortamet": "metformin", "riomet": "metformin",
    "zestril": "lisinopril", "prinivil": "lisinopril", "vasotec": "enalapril",
    "altace": "ramipril", "cozaar": "losartan", "diovan": "valsartan",
    "aldactone": "spironolactone", "inspra": "eplerenone",
    "zoloft": "sertraline", "prozac": "fluoxetine", "celexa": "citalopram",
    "lexapro": "escitalopram", "paxil": "paroxetine", "effexor": "venlafaxine",
    "cymbalta": "duloxetine",
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine",
    "advil": "ibuprofen", "motrin": "ibuprofen", "aleve": "naproxen",
    "norvasc": "amlodipine", "procardia": "nifedipine",
    "prograf": "tacrolimus", "neoral": "ciclosporin", "sandimmune": "ciclosporin",
    "nardil": "phenelzine", "parnate": "tranylcypromine",
    "cipro": "ciprofloxacin", "levaquin": "levofloxacin", "vibramycin": "doxycycline",
    "amoxil": "amoxicillin", "augmentin": "amoxicillin",
}

#: Ingredient -> the classes it belongs to. An ingredient may be in several.
CLASSES: dict[str, tuple[str, ...]] = {
    # anticoagulation / antiplatelet
    "warfarin": ("anticoagulant", "vka"),
    "apixaban": ("anticoagulant", "doac"), "rivaroxaban": ("anticoagulant", "doac"),
    "dabigatran": ("anticoagulant", "doac"), "edoxaban": ("anticoagulant", "doac"),
    "clopidogrel": ("antiplatelet",), "ticagrelor": ("antiplatelet",),
    # statins - all CYP3A4 substrates except pravastatin/rosuvastatin
    "simvastatin": ("statin", "cyp3a4_substrate"),
    "atorvastatin": ("statin", "cyp3a4_substrate"),
    "lovastatin": ("statin", "cyp3a4_substrate"),
    "rosuvastatin": ("statin",), "pravastatin": ("statin",),
    # renin-angiotensin and potassium-sparing
    "lisinopril": ("ace_inhibitor", "potassium_sparing"),
    "enalapril": ("ace_inhibitor", "potassium_sparing"),
    "ramipril": ("ace_inhibitor", "potassium_sparing"),
    "losartan": ("arb", "potassium_sparing"), "valsartan": ("arb", "potassium_sparing"),
    "spironolactone": ("potassium_sparing", "diuretic"),
    "eplerenone": ("potassium_sparing", "diuretic"), "amiloride": ("potassium_sparing",),
    # serotonergic
    "sertraline": ("ssri", "serotonergic"), "fluoxetine": ("ssri", "serotonergic"),
    "citalopram": ("ssri", "serotonergic"), "escitalopram": ("ssri", "serotonergic"),
    "paroxetine": ("ssri", "serotonergic"),
    "venlafaxine": ("snri", "serotonergic"), "duloxetine": ("snri", "serotonergic"),
    "phenelzine": ("maoi", "serotonergic"), "tranylcypromine": ("maoi", "serotonergic"),
    "isocarboxazid": ("maoi", "serotonergic"), "selegiline": ("maoi",),
    # NSAIDs
    "ibuprofen": ("nsaid",), "naproxen": ("nsaid",), "diclofenac": ("nsaid",),
    "celecoxib": ("nsaid",), "aspirin": ("nsaid", "antiplatelet"),
    # narrow therapeutic index / absorption-sensitive
    "levothyroxine": ("thyroid_hormone", "chelation_sensitive"),
    "doxycycline": ("tetracycline", "chelation_sensitive"),
    "ciprofloxacin": ("fluoroquinolone", "chelation_sensitive"),
    "levofloxacin": ("fluoroquinolone", "chelation_sensitive"),
    "tacrolimus": ("immunosuppressant", "cyp3a4_substrate"),
    "ciclosporin": ("immunosuppressant", "cyp3a4_substrate"),
    # calcium channel blockers
    "amlodipine": ("ccb", "cyp3a4_substrate"), "felodipine": ("ccb", "cyp3a4_substrate"),
    "nifedipine": ("ccb", "cyp3a4_substrate"),
    # other
    "metformin": ("biguanide",),
    "amoxicillin": ("penicillin", "beta_lactam"),
    "cephalexin": ("cephalosporin", "beta_lactam"),
    "penicillin": ("penicillin", "beta_lactam"),
}

#: Allergy cross-reactivity. Reacting to one member implies risk from the other.
CROSS_REACTIVITY: dict[str, tuple[str, ...]] = {
    "penicillin": ("cephalosporin",),
    "cephalosporin": ("penicillin",),
    "nsaid": ("nsaid",),
    "sulfonamide": ("sulfonamide",),
}


@dataclass(frozen=True)
class Medication:
    """One entry from a patient's medication list, normalized."""

    raw: str
    ingredient: str
    classes: frozenset[str]

    @property
    def known(self) -> bool:
        """False when the name is not in the ingredient table.

        Surfaced rather than swallowed: an unrecognised medication means the
        relational check cannot reason about it, and the answer says so instead
        of implying the list was fully screened.
        """
        return bool(self.classes)


def strip_noise(text: str) -> str:
    return " ".join(_NOISE.sub(" ", text.lower()).replace(",", " ").split())


def to_ingredient(text: str) -> str:
    """Best-effort ingredient name from a free-text medication entry."""
    cleaned = strip_noise(text)
    if not cleaned:
        return ""
    for token in cleaned.split():
        token = token.strip("().")
        if token in BRANDS:
            return BRANDS[token]
        if token in CLASSES:
            return token
    # Fall back to the first word so an unknown drug still has a stable label.
    return cleaned.split()[0].strip("().")


def normalize(entries: list[str]) -> list[Medication]:
    out: list[Medication] = []
    for entry in entries:
        ingredient = to_ingredient(entry)
        if not ingredient:
            continue
        out.append(
            Medication(
                raw=entry.strip(),
                ingredient=ingredient,
                classes=frozenset(CLASSES.get(ingredient, ())),
            )
        )
    return out


def classes_of(entries: list[str]) -> set[str]:
    return {c for med in normalize(entries) for c in med.classes}


def mentions_drug(text: str) -> list[str]:
    """Ingredients named in free text, resolving brands.

    Used to detect a claim *recommending* a drug, so it can be checked against
    what the patient already takes.
    """
    found: list[str] = []
    lowered = text.lower()
    for name in (*CLASSES, *BRANDS):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            resolved = BRANDS.get(name, name)
            if resolved not in found:
                found.append(resolved)
    return found
