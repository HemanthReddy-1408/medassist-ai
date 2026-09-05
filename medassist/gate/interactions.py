"""Curated interaction rules, keyed on drug *classes*.

Curated, versioned data with a citation per row - not model knowledge. An LLM
asked "does this food interact with warfarin?" answers fluently and
inconsistently; the same question against a table answers identically every
time, in microseconds, and can be reviewed by someone who knows medicine.

Rules are written against classes rather than individual drugs, so one row
covers every member. A rule naming simvastatin protects nobody on lovastatin,
though the mechanism is identical - which is exactly the kind of gap that looks
like coverage until someone is harmed by it.

Three rule families:

* ``FOOD_RULES``     - a dietary suggestion that is unsafe given the patient's drugs
* ``DRUG_RULES``     - a drug the answer *recommends*, against drugs the patient takes
* allergy            - a recommended drug the patient reacts to, incl. cross-reactivity

Coverage is deliberately narrow and stated as a limitation (§07.3) rather than
implied to be exhaustive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medassist.core.enums import ClaimDecision, Severity
from medassist.patient.normalize import CLASSES, CROSS_REACTIVITY, Medication, mentions_drug

#: Bumped whenever a row changes, so a decision record says which table it used.
TABLE_VERSION = "2026.09.05-2"


@dataclass(frozen=True)
class Finding:
    """One relational conflict, with everything needed to explain it."""

    kind: str                 # "food" | "drug" | "allergy"
    subject: str              # what the answer suggested
    trigger: str              # the patient's medication, class or allergy
    mechanism: str
    severity: Severity
    action: ClaimDecision
    caveat: str
    source: str


@dataclass(frozen=True)
class FoodRule:
    subject: str
    patterns: tuple[str, ...]
    classes: tuple[str, ...]          # any of these in the patient's list triggers it
    conditions: tuple[str, ...] = ()  # or any of these diagnoses
    mechanism: str = ""
    severity: Severity = Severity.MEDIUM
    action: ClaimDecision = ClaimDecision.QUALIFY
    caveat: str = ""
    source: str = ""

    def mentioned_in(self, text: str) -> str:
        for pattern in self.patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""


@dataclass(frozen=True)
class DrugRule:
    """A class-vs-class conflict between a recommended drug and a taken one."""

    recommended: str          # class of the drug the answer suggests
    taken: str                # class the patient is already on
    mechanism: str
    severity: Severity
    action: ClaimDecision
    caveat: str
    source: str


FOOD_RULES: tuple[FoodRule, ...] = (
    FoodRule(
        subject="vitamin K / leafy green vegetables",
        patterns=(r"\bleafy green\w*\b", r"\bvitamin k\b", r"\bspinach\b", r"\bkale\b",
                  r"\bbroccoli\b", r"\bcollard\w*\b", r"\bbrussels sprouts\b"),
        classes=("vka",),
        mechanism="Vitamin K antagonises warfarin, reducing anticoagulant effect and INR.",
        severity=Severity.HIGH,
        action=ClaimDecision.QUALIFY,
        caveat=("You take warfarin. Vitamin K in leafy greens affects how warfarin works - "
                "the guidance is to keep intake *consistent* rather than to increase or "
                "avoid it. Discuss dietary changes with your anticoagulation clinic."),
        source="FDA warfarin label, Drug Interactions",
    ),
    FoodRule(
        subject="grapefruit",
        patterns=(r"\bgrapefruit\w*\b", r"\bpomelo\b", r"\bseville orange\w*\b"),
        classes=("cyp3a4_substrate",),
        mechanism="Grapefruit inhibits intestinal CYP3A4, raising exposure to the drug.",
        severity=Severity.HIGH,
        action=ClaimDecision.REMOVE,
        caveat="Grapefruit raises blood levels of one of your medications and should be avoided.",
        source="FDA statin, CCB and immunosuppressant labels, Drug Interactions",
    ),
    FoodRule(
        subject="potassium-rich foods / salt substitutes",
        patterns=(r"\bpotassium\b", r"\bsalt substitute\w*\b", r"\bbanana\w*\b",
                  r"\borange juice\b", r"\bavocado\w*\b"),
        classes=("potassium_sparing",),
        mechanism="These drugs retain potassium; added dietary potassium risks hyperkalaemia.",
        severity=Severity.HIGH,
        action=ClaimDecision.REMOVE,
        caveat="Increasing potassium is unsafe with your blood-pressure medication.",
        source="FDA ACE inhibitor and ARB labels, Warnings",
    ),
    FoodRule(
        subject="St John's wort",
        patterns=(r"\bst\.? john'?s? wort\b", r"\bhypericum\b"),
        classes=("serotonergic",),
        mechanism="Additive serotonergic effect; risk of serotonin syndrome.",
        severity=Severity.CRITICAL,
        action=ClaimDecision.REMOVE,
        caveat="St John's wort is dangerous with your antidepressant.",
        source="FDA SSRI labels, Warnings and Precautions",
    ),
    FoodRule(
        subject="tyramine-rich foods",
        patterns=(r"\btyramine\b", r"\baged cheese\w*\b", r"\bcured meat\w*\b", r"\bfermented\b"),
        classes=("maoi",),
        mechanism="Tyramine with MAO inhibition can precipitate hypertensive crisis.",
        severity=Severity.CRITICAL,
        action=ClaimDecision.REMOVE,
        caveat="Aged and fermented foods are dangerous with MAO inhibitors.",
        source="FDA MAOI labels, Boxed Warning",
    ),
    FoodRule(
        subject="alcohol",
        patterns=(r"\balcohol\b", r"\bwine\b", r"\bbeer\b"),
        classes=("biguanide",),
        mechanism="Alcohol potentiates metformin's effect on lactate metabolism.",
        severity=Severity.MEDIUM,
        action=ClaimDecision.QUALIFY,
        caveat="Limit alcohol while taking metformin - it raises the risk of lactic acidosis.",
        source="FDA metformin label, Warnings and Precautions",
    ),
    FoodRule(
        subject="calcium / dairy / iron supplements",
        patterns=(r"\bcalcium\b", r"\bdairy\b", r"\bmilk\b", r"\bantacid\w*\b",
                  r"\biron supplement\w*\b"),
        classes=("chelation_sensitive",),
        mechanism="Polyvalent cations chelate these drugs and markedly reduce absorption.",
        severity=Severity.MEDIUM,
        action=ClaimDecision.QUALIFY,
        caveat="Separate calcium, dairy or iron from this medication by at least 4 hours.",
        source="FDA levothyroxine and fluoroquinolone labels, Drug Interactions",
    ),
    FoodRule(
        subject="high-protein diet",
        patterns=(r"\bhigh[- ]protein\b", r"\bprotein intake\b", r"\bmore protein\b"),
        classes=(),
        conditions=(r"\bckd\b", r"\bchronic kidney disease\b", r"\brenal failure\b", r"\bdialysis\b"),
        mechanism="Protein restriction is standard in advanced CKD; higher intake accelerates decline.",
        severity=Severity.HIGH,
        action=ClaimDecision.QUALIFY,
        caveat="With kidney disease, protein intake should be set by your nephrology team.",
        source="MedlinePlus, Chronic Kidney Disease diet",
    ),
)


DRUG_RULES: tuple[DrugRule, ...] = (
    DrugRule(
        recommended="nsaid", taken="anticoagulant",
        mechanism="Additive bleeding risk with impaired platelet function and GI injury.",
        severity=Severity.HIGH, action=ClaimDecision.REMOVE,
        caveat="NSAIDs substantially raise bleeding risk with your anticoagulant.",
        source="FDA anticoagulant labels, Warnings and Precautions",
    ),
    DrugRule(
        recommended="nsaid", taken="antiplatelet",
        mechanism="Additive bleeding risk.",
        severity=Severity.HIGH, action=ClaimDecision.REMOVE,
        caveat="NSAIDs raise bleeding risk alongside your antiplatelet medication.",
        source="FDA antiplatelet labels, Warnings",
    ),
    DrugRule(
        recommended="nsaid", taken="ace_inhibitor",
        mechanism="NSAIDs blunt antihypertensive effect and risk acute kidney injury.",
        severity=Severity.MEDIUM, action=ClaimDecision.QUALIFY,
        caveat="Regular NSAID use can raise blood pressure and strain the kidneys with an ACE inhibitor.",
        source="FDA ACE inhibitor labels, Drug Interactions",
    ),
    DrugRule(
        recommended="serotonergic", taken="serotonergic",
        mechanism="Additive serotonergic activity; risk of serotonin syndrome.",
        severity=Severity.CRITICAL, action=ClaimDecision.REMOVE,
        caveat="Combining two serotonergic medications risks serotonin syndrome.",
        source="FDA SSRI and SNRI labels, Warnings",
    ),
    DrugRule(
        recommended="potassium_sparing", taken="potassium_sparing",
        mechanism="Two potassium-retaining agents together risk severe hyperkalaemia.",
        severity=Severity.HIGH, action=ClaimDecision.REMOVE,
        caveat="Two potassium-sparing medications together risk dangerous potassium levels.",
        source="FDA spironolactone label, Warnings",
    ),
    DrugRule(
        recommended="anticoagulant", taken="anticoagulant",
        mechanism="Duplicate anticoagulation; major bleeding risk.",
        severity=Severity.CRITICAL, action=ClaimDecision.REMOVE,
        caveat="You are already on an anticoagulant; adding another is a major bleeding risk.",
        source="FDA anticoagulant labels, Boxed Warning",
    ),
)


def _condition_hit(rule: FoodRule, conditions: list[str]) -> str:
    joined = " ; ".join(conditions).lower()
    for pattern in rule.conditions:
        match = re.search(pattern, joined, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def find_food_conflicts(
    text: str, medications: list[Medication], conditions: list[str]
) -> list[Finding]:
    patient_classes = {c for med in medications for c in med.classes}
    findings: list[Finding] = []
    for rule in FOOD_RULES:
        subject = rule.mentioned_in(text)
        if not subject:
            continue
        matched = next((c for c in rule.classes if c in patient_classes), "")
        if not matched:
            matched = _condition_hit(rule, conditions)
        if not matched:
            continue
        trigger = next(
            (m.ingredient for m in medications if matched in m.classes), matched
        )
        findings.append(
            Finding(
                kind="food", subject=subject, trigger=trigger, mechanism=rule.mechanism,
                severity=rule.severity, action=rule.action, caveat=rule.caveat,
                source=rule.source,
            )
        )
    return findings


def find_drug_conflicts(text: str, medications: list[Medication]) -> list[Finding]:
    """Conflicts between a drug the answer recommends and one the patient takes."""
    recommended = mentions_drug(text)
    if not recommended:
        return []
    taken = {c: med.ingredient for med in medications for c in med.classes}
    findings: list[Finding] = []
    for ingredient in recommended:
        # Do not flag a drug against itself: mentioning a patient's own
        # medication by name is normal and expected.
        if any(med.ingredient == ingredient for med in medications):
            continue
        for cls in CLASSES.get(ingredient, ()):
            for rule in DRUG_RULES:
                if rule.recommended != cls or rule.taken not in taken:
                    continue
                findings.append(
                    Finding(
                        kind="drug", subject=ingredient, trigger=taken[rule.taken],
                        mechanism=rule.mechanism, severity=rule.severity,
                        action=rule.action, caveat=rule.caveat, source=rule.source,
                    )
                )
    return findings


def find_allergy_conflicts(text: str, allergies: list[str]) -> list[Finding]:
    """A recommended drug the patient reacts to, including cross-reactivity."""
    if not allergies:
        return []
    reactive: set[str] = set()
    for entry in allergies:
        token = entry.strip().lower()
        reactive.add(token)
        for cls in CLASSES.get(token, ()):
            reactive.add(cls)
        reactive.update(CROSS_REACTIVITY.get(token, ()))

    findings: list[Finding] = []
    for ingredient in mentions_drug(text):
        classes = set(CLASSES.get(ingredient, ()))
        overlap = ({ingredient} | classes) & reactive
        cross = {c for cls in classes for c in CROSS_REACTIVITY.get(cls, ())} & reactive
        if not overlap and not cross:
            continue
        direct = bool({ingredient} & reactive)
        findings.append(
            Finding(
                kind="allergy", subject=ingredient,
                trigger=next(iter(sorted(overlap or cross))),
                mechanism=("documented allergy" if direct
                           else "cross-reactivity with a documented allergy"),
                severity=Severity.CRITICAL,
                action=ClaimDecision.REMOVE,
                caveat=f"You have a documented allergy that contraindicates {ingredient}.",
                source="patient record",
            )
        )
    return findings


def find_conflicts(
    text: str,
    medications: list[Medication],
    conditions: list[str] | None = None,
    allergies: list[str] | None = None,
) -> list[Finding]:
    return [
        *find_food_conflicts(text, medications, list(conditions or [])),
        *find_drug_conflicts(text, medications),
        *find_allergy_conflicts(text, list(allergies or [])),
    ]
