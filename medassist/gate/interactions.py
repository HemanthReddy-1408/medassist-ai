"""A curated drug-food and drug-drug interaction table.

Curated, versioned data with a citation per row - not model knowledge. An LLM
asked "does this food interact with warfarin?" answers fluently and
inconsistently; the same question against a table answers identically every
time, in microseconds, and can be reviewed by someone who knows medicine.

Coverage is deliberately narrow and **stated as a limitation** rather than
implied to be exhaustive. These are the well-documented interactions where a
lay dietary suggestion is genuinely unsafe for someone on a common medication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medassist.core.enums import ClaimDecision, Severity


@dataclass(frozen=True)
class Interaction:
    subject: str
    subject_patterns: tuple[str, ...]
    medication: str
    medication_patterns: tuple[str, ...]
    mechanism: str
    severity: Severity
    action: ClaimDecision
    caveat: str
    source: str

    def mentions_subject(self, text: str) -> str:
        for pattern in self.subject_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""

    def matches_medication(self, medications: list[str]) -> str:
        joined = " ; ".join(medications).lower()
        for pattern in self.medication_patterns:
            match = re.search(pattern, joined, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""


TABLE: tuple[Interaction, ...] = (
    Interaction(
        subject="vitamin K / leafy green vegetables",
        subject_patterns=(
            r"\bleafy green\w*\b", r"\bvitamin k\b", r"\bspinach\b", r"\bkale\b",
            r"\bbroccoli\b", r"\bcollard\w*\b", r"\bbrussels sprouts\b",
        ),
        medication="warfarin",
        medication_patterns=(r"\bwarfarin\b", r"\bcoumadin\b", r"\bjantoven\b"),
        mechanism="Vitamin K antagonises warfarin, reducing anticoagulant effect and INR.",
        severity=Severity.HIGH,
        action=ClaimDecision.QUALIFY,
        caveat=(
            "You take warfarin. Vitamin K in leafy greens affects how warfarin works - the "
            "guidance is to keep your intake *consistent* rather than to increase or avoid it. "
            "Discuss any dietary change with your anticoagulation clinic."
        ),
        source="FDA warfarin label, Drug Interactions",
    ),
    Interaction(
        subject="grapefruit",
        subject_patterns=(r"\bgrapefruit\w*\b", r"\bpomelo\b", r"\bseville orange\w*\b"),
        medication="simvastatin / atorvastatin / lovastatin",
        medication_patterns=(r"\bsimvastatin\b", r"\batorvastatin\b", r"\blovastatin\b", r"\bzocor\b", r"\blipitor\b"),
        mechanism="Grapefruit inhibits intestinal CYP3A4, raising statin exposure and myopathy risk.",
        severity=Severity.HIGH,
        action=ClaimDecision.REMOVE,
        caveat="Grapefruit should not be recommended alongside your statin.",
        source="FDA statin labels, Drug Interactions",
    ),
    Interaction(
        subject="potassium-rich foods / salt substitutes",
        subject_patterns=(
            r"\bpotassium\b", r"\bsalt substitute\w*\b", r"\bbanana\w*\b",
            r"\borange juice\b", r"\bavocado\w*\b",
        ),
        medication="ACE inhibitor / ARB / potassium-sparing diuretic",
        medication_patterns=(
            r"\blisinopril\b", r"\bramipril\b", r"\benalapril\b", r"\blosartan\b",
            r"\bvalsartan\b", r"\bspironolactone\b", r"\beplerenone\b", r"\bamiloride\b",
        ),
        mechanism="These drugs retain potassium; added dietary potassium risks hyperkalaemia.",
        severity=Severity.HIGH,
        action=ClaimDecision.REMOVE,
        caveat="Increasing potassium is unsafe with your blood-pressure medication.",
        source="FDA ACE inhibitor and ARB labels, Warnings",
    ),
    Interaction(
        subject="St John's wort",
        subject_patterns=(r"\bst\.? john'?s? wort\b", r"\bhypericum\b"),
        medication="SSRI / SNRI",
        medication_patterns=(
            r"\bsertraline\b", r"\bfluoxetine\b", r"\bcitalopram\b", r"\bescitalopram\b",
            r"\bparoxetine\b", r"\bvenlafaxine\b", r"\bduloxetine\b",
        ),
        mechanism="Additive serotonergic effect; risk of serotonin syndrome.",
        severity=Severity.CRITICAL,
        action=ClaimDecision.REMOVE,
        caveat="St John's wort is dangerous with your antidepressant.",
        source="FDA SSRI labels, Warnings and Precautions",
    ),
    Interaction(
        subject="tyramine-rich foods",
        subject_patterns=(r"\btyramine\b", r"\baged cheese\w*\b", r"\bcured meat\w*\b", r"\bfermented\b"),
        medication="MAO inhibitor",
        medication_patterns=(r"\bphenelzine\b", r"\btranylcypromine\b", r"\bisocarboxazid\b", r"\bselegiline\b"),
        mechanism="Tyramine with MAO inhibition can precipitate hypertensive crisis.",
        severity=Severity.CRITICAL,
        action=ClaimDecision.REMOVE,
        caveat="Aged and fermented foods are dangerous with MAO inhibitors.",
        source="FDA MAOI labels, Boxed Warning",
    ),
    Interaction(
        subject="alcohol",
        subject_patterns=(r"\balcohol\b", r"\bwine\b", r"\bbeer\b", r"\bdrink\w* alcohol\b"),
        medication="metformin",
        medication_patterns=(r"\bmetformin\b", r"\bglucophage\b"),
        mechanism="Alcohol potentiates metformin's effect on lactate metabolism.",
        severity=Severity.MEDIUM,
        action=ClaimDecision.QUALIFY,
        caveat="Limit alcohol while taking metformin - it raises the risk of lactic acidosis.",
        source="FDA metformin label, Warnings and Precautions",
    ),
    Interaction(
        subject="calcium / dairy / mineral supplements",
        subject_patterns=(r"\bcalcium\b", r"\bdairy\b", r"\bmilk\b", r"\bantacid\w*\b", r"\biron supplement\w*\b"),
        medication="levothyroxine / tetracycline / fluoroquinolone",
        medication_patterns=(
            r"\blevothyroxine\b", r"\bsynthroid\b", r"\bdoxycycline\b",
            r"\bciprofloxacin\b", r"\blevofloxacin\b", r"\btetracycline\b",
        ),
        mechanism="Polyvalent cations chelate these drugs and markedly reduce absorption.",
        severity=Severity.MEDIUM,
        action=ClaimDecision.QUALIFY,
        caveat="Separate calcium, dairy or iron from this medication by at least 4 hours.",
        source="FDA levothyroxine and fluoroquinolone labels, Drug Interactions",
    ),
    Interaction(
        subject="NSAIDs",
        subject_patterns=(r"\bibuprofen\b", r"\bnaproxen\b", r"\bnsaid\w*\b", r"\baspirin\b", r"\bdiclofenac\b"),
        medication="anticoagulant",
        medication_patterns=(
            r"\bwarfarin\b", r"\bapixaban\b", r"\brivaroxaban\b", r"\bdabigatran\b",
            r"\bedoxaban\b", r"\bclopidogrel\b",
        ),
        mechanism="Additive bleeding risk with impaired platelet function.",
        severity=Severity.HIGH,
        action=ClaimDecision.REMOVE,
        caveat="NSAIDs substantially raise bleeding risk with your anticoagulant.",
        source="FDA anticoagulant labels, Warnings and Precautions",
    ),
    Interaction(
        subject="high-protein diet",
        subject_patterns=(r"\bhigh[- ]protein\b", r"\bprotein intake\b", r"\bmore protein\b"),
        medication="advanced chronic kidney disease",
        medication_patterns=(r"\bckd\b", r"\bchronic kidney disease\b", r"\brenal failure\b", r"\bdialysis\b"),
        mechanism="Protein restriction is standard in advanced CKD; higher intake accelerates decline.",
        severity=Severity.HIGH,
        action=ClaimDecision.QUALIFY,
        caveat="With kidney disease, protein intake should be set by your nephrology team, not increased generally.",
        source="MedlinePlus, Chronic Kidney Disease diet",
    ),
    Interaction(
        subject="grapefruit",
        subject_patterns=(r"\bgrapefruit\w*\b",),
        medication="calcium channel blocker / tacrolimus / ciclosporin",
        medication_patterns=(r"\bamlodipine\b", r"\bfelodipine\b", r"\bnifedipine\b", r"\btacrolimus\b", r"\bciclosporin\b", r"\bcyclosporine\b"),
        mechanism="CYP3A4 inhibition raises drug exposure unpredictably.",
        severity=Severity.HIGH,
        action=ClaimDecision.REMOVE,
        caveat="Grapefruit interacts with this medication and should be avoided.",
        source="FDA labels, Drug Interactions",
    ),
)

#: Bumped whenever a row changes, so a decision record says which table it used.
TABLE_VERSION = "2026.09.05-1"


def find_conflicts(text: str, medications: list[str], conditions: list[str] | None = None) -> list[tuple[Interaction, str, str]]:
    """Rows where the text suggests a subject the patient's record contraindicates.

    Returns ``(interaction, matched_subject, matched_medication)``. Conditions
    are searched alongside medications because some rows key on a diagnosis
    (advanced CKD) rather than on a drug.
    """
    haystack = list(medications) + list(conditions or [])
    found: list[tuple[Interaction, str, str]] = []
    for interaction in TABLE:
        subject = interaction.mentions_subject(text)
        if not subject:
            continue
        medication = interaction.matches_medication(haystack)
        if not medication:
            continue
        found.append((interaction, subject, medication))
    return found
