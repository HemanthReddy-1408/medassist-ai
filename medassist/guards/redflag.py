"""Red-flag triage. Runs first, always - before retrieval, before planning.

Tuned deliberately for **recall over precision**. A false positive costs a user
an unnecessary "seek care now". A false negative is the worst thing this system
can do. Both rates are reported separately (§08) so the trade-off stays visible
rather than hidden inside one F1.

The referral text is a constant, not a generation: no downstream node may
soften it, and no model is asked to decide how urgent it sounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REFERRAL = (
    "This may be a medical emergency. Stop and seek immediate care now - call your "
    "local emergency number or go to an emergency department. Do not wait to see "
    "whether it improves, and do not rely on this system for triage."
)


@dataclass(frozen=True)
class RedFlag:
    name: str
    patterns: tuple[str, ...]
    presentation: str

    def match(self, text: str) -> str:
        for pattern in self.patterns:
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                return found.group(0)
        return ""


FLAGS: tuple[RedFlag, ...] = (
    RedFlag("acute_coronary_syndrome",
        (r"\bchest (?:pain|pressure|tightness|discomfort)\b",
         r"\bcrushing (?:chest )?pain\b",
         r"\bpain (?:radiat\w+|spreading) (?:to|down|into) (?:my |the )?(?:left )?(?:arm|jaw|neck|shoulder)\b",
         r"\bheart attack\b"),
        "possible acute coronary syndrome"),
    RedFlag("stroke",
        (r"\bface (?:is )?droop\w*\b", r"\bfacial droop\b",
         r"\bsudden(?:ly)? (?:weak|numb)\w*\b",
         r"\bweakness (?:on )?one side\b", r"\bone[- ]sided weakness\b",
         r"\bslurr\w+ speech\b", r"\bcan'?t speak\b", r"\bsudden confusion\b",
         r"\bstroke\b"),
        "possible stroke"),
    RedFlag("subarachnoid_haemorrhage",
        (r"\bthunderclap headache\b", r"\bworst headache of my life\b",
         r"\bsudden(?:ly)? severe headache\b"),
        "possible subarachnoid haemorrhage"),
    RedFlag("anaphylaxis",
        (r"\banaphyla\w+\b", r"\bthroat (?:is )?clos\w+\b", r"\btongue swell\w*\b",
         r"\btrouble breathing after\b", r"\bcan'?t breathe\b"),
        "possible anaphylaxis"),
    RedFlag("suicidal_ideation",
        (r"\bkill myself\b", r"\bend my life\b", r"\bsuicid\w+\b",
         r"\bwant to die\b", r"\bharm myself\b", r"\bself[- ]harm\b"),
        "risk of self-harm"),
    RedFlag("sepsis",
        (r"\bsepsis\b", r"\bseptic\b",
         r"\b(?:high )?fever with (?:confusion|rigors|shaking chills)\b",
         r"\bconfus\w+ and (?:feverish|hot)\b"),
        "possible sepsis"),
    RedFlag("pulmonary_embolism",
        (r"\bsudden(?:ly)? short(?:ness)? of breath\b",
         r"\bcough\w* (?:up )?blood\b", r"\bhaemoptysis\b", r"\bhemoptysis\b",
         r"\bpulmonary embol\w+\b"),
        "possible pulmonary embolism"),
    RedFlag("gi_bleed",
        (r"\bvomit\w* blood\b", r"\bblack tarry stool\w*\b", r"\bmelaena\b", r"\bmelena\b"),
        "possible gastrointestinal bleed"),
    RedFlag("testicular_torsion",
        (r"\bsudden(?:ly)? severe testic\w+ pain\b", r"\btesticular torsion\b"),
        "possible testicular torsion"),
    RedFlag("meningitis",
        (r"\bstiff neck (?:and|with) (?:fever|rash)\b", r"\bmeningitis\b",
         r"\brash that does(?:n'?t| not) fade\b"),
        "possible meningitis"),
    RedFlag("overdose",
        (r"\boverdos\w+\b", r"\btook (?:too many|a whole bottle)\b",
         r"\bpoison\w*\b"),
        "possible overdose or poisoning"),
    RedFlag("obstetric_emergency",
        (r"\bheavy bleeding (?:while |during )?pregnan\w+\b",
         r"\bpregnan\w+ (?:and )?severe abdominal pain\b"),
        "possible obstetric emergency"),
)


@dataclass(frozen=True)
class TriageResult:
    triggered: bool
    flag: str = ""
    presentation: str = ""
    matched: str = ""

    @property
    def message(self) -> str:
        if not self.triggered:
            return ""
        return f"{REFERRAL}\n\nDetected: {self.presentation} (matched {self.matched!r})."


def triage(text: str) -> TriageResult:
    for flag in FLAGS:
        matched = flag.match(text)
        if matched:
            return TriageResult(True, flag.name, flag.presentation, matched)
    return TriageResult(False)
