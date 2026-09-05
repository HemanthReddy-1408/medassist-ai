"""The release gate: serving-time admission control for clinical answers.

At serving time there is no gold label and no second chance, so every check
here is reference-free and affordable enough to run in-band.
"""

from medassist.gate.cascade import GateOutcome, GateSpec, ReleaseGate
from medassist.gate.checks import DETERMINISTIC_CHECKS, GateContext
from medassist.gate.decisions import CheckOutcome, ClaimJudgement, GateReason
from medassist.gate.entailment import EntailmentCheck, EntailmentUnavailable

__all__ = [
    "DETERMINISTIC_CHECKS", "CheckOutcome", "ClaimJudgement", "EntailmentCheck",
    "EntailmentUnavailable", "GateContext", "GateOutcome", "GateReason",
    "GateSpec", "ReleaseGate",
]
