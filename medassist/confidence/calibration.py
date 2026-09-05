"""Calibration and selective prediction.

A confidence number is only useful if 0.8 means *right about 80% of the time*.
This module measures whether that holds, and picks the abstention threshold
from the resulting curve rather than from taste.

The operationally important metric is **selective risk**: if we abstain on the
least-confident 20%, how much does accuracy improve on the remaining 80%? If
the answer is "not at all", the confidence signal is worthless however good its
ECE looks - it is well-calibrated noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.accuracy)

    @property
    def direction(self) -> str:
        if self.count == 0:
            return "empty"
        if self.mean_confidence > self.accuracy:
            return "overconfident"
        return "underconfident" if self.mean_confidence < self.accuracy else "calibrated"


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    bins: list[Bin]
    n: int

    def explain(self) -> str:
        worst = max((b for b in self.bins if b.count), key=lambda b: b.gap, default=None)
        detail = (
            f"worst bin [{worst.lower:.1f},{worst.upper:.1f}) "
            f"{worst.direction} by {worst.gap:.2f} (n={worst.count})"
            if worst else "no populated bins"
        )
        return f"ECE={self.ece:.3f} MCE={self.mce:.3f} Brier={self.brier:.3f} n={self.n}; {detail}"


def _validate(confidences: list[float], correct: list[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and outcomes must be the same length")
    if any(not 0.0 <= c <= 1.0 for c in confidences):
        raise ValueError("confidences must lie in [0, 1]")


def calibration(confidences: list[float], correct: list[bool], bins: int = 10) -> CalibrationReport:
    """Expected and maximum calibration error, plus the Brier score."""
    _validate(confidences, correct)
    n = len(confidences)
    if n == 0:
        return CalibrationReport(0.0, 0.0, 0.0, [], 0)

    edges = [i / bins for i in range(bins + 1)]
    out: list[Bin] = []
    ece = 0.0
    mce = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        # The final bin is closed so a confidence of exactly 1.0 is counted.
        members = [
            (c, ok) for c, ok in zip(confidences, correct, strict=True)
            if (low <= c < high) or (i == bins - 1 and c == 1.0)
        ]
        if not members:
            out.append(Bin(low, high, 0, 0.0, 0.0))
            continue
        mean_conf = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, ok in members if ok) / len(members)
        out.append(Bin(low, high, len(members), mean_conf, accuracy))
        gap = abs(mean_conf - accuracy)
        ece += (len(members) / n) * gap
        mce = max(mce, gap)

    brier = sum(
        (c - (1.0 if ok else 0.0)) ** 2
        for c, ok in zip(confidences, correct, strict=True)
    ) / n
    return CalibrationReport(ece=ece, mce=mce, brier=brier, bins=out, n=n)


@dataclass(frozen=True)
class CoveragePoint:
    threshold: float
    coverage: float      # fraction answered
    risk: float          # error rate among those answered
    accuracy: float

    def describe(self) -> str:
        return (
            f"t={self.threshold:.2f} coverage={self.coverage:.0%} "
            f"accuracy={self.accuracy:.0%} risk={self.risk:.0%}"
        )


@dataclass(frozen=True)
class SelectiveReport:
    points: list[CoveragePoint]
    aurc: float
    base_accuracy: float

    def at_coverage(self, target: float) -> CoveragePoint | None:
        """Accuracy at the operating point that answers ~``target`` of questions.

        The point with the *smallest* coverage still at or above the target -
        not the best-scoring point among all of them. Taking the maximum over
        every qualifying threshold cherry-picks across dozens of noisy
        estimates, and on a purely random signal reliably returns a point that
        beats base accuracy by chance. That made an uninformative confidence
        score look useful, which is the one conclusion this method exists to
        prevent.
        """
        eligible = [p for p in self.points if p.coverage >= target]
        return min(eligible, key=lambda p: p.coverage) if eligible else None

    @property
    def useful(self) -> bool:
        """Does abstaining actually help?

        If accuracy at 70% coverage is no better than answering everything, the
        confidence signal carries no ordering information and abstention is
        just lost coverage.
        """
        point = self.at_coverage(0.7)
        return bool(point and point.accuracy > self.base_accuracy + 0.01)


def risk_coverage(confidences: list[float], correct: list[bool]) -> SelectiveReport:
    """Accuracy and risk as the abstention threshold sweeps upward."""
    _validate(confidences, correct)
    n = len(confidences)
    if n == 0:
        return SelectiveReport([], 0.0, 0.0)

    base_accuracy = sum(1 for ok in correct if ok) / n
    thresholds = sorted({0.0, *confidences, 1.0})
    points: list[CoveragePoint] = []
    for threshold in thresholds:
        answered = [ok for c, ok in zip(confidences, correct, strict=True) if c >= threshold]
        if not answered:
            continue
        accuracy = sum(1 for ok in answered if ok) / len(answered)
        points.append(
            CoveragePoint(
                threshold=threshold, coverage=len(answered) / n,
                risk=1.0 - accuracy, accuracy=accuracy,
            )
        )

    # AURC by the trapezoid rule over coverage. Lower is better: it is the
    # average risk incurred across every operating point.
    ordered = sorted(points, key=lambda p: p.coverage)
    aurc = 0.0
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        width = later.coverage - earlier.coverage
        aurc += width * (earlier.risk + later.risk) / 2
    span = ordered[-1].coverage - ordered[0].coverage if len(ordered) > 1 else 0.0
    aurc = aurc / span if span > 0 else (ordered[0].risk if ordered else 0.0)

    return SelectiveReport(points=points, aurc=aurc, base_accuracy=base_accuracy)


def select_threshold(
    confidences: list[float],
    correct: list[bool],
    *,
    target_risk: float = 0.1,
    min_coverage: float = 0.5,
) -> tuple[float, CoveragePoint | None]:
    """Lowest threshold meeting a risk target without collapsing coverage.

    Returning the *lowest* qualifying threshold is deliberate: among thresholds
    that all hit the risk target, the one that answers the most questions is
    the right operating point. Abstaining more than necessary is a real cost
    (§08.3 over-refusal), not free safety.
    """
    report = risk_coverage(confidences, correct)
    qualifying = [
        p for p in report.points if p.risk <= target_risk and p.coverage >= min_coverage
    ]
    if not qualifying:
        # No operating point satisfies both. Fail toward silence and say so.
        return 1.0, None
    best = min(qualifying, key=lambda p: p.threshold)
    return best.threshold, best


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - not the normal approximation.

    n is small here and p sits near 1, exactly where the normal approximation
    produces intervals that extend past 1.0 and understate uncertainty.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))
