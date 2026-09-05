"""Confidence is computed from observable signals, never asked of a model.

Asking a model for its confidence measures its fluency at producing confident
prose. Every factor below is something the runtime already observed: what was
retrieved, what survived verification, whether sources agreed, how old they
were. The number is then only as good as its calibration (§06.6), which is
measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medassist.core.enums import ClaimDecision, SourceKind
from medassist.core.ids import ChunkId
from medassist.core.models import Chunk

#: Study designs, strongest first. Read from PubMed's curated PublicationType
#: rather than inferred from the abstract, which would be a second model call
#: with a second failure mode.
_DESIGN_WEIGHT: dict[str, float] = {
    "meta-analysis": 1.0,
    "systematic review": 1.0,
    "randomized controlled trial": 0.9,
    "clinical trial": 0.8,
    "review": 0.6,
    "observational study": 0.55,
    "case reports": 0.35,
    "comment": 0.2,
    "editorial": 0.2,
}

_AUTHORITY: dict[SourceKind, float] = {
    SourceKind.FDA_LABEL: 1.0,
    SourceKind.MEDLINEPLUS: 0.7,
    SourceKind.PUBMED_ABSTRACT: 0.7,
    SourceKind.UNKNOWN: 0.3,
}


@dataclass(frozen=True)
class ConfidenceWeights:
    """Exponents, not multipliers.

    A factor with weight 0 contributes 1.0 and drops out; a factor with weight
    2 penalises twice as hard in log space. This keeps the product bounded in
    [0, 1] whatever the weights, which a weighted sum does not.
    """

    evidence_quality: float = 1.0
    evidence_coverage: float = 1.6
    source_agreement: float = 1.0
    retrieval_relevance: float = 0.8
    model_consistency: float = 1.0
    temporal_validity: float = 0.6


@dataclass
class ConfidenceReport:
    value: float
    factors: dict[str, float] = field(default_factory=dict)
    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)

    @property
    def weakest(self) -> str:
        """The factor dragging the score down.

        A single 0.62 is unactionable; ``evidence_coverage=0.55`` driving it is
        a bug report, which is why factors are stored individually.
        """
        return min(self.factors, key=lambda k: self.factors[k]) if self.factors else ""

    def explain(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.factors.items()))
        return f"confidence={self.value:.2f} ({parts}); weakest: {self.weakest}"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def evidence_quality(chunks: list[Chunk]) -> float:
    """Source authority blended with study design."""
    if not chunks:
        return 0.0
    scores: list[float] = []
    for chunk in chunks:
        authority = _AUTHORITY.get(chunk.source, 0.3)
        design = 1.0
        types = [str(t).lower() for t in chunk.meta.get("publication_types", [])]
        if types:
            design = max((_DESIGN_WEIGHT.get(t, 0.5) for t in types), default=0.5)
        scores.append(authority * design)
    # The best available source dominates, but a corpus of only weak sources
    # should not score as highly as one strong source - so blend max with mean.
    return _clamp(0.7 * max(scores) + 0.3 * (sum(scores) / len(scores)))


def evidence_coverage(judgements: list) -> float:
    """Fraction of claims that survived verification intact."""
    if not judgements:
        return 0.0
    retained = sum(1 for j in judgements if j.decision is ClaimDecision.RETAIN)
    qualified = sum(1 for j in judgements if j.decision is ClaimDecision.QUALIFY)
    # A qualified claim is real but caveated; half credit.
    return _clamp((retained + 0.5 * qualified) / len(judgements))


def source_agreement(conflicts: int, total_claims: int) -> float:
    if total_claims <= 0:
        return 1.0
    return _clamp(1.0 - conflicts / total_claims)


def retrieval_relevance(scores: dict[str, float], selected: list[ChunkId]) -> float:
    """Mean rerank score of what actually reached the context window."""
    if not selected:
        return 0.0
    values = [scores.get(str(cid), 0.0) for cid in selected]
    return _clamp(sum(values) / len(values))


def model_consistency(samples: list[list[str]]) -> float:
    """Agreement across repeated samples of the same question.

    A model that answers differently each time is uncertain regardless of how
    it phrases itself. Costs k times as much, so it runs on high-risk
    capabilities only.
    """
    if len(samples) < 2:
        return 1.0  # not measured; must not penalise
    signatures = [{" ".join(sorted(c.lower().split()))[:120] for c in s} for s in samples]
    reference = signatures[0]
    if not reference:
        return 0.0
    overlaps = [
        len(reference & other) / len(reference | other) if (reference | other) else 0.0
        for other in signatures[1:]
    ]
    return _clamp(sum(overlaps) / len(overlaps))


def temporal_validity(years: list[int], now: int = 2026, horizon: int = 8) -> float:
    """How current the evidence is, relative to the intent's horizon."""
    usable = [y for y in years if y > 1900]
    if not usable:
        return 0.6  # unknown age is not the same as old, nor as fresh
    ages = [max(0, now - y) for y in usable]
    freshest = min(ages)
    return _clamp(1.0 - freshest / (horizon * 2))


def combine(factors: dict[str, float], weights: ConfidenceWeights | None = None) -> ConfidenceReport:
    weights = weights or ConfidenceWeights()
    value = 1.0
    for name, score in factors.items():
        exponent = getattr(weights, name, 1.0)
        if exponent <= 0:
            continue
        # A zero on any weighted factor drives the product to zero, which is
        # intended: no evidence coverage means no confidence, whatever else held.
        value *= max(score, 1e-6) ** exponent
    return ConfidenceReport(value=_clamp(value), factors=dict(factors), weights=weights)
