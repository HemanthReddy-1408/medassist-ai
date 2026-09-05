"""The capability registry.

Twenty independently-written agents share no invariants. Every one re-decides
what it may call, how long it may run, and what counts as done - and those
decisions live in prose inside twenty prompts, where nothing can enforce them.

A capability is instead a **declared, typed, authorizable unit of work**. The
registry is the single place that answers: may this run, with what tools, for
how long, and what must be true before its output is released.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from medassist.core.models import Budget


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    risk: RiskLevel
    #: A closed set, not a suggestion. An agent proposing a tool outside it
    #: fails at the registry rather than at the model's discretion - which is
    #: the containment layer that actually holds against prompt injection,
    #: because it does not depend on recognising the attack.
    allowed_tools: frozenset[str]
    evidence_required: bool = True
    min_independent_sources: int = 1
    requires_human_review: bool = False
    #: Gate strictness, per capability. A dosing answer should abstain sooner
    #: than a general-overview answer.
    min_supported_fraction: float = 0.6
    budget: Budget = field(default_factory=Budget)
    fallback: str = "abstain"

    def authorizes(self, tool: str) -> bool:
        return tool in self.allowed_tools


_CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="evidence_qa",
        description="Answer a clinical question from retrieved literature and labels.",
        risk=RiskLevel.LOW,
        allowed_tools=frozenset({"retrieve", "expand_query"}),
        min_independent_sources=1,
        budget=Budget(max_steps=8, max_cost_usd=0.05, max_wall_time_s=45.0),
    ),
    Capability(
        name="literature_synthesis",
        description="Synthesize findings across multiple papers, surfacing disagreement.",
        risk=RiskLevel.LOW,
        allowed_tools=frozenset({"retrieve", "expand_query"}),
        min_independent_sources=3,
        budget=Budget(max_steps=16, max_cost_usd=0.12, max_wall_time_s=90.0),
    ),
    Capability(
        name="treatment_comparison",
        description="Compare two treatments on efficacy, risk and evidence quality.",
        risk=RiskLevel.MEDIUM,
        allowed_tools=frozenset({"retrieve", "expand_query"}),
        min_independent_sources=2,
        min_supported_fraction=0.7,
        budget=Budget(max_steps=16, max_cost_usd=0.12, max_wall_time_s=90.0),
    ),
    Capability(
        name="report_interpretation",
        description="Interpret a lab or clinical report against reference intervals.",
        risk=RiskLevel.MEDIUM,
        allowed_tools=frozenset({"retrieve", "parse_report"}),
        min_supported_fraction=0.7,
        budget=Budget(max_steps=10, max_cost_usd=0.08, max_wall_time_s=60.0),
    ),
    Capability(
        name="lifestyle_guidance",
        description="Diet and activity guidance, cross-checked against the patient's record.",
        risk=RiskLevel.MEDIUM,
        allowed_tools=frozenset({"retrieve"}),
        min_supported_fraction=0.7,
        budget=Budget(max_steps=10, max_cost_usd=0.08, max_wall_time_s=60.0),
    ),
    Capability(
        name="interaction_check",
        description="Check a drug against the patient's medications and allergies.",
        risk=RiskLevel.HIGH,
        allowed_tools=frozenset({"retrieve", "interaction_table"}),
        min_independent_sources=2,
        # A false negative here is silent: an answer that fails to mention an
        # interaction looks exactly like a correct answer, and no groundedness
        # metric detects it because everything it *does* say is well supported.
        requires_human_review=True,
        min_supported_fraction=0.8,
        budget=Budget(max_steps=12, max_cost_usd=0.10, max_wall_time_s=60.0),
    ),
)

REGISTRY: dict[str, Capability] = {c.name: c for c in _CAPABILITIES}


class UnknownCapability(KeyError):
    """Raised rather than defaulted: an undeclared capability is unauthorized."""


def get(name: str) -> Capability:
    try:
        return REGISTRY[name]
    except KeyError:
        raise UnknownCapability(
            f"{name!r} is not a declared capability. Known: {sorted(REGISTRY)}"
        ) from None


def names() -> list[str]:
    return sorted(REGISTRY)
