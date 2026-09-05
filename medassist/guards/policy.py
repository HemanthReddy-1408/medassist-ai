"""A deny-by-default policy engine.

Rules are data evaluated in code, and every decision records the rule that
produced it. "Denied" without the rule is not auditable, and an unauditable
policy engine is decoration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Effect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyContext:
    capability: str
    risk: str = "low"
    role: str = "patient"
    tool: str = ""
    confidence: float = 1.0
    sources: int = 1
    min_sources: int = 1
    has_contradiction: bool = False
    red_flag: bool = False
    injection_detected: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True)
class Rule:
    name: str
    effect: Effect
    when: Callable[[PolicyContext], bool]
    reason: str

    def applies(self, ctx: PolicyContext) -> bool:
        return self.when(ctx)


@dataclass(frozen=True)
class Decision:
    effect: Effect
    rule: str
    reason: str


#: Ordered. The first matching rule wins, so denials precede approvals and
#: approvals precede allows.
RULES: tuple[Rule, ...] = (
    Rule("red_flag_blocks_everything", Effect.DENY,
         lambda c: c.red_flag,
         "an emergency presentation was detected; no answer is released"),
    Rule("contradicted_claim_blocks", Effect.DENY,
         lambda c: c.has_contradiction,
         "a cited source contradicts its own claim"),
    Rule("insufficient_independent_sources", Effect.DENY,
         lambda c: c.sources < c.min_sources,
         "the capability requires more independent sources than were retrieved"),
    Rule("unauthorized_role_for_patient_data", Effect.DENY,
         lambda c: c.tool == "patient_data" and c.role not in {"clinician", "admin"},
         "this role may not read patient data"),
    Rule("high_risk_low_confidence_needs_review", Effect.REQUIRE_APPROVAL,
         lambda c: c.risk == "high" and c.confidence < 0.75,
         "a high-risk answer below the confidence threshold escalates"),
    Rule("injection_detected_needs_review", Effect.REQUIRE_APPROVAL,
         lambda c: c.injection_detected,
         "injection patterns were found in retrieved context"),
    Rule("high_risk_capability_escalates", Effect.REQUIRE_APPROVAL,
         lambda c: c.risk == "high",
         "this capability escalates on any positive finding"),
)


@dataclass
class PolicyEngine:
    rules: tuple[Rule, ...] = field(default_factory=lambda: RULES)

    def evaluate(self, ctx: PolicyContext) -> Decision:
        for rule in self.rules:
            if rule.applies(ctx):
                return Decision(effect=rule.effect, rule=rule.name, reason=rule.reason)
        # Deny-by-default applies to *actions*; a fully-checked answer with no
        # rule against it is allowed, and that allowance is still recorded.
        return Decision(Effect.ALLOW, "no_rule_matched", "no policy rule applied")

    def authorize_tool(self, tool: str, surface: frozenset[str], ctx: PolicyContext) -> Decision:
        """Tool authorization *is* deny-by-default: absent from the surface, denied."""
        if tool not in surface:
            return Decision(
                Effect.DENY, "tool_not_in_surface",
                f"{tool!r} is not in the authorized surface {sorted(surface)}",
            )
        return self.evaluate(ctx)
