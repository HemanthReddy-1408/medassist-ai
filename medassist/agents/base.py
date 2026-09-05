"""Agent contracts.

Every agent declares what it may do, and the runtime enforces it. The contract
is data the executor reads, not prose inside a prompt that nothing checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from medassist.core.ids import NodeId
from medassist.core.models import AgentResult, Budget
from medassist.orchestration.blackboard import Blackboard


class ToolNotAuthorized(PermissionError):
    """An agent asked for a tool outside its node's surface.

    Raised by the runtime, not decided by the agent - which is what makes the
    closed tool set a containment boundary rather than a convention.
    """


@dataclass(frozen=True)
class AgentContract:
    name: str
    purpose: str
    tools: frozenset[str]
    max_iterations: int = 6
    #: The default behaviour of every LLM agent is to answer anyway.
    #: Abstention has to be explicit, or it never happens.
    on_insufficient_evidence: str = "abstain"


@dataclass
class NodeCall:
    """What the executor hands an agent: identity, inputs, and its limits."""

    node_id: NodeId
    capability: str
    inputs: dict
    budget: Budget
    tools: frozenset[str]
    upstream: list[AgentResult] = field(default_factory=list)

    def authorize(self, tool: str) -> None:
        if tool not in self.tools:
            raise ToolNotAuthorized(
                f"{tool!r} is not in this node's surface {sorted(self.tools)}. "
                "Delegation cannot escalate privilege."
            )


class Agent(Protocol):
    contract: AgentContract

    def run(self, call: NodeCall, board: Blackboard) -> AgentResult: ...
