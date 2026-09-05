"""Plans are DAGs, validated before a single node runs.

The supervisor emits a plan; the runtime executes it. A router selects among
**declared** edges only - fed a node naming an unregistered agent, or an edge to
a node that does not exist, construction fails rather than the runtime doing its
best. This is the most important property of the orchestrator: a model that can
name arbitrary next steps is a model with arbitrary privileges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medassist.capabilities.registry import UnknownCapability
from medassist.capabilities.registry import get as get_capability
from medassist.core.ids import NodeId


class PlanInvalid(ValueError):
    """A plan that cannot be executed safely. Never repaired, always rejected."""


@dataclass(frozen=True)
class PlanNode:
    id: NodeId
    agent: str
    capability: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[NodeId, ...] = ()
    #: Share of the *remaining* parent budget. Shares of nodes that run in
    #: parallel must sum to <= 1.0; see Plan._validate_budget.
    budget_share: float = 1.0


@dataclass(frozen=True)
class Plan:
    nodes: tuple[PlanNode, ...]

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_edges()
        self._validate_acyclic()
        self._validate_budget()

    # -- validation --------------------------------------------------------

    def _validate_identity(self) -> None:
        if not self.nodes:
            raise PlanInvalid("a plan must contain at least one node")
        seen: set[NodeId] = set()
        for node in self.nodes:
            if node.id in seen:
                raise PlanInvalid(f"duplicate node id {node.id}")
            seen.add(node.id)
            try:
                get_capability(node.capability)
            except UnknownCapability as exc:
                raise PlanInvalid(str(exc)) from None
            if not 0 < node.budget_share <= 1:
                raise PlanInvalid(
                    f"node {node.id}: budget_share must be in (0, 1], got {node.budget_share}"
                )

    def _validate_edges(self) -> None:
        known = {n.id for n in self.nodes}
        for node in self.nodes:
            for dependency in node.depends_on:
                if dependency not in known:
                    raise PlanInvalid(
                        f"node {node.id} depends on {dependency}, which is not in the plan. "
                        "The runtime does not invent edges."
                    )
                if dependency == node.id:
                    raise PlanInvalid(f"node {node.id} depends on itself")

    def _validate_acyclic(self) -> None:
        # Kahn's algorithm; whatever is left over is inside a cycle.
        indegree = {n.id: len(n.depends_on) for n in self.nodes}
        dependents: dict[NodeId, list[NodeId]] = {n.id: [] for n in self.nodes}
        for node in self.nodes:
            for dependency in node.depends_on:
                dependents[dependency].append(node.id)

        queue = [nid for nid, degree in indegree.items() if degree == 0]
        if not queue:
            raise PlanInvalid("every node has a dependency: the plan is cyclic")
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for child in dependents[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(self.nodes):
            raise PlanInvalid("the plan contains a cycle")

    def _validate_budget(self) -> None:
        """Parallel shares must sum to <= 1.0.

        Router branches are deliberately exempt elsewhere - exactly one runs, so
        requiring a 4-way route's shares to sum to 1.0 would force every branch
        down to 0.25 for no reason. Here every node in a layer *does* run.
        """
        for depth, layer in enumerate(self.layers()):
            if len(layer) < 2:
                continue
            total = sum(n.budget_share for n in layer)
            if total > 1.0 + 1e-9:
                raise PlanInvalid(
                    f"layer {depth} runs {len(layer)} nodes in parallel whose budget "
                    f"shares sum to {total:.2f}; a budget that can be exceeded is not a budget"
                )

    # -- structure ---------------------------------------------------------

    def by_id(self) -> dict[NodeId, PlanNode]:
        return {n.id: n for n in self.nodes}

    def layers(self) -> list[list[PlanNode]]:
        """Topological levels. Nodes in one layer are independent, so they run
        concurrently."""
        remaining = {n.id: set(n.depends_on) for n in self.nodes}
        index = self.by_id()
        done: set[NodeId] = set()
        out: list[list[PlanNode]] = []
        while remaining:
            ready = sorted(
                (nid for nid, deps in remaining.items() if deps <= done), key=str
            )
            if not ready:  # pragma: no cover - _validate_acyclic rejects this first
                raise PlanInvalid("the plan contains a cycle")
            out.append([index[nid] for nid in ready])
            done.update(ready)
            for nid in ready:
                remaining.pop(nid)
        return out

    def ancestors(self, node_id: NodeId) -> set[NodeId]:
        index = self.by_id()
        seen: set[NodeId] = set()
        stack = list(index[node_id].depends_on)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(index[current].depends_on)
        return seen

    def tool_surface(self, node_id: NodeId) -> frozenset[str]:
        """A node's tools are the intersection with every ancestor's.

        Delegation cannot escalate privilege: a child of a node restricted to
        ``retrieve`` cannot obtain ``interaction_table`` by declaring a
        capability that has it.
        """
        index = self.by_id()
        surface = get_capability(index[node_id].capability).allowed_tools
        for ancestor in self.ancestors(node_id):
            surface = surface & get_capability(index[ancestor].capability).allowed_tools
        return frozenset(surface)
