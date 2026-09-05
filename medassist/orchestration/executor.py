"""The DAG executor.

Independent nodes run concurrently; budgets subdivide rather than reset; and
every failure resolves into a recorded ``TerminationReason`` rather than an
exception escaping to the caller.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from medassist.core.enums import NodeState, TerminationReason
from medassist.core.ids import NodeId
from medassist.core.models import AgentResult, Budget, Usage
from medassist.orchestration.blackboard import Blackboard
from medassist.orchestration.plan import Plan, PlanNode


@dataclass
class RunResult:
    results: list[AgentResult]
    usage: Usage
    termination: TerminationReason
    took_s: float = 0.0
    skipped: list[NodeId] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.termination is TerminationReason.COMPLETED


class Executor:
    def __init__(
        self,
        agents: dict[str, object],
        *,
        max_workers: int = 4,
        clock=time.monotonic,
    ) -> None:
        self.agents = agents
        # Concurrency is capped by the provider's rate limit, not by the DAG's
        # width. A plan fanning out to 12 nodes against a 30 RPM limit produces
        # 12 simultaneous 429s and a retry storm.
        self.max_workers = max(1, max_workers)
        self.clock = clock

    def run(self, plan: Plan, board: Blackboard, budget: Budget) -> RunResult:
        started = self.clock()
        usage = Usage()
        skipped: list[NodeId] = []
        termination = TerminationReason.COMPLETED
        detail = ""

        for layer in plan.layers():
            # Budgets are checked BEFORE dispatch, never after. A budget checked
            # afterwards has already spent what it exists to prevent.
            stop, reason, why = self._exhausted(usage, budget, started)
            if stop:
                skipped.extend(n.id for n in layer)
                termination, detail = reason, why
                continue

            runnable = [n for n in layer if self._deps_ok(n, board)]
            for node in layer:
                if node not in runnable:
                    skipped.append(node.id)

            if not runnable:
                continue

            layer_results = self._run_layer(runnable, plan, board, budget)
            for node, result in layer_results:
                board.put(node.id, result)
                usage.absorb(result.usage)
                if result.state is NodeState.FAILED:
                    usage.agent_errors += 1

            if usage.agent_errors > budget.max_agent_errors:
                termination = TerminationReason.AGENT_ERROR
                detail = f"{usage.agent_errors} agent errors exceeded the limit"
                break

        usage.wall_time_s = self.clock() - started
        return RunResult(
            results=board.results(), usage=usage, termination=termination,
            took_s=usage.wall_time_s, skipped=skipped, detail=detail,
        )

    def _run_layer(
        self, layer: list[PlanNode], plan: Plan, board: Blackboard, budget: Budget
    ) -> list[tuple[PlanNode, AgentResult]]:
        from medassist.agents.base import NodeCall

        def invoke(node: PlanNode) -> tuple[PlanNode, AgentResult]:
            agent = self.agents.get(node.agent)
            if agent is None:
                return node, AgentResult(
                    agent=node.agent, node_id=node.id, state=NodeState.FAILED,
                    error=f"no agent registered as {node.agent!r}",
                )
            call = NodeCall(
                node_id=node.id,
                capability=node.capability,
                inputs=dict(node.inputs),
                budget=budget.subdivide(node.budget_share),
                tools=plan.tool_surface(node.id),
                upstream=board.upstream(node.depends_on),
            )
            try:
                return node, agent.run(call, board)  # type: ignore[attr-defined]
            except Exception as exc:
                # An agent failure is a recorded node state, not an exception
                # that aborts the run: sibling nodes may still produce a usable
                # answer, and whether the graph degrades gracefully is itself
                # worth observing.
                return node, AgentResult(
                    agent=node.agent, node_id=node.id, state=NodeState.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )

        if len(layer) == 1:
            return [invoke(layer[0])]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(layer))) as pool:
            return list(pool.map(invoke, layer))

    @staticmethod
    def _deps_ok(node: PlanNode, board: Blackboard) -> bool:
        for dependency in node.depends_on:
            result = board.get(dependency)
            if result is None or result.state is not NodeState.SUCCEEDED:
                return False
        return True

    def _exhausted(
        self, usage: Usage, budget: Budget, started: float
    ) -> tuple[bool, TerminationReason, str]:
        if usage.steps >= budget.max_steps:
            return True, TerminationReason.MAX_STEPS, f"{usage.steps} steps"
        if usage.total_tokens >= budget.max_tokens:
            return True, TerminationReason.MAX_TOKENS, f"{usage.total_tokens} tokens"
        if usage.cost_usd >= budget.max_cost_usd:
            return True, TerminationReason.MAX_COST, f"${usage.cost_usd:.4f}"
        if self.clock() - started >= budget.max_wall_time_s:
            return True, TerminationReason.MAX_WALL_TIME, f"{self.clock() - started:.1f}s"
        return False, TerminationReason.COMPLETED, ""
