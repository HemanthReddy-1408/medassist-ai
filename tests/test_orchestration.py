"""Wave 3: capability registry, plan validation, and the parallel executor."""

from __future__ import annotations

import threading
import time

import pytest

from medassist.agents.base import AgentContract, NodeCall, ToolNotAuthorized
from medassist.capabilities.registry import REGISTRY, RiskLevel, UnknownCapability
from medassist.capabilities.registry import get as get_capability
from medassist.core.enums import NodeState, TerminationReason
from medassist.core.ids import NodeId
from medassist.core.models import AgentResult, Budget, Usage
from medassist.orchestration.blackboard import Blackboard
from medassist.orchestration.executor import Executor
from medassist.orchestration.plan import Plan, PlanInvalid, PlanNode


def node(agent: str, capability: str = "evidence_qa", **kw) -> PlanNode:
    return PlanNode(id=NodeId.new(), agent=agent, capability=capability, **kw)


class RecordingAgent:
    contract = AgentContract(name="rec", purpose="test", tools=frozenset({"retrieve"}))

    def __init__(self, delay: float = 0.0, fail: bool = False, cost: float = 0.0):
        self.delay, self.fail, self.cost = delay, fail, cost
        self.entered: list[float] = []
        self.calls: list[NodeCall] = []
        self._lock = threading.Lock()

    def run(self, call: NodeCall, board: Blackboard) -> AgentResult:
        with self._lock:
            self.entered.append(time.monotonic())
            self.calls.append(call)
        if self.fail:
            raise RuntimeError("agent exploded")
        time.sleep(self.delay)
        return AgentResult(
            agent="rec", node_id=call.node_id,
            usage=Usage(steps=1, cost_usd=self.cost), state=NodeState.SUCCEEDED,
        )


class TestRegistry:
    def test_all_six_capabilities_declared(self):
        assert len(REGISTRY) == 6

    def test_unknown_capability_raises_rather_than_defaulting(self):
        with pytest.raises(UnknownCapability):
            get_capability("delete_everything")

    def test_interaction_check_is_high_risk_and_always_escalates(self):
        """A false negative here is silent - it looks like a correct answer."""
        cap = get_capability("interaction_check")
        assert cap.risk is RiskLevel.HIGH
        assert cap.requires_human_review is True

    def test_higher_risk_capabilities_gate_more_strictly(self):
        assert (
            get_capability("interaction_check").min_supported_fraction
            > get_capability("evidence_qa").min_supported_fraction
        )

    def test_tool_sets_are_closed(self):
        assert not get_capability("evidence_qa").authorizes("interaction_table")

    def test_every_capability_carries_a_budget(self):
        assert all(c.budget.max_cost_usd > 0 for c in REGISTRY.values())


class TestPlanValidation:
    def test_layers_group_independent_nodes(self):
        a = node("retrieval")
        b = node("answer", depends_on=(a.id,), budget_share=0.5)
        c = node("answer", depends_on=(a.id,), budget_share=0.5)
        d = node("synthesizer", depends_on=(b.id, c.id))
        layers = Plan(nodes=(a, b, c, d)).layers()
        assert [len(x) for x in layers] == [1, 2, 1]

    def test_cycle_rejected(self):
        a, b = NodeId.new(), NodeId.new()
        with pytest.raises(PlanInvalid, match="cyclic|cycle"):
            Plan(nodes=(
                PlanNode(id=a, agent="x", capability="evidence_qa", depends_on=(b,)),
                PlanNode(id=b, agent="y", capability="evidence_qa", depends_on=(a,)),
            ))

    def test_edge_to_a_nonexistent_node_rejected(self):
        """The runtime does not invent edges."""
        with pytest.raises(PlanInvalid, match="not in the plan"):
            Plan(nodes=(PlanNode(
                id=NodeId.new(), agent="x", capability="evidence_qa",
                depends_on=(NodeId.new(),),
            ),))

    def test_undeclared_capability_rejected(self):
        with pytest.raises(PlanInvalid, match="not a declared capability"):
            Plan(nodes=(node("x", capability="delete_everything"),))

    def test_self_dependency_rejected(self):
        nid = NodeId.new()
        with pytest.raises(PlanInvalid, match="itself"):
            Plan(nodes=(PlanNode(id=nid, agent="x", capability="evidence_qa", depends_on=(nid,)),))

    def test_duplicate_node_ids_rejected(self):
        nid = NodeId.new()
        with pytest.raises(PlanInvalid, match="duplicate"):
            Plan(nodes=(
                PlanNode(id=nid, agent="a", capability="evidence_qa"),
                PlanNode(id=nid, agent="b", capability="evidence_qa"),
            ))

    def test_empty_plan_rejected(self):
        with pytest.raises(PlanInvalid, match="at least one node"):
            Plan(nodes=())

    def test_parallel_budget_shares_may_not_exceed_the_whole(self):
        with pytest.raises(PlanInvalid, match="sum to"):
            Plan(nodes=(node("a", budget_share=0.8), node("b", budget_share=0.8)))

    def test_sequential_nodes_may_each_take_the_full_share(self):
        a = node("a", budget_share=1.0)
        b = node("b", depends_on=(a.id,), budget_share=1.0)
        assert len(Plan(nodes=(a, b)).layers()) == 2

    @pytest.mark.parametrize("share", [0.0, -0.5, 1.5])
    def test_invalid_share_rejected(self, share):
        with pytest.raises(PlanInvalid, match="budget_share"):
            Plan(nodes=(node("a", budget_share=share),))


class TestPrivilegeNarrowing:
    def test_child_tools_are_the_intersection_with_ancestors(self):
        """Delegation cannot escalate privilege."""
        parent = node("a", capability="lifestyle_guidance")          # {retrieve}
        child = node("b", capability="interaction_check", depends_on=(parent.id,))
        plan = Plan(nodes=(parent, child))
        assert plan.tool_surface(child.id) == frozenset({"retrieve"})
        assert "interaction_table" in get_capability("interaction_check").allowed_tools

    def test_a_root_node_keeps_its_full_surface(self):
        root = node("a", capability="interaction_check")
        assert "interaction_table" in Plan(nodes=(root,)).tool_surface(root.id)

    def test_narrowing_is_transitive_through_the_chain(self):
        a = node("a", capability="interaction_check")
        b = node("b", capability="lifestyle_guidance", depends_on=(a.id,))
        c = node("c", capability="interaction_check", depends_on=(b.id,))
        assert Plan(nodes=(a, b, c)).tool_surface(c.id) == frozenset({"retrieve"})

    def test_unauthorized_tool_raises_at_the_runtime(self):
        call = NodeCall(
            node_id=NodeId.new(), capability="evidence_qa", inputs={},
            budget=Budget(), tools=frozenset({"retrieve"}),
        )
        with pytest.raises(ToolNotAuthorized, match="escalate privilege"):
            call.authorize("interaction_table")


class TestExecutor:
    def test_independent_nodes_run_concurrently(self):
        agent = RecordingAgent(delay=0.05)
        a = node("rec", budget_share=0.3)
        b = node("rec", budget_share=0.3)
        c = node("rec", budget_share=0.3)
        plan = Plan(nodes=(a, b, c))
        result = Executor({"rec": agent}, max_workers=3).run(
            plan, Blackboard(question="q"), Budget()
        )
        assert result.ok
        # Three 50ms nodes in parallel finish well under the 150ms serial floor.
        assert result.took_s < 0.13
        assert len(agent.entered) == 3

    def test_budgets_subdivide_and_do_not_reset(self):
        agent = RecordingAgent()
        a = node("rec", budget_share=0.5)
        b = node("rec", budget_share=0.5)
        Executor({"rec": agent}).run(
            Plan(nodes=(a, b)), Blackboard(question="q"), Budget(max_cost_usd=1.0)
        )
        assert all(call.budget.max_cost_usd == pytest.approx(0.5) for call in agent.calls)

    def test_a_failing_agent_does_not_abort_its_siblings(self):
        ok, bad = RecordingAgent(), RecordingAgent(fail=True)
        a = node("ok", budget_share=0.5)
        b = node("bad", budget_share=0.5)
        result = Executor({"ok": ok, "bad": bad}).run(
            Plan(nodes=(a, b)), Blackboard(question="q"), Budget()
        )
        states = {r.state for r in result.results}
        assert states == {NodeState.SUCCEEDED, NodeState.FAILED}
        assert result.usage.agent_errors == 1

    def test_dependents_of_a_failed_node_are_skipped_not_run(self):
        bad, ok = RecordingAgent(fail=True), RecordingAgent()
        a = node("bad")
        b = node("ok", depends_on=(a.id,))
        result = Executor({"bad": bad, "ok": ok}).run(
            Plan(nodes=(a, b)), Blackboard(question="q"), Budget()
        )
        assert b.id in result.skipped
        assert ok.entered == []

    def test_too_many_agent_errors_terminates_the_run(self):
        bad = RecordingAgent(fail=True)
        nodes = tuple(
            node("bad", depends_on=() if i == 0 else ()) for i in range(5)
        )
        nodes = tuple(
            PlanNode(id=n.id, agent=n.agent, capability=n.capability, budget_share=0.2)
            for n in nodes
        )
        result = Executor({"bad": bad}).run(
            Plan(nodes=nodes), Blackboard(question="q"), Budget(max_agent_errors=2)
        )
        assert result.termination is TerminationReason.AGENT_ERROR

    def test_cost_budget_is_checked_before_dispatch(self):
        """A budget checked afterwards has already spent what it prevents."""
        expensive = RecordingAgent(cost=1.0)
        a = node("rec")
        b = node("rec", depends_on=(a.id,))
        result = Executor({"rec": expensive}).run(
            Plan(nodes=(a, b)), Blackboard(question="q"), Budget(max_cost_usd=0.5)
        )
        assert result.termination is TerminationReason.MAX_COST
        assert b.id in result.skipped
        assert len(expensive.entered) == 1  # the second node never ran

    def test_missing_agent_is_a_failed_node_not_a_crash(self):
        result = Executor({}).run(
            Plan(nodes=(node("nonexistent"),)), Blackboard(question="q"), Budget()
        )
        assert result.results[0].state is NodeState.FAILED
        assert "no agent registered" in result.results[0].error

    def test_upstream_results_reach_the_dependent_node(self):
        agent = RecordingAgent()
        a = node("rec")
        b = node("rec", depends_on=(a.id,))
        Executor({"rec": agent}).run(
            Plan(nodes=(a, b)), Blackboard(question="q"), Budget()
        )
        downstream = next(c for c in agent.calls if c.node_id == b.id)
        assert len(downstream.upstream) == 1


class TestBlackboard:
    def test_context_ids_union_upstream_retrievals_without_duplicates(self):
        from medassist.core.enums import RetrievalStage
        from medassist.core.ids import ChunkId
        from medassist.core.models import RetrievalTrace, StageRecord

        shared, only_a = ChunkId.new(), ChunkId.new()
        board = Blackboard(question="q")

        def trace(ids):
            return RetrievalTrace(
                query="q",
                stages=[StageRecord(stage=RetrievalStage.SELECTED, chunk_ids=ids)],
                selected=ids,
            )

        board.put(NodeId.new(), AgentResult(agent="a", node_id=NodeId.new(), retrieval=trace([shared, only_a])))
        board.put(NodeId.new(), AgentResult(agent="b", node_id=NodeId.new(), retrieval=trace([shared])))
        assert sorted(board.context_ids()) == sorted({shared, only_a})

    def test_writes_are_thread_safe(self):
        board = Blackboard(question="q")

        def write() -> None:
            for _ in range(200):
                board.put(NodeId.new(), AgentResult(agent="a", node_id=NodeId.new()))

        threads = [threading.Thread(target=write) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(board.results()) == 800
