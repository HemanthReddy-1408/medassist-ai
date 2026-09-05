"""Concrete agents.

Each does real work against the retriever and the model gateway. They are
deliberately small: the interesting behaviour lives in the contracts the
runtime enforces around them, not in prompt cleverness inside them.
"""

from __future__ import annotations

import time

from medassist.agents.base import Agent, AgentContract, NodeCall
from medassist.core.enums import NodeState
from medassist.core.models import AgentResult, Claim, Usage
from medassist.generate import generate_claims
from medassist.orchestration.blackboard import Blackboard


class RetrievalAgent:
    """Gathers evidence. Produces no claims - only context and a trace."""

    contract = AgentContract(
        name="retrieval",
        purpose="retrieve evidence for a sub-question",
        tools=frozenset({"retrieve", "expand_query"}),
    )

    def __init__(self, retriever) -> None:
        self.retriever = retriever

    def run(self, call: NodeCall, board: Blackboard) -> AgentResult:
        call.authorize("retrieve")
        query = str(call.inputs.get("query") or board.question)
        intent = str(call.inputs.get("intent", ""))
        result = self.retriever.retrieve(query, intent=intent)
        return AgentResult(
            agent=self.contract.name,
            node_id=call.node_id,
            retrieval=result.trace,
            notes=f"{len(result.chunk_ids)} chunks for {query!r}",
            usage=Usage(steps=1),
            state=NodeState.SUCCEEDED,
        )


class AnswerAgent:
    """Turns retrieved context into claims with citation handles."""

    contract = AgentContract(
        name="answer", purpose="produce cited claims", tools=frozenset({"retrieve"})
    )

    def __init__(self, client, retriever) -> None:
        self.client = client
        self.retriever = retriever

    def run(self, call: NodeCall, board: Blackboard) -> AgentResult:
        call.authorize("retrieve")
        started = time.perf_counter()

        chunk_ids = [
            cid
            for upstream in call.upstream
            if upstream.retrieval is not None
            for cid in upstream.retrieval.selected
        ]
        if not chunk_ids:
            # No upstream evidence: abstain rather than answer from parametric
            # memory. This is the contract's on_insufficient_evidence, enforced.
            return AgentResult(
                agent=self.contract.name, node_id=call.node_id,
                notes="no upstream evidence; abstained",
                usage=Usage(steps=1, wall_time_s=time.perf_counter() - started),
                state=NodeState.SUCCEEDED,
            )

        seen: list = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.append(cid)
        from medassist.index.rerank import format_context

        context = format_context(seen, self.retriever.chunks)
        handles = {f"C{i}": cid for i, cid in enumerate(seen, start=1)}
        question = str(call.inputs.get("question") or board.question)

        claims, usage, insufficient = generate_claims(
            self.client, question, context, handles, self.retriever.chunks,
            profile=board.profile,
        )
        usage.steps = max(usage.steps, 1)
        usage.wall_time_s = time.perf_counter() - started
        return AgentResult(
            agent=self.contract.name, node_id=call.node_id, claims=claims,
            notes="insufficient evidence declared" if insufficient else f"{len(claims)} claims",
            usage=usage, state=NodeState.SUCCEEDED,
        )


class SynthesizerAgent:
    """Merges upstream claims, dropping near-duplicates.

    Fan-out produces the same fact from several branches. Deduplicating here
    keeps the gate from spending an entailment slot on each restatement, and
    keeps the answer from repeating itself.
    """

    contract = AgentContract(
        name="synthesizer", purpose="merge and deduplicate claims", tools=frozenset()
    )

    def run(self, call: NodeCall, board: Blackboard) -> AgentResult:
        merged: list[Claim] = []
        signatures: set[str] = set()
        for upstream in call.upstream:
            for claim in upstream.claims:
                signature = " ".join(sorted(claim.text.lower().split()))[:180]
                if signature in signatures:
                    continue
                signatures.add(signature)
                merged.append(claim)
        return AgentResult(
            agent=self.contract.name, node_id=call.node_id, claims=merged,
            notes=f"merged {len(merged)} claims from {len(call.upstream)} node(s)",
            usage=Usage(steps=1), state=NodeState.SUCCEEDED,
        )


def default_agents(client, retriever) -> dict[str, Agent]:
    return {
        "retrieval": RetrievalAgent(retriever),
        "answer": AnswerAgent(client, retriever),
        "synthesizer": SynthesizerAgent(),
    }
