"""Shared state between plan nodes. Thread-safe, because layers run in parallel."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from medassist.core.ids import ChunkId, NodeId
from medassist.core.models import AgentResult, Claim, PatientProfile


@dataclass
class Blackboard:
    question: str
    profile: PatientProfile | None = None
    _results: dict[NodeId, AgentResult] = field(default_factory=dict)
    _facts: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def put(self, node_id: NodeId, result: AgentResult) -> None:
        with self._lock:
            self._results[node_id] = result

    def get(self, node_id: NodeId) -> AgentResult | None:
        with self._lock:
            return self._results.get(node_id)

    def results(self) -> list[AgentResult]:
        with self._lock:
            return list(self._results.values())

    def upstream(self, node_ids: tuple[NodeId, ...]) -> list[AgentResult]:
        with self._lock:
            return [self._results[n] for n in node_ids if n in self._results]

    def claims(self) -> list[Claim]:
        return [c for r in self.results() for c in r.claims]

    def context_ids(self) -> list[ChunkId]:
        """Every chunk any node selected, de-duplicated, order preserved.

        The gate checks a claim against the union: a claim from the pharmacology
        node may legitimately cite a chunk the evidence node retrieved.
        """
        seen: list[ChunkId] = []
        for result in self.results():
            if result.retrieval is None:
                continue
            for chunk_id in result.retrieval.selected:
                if chunk_id not in seen:
                    seen.append(chunk_id)
        return seen

    def set_fact(self, key: str, value: Any) -> None:
        with self._lock:
            self._facts[key] = value

    def fact(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._facts.get(key, default)
