"""Decision records.

Recorded not for debugging but because **a gate that cannot explain a refusal
is indistinguishable from one that is broken**. The same record renders to the
user: "why do you believe this?" is answerable by showing the claim, its cited
span, and which checks it passed.

Per-check attribution is the load-bearing field. "Claim 3 was removed" is not
actionable. "Removed by numeric_grounding: 2000 mg appears in no cited span" is
a bug report against the generator, and "removed by relational_safety: vitamin
K x warfarin" is correct behaviour that should never be tuned away.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medassist.core.enums import ClaimDecision, ResponseDecision
from medassist.core.ids import ChunkId, RunId
from medassist.core.models import Chunk, Claim, PatientProfile
from medassist.gate.cascade import GateOutcome


def profile_digest(profile: PatientProfile | None, *, salt: str = "medassist") -> str:
    """A salted hash. Raw age, medications and conditions never enter a record."""
    if profile is None:
        return ""
    material = json.dumps(profile.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(f"{salt}:{material}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ClaimRecord:
    text: str
    decision: ClaimDecision
    firing_check: str
    explanation: str
    citations: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    run_id: RunId
    question: str
    decision: ResponseDecision
    rule: str
    detail: str
    claims: list[ClaimRecord] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)
    corpus_snapshot: str = ""
    subject_model: str = ""
    judge_model: str = ""
    profile_digest: str = ""
    gate_latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    entailment_ran: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def released(self) -> bool:
        return self.decision in (
            ResponseDecision.RELEASE, ResponseDecision.RELEASE_WITH_CAVEAT
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            k: v for k, v in self.__dict__.items() if k != "claims"
        }
        payload["decision"] = self.decision.value
        payload["claims"] = [
            {**c.__dict__, "decision": c.decision.value} for c in self.claims
        ]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def explain(self) -> str:
        """The user-facing rendering. Refusals name the check that refused."""
        lines = [f"{self.decision.value.upper()}: {self.detail or self.rule}"]
        for claim in self.claims:
            mark = {"retain": "kept", "qualify": "caveated", "remove": "removed"}[
                claim.decision.value
            ]
            lines.append(f"  [{mark}] {claim.text[:96]}")
            if claim.decision is not ClaimDecision.RETAIN:
                lines.append(f"          why: {claim.explanation}")
        return "\n".join(lines)


def build_record(
    *,
    run_id: RunId,
    question: str,
    outcome: GateOutcome,
    claims: list[Claim],
    chunks: dict[ChunkId, Chunk],
    context_ids: list[ChunkId],
    profile: PatientProfile | None = None,
    corpus_snapshot: str = "",
    subject_model: str = "",
    judge_model: str = "",
) -> DecisionRecord:
    by_id = {c.id: c for c in claims}
    claim_records: list[ClaimRecord] = []
    for judgement in outcome.judgements:
        claim = by_id.get(judgement.claim_id)
        if claim is None:
            continue
        claim_records.append(
            ClaimRecord(
                text=claim.text,
                decision=judgement.decision,
                firing_check=judgement.firing_check.value if judgement.firing_check else "",
                explanation=judgement.explanation,
                citations=[
                    f"{chunks[c.chunk_id].source.value}/{chunks[c.chunk_id].section or 'body'}"
                    for c in claim.citations
                    if c.chunk_id in chunks
                ],
                # Every check's verdict, not only the one that fired: a claim
                # that barely passed three checks reads differently from one
                # that passed them cleanly.
                checks={o.check.value: o.reason for o in judgement.outcomes},
            )
        )

    return DecisionRecord(
        run_id=run_id,
        question=question,
        decision=outcome.decision,
        rule=outcome.reason.rule,
        detail=outcome.reason.detail,
        claims=claim_records,
        context_chunks=[
            f"{chunks[cid].source.value}/{chunks[cid].section or 'body'}"
            for cid in context_ids
            if cid in chunks
        ],
        corpus_snapshot=corpus_snapshot,
        subject_model=subject_model,
        judge_model=judge_model,
        profile_digest=profile_digest(profile),
        gate_latency_ms=outcome.latency_ms,
        stage_latency_ms=dict(outcome.stage_latency_ms),
        cost_usd=outcome.usage.cost_usd,
        entailment_ran=outcome.entailment_ran,
    )


class RecordStore:
    """Line-delimited JSON, one record per line.

    This is the boundary with offline evaluation (§00 §0): an evaluation
    platform ingests these to compare gate versions across a labelled set, and
    this repository does not attempt that itself.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def release_rate(self) -> float:
        records = self.read()
        if not records:
            return 0.0
        released = sum(
            1 for r in records
            if r["decision"] in ("release", "release_with_caveat")
        )
        return released / len(records)
