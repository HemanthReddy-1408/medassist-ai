"""The end-to-end pipeline: question in, gated answer out.

Order is the specification, and every step of it is an enforcement point:

    triage -> redact -> retrieve -> generate -> gate -> record -> confidence

Triage runs before retrieval because an emergency presentation must not wait
for a corpus lookup. Redaction runs before the model call because nothing
unredacted may reach a provider. The gate runs before anything is returned,
because there is no path from generation to a user that does not pass it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from medassist.audit.records import DecisionRecord, build_record
from medassist.capabilities.registry import Capability
from medassist.capabilities.registry import get as get_capability
from medassist.confidence.compute import (
    ConfidenceReport,
    combine,
    evidence_coverage,
    evidence_quality,
    retrieval_relevance,
    source_agreement,
    temporal_validity,
)
from medassist.core.enums import ClaimDecision, ResponseDecision, RetrievalStage
from medassist.core.ids import RunId
from medassist.core.models import Claim, PatientProfile, Usage
from medassist.gate.cascade import GateOutcome, GateSpec, ReleaseGate
from medassist.gate.checks import GateContext
from medassist.generate import generate_claims
from medassist.guards.pii import redact
from medassist.guards.redflag import TriageResult, triage
from medassist.index.rerank import format_context


@dataclass
class AnswerResult:
    run_id: RunId
    question: str
    decision: ResponseDecision
    prose: str
    claims: list[Claim] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    confidence: ConfidenceReport | None = None
    outcome: GateOutcome | None = None
    record: DecisionRecord | None = None
    triage: TriageResult | None = None
    usage: Usage = field(default_factory=Usage)
    took_ms: float = 0.0

    @property
    def released(self) -> bool:
        return self.decision in (
            ResponseDecision.RELEASE, ResponseDecision.RELEASE_WITH_CAVEAT
        )

    def explain(self) -> str:
        if self.record is not None:
            return self.record.explain()
        return f"{self.decision.value.upper()}: {self.prose}"


ABSTAIN_TEXT = (
    "I could not verify an answer to that from the sources available, so I am "
    "not going to guess. {reason}"
)


class Pipeline:
    def __init__(
        self,
        *,
        retriever,
        subject_client,
        judge_client,
        gate: ReleaseGate | None = None,
        record_store=None,
    ) -> None:
        self.retriever = retriever
        self.subject = subject_client
        self.judge = judge_client
        self.gate = gate
        self.record_store = record_store

    def _gate_for(self, capability: Capability) -> ReleaseGate:
        if self.gate is not None:
            return self.gate
        from medassist.gate.entailment import EntailmentCheck

        return ReleaseGate(
            entailment=EntailmentCheck(self.judge),
            spec=GateSpec(
                min_supported_fraction=capability.min_supported_fraction,
                requires_human_review=capability.requires_human_review,
            ),
        )

    def ask(
        self,
        question: str,
        *,
        capability: str = "evidence_qa",
        profile: PatientProfile | None = None,
        intent: str = "",
    ) -> AnswerResult:
        started = time.perf_counter()
        run_id = RunId.new()
        cap = get_capability(capability)

        # 1. Triage pre-empts everything. No retrieval, no generation.
        triage_result = triage(question)
        if triage_result.triggered:
            return AnswerResult(
                run_id=run_id, question=question, decision=ResponseDecision.BLOCK,
                prose=triage_result.message, triage=triage_result,
                took_ms=(time.perf_counter() - started) * 1000,
            )

        # 2. Nothing unredacted reaches a provider.
        redaction = redact(question)
        safe_question = redaction.text

        # 3. Retrieve.
        result = self.retriever.retrieve(safe_question, intent=intent or capability)
        if not result.chunk_ids:
            return self._abstain(
                run_id, question, "retrieval found no relevant sources", started
            )

        handles = self.retriever.handles(result.chunk_ids)
        context = format_context(result.chunk_ids, self.retriever.chunks)

        # 4. Generate claims.
        usage = Usage()
        try:
            claims, gen_usage, insufficient = generate_claims(
                self.subject, safe_question, context, handles,
                self.retriever.chunks, profile=profile,
            )
            usage.absorb(gen_usage)
        except Exception as exc:
            return self._abstain(
                run_id, question, f"generation failed ({type(exc).__name__})", started, usage
            )

        if insufficient and not claims:
            return self._abstain(
                run_id, question, "the sources do not answer this question", started, usage
            )

        # 5. Gate. The only path to release.
        ctx = GateContext(
            chunks=self.retriever.chunks,
            context_ids=result.chunk_ids,
            profile=profile,
            evidence_required=cap.evidence_required,
        )
        outcome = self._gate_for(cap).evaluate(claims, ctx)
        usage.absorb(outcome.usage)

        kept = {
            j.claim_id for j in outcome.judgements if j.decision is not ClaimDecision.REMOVE
        }
        surviving = [c for c in claims if c.id in kept]
        prose = (
            " ".join(c.text for c in surviving)
            if outcome.released
            else ABSTAIN_TEXT.format(reason=outcome.reason.detail or outcome.reason.rule)
        )

        record = build_record(
            run_id=run_id, question=safe_question, outcome=outcome, claims=claims,
            chunks=self.retriever.chunks, context_ids=result.chunk_ids, profile=profile,
            corpus_snapshot=self.retriever.snapshot_id,
            subject_model=getattr(self.subject, "model", ""),
            judge_model=getattr(self.judge, "model", ""),
        )
        if self.record_store is not None:
            self.record_store.append(record)

        return AnswerResult(
            run_id=run_id, question=question, decision=outcome.decision, prose=prose,
            claims=surviving, caveats=outcome.caveats,
            citations=self._citations(surviving), outcome=outcome, record=record,
            confidence=self._confidence(outcome, result, surviving),
            usage=usage, took_ms=(time.perf_counter() - started) * 1000,
        )

    def _abstain(
        self, run_id: RunId, question: str, reason: str, started: float,
        usage: Usage | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            run_id=run_id, question=question, decision=ResponseDecision.ABSTAIN,
            prose=ABSTAIN_TEXT.format(reason=reason), usage=usage or Usage(),
            took_ms=(time.perf_counter() - started) * 1000,
        )

    def _citations(self, claims: list[Claim]) -> list[dict]:
        seen: dict[str, dict] = {}
        for claim in claims:
            for citation in claim.citations:
                chunk = self.retriever.chunks.get(citation.chunk_id)
                if chunk is None or str(chunk.id) in seen:
                    continue
                seen[str(chunk.id)] = {
                    "chunk_id": str(chunk.id),
                    "source": chunk.source.value,
                    "section": chunk.section,
                    "title": chunk.meta.get("doc_title", ""),
                    "url": chunk.meta.get("url", ""),
                    "quote": chunk.text[:400],
                }
        return list(seen.values())

    def _confidence(self, outcome: GateOutcome, result, claims: list[Claim]) -> ConfidenceReport:
        chunks = [
            self.retriever.chunks[cid]
            for cid in result.chunk_ids
            if cid in self.retriever.chunks
        ]
        scores: dict[str, float] = {}
        for stage in result.trace.stages:
            if stage.stage is RetrievalStage.RERANKED:
                scores = stage.scores
        years = [
            int(y)
            for c in chunks
            if (y := str(c.meta.get("published", "") or "0")).isdigit()
        ]
        return combine({
            "evidence_quality": evidence_quality(chunks),
            "evidence_coverage": evidence_coverage(outcome.judgements),
            "source_agreement": source_agreement(0, max(len(claims), 1)),
            "retrieval_relevance": retrieval_relevance(scores, result.chunk_ids),
            "temporal_validity": temporal_validity(years),
        })
