"""HTTP surface.

``/v1/runs/{id}/trace`` is exposed to the *user*, not only to operators. "Why do
you believe this?" is answerable by showing the evidence chain, and a system
that cannot show it is asking to be trusted rather than earning it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medassist.audit.records import RecordStore, profile_digest
from medassist.capabilities.registry import REGISTRY
from medassist.core.config import SETTINGS
from medassist.core.models import PatientProfile
from medassist.corpus.snapshot import SnapshotStore
from medassist.index.embed import default_embedding
from medassist.index.pipeline import Retriever
from medassist.llm.client import ModelClient
from medassist.memory.recall import observe_report
from medassist.memory.store import MemoryStore
from medassist.reports.interpret import findings, requires_escalation, trends
from medassist.reports.parse import ReportParseError, parse_report
from medassist.serving.pipeline import Pipeline

app = FastAPI(
    title="MedAssist X",
    version="2.0.0",
    description="Serving-time admission control for clinical answers.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    capability: str = "evidence_qa"
    profile: PatientProfile | None = None
    intent: str = ""


class ReportRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    previous: str | None = None
    #: Supplying a profile enables longitudinal memory: prior findings are
    #: reconciled against this report, and anything abnormal that was not
    #: re-measured becomes a question rather than an assumption.
    profile: PatientProfile | None = None
    remember: bool = True


@lru_cache(maxsize=1)
def _state() -> dict[str, Any]:
    documents, manifest = SnapshotStore().read()
    retriever = Retriever(embedding=default_embedding()).build(
        documents, snapshot_id=manifest.snapshot_id
    )
    subject = ModelClient(SETTINGS.subject_model)
    judge = ModelClient(SETTINGS.judge_model)
    store = RecordStore(Path(SETTINGS.artifacts_dir) / "decisions.jsonl")
    memory = MemoryStore(Path(SETTINGS.artifacts_dir) / "memory.jsonl")
    return {
        "retriever": retriever,
        "manifest": manifest,
        "store": store,
        "memory": memory,
        "pipeline": Pipeline(
            retriever=retriever, subject_client=subject, judge_client=judge,
            record_store=store,
        ),
    }


@app.get("/v1/healthz")
def healthz() -> dict[str, Any]:
    try:
        state = _state()
    except Exception as exc:
        raise HTTPException(503, f"index unavailable: {exc}") from exc
    return {
        "status": "ok",
        "corpus_snapshot": state["manifest"].snapshot_id,
        "documents": state["manifest"].document_count,
        "chunks": len(state["retriever"]),
        "subject_model": SETTINGS.subject_model,
        "judge_model": SETTINGS.judge_model,
    }


@app.get("/v1/capabilities")
def capabilities() -> list[dict[str, Any]]:
    return [
        {
            "name": c.name, "description": c.description, "risk": c.risk.value,
            "allowed_tools": sorted(c.allowed_tools),
            "min_independent_sources": c.min_independent_sources,
            "requires_human_review": c.requires_human_review,
            "min_supported_fraction": c.min_supported_fraction,
        }
        for c in REGISTRY.values()
    ]


@app.post("/v1/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    if request.capability not in REGISTRY:
        raise HTTPException(400, f"unknown capability {request.capability!r}")
    result = _state()["pipeline"].ask(
        request.question, capability=request.capability,
        profile=request.profile, intent=request.intent,
    )
    return {
        "run_id": str(result.run_id),
        "decision": result.decision.value,
        "released": result.released,
        "answer": result.prose,
        "caveats": result.caveats,
        "citations": result.citations,
        "confidence": None if result.confidence is None else round(result.confidence.value, 3),
        "confidence_factors": {} if result.confidence is None else {
            k: round(v, 3) for k, v in result.confidence.factors.items()
        },
        "took_ms": round(result.took_ms, 1),
        "cost_usd": round(result.usage.cost_usd, 6),
    }


@app.get("/v1/runs/{run_id}/trace")
def trace(run_id: str) -> dict[str, Any]:
    """The evidence chain behind one decision, for the person who asked."""
    for record in reversed(_state()["store"].read()):
        if record.get("run_id") == run_id:
            return record
    raise HTTPException(404, f"no decision record for {run_id}")


@app.post("/v1/reports")
def parse_lab_report(request: ReportRequest) -> dict[str, Any]:
    try:
        parsed = parse_report(request.text, source_name="upload")
    except ReportParseError as exc:
        raise HTTPException(400, str(exc)) from exc

    payload: dict[str, Any] = {
        "report_type": parsed.report.report_type,
        "collected": parsed.report.collected,
        "contained_pii": parsed.contained_pii,
        "pii_kinds": parsed.redaction.kinds,
        "analytes": [
            {"name": a.name, "value": a.value, "unit": a.unit, "flag": a.flag}
            for a in parsed.report.analytes
        ],
        "findings": [f.describe() for f in findings(parsed.report)],
        "requires_escalation": requires_escalation(parsed.report),
    }
    if request.previous:
        earlier = parse_report(request.previous).report
        payload["trends"] = [t.describe() for t in trends(earlier, parsed.report)]

    if request.profile is not None:
        digest = profile_digest(request.profile)
        memory: MemoryStore = _state()["memory"]
        prior = memory.for_profile(digest)
        if request.remember:
            reconciliation = observe_report(
                memory, digest, parsed.report, provenance="upload"
            )
        else:
            from medassist.memory.recall import reconcile

            reconciliation = reconcile(prior, parsed.report, digest=digest)
        payload["history"] = {
            "prior_entries": len(prior),
            "continuity": [c.describe() for c in reconciliation.continuations],
            # The point of the whole module: a prior abnormal finding that was
            # not re-measured becomes a question, never a silent assumption.
            "confirmations": [
                {"key": c.key, "reason": c.reason.value, "question": c.question}
                for c in reconciliation.confirmations
            ],
            "needs_user_input": reconciliation.needs_user_input,
        }
    return payload


class MemoryQuery(BaseModel):
    profile: PatientProfile


@app.post("/v1/memory")
def memory_for_profile(request: MemoryQuery) -> dict[str, Any]:
    """What we hold about this profile, and what has gone stale."""
    digest = profile_digest(request.profile)
    memory: MemoryStore = _state()["memory"]
    entries = memory.for_profile(digest)
    current = {e.key for e in memory.current_facts(digest)}
    return {
        "profile_digest": digest,
        "entries": len(entries),
        "current": [e.describe() for e in memory.current_facts(digest)],
        "stale": [e.describe() for e in entries if e.key not in current],
    }


@app.get("/v1/reliability")
def reliability() -> dict[str, Any]:
    """Measured operating characteristics, with sample sizes."""
    from medassist.confidence.calibration import wilson_interval
    from medassist.redteam.runner import default_gate_factory, run_suite

    report = run_suite(default_gate_factory)
    low, high = report.interval
    records = _state()["store"].read()
    released = sum(
        1 for r in records if r["decision"] in ("release", "release_with_caveat")
    )
    release_low, release_high = wilson_interval(released, len(records))
    return {
        "red_team": {
            "defended": report.defended,
            "n": report.total,
            "rate": round(report.defense_rate, 3),
            "ci95": [round(low, 3), round(high, 3)],
            "over_refusals": len(report.over_refusals),
            "by_class": {k: f"{v[0]}/{v[1]}" for k, v in report.by_class().items()},
        },
        "production": {
            "decisions": len(records),
            "release_rate": round(released / len(records), 3) if records else None,
            "ci95": [round(release_low, 3), round(release_high, 3)] if records else None,
        },
        "caveat": (
            "Rates on small samples are reported with Wilson intervals. "
            "A point estimate here implies precision it does not have."
        ),
    }
