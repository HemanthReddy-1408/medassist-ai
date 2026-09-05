"""Command line surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medassist.core.config import SETTINGS


def _retriever(snapshot: str = ""):
    from medassist.corpus.snapshot import SnapshotStore
    from medassist.index.embed import default_embedding
    from medassist.index.pipeline import Retriever

    documents, manifest = SnapshotStore().read(snapshot or None)
    return Retriever(embedding=default_embedding()).build(
        documents, snapshot_id=manifest.snapshot_id
    ), manifest


def cmd_corpus_build(args: argparse.Namespace) -> int:
    from medassist.demo import build_corpus

    _documents, snapshot_id = build_corpus()
    print(f"snapshot {snapshot_id}")
    return 0


def cmd_corpus_list(args: argparse.Namespace) -> int:
    from medassist.corpus.snapshot import SnapshotStore

    store = SnapshotStore()
    latest = store.latest()
    for snapshot in store.list_snapshots():
        _docs, manifest = store.read(snapshot)
        mark = "*" if snapshot == latest else " "
        print(f"{mark} {snapshot}  {manifest.document_count:4d} docs  {manifest.built_at}  {manifest.by_source}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from medassist.audit.records import RecordStore
    from medassist.core.models import PatientProfile
    from medassist.llm.client import ModelClient
    from medassist.serving.pipeline import Pipeline

    profile = None
    if args.profile:
        profile = PatientProfile(**json.loads(Path(args.profile).read_text()))

    retriever, manifest = _retriever(args.snapshot)
    pipeline = Pipeline(
        retriever=retriever,
        subject_client=ModelClient(SETTINGS.subject_model),
        judge_client=ModelClient(SETTINGS.judge_model),
        record_store=RecordStore(SETTINGS.artifacts_dir / "decisions.jsonl"),
    )
    result = pipeline.ask(args.question, capability=args.capability, profile=profile)

    print(f"\n{result.decision.value.upper()}  ({result.took_ms:.0f}ms, ${result.usage.cost_usd:.5f})")
    print(f"\n{result.prose}\n")
    for caveat in result.caveats:
        print(f"  ! {caveat}")
    if result.confidence is not None:
        print(f"\n  {result.confidence.explain()}")
    if args.explain and result.record is not None:
        print(f"\n{result.record.explain()}")
    if result.citations:
        print("\nsources:")
        for citation in result.citations:
            print(f"  - {citation['source']}/{citation['section'] or 'body'}: {citation['title'][:60]}")
    return 0 if result.released else 2


def cmd_report(args: argparse.Namespace) -> int:
    from medassist.reports.interpret import findings, requires_escalation
    from medassist.reports.parse import load_report

    parsed = load_report(args.path)
    print(f"type: {parsed.report.report_type or 'unknown'}  collected: {parsed.report.collected or 'unknown'}")
    print(f"PII removed: {parsed.redaction.kinds or 'none found'}\n")
    for analyte in parsed.report.analytes:
        print(f"  {analyte.describe()}")
    abnormal = findings(parsed.report)
    if abnormal:
        print("\nfindings (most severe first):")
        for finding in abnormal:
            print(f"  {finding.describe()}")
    if requires_escalation(parsed.report):
        print("\n  ESCALATE: a value is in the critical range.")
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from medassist.redteam.runner import default_gate_factory, run_suite

    report = run_suite(default_gate_factory)
    print(report.summary())
    return 0 if not report.failures else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("medassist.serving.api:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medassist", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="corpus snapshots").add_subparsers(
        dest="corpus_command", required=True
    )
    corpus.add_parser("build", help="scrape and snapshot").set_defaults(func=cmd_corpus_build)
    corpus.add_parser("list", help="list snapshots").set_defaults(func=cmd_corpus_list)

    ask = sub.add_parser("ask", help="ask a question through the full pipeline")
    ask.add_argument("question")
    ask.add_argument("--capability", default="evidence_qa")
    ask.add_argument("--profile", help="path to a patient profile JSON file")
    ask.add_argument("--snapshot", default="")
    ask.add_argument("--explain", action="store_true", help="print the decision record")
    ask.set_defaults(func=cmd_ask)

    report = sub.add_parser("report", help="parse and interpret a lab report")
    report.add_argument("path")
    report.set_defaults(func=cmd_report)

    sub.add_parser("redteam", help="run the adversarial suite").set_defaults(func=cmd_redteam)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
