"""Live demonstration of the release gate.

Retrieves from the real corpus, generates claim-structured answers with a real
model, and gates them with a real judge. The third scenario tampers with a
generated claim - changing one dose digit - to show the cheapest check catching
what the most expensive one would wave through.

    make demo-gate
"""

from __future__ import annotations

import sys

from medassist.core.config import SETTINGS
from medassist.core.models import Claim, PatientProfile
from medassist.corpus.snapshot import SnapshotStore
from medassist.gate import EntailmentCheck, GateContext, GateSpec, ReleaseGate
from medassist.generate import generate_claims
from medassist.index.embed import HashingEmbedding
from medassist.index.pipeline import Retriever
from medassist.llm.client import ModelClient

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
COLOUR = {
    "release": "\033[32m", "release_with_caveat": "\033[33m",
    "abstain": "\033[36m", "block": "\033[31m", "escalate": "\033[35m",
}


def show(outcome, claims) -> None:
    by_id = {c.id: c for c in claims}
    tint = COLOUR.get(outcome.decision.value, "")
    print(f"  {tint}{BOLD}{outcome.decision.value.upper()}{RESET}"
          f"  ({outcome.reason.rule}: {outcome.reason.detail})")
    for judgement in outcome.judgements:
        claim = by_id[judgement.claim_id]
        mark = {"retain": "✓", "qualify": "~", "remove": "✗"}[judgement.decision.value]
        print(f"    {mark} {claim.text[:78]}")
        if judgement.decision.value != "retain":
            print(f"      {DIM}{judgement.explanation[:96]}{RESET}")
    stages = "  ".join(f"{k}={v:.2f}ms" for k, v in outcome.stage_latency_ms.items())
    print(f"  {DIM}{stages}{RESET}")
    print(f"  {DIM}total {outcome.latency_ms:.0f}ms · ${outcome.usage.cost_usd:.5f}{RESET}\n")


def main() -> int:
    if not SETTINGS.configured:
        print("No API key. Set MEDASSIST_API_KEY in .env")
        return 1

    documents, manifest = SnapshotStore().read()
    print(f"\n{BOLD}Corpus{RESET} snapshot {manifest.snapshot_id} · {len(documents)} documents")
    retriever = Retriever(embedding=HashingEmbedding()).build(
        documents, snapshot_id=manifest.snapshot_id
    )
    print(f"{BOLD}Index{RESET}  {len(retriever)} chunks\n")

    subject = ModelClient(SETTINGS.subject_model)
    # The judge is a different model. One that grades its own output is
    # performing a self-assessment, not a verification.
    judge = ModelClient(SETTINGS.judge_model)
    gate = ReleaseGate(entailment=EntailmentCheck(judge), spec=GateSpec())

    warfarin_patient = PatientProfile(
        age=64, sex="female", conditions=["atrial fibrillation", "hypertension"],
        medications=["warfarin 5mg", "lisinopril 10mg"],
    )

    scenarios = [
        ("What is the recommended starting dose of metformin?", "drug_info", None, False),
        ("What should I eat to keep my heart healthy?", "condition_overview", warfarin_patient, False),
        ("What is the recommended starting dose of metformin?", "drug_info", None, True),
    ]

    for question, intent, profile, tamper in scenarios:
        label = question if not tamper else f"{question}   {DIM}[dose digit tampered]{RESET}"
        print(f"{BOLD}? {label}{RESET}")
        if profile is not None:
            print(f"  {DIM}patient: {profile.describe()}{RESET}")

        result = retriever.retrieve(question, intent=intent)
        handles = retriever.handles(result.chunk_ids)
        claims, usage, insufficient = generate_claims(
            subject, question, result.context, handles, retriever.chunks, profile=profile
        )
        if insufficient and not claims:
            print("  ABSTAIN (generator declared the passages insufficient)\n")
            continue

        if tamper and claims:
            # Change exactly one digit of a dose. Entailment often accepts this
            # - the sentence is otherwise identical and both numbers are the
            # same kind of thing. Arithmetic does not.
            target = next((c for c in claims if c.is_dosage), claims[0])
            claims = [
                Claim(
                    id=target.id,
                    text=target.text.replace("500", "800").replace("2550", "2000"),
                    citations=list(target.citations),
                    is_dosage=True,
                )
                if c.id == target.id else c
                for c in claims
            ]

        ctx = GateContext(
            chunks=retriever.chunks, context_ids=result.chunk_ids, profile=profile
        )
        outcome = gate.evaluate(claims, ctx)
        outcome.usage.absorb(usage)
        show(outcome, claims)

    subject.close()
    judge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
