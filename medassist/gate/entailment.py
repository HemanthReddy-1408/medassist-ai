"""The one check that needs a model.

Reference-free by construction: the judge sees a claim and the spans that claim
cites, and is asked only whether those spans entail it. It never sees a
known-correct answer, because at serving time none exists.

All surviving claims go in **one batched call**. Per-claim calls would multiply
latency by the claim count, which on an 8-claim answer is the difference
between a gate that can run in-band and one that cannot.
"""

from __future__ import annotations

import time

from medassist.core.enums import CheckName, ClaimDecision, ClaimVerdict, Severity
from medassist.core.errors import MedAssistError
from medassist.core.ids import ClaimId
from medassist.core.models import Claim, Usage
from medassist.gate.checks import GateContext
from medassist.gate.decisions import CheckOutcome

SYSTEM = """You are a strict entailment judge for clinical text.

For each numbered claim you are given the exact source passages it cites. Decide
ONLY whether those passages support the claim. Do not use outside knowledge, and
do not judge whether the claim is true in general.

verdict must be exactly one of:
  supported    - the passages state or directly entail the claim
  contradicted - the passages state something incompatible with the claim
  unsupported  - the passages neither entail nor contradict it
  unverifiable - the claim is not a factual assertion (advice to see a doctor,
                 a hedge, a recommendation to consult someone)

Reply with JSON only:
{"judgements":[{"id":1,"verdict":"supported","evidence":"<short quote or reason>"}]}"""


class EntailmentUnavailable(MedAssistError):
    """The judge could not be reached or did not return a usable shape.

    Raised rather than defaulted, because the gate must fail closed: unverified
    claims cause an ABSTAIN, never a silent release.
    """


class EntailmentCheck:
    name = CheckName.ENTAILMENT

    def __init__(
        self,
        client,
        *,
        unsupported_action: ClaimDecision = ClaimDecision.QUALIFY,
        batch_size: int = 6,
    ) -> None:
        self.client = client
        self.unsupported_action = unsupported_action
        # Judges reliably drop claims from long lists - a 15-claim batch came
        # back with 14 verdicts, which fails closed and abstains on an answer
        # that was fine. Smaller batches cost more calls and lose far fewer
        # answers to a bookkeeping slip.
        self.batch_size = max(1, batch_size)

    def run_batch(
        self, claims: list[Claim], ctx: GateContext, *, max_tokens: int = 900
    ) -> tuple[dict[ClaimId, CheckOutcome], Usage]:
        if not claims:
            return {}, Usage()

        outcomes: dict[ClaimId, CheckOutcome] = {}
        usage = Usage()
        for start in range(0, len(claims), self.batch_size):
            window = claims[start : start + self.batch_size]
            batch_outcomes, batch_usage = self._judge(window, ctx, max_tokens=max_tokens)
            outcomes.update(batch_outcomes)
            usage.absorb(batch_usage)
        return outcomes, usage

    def _judge(
        self, claims: list[Claim], ctx: GateContext, *, max_tokens: int
    ) -> tuple[dict[ClaimId, CheckOutcome], Usage]:
        started = time.perf_counter()
        blocks: list[str] = []
        for index, claim in enumerate(claims, start=1):
            spans = ctx.spans(claim)
            rendered = "\n".join(f"  - {s}" for s in spans) or "  (no cited passages)"
            blocks.append(f"CLAIM {index}: {claim.text}\nCITED PASSAGES:\n{rendered}")

        try:
            payload, usage = self.client.structured(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "\n\n".join(blocks)},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # provider error, timeout, unparseable after repair
            raise EntailmentUnavailable(str(exc)) from exc

        judgements = payload.get("judgements") if isinstance(payload, dict) else payload
        if not isinstance(judgements, list):
            raise EntailmentUnavailable(f"expected a list of judgements, got {type(payload)}")

        took = (time.perf_counter() - started) * 1000
        by_index = {}
        for item in judgements:
            if not isinstance(item, dict):
                continue
            try:
                by_index[int(item.get("id", 0))] = item
            except (TypeError, ValueError):
                continue

        outcomes: dict[ClaimId, CheckOutcome] = {}
        for index, claim in enumerate(claims, start=1):
            item = by_index.get(index)
            if item is None:
                # A claim the judge skipped is unverified, not passed.
                raise EntailmentUnavailable(f"judge returned no verdict for claim {index}")
            outcomes[claim.id] = self._to_outcome(claim, item, ctx, took / len(claims))
        return outcomes, usage

    def _to_outcome(
        self, claim: Claim, item: dict, ctx: GateContext, took_ms: float
    ) -> CheckOutcome:
        raw = str(item.get("verdict", "")).strip().lower()
        try:
            verdict = ClaimVerdict(raw)
        except ValueError:
            raise EntailmentUnavailable(f"unrecognised verdict {raw!r}") from None

        evidence = str(item.get("evidence", ""))[:400]
        if verdict is ClaimVerdict.SUPPORTED:
            decision, severity = ClaimDecision.RETAIN, Severity.INFO
        elif verdict is ClaimVerdict.CONTRADICTED:
            # The cited source says otherwise. Worse than unsupported, and the
            # only claim-level verdict that blocks the whole response.
            decision, severity = ClaimDecision.REMOVE, Severity.CRITICAL
        elif verdict is ClaimVerdict.UNSUPPORTED:
            decision, severity = self.unsupported_action, Severity.MEDIUM
        else:  # UNVERIFIABLE - a hedge is not a factual assertion
            decision, severity = ClaimDecision.RETAIN, Severity.INFO

        detail = evidence
        if verdict is ClaimVerdict.UNSUPPORTED and self.unsupported_action is ClaimDecision.QUALIFY:
            detail = "Not established by the sources consulted."

        return CheckOutcome(
            check=self.name,
            decision=decision,
            reason=f"entailment:{verdict.value}",
            detail=detail,
            severity=severity,
            supporting_chunks=ctx.resolved(claim) if verdict is ClaimVerdict.SUPPORTED else [],
            took_ms=took_ms,
        )
