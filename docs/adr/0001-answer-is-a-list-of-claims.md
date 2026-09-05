# ADR-0001 — An answer is a list of claims, not a string

**Status:** accepted · **Wave:** 0

## Context
Hallucination detection over free prose can only ask "does this look grounded?"
and answer with a judge's overall impression.

## Decision
The generator emits atomic claims with citation handles. Prose is rendered from
claims. Every claim is verified independently.

## Consequences
Faithfulness becomes a count over verdicts rather than a score, individual
claims can be removed or qualified without regenerating, and citation precision
is computable. Cost: the generator prompt is more constrained, and some
fluency is lost — mitigated by rendering prose from the claim list rather than
presenting the list raw.

## Rejected
*Generate prose, then decompose.* The decomposition step hallucinates too, and
produces claims the answer never made — attributing a failure to the generator
that belongs to the extractor.
