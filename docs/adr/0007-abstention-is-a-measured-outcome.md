# ADR-0007 — Abstention is a first-class, measured outcome

**Status:** accepted · **Wave:** 2

## Context
The default behaviour of every LLM agent is to answer anyway. Abstention that
is merely encouraged in a prompt does not happen.

## Decision
`should_abstain` is a labelled property of gold cases. Abstention precision and
recall are reported, and `NOT_INDEXED` retrieval failure routes to abstain
rather than retry.

## Consequences
Over-refusal is reported alongside every safety metric, because refusing
everything scores perfectly on safety and is useless. Any safety improvement
must state its over-refusal cost in the same table.
