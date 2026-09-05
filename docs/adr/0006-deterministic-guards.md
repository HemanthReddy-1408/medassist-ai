# ADR-0006 — Safety guards are code wherever a deterministic rule exists

**Status:** accepted · **Wave:** 4

## Context
A guard implemented as a prompt instruction is a request. Requests are
negotiable, and adversarial input negotiates.

## Decision
Red-flag triage, dosage grounding, numeric verification, drug–food
cross-checking, and tool authorization are deterministic code over structured
data. LLM judgement is used only where no deterministic rule exists (entailment
between a claim and a span).

## Consequences
Guards are testable in isolation, run in microseconds, and cannot be
jailbroken by rephrasing. Cost: the drug–food interaction table is curated data
requiring maintenance, and its coverage is a stated limitation rather than an
implied guarantee.
