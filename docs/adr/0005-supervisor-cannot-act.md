# ADR-0005 — The supervisor plans and delegates; it never calls a tool

**Status:** accepted · **Wave:** 3

## Context
An orchestrator that can also act will, under pressure from a hard task,
quietly do the work itself — bypassing every per-agent tool restriction,
budget, and evidence requirement in the capability registry.

## Decision
`SupervisorAgent` has an empty `allowed_tools`. It emits a `Plan`; the runtime
executes it. A plan naming an unregistered agent or an undeclared capability
fails at construction with `plan_invalid`.

## Consequences
Every action is attributable to a contracted agent. A model cannot invent an
edge or a tool. Cost: some tasks need an extra planning round-trip that a
free-acting supervisor would have short-circuited.
