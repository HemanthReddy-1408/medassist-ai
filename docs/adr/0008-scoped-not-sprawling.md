# ADR-0008 — Six capabilities, not thirty subsystems

**Status:** accepted · **Wave:** 0

## Context
An ambitious architecture document proposed ~30 subsystems including a code
sandbox, synthetic patient simulation at scale, graph databases, and
video/audio modalities.

## Decision
Six capabilities, each exercising different machinery. Explicit non-goals
recorded in §00.5 with reasons.

## Consequences
Depth over breadth: an interviewer picking any box on the diagram reaches
working, measured code. Cost: the system does less. Accepted — a README
advertising a "Knowledge Fabric" over a stub is the exact failure mode this
project inherited from v1, where the README claimed multi-source RAG and
`pubmed_node` returned `"not implemented yet"`.

## Rejected
*Build all thirty as stubs.* Produces a diagram that cannot survive one
follow-up question.
