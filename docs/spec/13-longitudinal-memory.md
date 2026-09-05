# 13 — Longitudinal Memory  **BUILT**

`medassist/memory/` — episodic memory across visits, and the confirmations it
raises.

## 13.1 The case this exists for

> A profile uploads a report showing haemoglobin low at 9.1 g/dL. Five months
> later the same profile uploads a lipid panel, which does not include
> haemoglobin.

Three behaviours are available, and two of them are wrong.

**Forget it.** The person is still living with the finding, and the next answer
is worse for not knowing.

**Assume it still holds.** The system reasons from a value five months old as
though it were measured today. In this domain that is how confident, wrong
advice gets produced.

**Ask.** *"Your earlier record shows haemoglobin 9.1 g/dL (low) as of
2026-04-08, and this report does not include haemoglobin. Is that still the
case, has it been rechecked since, or was it treated?"*

Only the third preserves the finding without asserting it, and it is what the
system does.

## 13.2 Shelf life, per fact

A lab value from eight months ago is not a current fact about a person. Every
entry carries an assertion date and a shelf life, and expiry is a property of
the *kind* of fact:

| Kind | Shelf life |
|---|---:|
| Lab finding (default) | 180 days |
| Condition | 730 days |
| Medication | 365 days |
| User statement | 365 days |

Analytes that move faster expire sooner, because the rate of change is a
clinical property, not a storage-policy one:

```
INR 30 · Potassium 60 · Sodium 60 · Glucose 90 · WBC 90 · Platelets 90
Hemoglobin 120 · Creatinine 120 · eGFR 120
```

**An undated fact is treated as stale.** It cannot be shown to be current, and
assuming currency is the failure this whole mechanism exists to prevent.

The report's own `Collected:` date is used, never today's. Dating an old
uploaded report as current would defeat the shelf life entirely.

## 13.3 Continuity classification

Comparing memory against a new report yields one of six states:

| State | Meaning |
|---|---|
| `RESOLVED` | was abnormal, now in range |
| `PERSISTING` | abnormal then, abnormal now |
| `IMPROVING` | abnormal both times, moving toward the interval |
| `WORSENING` | abnormal both times, moving away |
| `NEW` | abnormal now, no prior record |
| `UNCHECKED` | abnormal before, absent from this report → **a question** |

Direction is computed relative to the reference interval, so a falling HbA1c
and a rising haemoglobin are both `IMPROVING`. A single abnormal value and a
value abnormal at every draw since March are different clinical facts, and only
the second is visible longitudinally.

## 13.4 Which findings become questions

Only abnormal history is raised. Confirming that a normal value is still normal
changes no answer and spends the user's patience.

Questions are **capped at three**, ordered by severity. A system that opens
with nine questions gets none of them answered, so a critical potassium is
asked about before a mildly raised HbA1c.

Two reasons are distinguished, because they call for different answers:

- `NOT_REMEASURED` — recent, simply absent from this report.
- `STALE` — past its shelf life; even a prior "yes" needs refreshing.

## 13.5 Read before write

`observe_report` reconciles *first*, then records. Recording first would make
every prior finding appear re-measured in the new report, and no confirmation
would ever be raised. This ordering is asserted by a test, because it is the
kind of invariant that a later refactor silently inverts.

## 13.6 Append-only

History is the value. A store that overwrites can express "haemoglobin is low";
only an append-only one can express "haemoglobin has been low at every draw
since March", which is the more clinically interesting statement.

Memory is partitioned by **salted profile digest** — raw age, conditions and
medications never enter the store (§10.4).

## 13.7 Not yet built

`CONTRADICTED` is defined and unused: reconciling a *user statement* against a
report ("I'm not on anticoagulants" vs. a record showing warfarin) is specified
in §05.7 but not implemented. Listed here rather than omitted, so the gap is
visible.
