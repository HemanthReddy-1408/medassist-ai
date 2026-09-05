# 07 — Safety, Policy, and the Decision Gate  **SPECIFIED**

Guards are deterministic code wherever a deterministic rule exists. A guard
implemented as a prompt is a request, and requests are negotiable.

## 7.1 Red-flag triage — runs first, always

Before retrieval, before planning. A pattern set over symptom descriptions
mapping to emergency presentations: crushing chest pain with radiation, sudden
unilateral weakness or facial droop, thunderclap headache, anaphylaxis,
suicidal ideation, sepsis triad, testicular/ovarian torsion.

On a hit the run **terminates immediately** with an emergency referral. It does
not retrieve, it does not synthesize, and no downstream node may soften the
message — the referral text is a constant, not a generation.

Tuned deliberately for recall over precision. A false positive costs a user an
unnecessary "seek care now". A false negative is the worst thing this system
can do. §08.3 reports both rates separately so the trade-off stays visible
rather than hidden inside one F1.

## 7.2 Dosage grounding

Any numeric dose in a released answer must cite a chunk whose `section` is
`dosage_and_administration` or `dosage_forms_and_strengths`, from a
`SourceKind.FDA_LABEL` document. Otherwise: `REWRITE`.

This is why sections are carried as offsets from the corpus layer up (§02.2).
The guard is a set membership test, not an inference.

Second check: the numeric verifier (§06.2) requires the digits to appear in the
cited span, unit-normalized. A dose that is well-cited but not actually present
in the citation fails.

## 7.3 The drug–food and drug–drug cross-check

The guard that justifies `lifestyle_guidance` being MEDIUM risk.

```
answer claims  ×  patient.medications  →  interaction table  →  finding
```

Worked cases:

| Advice | Medication | Interaction | Decision |
|---|---|---|---|
| "eat more leafy greens" | warfarin | vitamin K antagonism | `ANNOTATE` + consistency caveat |
| "grapefruit juice is healthy" | simvastatin | CYP3A4 inhibition | `REWRITE` |
| "increase potassium" | lisinopril / spironolactone | hyperkalaemia | `REWRITE` |
| "high-protein diet" | *CKD stage 4+ in profile* | protein restriction | `ANNOTATE` |
| "St John's wort for mood" | any SSRI | serotonin syndrome | `BLOCK` |

**No groundedness metric catches any of these.** Every one of these claims is
perfectly supported by the corpus — leafy greens *are* healthy. They are wrong
for *this person*. Correctness is relational, and a guard that only checks
claim-against-context cannot see the relation.

The table is curated, versioned data with citations, not model knowledge. Its
coverage is a stated limitation: it holds the well-documented interactions, and
the answer says so rather than implying exhaustiveness.

## 7.4 Policy engine and the decision gate

Deny by default. Rules are data, evaluated in code:

```
IF   task.risk == HIGH AND confidence < threshold   THEN require_human_review
IF   tool == patient_data AND user.role NOT IN authorized   THEN deny
IF   capability.evidence_required AND sources < min   THEN abstain
IF   any claim CONTRADICTED   THEN block
IF   red_flag_triggered   THEN block, escalate
```

Every decision is recorded with the rule that produced it. "Denied" without the
rule is not auditable, and an unauditable policy engine is decoration.

```
        verified answer
              │
              ▼
      ┌───────────────┐
      │ decision gate │
      └───────┬───────┘
      ┌───────┴────────┐
      ▼                ▼
   APPROVE         ESCALATE
      │                │
   release      human review queue
```

The gate is the only path to release. There is no code path that returns an
answer to a user without passing it.

## 7.5 Prompt injection through retrieved documents

The corpus is scraped from the public internet. **A retrieved document is an
untrusted input channel**, and this is the attack surface most RAG systems
leave open.

Defences, layered:

1. **Structural.** Retrieved text is delimited and labelled as data in the
   prompt. Instructions are never assembled from retrieved content.
2. **Detection.** Imperative patterns aimed at the model in retrieved chunks
   (`ignore previous`, `you are now`, `system:`) are flagged pre-prompt.
3. **Containment.** The model's proposed actions are validated against the
   capability's `allowed_tools`. An injected instruction to call a tool the
   capability does not declare fails at the registry, not at the model's
   discretion.
4. **Output.** Claims must cite retrieved chunks; an injected instruction that
   produces an uncited claim is stripped by §06.3.

Layer 3 is the one that actually holds. Layers 1 and 2 are pattern-matching
against an adversary who can rephrase; layer 3 is a closed set.

Memory and uploaded reports pass the same guard — user-supplied text re-entering
the prompt is the same channel wearing a different hat.

## 7.6 PII

Uploaded reports contain names, MRNs, dates of birth. Detection and redaction
run **before** any text reaches a provider, and the redaction map stays local so
the answer can be rehydrated for the user. Cache keys are computed on redacted
text, so the disk cache never holds identifiers.
