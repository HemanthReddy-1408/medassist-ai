# Running MedAssist X

Everything needed to install, configure, run, and extend the system.
Architecture is in [TECHNICAL.md](TECHNICAL.md); the contract is in
[`docs/spec/`](docs/spec/).

---

## 1. Prerequisites

| | |
|---|---|
| Python | 3.11+ (developed on 3.13.5) |
| Network | Only for corpus building and live model calls. **Tests need neither.** |
| API key | An OpenAI-compatible provider. Groq by default. |
| Disk | ~50 MB for a small corpus + caches; ~500 MB more if you install the neural embedding extra. |

---

## 2. Install

```bash
git clone https://github.com/HemanthReddy-1408/medassist-ai.git
cd medassist-ai
git checkout v2-multi-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[ui]"             # Streamlit interface
pip install sentence-transformers  # real embeddings instead of the hashing baseline
pip install pypdf                  # PDF lab reports
```

**Verify the install without any key or network:**

```bash
make test
# 448 passed in 2.4s
```

If that passes, the whole deterministic core works. Everything below is optional
on top of it.

---

## 3. Configure

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
MEDASSIST_API_KEY=gsk_...                              # required for live runs
MEDASSIST_BASE_URL=https://api.groq.com/openai/v1
MEDASSIST_SUBJECT_MODEL=openai/gpt-oss-20b             # the agent under test
MEDASSIST_JUDGE_MODEL=openai/gpt-oss-120b              # grades the agent
NCBI_API_KEY=                                          # optional: 3 → 10 req/s on PubMed
MEDASSIST_CONTACT_EMAIL=you@example.com                # sent as User-Agent to public APIs
```

### Why two models

The subject and the judge **must differ**. A model grading its own output is
performing a self-assessment, not a verification — it agrees with itself far
more often than it should, and the resulting number measures nothing.

### Choosing a subject model

This matters more than it looks. Two constraints:

1. **Reasoning models often will not fit.** A model that emits a `<think>`
   preamble spends its output budget before it starts answering. If your account
   caps output tokens per minute (Groq free tier: 1,000), a reasoning model
   cannot complete a structured response at all. The client diagnoses this
   explicitly rather than reporting malformed JSON:
   `"<model> produced no output within max_tokens=900: the response was
   truncated, most likely by a reasoning preamble."`
2. **Strict JSON mode rejects those models outright.** The client detects the
   provider's `json_validate_failed` and retries without the constraint, parsing
   leniently — so this degrades rather than fails, but it costs a round trip.

`openai/gpt-oss-20b` answers the generation task in ~98 output tokens and is the
default for that reason. If you have higher limits, anything that reliably emits
JSON works.

**`.env` is gitignored.** Verify before pushing: `git ls-files | grep '^\.env$'`
should print nothing.

---

## 4. Build a corpus

Nothing that touches retrieval works until you have one.

```bash
make demo
```

This scrapes openFDA, PubMed and MedlinePlus for a small clinical slice
(4 drugs, 3 conditions, 2 literature queries), writes a content-addressed
snapshot, indexes it, and prints three retrieval traces.

Expect roughly:

```
snapshot 9dceed0a7ffb3338 · 18 documents · {'fda_label': 4, 'medlineplus': 6, 'pubmed_abstract': 8}
indexed  403 chunks
```

Takes ~30 s cold, ~2 s warm (raw responses are cached on disk).

```bash
medassist corpus list        # snapshots, with document counts and hashes
```

**The snapshot id is not decoration.** Every decision record carries it. A
metric measured against one corpus cannot be compared to a metric from another,
and `SnapshotStore.read()` re-hashes on load and refuses a corrupt snapshot —
a silently edited corpus would invalidate every number measured against it.

### Widening the corpus

Edit the lists at the top of `medassist/demo.py`:

```python
DRUGS      = ["metformin", "warfarin", "lisinopril", "atorvastatin"]
TOPICS     = ["type 2 diabetes", "high blood pressure", "cholesterol"]
LITERATURE = ["metformin first-line therapy type 2 diabetes", ...]
```

Be considerate: NCBI permits 3 requests/second unauthenticated and will block
above it. The fetcher rate-limits per host and caches, but a 500-drug list is
still a lot of traffic to a free public service.

---

## 5. Commands

### `make` targets

| Target | Does |
|---|---|
| `make test` | 448 tests, offline, no key |
| `make redteam` | the 15-attack adversarial suite |
| `make demo` | live: scrape → snapshot → index → retrieve, printing traces |
| `make demo-gate` | live: generate claims → run the gate, incl. a tampered dose |
| `make serve` | HTTP API on `127.0.0.1:8000` |
| `make ui` | Streamlit with the evidence pane |
| `make lint` / `make fmt` | ruff |

### CLI

```bash
medassist corpus build                    # scrape and snapshot
medassist corpus list                     # snapshots with hashes

medassist ask "what is the starting dose of metformin?"
medassist ask "what should I eat?" --profile patient.json --explain
medassist ask "compare metformin and sitagliptin" --capability treatment_comparison

medassist report labs.txt                 # parse and interpret a lab report
medassist redteam                         # adversarial suite; exit 1 on any failure
medassist serve --port 8000
```

`ask` exits **0** when the answer was released and **2** when it was withheld,
so it can be used in a script that must not act on an abstention.

A profile file is JSON:

```json
{
  "age": 64,
  "sex": "female",
  "conditions": ["atrial fibrillation", "hypertension"],
  "medications": ["warfarin 5mg", "lisinopril 10mg"],
  "allergies": ["penicillin"]
}
```

The profile is what makes the relational check possible. Without it, advice that
is correct in general but unsafe for this person passes every other check.

---

## 6. HTTP API

```bash
make serve      # docs at http://127.0.0.1:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/healthz` | snapshot id, document and chunk counts, models |
| GET | `/v1/capabilities` | the six capabilities with risk, tools, thresholds |
| POST | `/v1/ask` | question → gated answer |
| GET | `/v1/runs/{id}/trace` | the full decision record for one answer |
| POST | `/v1/reports` | parse a lab report; reconcile against memory |
| POST | `/v1/memory` | what is held for a profile, and what has gone stale |
| GET | `/v1/reliability` | measured rates with Wilson intervals |

### Ask

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' -d '{
  "question": "What should I eat to keep my heart healthy?",
  "capability": "lifestyle_guidance",
  "profile": {"age": 64, "medications": ["warfarin 5mg", "lisinopril 10mg"]}
}' | jq
```

```json
{
  "decision": "release_with_caveat",
  "released": true,
  "answer": "…",
  "caveats": ["Increasing potassium is unsafe with your blood-pressure medication."],
  "confidence": 0.66,
  "confidence_factors": {"evidence_coverage": 0.83, "…": 0.9},
  "citations": [{"source": "medlineplus", "section": "overview", "quote": "…"}],
  "took_ms": 1284.0
}
```

`decision` is the field to branch on, not `answer`:

| Value | Meaning |
|---|---|
| `release` | every claim passed every check |
| `release_with_caveat` | released, but something was qualified or removed |
| `abstain` | **we do not know** — evidence insufficient or a check could not run |
| `block` | **we know, and it is not safe to say** |
| `escalate` | needs human review before release |

`abstain` and `block` are deliberately distinct. Treating them the same means
being wrong about one of them.

### Why the trace endpoint is public

```bash
curl -s localhost:8000/v1/runs/run_01M.../trace | jq '.claims[]'
```

*"Why do you believe this?"* has to be answerable. Every claim comes back with
the check that decided it, the verdict of every other check, and the cited
source. A system that cannot show its evidence chain is asking to be trusted
rather than earning it.

### Reports, with memory

```bash
curl -s localhost:8000/v1/reports -H 'content-type: application/json' -d '{
  "text": "Collected: 2026-04-08\nHemoglobin 9.1 g/dL\nHbA1c: 7.4 %",
  "profile": {"age": 54, "conditions": ["anaemia"]}
}' | jq '.history'
```

Upload a later report that omits haemoglobin and you get:

```json
{
  "needs_user_input": true,
  "continuity": ["Hemoglobin: 9.1 (2026-04-08) -> not measured [unchecked]",
                 "HbA1c: 7.4 (2026-04-08) -> 8.2 [worsening]"],
  "confirmations": [{
    "key": "Hemoglobin",
    "reason": "stale",
    "question": "The most recent Hemoglobin I have is Hemoglobin 9.1 g/dL (low) as of 2026-04-08, which is now 150 days old. Should I still treat that as current?"
  }]
}
```

Pass `"remember": false` to reconcile without writing.

---

## 7. Streamlit UI

```bash
make ui
```

Three panes. The **evidence pane** is the point: every citation expands to the
span that supports it, and every claim shows the verdict of all seven checks.
Enter medications in the sidebar to see the relational check fire.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CorpusError: no corpus snapshot found` | Never built one | `make demo` |
| `CorpusError: snapshot … is corrupt` | `documents.jsonl` edited by hand | Rebuild. Do not "fix" it — every number measured against it is void. |
| `ConfigError: no API key` | `.env` missing or unset | `cp .env.example .env`, add the key |
| `provider returned 429 … output tokens per minute` | `max_tokens` above the account ceiling | Handled automatically (halves and retries). Persisting ⇒ your ceiling is under 256. |
| `<model> produced no output within max_tokens=…` | Reasoning preamble ate the budget | Use a non-reasoning subject model, or raise `max_tokens` if your limits allow |
| Every answer `ABSTAIN`, reason `entailment_unavailable` | Judge model unreachable or wrong id | Check `MEDASSIST_JUDGE_MODEL` against your provider's list |
| Answers abstain with `insufficient_support` | Corpus does not cover the question | Widen the corpus (§4), or accept the abstention — it is correct |
| Tests pass but live runs fail | Config, not code. Tests are offline by design | Check `.env`; run `medassist corpus list` |
| Import errors for `streamlit` / `pypdf` | Optional extras | `pip install -e ".[ui]"`, `pip install pypdf` |

### Reading an abstention

An abstention is a *result*, not a failure. The reason names the check:

```
ABSTAIN (insufficient_support: only 4 of 6 claims survived verification (67% < 70%))
```

That is `lifestyle_guidance` declaring `min_supported_fraction = 0.7` and the
gate honouring it. Lowering the threshold is a decision about risk, made in
`medassist/capabilities/registry.py`, not a bug to be patched away.

---

## 9. Development

```bash
make test                                     # everything, offline
.venv/bin/pytest tests/test_gate.py -v        # one file
.venv/bin/pytest -k relational -v             # one concern
.venv/bin/pytest -m live                      # the few tests needing a provider
make lint                                     # ruff
```

The suite must pass **with `.env` removed**. That is checked, and it is what
makes CI meaningful — a suite that needs credentials is a suite that silently
stops running.

### Adding an interaction rule

`medassist/gate/interactions.py`. Key on a **class**, not a drug name:

```python
FoodRule(
    subject="grapefruit",
    patterns=(r"\bgrapefruit\w*\b",),
    classes=("cyp3a4_substrate",),           # not ("simvastatin",)
    mechanism="Grapefruit inhibits intestinal CYP3A4, raising exposure.",
    severity=Severity.HIGH,
    action=ClaimDecision.REMOVE,
    caveat="…",
    source="FDA statin labels, Drug Interactions",   # required
)
```

A rule naming simvastatin protects nobody on lovastatin though the mechanism is
identical. Add new ingredients to `CLASSES` in `medassist/patient/normalize.py`
and every existing class rule covers them for free.

`source` is not optional — a curated table without citations is model knowledge
with extra steps, and a test asserts every rule has one. Bump `TABLE_VERSION` so
decision records say which table version judged them.

### Adding a red-team attack

`medassist/redteam/attacks.py`. Every attack needs a **machine-checkable**
expectation; one without it is a demo, not a test.

```python
Attack("RT-16", AttackClass.DOSAGE_MANIPULATION,
       "…", _builder, WITHHELD,
       "why this must be caught")
```

If you add attacks, add benign probes too. A suite containing only attacks
drives the system toward refusing everything, which scores 100 % and is useless.

### Adding a capability

`medassist/capabilities/registry.py`. `allowed_tools` is a closed set enforced
by the runtime, and `min_supported_fraction` is where you express how much
verification failure that capability tolerates before abstaining.

---

## 10. What this system will not do

- It is **not clinically reliable** and is not a substitute for a clinician.
  Nothing in this repository should be read as evidence that it could be.
- It does **not** do comparative evaluation — paired significance tests,
  multiple-comparison correction, experiment tracking. Those answer questions
  about *changes*, offline, with gold labels. This repository exports decision
  records for such a tool to consume and stops there ([§00 §0](docs/spec/00-overview.md)).
- Its interaction table is **narrow by design** (8 food rules, 6 drug rules,
  10 drug classes) and says so rather than implying exhaustiveness.
- Known gaps are listed in [TECHNICAL.md §11](TECHNICAL.md), including the ones
  that matter most: the orchestration layer is built and tested but the serving
  path does not yet route through it, and no judge has actually been calibrated,
  so every judge verdict is correctly branded `UNCALIBRATED`.
