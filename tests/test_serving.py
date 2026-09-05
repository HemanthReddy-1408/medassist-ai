"""Wave 8: the end-to-end pipeline and the HTTP surface. No network."""

from __future__ import annotations

import pytest

from medassist.audit.records import RecordStore
from medassist.core.enums import ResponseDecision
from medassist.core.models import PatientProfile, Usage
from medassist.gate.cascade import GateSpec, ReleaseGate
from medassist.gate.entailment import EntailmentCheck
from medassist.index.embed import HashingEmbedding
from medassist.index.pipeline import Retriever
from medassist.redteam.runner import StubJudge
from medassist.serving.pipeline import Pipeline


class StubSubject:
    """A generator whose output is scripted, so pipeline wiring is tested alone."""

    model = "stub-subject"

    def __init__(self, payload, fail: bool = False):
        self.payload = payload
        self.fail = fail
        self.calls = 0
        self.seen_prompts: list[str] = []

    def structured(self, messages, **kwargs):
        self.calls += 1
        self.seen_prompts.append(messages[-1]["content"])
        if self.fail:
            raise RuntimeError("provider down")
        return self.payload, Usage(steps=1, prompt_tokens=80, completion_tokens=30, cost_usd=0.0002)


def build(corpus, subject, verdicts, *, store=None, spec=None):
    retriever = Retriever(embedding=HashingEmbedding()).build(corpus, snapshot_id="testsnap")
    gate = ReleaseGate(
        entailment=EntailmentCheck(StubJudge(verdicts)),
        spec=spec or GateSpec(min_supported_fraction=0.6),
    )
    return Pipeline(
        retriever=retriever, subject_client=subject,
        judge_client=StubJudge(verdicts), gate=gate, record_store=store,
    ), retriever


class TestTriagePreemption:
    def test_red_flag_blocks_before_any_model_call(self, corpus):
        subject = StubSubject({"claims": []})
        pipeline, _ = build(corpus, subject, [])
        result = pipeline.ask("I have crushing chest pain radiating to my left arm")
        assert result.decision is ResponseDecision.BLOCK
        assert subject.calls == 0  # no retrieval, no generation
        assert result.usage.cost_usd == 0.0

    def test_referral_text_is_returned_verbatim(self, corpus):
        pipeline, _ = build(corpus, StubSubject({"claims": []}), [])
        result = pipeline.ask("worst headache of my life")
        assert "emergency" in result.prose.lower()
        assert result.triage is not None and result.triage.triggered


class TestHappyPath:
    def test_a_grounded_answer_is_released(self, corpus):
        subject = StubSubject({
            "claims": [{
                "text": "The maximum recommended daily dose is 2550 mg.",
                "cites": ["C1"], "is_dosage": True,
            }],
            "insufficient": False,
        })
        pipeline, _ = build(corpus, subject, ["supported"])
        result = pipeline.ask("what is the maximum daily dose of metformin", intent="drug_info")
        assert result.decision is ResponseDecision.RELEASE
        assert result.citations
        assert result.confidence is not None and result.confidence.value > 0

    def test_prose_is_rendered_from_surviving_claims_only(self, corpus):
        subject = StubSubject({
            "claims": [
                {"text": "The maximum recommended daily dose is 2550 mg.", "cites": ["C1"], "is_dosage": True},
                {"text": "The maximum recommended daily dose is 9000 mg.", "cites": ["C1"], "is_dosage": True},
            ],
            "insufficient": False,
        })
        pipeline, _ = build(corpus, subject, ["supported"], spec=GateSpec(min_supported_fraction=0.4))
        result = pipeline.ask("maximum daily dose of metformin", intent="drug_info")
        assert "9000" not in result.prose
        assert "2550" in result.prose


class TestFailClosed:
    def test_generation_failure_abstains(self, corpus):
        pipeline, _ = build(corpus, StubSubject(None, fail=True), [])
        result = pipeline.ask("what is the dose of metformin")
        assert result.decision is ResponseDecision.ABSTAIN
        assert "generation failed" in result.prose

    def test_generator_declaring_insufficiency_is_honoured(self, corpus):
        """An abstention the generator volunteers is cheaper than one recovered."""
        subject = StubSubject({"claims": [], "insufficient": True})
        pipeline, _ = build(corpus, subject, [])
        result = pipeline.ask("what is the dose of a drug not in this corpus")
        assert result.decision is ResponseDecision.ABSTAIN

    def test_no_retrieval_hits_abstains_without_generating(self, corpus):
        subject = StubSubject({"claims": []})
        pipeline, retriever = build(corpus, subject, [])
        retriever.chunks = {}
        retriever._dense.chunk_ids = []
        retriever._sparse.chunk_ids = []
        result = pipeline.ask("anything at all")
        assert result.decision is ResponseDecision.ABSTAIN
        assert subject.calls == 0

    def test_abstain_prose_names_the_reason(self, corpus):
        pipeline, _ = build(corpus, StubSubject(None, fail=True), [])
        assert "not going to guess" in pipeline.ask("q").prose


class TestPrivacy:
    def test_pii_never_reaches_the_generator(self, corpus):
        subject = StubSubject({"claims": [], "insufficient": True})
        pipeline, _ = build(corpus, subject, [])
        pipeline.ask("My name is Jane Doe, MRN: A1234567 - what is metformin?")
        assert subject.seen_prompts
        assert "Jane Doe" not in subject.seen_prompts[0]
        assert "A1234567" not in subject.seen_prompts[0]


class TestRelationalIntegration:
    def test_patient_record_reaches_the_gate(self, corpus):
        subject = StubSubject({
            "claims": [{
                "text": "Leafy green vegetables are an excellent source of vitamins.",
                "cites": ["C1"],
            }],
            "insufficient": False,
        })
        pipeline, _ = build(corpus, subject, ["supported"])
        result = pipeline.ask(
            "what should I eat",
            profile=PatientProfile(medications=["warfarin 5mg"]),
            capability="lifestyle_guidance",
        )
        # Either caveated or withheld, but never released unqualified.
        assert result.decision is not ResponseDecision.RELEASE


class TestDecisionRecording:
    def test_records_are_persisted(self, corpus, tmp_path):
        store = RecordStore(tmp_path / "decisions.jsonl")
        subject = StubSubject({
            "claims": [{"text": "Metformin is first-line therapy.", "cites": ["C1"]}],
            "insufficient": False,
        })
        pipeline, _ = build(corpus, subject, ["supported"], store=store)
        result = pipeline.ask("is metformin first line", intent="evidence_qa")
        records = store.read()
        assert len(records) == 1
        assert records[0]["run_id"] == str(result.run_id)
        assert records[0]["corpus_snapshot"] == "testsnap"

    def test_record_carries_per_check_verdicts(self, corpus, tmp_path):
        store = RecordStore(tmp_path / "d.jsonl")
        subject = StubSubject({
            "claims": [{"text": "Metformin is first-line therapy.", "cites": ["C1"]}],
            "insufficient": False,
        })
        pipeline, _ = build(corpus, subject, ["supported"], store=store)
        pipeline.ask("is metformin first line")
        assert store.read()[0]["claims"][0]["checks"]


class TestCapabilityWiring:
    def test_unknown_capability_raises(self, corpus):
        from medassist.capabilities.registry import UnknownCapability

        pipeline, _ = build(corpus, StubSubject({"claims": []}), [])
        with pytest.raises(UnknownCapability):
            pipeline.ask("q", capability="delete_everything")

    def test_stricter_capability_gates_harder(self, corpus):
        """A MEDIUM-risk capability abstains where a LOW-risk one releases."""
        payload = {
            "claims": [
                {"text": "Metformin is first-line therapy.", "cites": ["C1"]},
                {"text": "The dose is 9999 mg.", "cites": ["C1"], "is_dosage": True},
                {"text": "Type 2 diabetes is common.", "cites": ["C1"]},
            ],
            "insufficient": False,
        }
        low, _ = build(corpus, StubSubject(payload), ["supported"] * 3,
                       spec=GateSpec(min_supported_fraction=0.6))
        high, _ = build(corpus, StubSubject(payload), ["supported"] * 3,
                        spec=GateSpec(min_supported_fraction=0.8))
        assert low.ask("q").decision is not ResponseDecision.ABSTAIN
        assert high.ask("q").decision is ResponseDecision.ABSTAIN


class TestApiSurface:
    def test_routes_are_registered(self):
        from medassist.serving.api import app

        paths = {r.path for r in app.routes if hasattr(r, "methods")}
        assert {"/v1/ask", "/v1/healthz", "/v1/capabilities", "/v1/reports",
                "/v1/reliability", "/v1/runs/{run_id}/trace"} <= paths

    def test_capabilities_endpoint_lists_all_six(self):
        from medassist.serving.api import capabilities

        listed = capabilities()
        assert len(listed) == 6
        assert all(c["allowed_tools"] for c in listed)

    def test_report_endpoint_parses_and_redacts(self):
        from medassist.serving.api import ReportRequest, parse_lab_report

        payload = parse_lab_report(ReportRequest(
            text="Patient Name: Jane Doe\nHbA1c: 8.2 %\nPotassium 6.8 mmol/L"
        ))
        assert payload["contained_pii"] is True
        assert payload["requires_escalation"] is True
        assert any(a["name"] == "HbA1c" for a in payload["analytes"])

    def test_report_endpoint_computes_trends(self):
        from medassist.serving.api import ReportRequest, parse_lab_report

        payload = parse_lab_report(ReportRequest(
            text="HbA1c: 8.2 %", previous="HbA1c: 6.9 %"
        ))
        assert payload["trends"] and "rising" in payload["trends"][0]

    def test_unknown_capability_is_a_400(self):
        from fastapi import HTTPException

        from medassist.serving.api import AskRequest, ask

        with pytest.raises(HTTPException) as excinfo:
            ask(AskRequest(question="q", capability="nope"))
        assert excinfo.value.status_code == 400


class TestCli:
    def test_parser_accepts_every_command(self):
        from medassist.cli import build_parser

        parser = build_parser()
        for argv in (
            ["corpus", "build"], ["corpus", "list"], ["ask", "q"],
            ["ask", "q", "--capability", "lifestyle_guidance", "--explain"],
            ["report", "x.txt"], ["redteam"], ["serve", "--port", "9000"],
        ):
            assert parser.parse_args(argv)

    def test_redteam_command_passes(self, capsys):
        from medassist.cli import main

        assert main(["redteam"]) == 0
        assert "defended" in capsys.readouterr().out


class TestUiImports:
    def test_ui_module_is_importable(self):
        pytest.importorskip("streamlit")
        import medassist.ui.app as ui

        assert hasattr(ui, "main")
