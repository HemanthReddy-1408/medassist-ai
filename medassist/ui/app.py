"""Streamlit surface.

Three panes: conversation, **evidence**, and **reliability**.

The evidence pane is the product. A chat window that hides its evidence is the
thing this project exists to argue against, so the cited span behind every
claim is one click away and every removed claim says which check removed it.

    streamlit run medassist/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from medassist.audit.records import RecordStore
from medassist.core.config import SETTINGS
from medassist.core.models import PatientProfile
from medassist.corpus.snapshot import SnapshotStore
from medassist.index.embed import default_embedding
from medassist.index.pipeline import Retriever
from medassist.llm.client import ModelClient
from medassist.reports.interpret import findings, requires_escalation
from medassist.reports.parse import parse_report
from medassist.serving.pipeline import Pipeline

DECISION_STYLE = {
    "release": ("✅", "#137333"),
    "release_with_caveat": ("⚠️", "#b06000"),
    "abstain": ("🤐", "#1a73e8"),
    "block": ("⛔", "#c5221f"),
    "escalate": ("🚑", "#8430ce"),
}
CLAIM_MARK = {"retain": "✓", "qualify": "~", "remove": "✗"}


@st.cache_resource(show_spinner="Loading corpus and building the index…")
def _pipeline():
    documents, manifest = SnapshotStore().read()
    retriever = Retriever(embedding=default_embedding()).build(
        documents, snapshot_id=manifest.snapshot_id
    )
    pipeline = Pipeline(
        retriever=retriever,
        subject_client=ModelClient(SETTINGS.subject_model),
        judge_client=ModelClient(SETTINGS.judge_model),
        record_store=RecordStore(SETTINGS.artifacts_dir / "decisions.jsonl"),
    )
    return pipeline, retriever, manifest


def sidebar() -> PatientProfile | None:
    st.sidebar.header("Patient record")
    st.sidebar.caption(
        "Used for the relational safety check: advice that is correct in "
        "general can be unsafe for a specific person."
    )
    age = st.sidebar.number_input("Age", 0, 120, 0)
    conditions = st.sidebar.text_input("Conditions (comma separated)")
    medications = st.sidebar.text_input("Medications (comma separated)", "warfarin 5mg")
    allergies = st.sidebar.text_input("Allergies (comma separated)")

    def split(value: str) -> list[str]:
        return [v.strip() for v in value.split(",") if v.strip()]

    if not (age or conditions or medications or allergies):
        return None
    return PatientProfile(
        age=age or None, conditions=split(conditions),
        medications=split(medications), allergies=split(allergies),
    )


def render_answer(result, retriever) -> None:
    icon, colour = DECISION_STYLE.get(result.decision.value, ("•", "#444"))
    st.markdown(
        f"<h4 style='color:{colour}'>{icon} {result.decision.value.replace('_',' ').upper()}</h4>",
        unsafe_allow_html=True,
    )
    st.write(result.prose)
    for caveat in result.caveats:
        st.warning(caveat)

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Evidence")
        if not result.citations:
            st.caption("No sources were cited.")
        for citation in result.citations:
            label = f"{citation['source']} · {citation['section'] or 'body'} — {citation['title'][:52]}"
            with st.expander(label):
                st.write(citation["quote"])
                if citation["url"]:
                    st.caption(citation["url"])

        if result.record is not None and result.record.claims:
            st.subheader("Per-claim verdicts")
            for claim in result.record.claims:
                mark = CLAIM_MARK[claim.decision.value]
                st.markdown(f"**{mark} {claim.text}**")
                if claim.decision.value != "retain":
                    st.caption(f"↳ {claim.explanation}")
                st.caption(" · ".join(f"{k}: {v}" for k, v in claim.checks.items()))

    with right:
        st.subheader("Reliability")
        if result.confidence is not None:
            st.metric("Confidence", f"{result.confidence.value:.2f}")
            st.caption(f"weakest factor: {result.confidence.weakest}")
            for name, value in sorted(result.confidence.factors.items()):
                st.progress(min(max(value, 0.0), 1.0), text=f"{name} {value:.2f}")
        if result.record is not None:
            st.caption(f"rule: {result.record.rule}")
            st.caption(f"snapshot: {result.record.corpus_snapshot}")
            st.caption(
                " · ".join(
                    f"{k} {v:.1f}ms" for k, v in result.record.stage_latency_ms.items()
                )
            )
        st.caption(f"{result.took_ms:.0f} ms · ${result.usage.cost_usd:.5f}")


def main() -> None:
    st.set_page_config(page_title="MedAssist X", page_icon="🩺", layout="wide")
    st.title("🩺 MedAssist X")
    st.caption(
        "Serving-time admission control for clinical answers. "
        "**Not clinically reliable — not a substitute for a clinician.**"
    )

    profile = sidebar()
    try:
        pipeline, retriever, manifest = _pipeline()
    except Exception as exc:
        st.error(f"No corpus available: {exc}\n\nBuild one with `medassist corpus build`.")
        return
    st.sidebar.divider()
    st.sidebar.caption(f"snapshot `{manifest.snapshot_id}` · {len(retriever)} chunks")

    ask_tab, report_tab = st.tabs(["Ask", "Interpret a report"])

    with ask_tab:
        capability = st.selectbox(
            "Capability",
            ["evidence_qa", "lifestyle_guidance", "interaction_check",
             "treatment_comparison", "literature_synthesis"],
        )
        question = st.text_input("Question", "What should I eat to keep my heart healthy?")
        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Retrieving, generating, gating…"):
                result = pipeline.ask(question, capability=capability, profile=profile)
            render_answer(result, retriever)

    with report_tab:
        text = st.text_area("Paste a lab report", height=200)
        if st.button("Interpret") and text.strip():
            parsed = parse_report(text, source_name="pasted")
            if parsed.contained_pii:
                st.info(f"Removed before processing: {', '.join(parsed.redaction.kinds)}")
            if requires_escalation(parsed.report):
                st.error("A value is in the critical range. Seek clinical review.")
            for analyte in parsed.report.analytes:
                st.write(f"- {analyte.describe()}")
            abnormal = findings(parsed.report)
            if abnormal:
                st.subheader("Findings, most severe first")
                for finding in abnormal:
                    st.write(f"- {finding.describe()}")


if __name__ == "__main__":
    main()
