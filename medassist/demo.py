"""Live end-to-end demo of what is currently BUILT.

Scrapes a small real corpus, snapshots it, indexes it, runs three queries, and
prints the retrieval trace with per-stage survivor counts and failure
attribution. Deliberately shows the machinery rather than a chat transcript -
the trace is the part that distinguishes this from a RAG chatbot.

    make demo
"""

from __future__ import annotations

import sys

from medassist.core.enums import RetrievalFailure
from medassist.corpus.http import Fetcher
from medassist.corpus.snapshot import SnapshotStore
from medassist.corpus.sources import medlineplus, openfda, pubmed
from medassist.index.embed import HashingEmbedding
from medassist.index.pipeline import Retriever

DRUGS = ["metformin", "warfarin", "lisinopril", "atorvastatin"]
TOPICS = ["type 2 diabetes", "high blood pressure", "cholesterol"]
LITERATURE = ["metformin first-line therapy type 2 diabetes", "warfarin vitamin K interaction"]

QUERIES = [
    ("what is the maximum daily dose of metformin", "drug_info"),
    ("can I eat leafy greens while taking warfarin", "drug_interaction"),
    ("what should I eat if I have type 2 diabetes", "condition_overview"),
]


def build_corpus() -> tuple[list, str]:
    documents = []
    with Fetcher() as fetcher:
        for drug in DRUGS:
            print(f"  openFDA      {drug}")
            documents += openfda.fetch_drug(fetcher, drug, limit=1)
        for topic in TOPICS:
            print(f"  MedlinePlus  {topic}")
            documents += medlineplus.search(fetcher, topic, retmax=2)
        for query in LITERATURE:
            print(f"  PubMed       {query}")
            documents += pubmed.fetch(fetcher, pubmed.search(fetcher, query, retmax=4))

    # Sources overlap - the same MedlinePlus topic answers several queries.
    unique = {d.source_uid: d for d in documents}
    manifest = SnapshotStore().write(list(unique.values()), spec_name="demo")
    return list(unique.values()), manifest.snapshot_id


def main() -> int:
    print("\n\033[1mBuilding corpus from live public APIs\033[0m")
    documents, snapshot_id = build_corpus()
    by_source: dict[str, int] = {}
    for doc in documents:
        by_source[doc.source.value] = by_source.get(doc.source.value, 0) + 1
    print(f"\n  snapshot {snapshot_id}  ·  {len(documents)} documents  ·  {by_source}")

    retriever = Retriever(embedding=HashingEmbedding()).build(documents, snapshot_id=snapshot_id)
    print(f"  indexed  {len(retriever)} chunks\n")

    for query, intent in QUERIES:
        print(f"\033[1m? {query}\033[0m   [{intent}]")
        result = retriever.retrieve(query, intent=intent)

        survivors = "  ".join(
            f"{s.stage.value}:{len(s.chunk_ids)}" for s in result.trace.stages
        )
        print(f"  stages   {survivors}")
        print(f"  dropped by context budget: {len(result.trace.truncated_by_budget)}")

        for i, chunk_id in enumerate(result.chunk_ids[:3], start=1):
            chunk = retriever.get(chunk_id)
            assert chunk is not None
            label = f"{chunk.source.value}/{chunk.section or 'body'}"
            print(f"    [C{i}] {label:<46} {chunk.text[:66].strip()}...")

        # Attribution needs hand-labelled gold chunks; with none, the honest
        # output is that the question was not asked, not a fabricated verdict.
        verdict = result.trace.attribute(set(), retriever.indexed_ids)
        assert verdict is RetrievalFailure.NONE
        print("    (failure attribution requires gold labels - see docs/spec/08)\n")

    print("\033[1mBuilt today:\033[0m corpus, snapshots, hybrid retrieval, full trace.")
    print("\033[1mNext (docs/spec/12):\033[0m gold set + retrieval metrics with attribution.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
