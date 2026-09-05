"""Content-addressed corpus snapshots.

An evaluation number is meaningless without the corpus it was measured
against. "Recall@10 went from 0.61 to 0.74" is only a result if the corpus was
identical; otherwise it may just record that someone re-scraped PubMed and got
three more papers.

So a snapshot is written once, hashed over its content, and referred to by that
hash. Every ``RunRecord`` carries the hash it ran against, and the eval report
refuses to compare two runs from different snapshots without saying so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from medassist.core.config import SETTINGS
from medassist.core.enums import SourceKind
from medassist.core.errors import CorpusError
from medassist.core.models import Document


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    document_count: int
    total_chars: int
    by_source: dict[str, int]
    built_at: str
    spec_name: str = ""
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def content_hash(documents: list[Document]) -> str:
    """Hash over content, not over object identity or fetch order.

    ``source_uid`` and text are the only inputs. Document ids are freshly
    minted ULIDs on every scrape, so hashing them would make every rebuild a
    different snapshot even when nothing changed - defeating the entire point.
    """
    digest = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.source_uid):
        digest.update(doc.source_uid.encode())
        digest.update(b"\x00")
        digest.update(doc.text.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class SnapshotStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or SETTINGS.corpus_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, snapshot_id: str) -> Path:
        return self.root / snapshot_id

    def write(self, documents: list[Document], *, spec_name: str = "", notes: str = "") -> SnapshotManifest:
        if not documents:
            raise CorpusError("refusing to write an empty snapshot")
        snapshot_id = content_hash(documents)
        target = self.path(snapshot_id)
        target.mkdir(parents=True, exist_ok=True)

        by_source: dict[str, int] = {}
        with (target / "documents.jsonl").open("w", encoding="utf-8") as handle:
            for doc in sorted(documents, key=lambda d: d.source_uid):
                handle.write(doc.model_dump_json() + "\n")
                by_source[doc.source.value] = by_source.get(doc.source.value, 0) + 1

        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            document_count=len(documents),
            total_chars=sum(len(d.text) for d in documents),
            by_source=by_source,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            spec_name=spec_name,
            notes=notes,
        )
        (target / "manifest.json").write_text(manifest.to_json())
        (self.root / "LATEST").write_text(snapshot_id)
        return manifest

    def read(self, snapshot_id: str | None = None) -> tuple[list[Document], SnapshotManifest]:
        snapshot_id = snapshot_id or self.latest()
        if not snapshot_id:
            raise CorpusError(
                "no corpus snapshot found. Build one with: medassist corpus build"
            )
        target = self.path(snapshot_id)
        if not (target / "documents.jsonl").exists():
            raise CorpusError(f"snapshot {snapshot_id} not found under {self.root}")

        documents = [
            Document.model_validate_json(line)
            for line in (target / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest_data = json.loads((target / "manifest.json").read_text())
        manifest_data.pop("extra", None)
        manifest = SnapshotManifest(**manifest_data, extra={})

        # Verify rather than trust. A silently-edited snapshot would invalidate
        # every number measured against it, and the failure would be invisible.
        actual = content_hash(documents)
        if actual != snapshot_id:
            raise CorpusError(
                f"snapshot {snapshot_id} is corrupt: content hashes to {actual}. "
                "Results measured against it cannot be trusted."
            )
        return documents, manifest

    def latest(self) -> str:
        pointer = self.root / "LATEST"
        return pointer.read_text().strip() if pointer.exists() else ""

    def list_snapshots(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.json").exists())


def source_of(uid: str) -> SourceKind:
    prefix = uid.split(":", 1)[0]
    return {
        "PMID": SourceKind.PUBMED_ABSTRACT,
        "FDA": SourceKind.FDA_LABEL,
        "MEDLINEPLUS": SourceKind.MEDLINEPLUS,
    }.get(prefix, SourceKind.UNKNOWN)
