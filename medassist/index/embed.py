"""Embedding backends behind one port.

Two implementations ship, and the point of having both is that the evaluation
harness can measure the difference rather than assume it:

``HashingEmbedding`` needs no model download and is fully deterministic, which
makes it the honest baseline and the CI default.

``SentenceTransformerEmbedding`` loads a real encoder. It should win on
paraphrase queries ("what can I eat" vs "dietary recommendations") where the
hashing backend has no signal at all. ``make eval-embeddings`` runs both arms
over the same gold set and reports whether the difference survives a
significance test - so the choice is a measurement, not a preference.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from medassist.core.config import SETTINGS

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "its", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with", "what", "which", "how", "when", "who", "whom", "why", "can", "could", "should", "would", "do", "does", "did", "have", "has", "had", "i", "you", "my", "your", "me", "we", "our"]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping intra-word hyphens and decimals.

    ``hba1c``, ``type-2`` and ``2.5`` must survive as single tokens: splitting
    ``2.5`` into ``2`` and ``5`` destroys precisely the dosage terms that matter
    most, and they are also the terms a lexical index is best at.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@runtime_checkable
class EmbeddingPort(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class HashingEmbedding:
    """Hashed word + character-trigram features, sublinear-tf weighted.

    Character trigrams are included so that morphological variants
    (``nephropathy`` / ``nephropathies``) and misspellings share signal, which
    plain word hashing does not give. Deterministic across processes because
    the hash is blake2b rather than Python's salted ``hash()``.
    """

    def __init__(self, dim: int = 768, char_ngrams: bool = True) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.char_ngrams = char_ngrams

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % dim

    def _features(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        words = tokenize(text)
        for word in words:
            counts[self._bucket(word, self.dim)] = counts.get(self._bucket(word, self.dim), 0.0) + 1.0
        if self.char_ngrams:
            for word in words:
                padded = f"^{word}$"
                for i in range(len(padded) - 2):
                    bucket = self._bucket("#" + padded[i : i + 3], self.dim)
                    counts[bucket] = counts.get(bucket, 0.0) + 0.35
        return counts

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for bucket, count in self._features(text).items():
                # Sublinear tf: a term repeated 40 times in a label section is
                # not 40 times as relevant.
                matrix[row, bucket] = 1.0 + np.log(count)
        return _l2(matrix)


class SentenceTransformerEmbedding:
    """A real encoder, loaded lazily so importing this module stays cheap."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.name = model_name.rsplit("/", 1)[-1]
        self._model = None
        self.dim = 384

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class CachedEmbedding:
    """Memoize encodings on disk, keyed by backend name and text hash.

    Re-embedding an unchanged corpus on every evaluation arm is the single
    largest avoidable cost in the harness.
    """

    def __init__(self, inner: EmbeddingPort, root: Path | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.root = (root or SETTINGS.cache_dir.parent / "embed") / self.name
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(text.encode()).hexdigest()
        return self.root / key[:2] / f"{key}.npy"

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                try:
                    out[i] = np.load(path)
                    continue
                except (OSError, ValueError):
                    pass
            missing.append(i)

        if missing:
            fresh = self.inner.encode([texts[i] for i in missing])
            self.dim = fresh.shape[1]
            for slot, i in enumerate(missing):
                vector = fresh[slot]
                out[i] = vector
                path = self._path(texts[i])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vector)

        return np.vstack([v for v in out if v is not None]).astype(np.float32)


def default_embedding(prefer_neural: bool = True) -> EmbeddingPort:
    """Neural if the dependency is present, hashing otherwise. Never fails."""
    if prefer_neural:
        try:
            import sentence_transformers  # noqa: F401

            return CachedEmbedding(SentenceTransformerEmbedding())
        except ImportError:
            pass
    return CachedEmbedding(HashingEmbedding())
