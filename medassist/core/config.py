"""Settings. Read once, at the composition root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Load this project's own .env, and only this project's.

    An earlier version also read a sibling project's .env as a convenience.
    That silently pinned a different subject model than the one configured
    here, and the resulting failures looked like model incompetence rather
    than a config leak. A standalone repository reads its own settings.
    """
    for candidate in (ROOT / ".env",):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    api_key: str = field(
        default_factory=lambda: _env("MEDASSIST_API_KEY", "GROQ_API_KEY", "AEGIS_GROQ_API_KEY")
    )
    base_url: str = field(
        default_factory=lambda: _env(
            "MEDASSIST_BASE_URL",
            "AEGIS_GROQ_BASE_URL",
            default="https://api.groq.com/openai/v1",
        )
    )
    # The agent under test. Deliberately separate from the judge: a model that
    # grades its own output is not an evaluation, it is a self-assessment.
    subject_model: str = field(
        default_factory=lambda: _env(
            "MEDASSIST_SUBJECT_MODEL", "AEGIS_SUBJECT_MODEL", default="openai/gpt-oss-20b"
        )
    )
    judge_model: str = field(
        default_factory=lambda: _env(
            "MEDASSIST_JUDGE_MODEL", "AEGIS_JUDGE_MODEL", default="openai/gpt-oss-120b"
        )
    )
    tavily_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    ncbi_key: str = field(default_factory=lambda: _env("NCBI_API_KEY"))
    contact_email: str = field(
        default_factory=lambda: _env("MEDASSIST_CONTACT_EMAIL", default="medassist@example.com")
    )

    corpus_dir: Path = ROOT / "datasets" / "corpus"
    gold_dir: Path = ROOT / "datasets" / "gold"
    cache_dir: Path = ROOT / ".cache" / "llm"
    artifacts_dir: Path = ROOT / "artifacts"

    seed: int = 20260905
    request_timeout_s: float = 90.0
    max_retries: int = 4

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


SETTINGS = Settings()
