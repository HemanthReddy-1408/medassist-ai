"""A polite HTTP fetcher for public medical APIs.

Rate limiting is not optional here. NCBI E-utilities permits 3 requests/second
without an API key and will start returning 429s - and eventually block a
client - above that. The raw-response cache matters for the same reason it
matters for the model: a corpus rebuild should not re-hammer a public service
that is offering the data for free.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from pathlib import Path

import httpx

from medassist.core.config import SETTINGS
from medassist.core.errors import CorpusError


class RateLimiter:
    """Minimum interval between calls, per host, thread-safe."""

    def __init__(self) -> None:
        self._last: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def wait(self, host: str, min_interval: float) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last[host]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last[host] = time.monotonic()


_LIMITER = RateLimiter()


class Fetcher:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        min_interval: float = 0.34,
        timeout: float = 30.0,
        user_agent: str = "",
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = cache_dir or (SETTINGS.corpus_dir.parent / ".rawcache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        agent = user_agent or f"medassist/2.0 (+{SETTINGS.contact_email})"
        self._client = httpx.Client(
            timeout=timeout, headers={"User-Agent": agent}, follow_redirects=True
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, url: str, params: dict[str, str | int] | None = None, *, use_cache: bool = True) -> str:
        key = hashlib.sha256(f"{url}?{sorted((params or {}).items())}".encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.txt"
        if use_cache and path.exists():
            return path.read_text(encoding="utf-8")

        host = httpx.URL(url).host or "unknown"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            _LIMITER.wait(host, self.min_interval)
            try:
                response = self._client.get(url, params=params)
            except httpx.RequestError as exc:
                last = CorpusError(f"GET {url} failed: {exc}")
            else:
                if response.status_code < 300:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(response.text, encoding="utf-8")
                    return response.text
                if response.status_code in (429, 500, 502, 503, 504):
                    last = CorpusError(f"GET {url} -> {response.status_code}")
                else:
                    raise CorpusError(f"GET {url} -> {response.status_code}: {response.text[:200]}")
            time.sleep(2**attempt)
        raise last or CorpusError(f"GET {url}: retries exhausted")
