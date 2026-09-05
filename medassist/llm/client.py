"""Model gateway: retries, cost accounting, a content-addressed cache, and
structured output with a repair pass.

Two decisions worth explaining.

**The cache is keyed by the full request, on disk.** Evaluation runs the same
prompts many times across arms; without a cache the harness is dominated by
provider latency and cost, and worse, results drift between runs for reasons
that have nothing to do with the change under test. Keying on a hash of
(model, messages, temperature, max_tokens, response_format) makes a re-run of
an unchanged arm free and byte-identical.

**Reasoning traces are stripped before parsing.** Several current models emit a
``<think>`` block ahead of their answer. That text is not part of the response
contract, and letting it reach a JSON parser produces failures that look like
model errors but are formatting artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from medassist.core.config import SETTINGS, Settings
from medassist.core.errors import ConfigError, ProviderError, StructuredOutputError
from medassist.core.models import Usage

#: Floor for the adaptive max_tokens reduction below; under this a response
#: is too short to be worth returning, so the error surfaces instead.
_MIN_MAX_TOKENS = 256

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Approximate USD per 1M tokens. Kept explicit rather than looked up at runtime
# so a cost number in a report is reproducible; refresh when pricing moves.
_PRICES: dict[str, tuple[float, float]] = {
    "qwen/qwen3.6-27b": (0.29, 0.59),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks, including one left unterminated by truncation."""
    text = _THINK.sub("", text)
    if "<think>" in text.lower():
        idx = text.lower().rindex("<think>")
        closing = text.lower().find("</think>", idx)
        text = text[:idx] if closing == -1 else text[:idx] + text[closing + 8 :]
    return text.strip()


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Tries, in order: the whole string, a fenced block, then the outermost
    brace- or bracket-delimited span. Models wrap JSON in prose often enough
    that failing on the first attempt would misreport a formatting quirk as a
    capability failure.
    """
    text = strip_reasoning(text).strip()
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise StructuredOutputError(f"no JSON value found in response: {text[:300]!r}")


def _json_candidates(text: str) -> list[str]:
    out = [text]
    fenced = _FENCE.search(text)
    if fenced:
        out.append(fenced.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            out.append(text[start : end + 1])
    return out


@dataclass
class Completion:
    text: str
    usage: Usage
    model: str
    cached: bool = False
    finish_reason: str = "stop"

    def json(self) -> Any:
        return extract_json(self.text)


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, path)


class ModelClient:
    """Synchronous OpenAI-compatible chat client."""

    def __init__(
        self,
        model: str | None = None,
        *,
        settings: Settings = SETTINGS,
        use_cache: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.model = model or settings.subject_model
        self.cache = DiskCache(settings.cache_dir) if use_cache else None
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.Client:
        if self._client is None:
            if not self.settings.api_key:
                raise ConfigError(
                    "no API key. Set MEDASSIST_API_KEY (or GROQ_API_KEY) in .env"
                )
            self._client = httpx.Client(
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout_s,
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        json_mode: bool = False,
        model: str | None = None,
        reasoning_effort: str = "",
    ) -> Completion:
        model = model or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            # Reasoning tokens count against the output budget. Where a
            # provider exposes a dial for them, turning it down is the
            # difference between fitting the budget and spending all of it
            # thinking. Accepted values differ per model, so this is a
            # passthrough rather than an enum.
            payload["reasoning_effort"] = reasoning_effort
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        cache_key = DiskCache.key(payload)
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                usage = Usage(**hit["usage"])
                usage.cache_hits = 1
                return Completion(
                    text=hit["text"], usage=usage, model=model, cached=True,
                    finish_reason=hit.get("finish_reason", "stop"),
                )

        try:
            raw = self._post(payload)
        except ProviderError as exc:
            # Reasoning models emit a <think> preamble, which a provider's
            # strict JSON mode rejects before the model ever finishes. The
            # content is usually fine - `extract_json` strips the preamble and
            # recovers the object - so drop the constraint and parse leniently
            # rather than failing a request the model could answer.
            if not (json_mode and exc.status == 400 and "json_validate_failed" in str(exc)):
                raise
            payload.pop("response_format", None)
            raw = self._post(payload)

        choice = raw["choices"][0]
        text = strip_reasoning(choice["message"].get("content") or "")
        raw_usage = raw.get("usage", {})
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
        completion_tokens = int(raw_usage.get("completion_tokens", 0))
        usage = Usage(
            steps=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self.cost(model, prompt_tokens, completion_tokens),
        )
        result = Completion(
            text=text, usage=usage, model=model,
            finish_reason=choice.get("finish_reason", "stop"),
        )
        if self.cache is not None:
            self.cache.put(
                cache_key,
                {
                    "text": text,
                    "usage": usage.model_dump(),
                    "finish_reason": result.finish_reason,
                },
            )
        return result

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with backoff. Retries only what is actually retryable."""
        last: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                response = self._http().post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                last = ProviderError(f"transport error: {exc}", retryable=True)
            else:
                if response.status_code < 300:
                    return response.json()
                retryable = response.status_code == 429 or response.status_code >= 500
                last = ProviderError(
                    f"provider returned {response.status_code}: {response.text[:200]}",
                    status=response.status_code,
                    retryable=retryable,
                )
                if not retryable:
                    raise last

                # A 429 saying the *request* is too large is not congestion and
                # will never succeed on retry - the account's output-tokens-per-
                # minute ceiling is below what we asked for. Waiting changes
                # nothing; asking for less does. Halve and retry immediately.
                if (
                    response.status_code == 429
                    and "too large" in response.text.lower()
                    and payload.get("max_tokens", 0) > _MIN_MAX_TOKENS
                ):
                    payload["max_tokens"] = max(
                        _MIN_MAX_TOKENS, int(payload["max_tokens"]) // 2
                    )
                    continue
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 30.0))
                        continue
                    except ValueError:
                        pass
            # Exponential backoff with jitter: synchronised retries from a
            # parallel fan-out would otherwise re-collide on every attempt.
            time.sleep(min(2**attempt + random.random(), 20.0))
        raise last or ProviderError("exhausted retries")

    def structured(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        model: str | None = None,
        repair: bool = True,
        reasoning_effort: str = "",
    ) -> tuple[Any, Usage]:
        """Ask for JSON and get JSON, or raise.

        One repair round-trip is allowed: the model is shown its own malformed
        output and the parse error. This recovers the common near-miss (a
        trailing comma, a stray sentence) without masking a model that cannot
        produce the shape at all - which is a real capability signal we want to
        keep visible rather than retry into the ground.
        """
        completion = self.complete(
            messages, temperature=temperature, max_tokens=max_tokens,
            json_mode=True, model=model, reasoning_effort=reasoning_effort,
        )
        usage = completion.usage

        # A reasoning model that spends its whole completion budget on the
        # preamble returns an empty string once the trace is stripped. That is
        # budget exhaustion, not malformed output, and a repair round-trip
        # would truncate identically - so name it and stop.
        if completion.finish_reason == "length" and not completion.text.strip():
            raise StructuredOutputError(
                f"{completion.model} produced no output within max_tokens="
                f"{max_tokens}: the response was truncated, most likely by a "
                "reasoning preamble. Raise max_tokens, lower reasoning_effort, "
                "or use a model that does not emit one."
            )

        try:
            return completion.json(), usage
        except StructuredOutputError as first_error:
            if not repair:
                raise
            retry = self.complete(
                [
                    *messages,
                    {"role": "assistant", "content": completion.text[:2000]},
                    {
                        "role": "user",
                        "content": (
                            f"That response could not be parsed as JSON ({first_error}). "
                            "Reply with the JSON object only - no prose, no code fence."
                        ),
                    },
                ],
                temperature=0.0, max_tokens=max_tokens, json_mode=True, model=model,
            )
            usage.absorb(retry.usage)
            return retry.json(), usage

    @staticmethod
    def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_price, completion_price = _PRICES.get(model, (0.0, 0.0))
        return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
