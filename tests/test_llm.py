"""Model gateway: response parsing, caching, cost. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from medassist.core.config import Settings
from medassist.core.errors import ProviderError, StructuredOutputError
from medassist.llm.client import DiskCache, ModelClient, extract_json, strip_reasoning


class TestReasoningStripping:
    def test_removes_a_think_block(self):
        assert strip_reasoning("<think>deliberating</think>The answer.") == "The answer."

    def test_handles_a_block_truncated_by_max_tokens(self):
        # A reasoning model cut off mid-thought must not leak the trace.
        assert strip_reasoning("prefix <think>cut off here") == "prefix"

    def test_leaves_ordinary_text_alone(self):
        assert strip_reasoning("  plain answer  ") == "plain answer"


class TestJsonExtraction:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"a": 1}',
            'Here you go:\n```json\n{"a": 1}\n```',
            'prose before {"a": 1} prose after',
            '<think>reasoning</think>{"a": 1}',
        ],
    )
    def test_recovers_json_from_common_wrappings(self, raw):
        assert extract_json(raw) == {"a": 1}

    def test_recovers_a_top_level_array(self):
        assert extract_json("results: [1, 2]") == [1, 2]

    def test_raises_when_there_is_no_json(self):
        with pytest.raises(StructuredOutputError, match="no JSON"):
            extract_json("there is no json here at all")


class TestDiskCache:
    def test_key_is_order_independent(self):
        assert DiskCache.key({"a": 1, "b": 2}) == DiskCache.key({"b": 2, "a": 1})

    def test_roundtrip(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.put("k" * 64, {"text": "hi"})
        assert cache.get("k" * 64) == {"text": "hi"}

    def test_missing_key_returns_none(self, tmp_path):
        assert DiskCache(tmp_path).get("f" * 64) is None

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.put("a" * 64, {"text": "hi"})
        next(tmp_path.rglob("*.json")).write_text("{not json")
        assert cache.get("a" * 64) is None


def _stub(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://stub")


def _ok(text: str = "answer", prompt: int = 10, completion: int = 5):
    def handler(request: httpx.Request) -> httpx.Response:
        handler.calls += 1  # type: ignore[attr-defined]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            },
        )

    handler.calls = 0  # type: ignore[attr-defined]
    return handler


class TestModelClient:
    def test_records_usage_and_cost(self, tmp_path):
        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient("llama-3.1-8b-instant", settings=settings, client=_stub(_ok()))
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result.text == "answer"
        assert result.usage.total_tokens == 15
        assert result.usage.cost_usd == pytest.approx((10 * 0.05 + 5 * 0.08) / 1_000_000)

    def test_second_identical_call_hits_the_cache(self, tmp_path):
        handler = _ok()
        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient("llama-3.1-8b-instant", settings=settings, client=_stub(handler))
        messages = [{"role": "user", "content": "hi"}]
        client.complete(messages)
        second = client.complete(messages)
        assert second.cached is True
        assert handler.calls == 1  # type: ignore[attr-defined]

    def test_reasoning_is_stripped_before_caching(self, tmp_path):
        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient(
            "llama-3.1-8b-instant", settings=settings,
            client=_stub(_ok("<think>hmm</think>final")),
        )
        assert client.complete([{"role": "user", "content": "x"}]).text == "final"

    def test_client_error_is_not_retried(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            handler.calls += 1  # type: ignore[attr-defined]
            return httpx.Response(400, text="bad request")

        handler.calls = 0  # type: ignore[attr-defined]
        settings = Settings(api_key="test", cache_dir=tmp_path, max_retries=4)
        client = ModelClient(settings=settings, client=_stub(handler))
        with pytest.raises(ProviderError) as excinfo:
            client.complete([{"role": "user", "content": "x"}])
        assert excinfo.value.retryable is False
        assert handler.calls == 1  # type: ignore[attr-defined]

    def test_structured_repairs_one_malformed_response(self, tmp_path):
        responses = ["not json at all", '{"ok": true}']

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            index = min(len(body["messages"]) > 1, len(responses) - 1)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": responses[index]}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                },
            )

        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient(settings=settings, client=_stub(handler))
        value, usage = client.structured([{"role": "user", "content": "give me json"}])
        assert value == {"ok": True}
        assert usage.steps == 2  # the repair round-trip is accounted for, not hidden


class TestJsonModeFallback:
    """Reasoning models cannot satisfy strict JSON mode - they emit a preamble."""

    def test_json_validate_failure_retries_without_the_constraint(self, tmp_path):
        seen: list[bool] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            constrained = "response_format" in body
            seen.append(constrained)
            if constrained:
                return httpx.Response(
                    400,
                    json={"error": {"code": "json_validate_failed", "message": "Failed to validate JSON."}},
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "<think>hmm</think>{\"ok\": true}"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                },
            )

        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient(settings=settings, client=_stub(handler))
        value, _usage = client.structured([{"role": "user", "content": "json please"}])
        assert value == {"ok": True}
        assert seen == [True, False]  # constrained first, then relaxed

    def test_other_400s_still_raise(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "context too long"}})

        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient(settings=settings, client=_stub(handler))
        with pytest.raises(ProviderError, match="context too long"):
            client.complete([{"role": "user", "content": "x"}], json_mode=True)


class TestAdaptiveTokenBudget:
    """A 429 about request *size* is not congestion; retrying unchanged never works."""

    def test_request_too_large_halves_max_tokens_and_retries(self, tmp_path):
        asked: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            asked.append(body["max_tokens"])
            if body["max_tokens"] > 1000:
                return httpx.Response(
                    429,
                    json={"error": {"message": "Request too large on output tokens per minute (OTPM): Limit 1000"}},
                )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                },
            )

        settings = Settings(api_key="test", cache_dir=tmp_path, max_retries=6)
        client = ModelClient(settings=settings, client=_stub(handler))
        result = client.complete([{"role": "user", "content": "x"}], max_tokens=4000)
        assert result.text == "ok"
        assert asked == [4000, 2000, 1000]

    def test_ordinary_429_is_not_shrunk(self, tmp_path):
        asked: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            asked.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }) if len(asked) > 1 else httpx.Response(
                429, json={"error": {"message": "rate limit exceeded, slow down"}}
            )

        settings = Settings(api_key="test", cache_dir=tmp_path, max_retries=3)
        client = ModelClient(settings=settings, client=_stub(handler))
        client.complete([{"role": "user", "content": "x"}], max_tokens=800)
        assert asked == [800, 800]  # congestion: same request, after a backoff


class TestTruncationDiagnosis:
    def test_empty_truncated_response_is_named_not_reported_as_bad_json(self, tmp_path):
        """A reasoning preamble that eats the budget is budget exhaustion."""

        def handler(request: httpx.Request) -> httpx.Response:
            handler.calls += 1  # type: ignore[attr-defined]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "<think>thinking and thinking"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 900},
                },
            )

        handler.calls = 0  # type: ignore[attr-defined]
        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient(settings=settings, client=_stub(handler))
        with pytest.raises(StructuredOutputError, match="truncated"):
            client.structured([{"role": "user", "content": "x"}], max_tokens=900)
        assert handler.calls == 1  # no repair attempt: it would truncate identically

    def test_reasoning_effort_is_passed_through(self, tmp_path):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })

        settings = Settings(api_key="test", cache_dir=tmp_path)
        client = ModelClient(settings=settings, client=_stub(handler))
        client.complete([{"role": "user", "content": "x"}], reasoning_effort="low")
        assert seen[0]["reasoning_effort"] == "low"
