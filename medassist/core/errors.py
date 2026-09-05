"""Typed failures. Each one names a distinct recovery."""

from __future__ import annotations


class MedAssistError(Exception):
    """Base for everything this package raises deliberately."""


class ConfigError(MedAssistError):
    """A required setting is missing. Not retryable."""


class ProviderError(MedAssistError):
    """The model provider failed. Retryable if ``retryable`` is set."""

    def __init__(self, message: str, *, status: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class BudgetExhausted(MedAssistError):
    """A limit was reached. Carries which one, because that is the measurement."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class StructuredOutputError(MedAssistError):
    """The model returned something that is not the requested shape.

    Deliberately distinct from ProviderError: the call succeeded, the content
    is wrong. Retrying with a repair prompt is reasonable; retrying the HTTP
    request is not.
    """


class CorpusError(MedAssistError):
    """A source could not be fetched or normalized."""


class GuardBlocked(MedAssistError):
    """A safety guard refused to release an answer."""

    def __init__(self, guard: str, message: str) -> None:
        super().__init__(f"[{guard}] {message}")
        self.guard = guard
