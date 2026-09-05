"""Prefixed, time-sortable identifiers.

Three properties earn the small amount of complexity here:

1. They sort lexicographically by creation time, so paginating by id is an
   index seek rather than a sort.
2. The prefix makes an id self-describing in a log line or a trace.
3. A wrong id *type* fails at construction rather than silently missing a
   lookup later - ``ChunkId("doc_...")`` raises.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32, no I L O U


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


_MAX_RAND = (1 << 80) - 1
_lock = threading.Lock()
_last_ms = -1
_last_rand = 0


def _ulid() -> str:
    """48 bits of milliseconds + 80 bits of randomness, base32-encoded.

    **Monotonic within a millisecond.** Plain ULIDs draw fresh randomness every
    call, so two ids minted in the same millisecond sort arbitrarily. That
    would quietly break the sortability property this module advertises:
    chunking one document mints ~100 ids in well under a millisecond, and none
    of them would order by creation.

    So within a millisecond the random component is incremented rather than
    redrawn, which is the ULID spec's own monotonicity rule. The counter is
    seeded with the high bit clear, leaving 2^79 headroom before it could
    overflow into the next millisecond - unreachable in practice, and it
    re-seeds on the next tick regardless.
    """
    global _last_ms, _last_rand
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    with _lock:
        if ms == _last_ms:
            _last_rand = (_last_rand + 1) & _MAX_RAND
        else:
            _last_ms = ms
            _last_rand = int.from_bytes(os.urandom(10), "big") & (_MAX_RAND >> 1)
        rand = _last_rand
    return _encode(ms, 10) + _encode(rand, 16)


class PrefixedId(str):
    """A ``str`` subclass that validates its own prefix."""

    prefix: str = "id"
    __test__ = False  # keep pytest from collecting TestCaseId as a test class

    def __new__(cls, value: str) -> PrefixedId:
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} must be built from a str, got {type(value)!r}")
        expected = f"{cls.prefix}_"
        if not value.startswith(expected):
            raise ValueError(
                f"{cls.__name__} must start with {expected!r}, got {value!r}. "
                "This is a type error, not a missing record."
            )
        return super().__new__(cls, value)

    @classmethod
    def new(cls) -> PrefixedId:
        return cls(f"{cls.prefix}_{_ulid()}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema(), serialization=core_schema.to_string_ser_schema()
        )


class DocumentId(PrefixedId):
    prefix = "doc"


class ChunkId(PrefixedId):
    prefix = "chk"


class RunId(PrefixedId):
    prefix = "run"


class NodeId(PrefixedId):
    prefix = "node"


class ClaimId(PrefixedId):
    prefix = "clm"


class CaseId(PrefixedId):
    prefix = "case"


class EvalId(PrefixedId):
    prefix = "eval"
