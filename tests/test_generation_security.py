"""Security-focused tests for rag.generation.chain.AnswerGenerator.

These guard against a real incident: a malformed GROQ_API_KEY (trailing
newline) caused httpx/httpcore to raise an exception whose message embedded
the raw 'Authorization: Bearer <key>' header value, which then ended up in
server logs verbatim. AnswerGenerator must never let that raw value escape
into a GenerationError message (which is what API responses and logs see).
"""
from __future__ import annotations

import httpx
import pytest

from rag.generation.chain import AnswerGenerator, GenerationError

FAKE_KEY = "gsk_super_secret_value_that_must_never_leak_12345"


class _ExplodingChain:
    """Stand-in for the LCEL chain that always raises on invoke()."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, _input):
        raise self._exc


def _generator_with_chain(monkeypatch, exc: Exception) -> AnswerGenerator:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    gen = AnswerGenerator()
    # Bypass _get_chain()/ChatGroq construction entirely — we only care
    # about how generate() translates a raised exception.
    gen._chain = _ExplodingChain(exc)
    return gen


def test_api_key_is_stripped_of_whitespace(monkeypatch):
    """A trailing newline/space in the secret (the actual bug we hit) must
    not survive into the value AnswerGenerator hands to the Groq client."""
    monkeypatch.setenv("GROQ_API_KEY", f"{FAKE_KEY}\n")
    gen = AnswerGenerator()
    assert gen._api_key == FAKE_KEY
    assert "\n" not in gen._api_key


def test_malformed_header_error_does_not_leak_key_in_message(monkeypatch):
    """Simulates the real incident: httpx raises LocalProtocolError whose
    message contains the raw Authorization header value."""
    leaking_exc = httpx.LocalProtocolError(
        f"Illegal header value b'Bearer {FAKE_KEY}\\n'"
    )
    gen = _generator_with_chain(monkeypatch, leaking_exc)

    with pytest.raises(GenerationError) as exc_info:
        gen.generate("question", [])

    err = exc_info.value
    assert err.status_code == 502
    assert FAKE_KEY not in err.message
    assert FAKE_KEY not in str(err)


def test_generation_error_str_never_contains_key(monkeypatch):
    """Belt-and-suspenders: whatever the underlying Groq/httpx exception
    says, str(GenerationError) — which is what ends up in logs/API
    responses — must never contain the configured API key."""
    import groq

    leaking_exc = groq.APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )
    gen = _generator_with_chain(monkeypatch, leaking_exc)

    with pytest.raises(GenerationError) as exc_info:
        gen.generate("question", [])

    assert FAKE_KEY not in str(exc_info.value)
    assert FAKE_KEY not in exc_info.value.message