"""OpenAIProvider tests for issue #5.

Verifies:
1. Default model is 'gpt-4o-mini' when OPENAI_MODEL env var is unset
   (previously the default was the non-existent 'gpt-5.4-nano').
2. AuthenticationError from JSON-mode propagates instead of being
   swallowed by a misleading 'fell back' warning.
3. BadRequestError from JSON-mode triggers the plain-completion fallback.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xclinvision.agent.llm_provider import OpenAIProvider  # noqa: E402


def _api_status_error(cls, status_code: int, body: dict | None = None):
    """Build an OpenAI APIStatusError-derived exception with the right shape."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return cls(message="boom", response=response, body=body or {})


def test_default_model_is_gpt_4o_mini(monkeypatch):
    """Issue #5: unconfigured deploys must default to a real model id."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    p = OpenAIProvider(api_key="sk-test")
    assert p._model == "gpt-4o-mini"


def test_authentication_error_propagates_from_json_mode(monkeypatch):
    """Issue #5: auth errors must surface, not be hidden as 'falling back'."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    p = OpenAIProvider(api_key="sk-test")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _api_status_error(
        AuthenticationError, status_code=401
    )

    with patch.object(p, "_get_client", return_value=fake_client):
        with pytest.raises(AuthenticationError):
            p.call("sys", "user")

    # JSON-mode call attempted exactly once — no silent fallback retry.
    assert fake_client.chat.completions.create.call_count == 1


def test_bad_request_triggers_plain_fallback(monkeypatch):
    """Issue #5: BadRequestError (response_format unsupported) still falls back."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    p = OpenAIProvider(api_key="sk-test", model="some-model-without-json-mode")

    fake_client = MagicMock()
    json_mode_err = _api_status_error(BadRequestError, status_code=400)

    plain_response = MagicMock()
    plain_response.choices = [MagicMock(message=MagicMock(content='{"k": 1}'))]
    plain_response.usage = None

    # First call (JSON mode) raises BadRequestError; subsequent calls return ok.
    fake_client.chat.completions.create.side_effect = [json_mode_err, plain_response]

    with patch.object(p, "_get_client", return_value=fake_client):
        out = p.call("sys", "user")

    assert out == '{"k": 1}'
    assert fake_client.chat.completions.create.call_count == 2


def test_all_candidate_models_failing_raises_not_none(monkeypatch):
    """Issue 6: the plain-completion loop must not fall through returning None.

    When OPENAI_MODEL equals the hardcoded fallback (the default — both are
    'gpt-4o-mini'), dict.fromkeys collapses the loop to one iteration. The old
    code logged and fell off the end, so LLMManager.call_llm saw a non-raising
    None and never tried the next provider.
    """
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    bad_request = BadRequestError(
        "response_format unsupported",
        response=httpx.Response(400, request=request),
        body=None,
    )
    rate_limited = RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=request),
        body=None,
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [bad_request, rate_limited]
    provider._client = client

    with pytest.raises(RuntimeError, match="all candidate models failed"):
        provider.call("system", "user")


def test_both_distinct_models_rate_limited_raises(monkeypatch):
    """Issue 6: the two-iteration path must raise too, not fall through.

    test_all_candidate_models_failing_raises_not_none covers the default
    config, where OPENAI_MODEL equals the fallback and dict.fromkeys collapses
    the loop to one entry. This covers a distinct primary model, so the loop
    actually runs twice before giving up.
    """
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
    assert provider._model != provider._FALLBACK_MODEL

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    bad_request = BadRequestError(
        "response_format unsupported",
        response=httpx.Response(400, request=request),
        body=None,
    )
    rate_limited = [
        RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=request),
            body=None,
        )
        for _ in range(2)
    ]

    client = MagicMock()
    client.chat.completions.create.side_effect = [bad_request, *rate_limited]
    provider._client = client

    with pytest.raises(RuntimeError, match="all candidate models failed"):
        provider.call("system", "user")

    # JSON mode + primary plain + fallback plain
    assert client.chat.completions.create.call_count == 3
