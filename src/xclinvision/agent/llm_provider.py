"""LLM Provider Interface — abstracts multiple LLM backends behind a uniform API.

Supports both synchronous calls and streaming (async generator) responses.
Implementations: OpenAI-compatible APIs and local models (e.g. Ollama, vLLM).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterator, Optional

logger = logging.getLogger(__name__)

#: How long a health-check verdict stays valid. The probe below is a real
#: billed completion, so it must not run once per inbound request.
HEALTH_CACHE_TTL_SECONDS = 60.0


def _completion_tokens_kwarg(model: str, value: int = 700) -> Dict[str, int]:
    """Return the correct token-limit parameter for the given model.

    OpenAI's newer *o-* and *gpt-5* family models require
    ``max_completion_tokens`` instead of the legacy ``max_tokens``.
    """
    if "gpt-5" in model or model.startswith("o"):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


class LLMProvider(ABC):
    """Base interface for all LLM providers.

    Every provider must implement:
    - ``call()``: synchronous completion returning the full text.
    - ``stream()``: generator yielding text chunks for real-time UI.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'openai', 'local')."""

    @abstractmethod
    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Synchronous LLM call — returns the full response text."""

    @abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Streaming LLM call — yields text chunks as they arrive."""

    def _probe(self) -> bool:
        """Issue one minimal live completion to confirm the provider works.

        This costs a real, billed request — call it through
        :meth:`health_check`, which caches the verdict.
        """
        try:
            response = self.call(
                "You are a test assistant.",
                "Reply with exactly: OK",
                temperature=0.0,
            )
            return bool(response and response.strip())
        except Exception:
            return False

    def health_check(self) -> bool:
        """Return True if the provider is reachable and functional.

        The verdict is cached for :data:`HEALTH_CACHE_TTL_SECONDS`. A racing
        pair of callers may both probe once; that is bounded and harmless,
        whereas probing per request is not.
        """
        cached = getattr(self, "_health_cache", None)
        now = time.monotonic()
        if cached is not None and now - cached[0] < HEALTH_CACHE_TTL_SECONDS:
            return cached[1]
        result = self._probe()
        self._health_cache = (now, result)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible provider
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (works with OpenAI, Azure, and compatible APIs).

    Configuration via environment variables:
    - ``OPENAI_API_KEY``  — required
    - ``OPENAI_MODEL``    — defaults to ``gpt-4o-mini`` (issue #5)
    - ``OPENAI_API_BASE`` — optional custom base URL
    """

    _FALLBACK_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        # Issue #5: previous default 'gpt-5.4-nano' is not a real model id;
        # every unconfigured deploy hit the fallback path on every call.
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._base_url = base_url or os.environ.get("OPENAI_API_BASE")
        self._client: Any = None
        logger.info("OpenAIProvider: model=%s, base_url=%s, key_set=%s",
                    self._model, self._base_url or "(default)", bool(self._api_key))

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install the 'openai' package: pip install openai"
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Try structured JSON mode first
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                **_completion_tokens_kwarg(self._model),
                response_format={"type": "json_object"},
            )
            if response.usage:
                logger.info(
                    "OpenAI usage [%s] prompt=%d completion=%d total=%d",
                    self._model, response.usage.prompt_tokens,
                    response.usage.completion_tokens, response.usage.total_tokens,
                )
            return response.choices[0].message.content or ""
        except Exception as exc:
            # Issue #5: only fall back when the failure is a feature/model
            # mismatch (BadRequestError = response_format unsupported,
            # NotFoundError = model id wrong). Auth, rate-limit, and
            # network errors must propagate so callers see real failures
            # instead of a misleading "fallback" warning.
            from openai import BadRequestError, NotFoundError
            if not isinstance(exc, (BadRequestError, NotFoundError)):
                raise
            logger.warning(
                "Structured JSON mode failed (%s); falling back to plain completion.",
                exc,
            )

        # Fallback: plain completion (try primary model, then fallback model)
        # dict.fromkeys de-duplicates when primary == fallback.
        for model in dict.fromkeys((self._model, self._FALLBACK_MODEL)):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **_completion_tokens_kwarg(model),
                )
                raw = response.choices[0].message.content or ""
                if response.usage:
                    logger.info(
                        "OpenAI usage [%s] prompt=%d completion=%d total=%d",
                        model, response.usage.prompt_tokens,
                        response.usage.completion_tokens, response.usage.total_tokens,
                    )
                if model != self._model:
                    logger.info("OpenAI call succeeded with fallback model '%s'", model)
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                return match.group() if match else raw
            except Exception as exc:
                if model == self._model:
                    logger.warning("Primary model '%s' failed: %s. Trying fallback '%s'.",
                                   model, exc, self._FALLBACK_MODEL)
                else:
                    raise

    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                **_completion_tokens_kwarg(self._model),
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.warning(
                "OpenAI stream with '%s' failed: %s. Trying fallback '%s'.",
                self._model, exc, self._FALLBACK_MODEL,
            )
            response = client.chat.completions.create(
                model=self._FALLBACK_MODEL,
                messages=messages,
                temperature=temperature,
                **_completion_tokens_kwarg(self._FALLBACK_MODEL),
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def health_check(self) -> bool:
        if not self._api_key:
            return False
        return super().health_check()


# ═══════════════════════════════════════════════════════════════════════════════
# Local model provider (Ollama / vLLM / any OpenAI-compatible local server)
# ═══════════════════════════════════════════════════════════════════════════════


class LocalProvider(LLMProvider):
    """Local model provider — connects to a locally-running LLM server.

    Supports any server exposing an OpenAI-compatible ``/v1/chat/completions``
    endpoint (e.g. Ollama, vLLM, llama.cpp, LocalAI).

    Configuration via environment variables:
    - ``LOCAL_LLM_BASE_URL`` — defaults to ``http://localhost:11434/v1``
    - ``LOCAL_LLM_MODEL``    — defaults to ``llama3.2``
    - ``LOCAL_LLM_API_KEY``  — defaults to ``ollama`` (required by some clients)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._base_url = base_url or os.environ.get(
            "LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"
        )
        self._model = model or os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
        self._api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY", "ollama")
        self._client: Any = None
        self._reachable: Optional[bool] = None

    @property
    def name(self) -> str:
        return "local"

    def _is_reachable(self) -> bool:
        """Quick TCP-level check to avoid long timeouts when server is down."""
        if self._reachable is not None:
            return self._reachable
        import socket
        try:
            # Parse host:port from base_url
            from urllib.parse import urlparse
            parsed = urlparse(self._base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 11434
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            self._reachable = True
        except (OSError, socket.timeout):
            self._reachable = False
            logger.info("LocalProvider: server at %s not reachable — disabled.", self._base_url)
        return self._reachable

    def _get_client(self) -> Any:
        if not self._is_reachable():
            raise ConnectionError(
                f"Local LLM server at {self._base_url} is not reachable"
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install the 'openai' package: pip install openai"
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=15.0,
            )
        return self._client

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        raw = response.choices[0].message.content or ""

        # Try to extract JSON if present
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return match.group()
        return raw

    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def health_check(self) -> bool:
        if not self._is_reachable():
            return False
        try:
            return super().health_check()
        except Exception:
            return False
