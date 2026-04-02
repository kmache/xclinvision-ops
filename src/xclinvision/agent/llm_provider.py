"""LLM Provider Interface — abstracts multiple LLM backends behind a uniform API.

Supports both synchronous calls and streaming (async generator) responses.
Implementations: OpenAI-compatible APIs and local models (e.g. Ollama, vLLM).
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterator, Optional

logger = logging.getLogger(__name__)


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

    def health_check(self) -> bool:
        """Return True if the provider is reachable and functional."""
        try:
            response = self.call(
                "You are a test assistant.",
                "Reply with exactly: OK",
                temperature=0.0,
            )
            return bool(response and response.strip())
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible provider
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (works with OpenAI, Azure, and compatible APIs).

    Configuration via environment variables:
    - ``OPENAI_API_KEY``  — required
    - ``OPENAI_MODEL``    — defaults to ``gpt-4o-mini``
    - ``OPENAI_API_BASE`` — optional custom base URL
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._base_url = base_url or os.environ.get("OPENAI_API_BASE")
        self._client: Any = None

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
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(
                "Structured JSON mode failed (%s); falling back to plain completion.",
                exc,
            )

        # Fallback: plain completion
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        raw = response.choices[0].message.content or ""

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

    @property
    def name(self) -> str:
        return "local"

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
        try:
            import requests
            resp = requests.get(
                self._base_url.rstrip("/").rsplit("/v1", 1)[0] + "/api/tags",
                timeout=3,
            )
            return resp.status_code == 200
        except Exception:
            # Fallback: try via the OpenAI client
            return super().health_check()
