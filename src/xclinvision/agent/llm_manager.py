"""LLM Manager — runtime provider switching and fallback orchestration.

Maintains a registry of :class:`LLMProvider` instances and exposes a
single ``call_llm`` / ``stream_llm`` interface that the rest of the agent
module can use, regardless of which backend is currently active.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Iterator, List, Optional

from xclinvision.agent.llm_provider import (
    LLMProvider,
    LocalProvider,
    OpenAIProvider,
)

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages multiple LLM providers with runtime switching and fallback.

    Usage::

        manager = LLMManager.from_env()
        manager.set_active("openai")

        # Synchronous call (drop-in replacement for _default_call_llm)
        text = manager.call_llm(system_prompt, user_prompt, temperature=0.2)

        # Streaming call
        for chunk in manager.stream_llm(system_prompt, user_prompt):
            print(chunk, end="", flush=True)
    """

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._active_name: Optional[str] = None
        self._lock = threading.Lock()

    # ── Provider registry ─────────────────────────────────────────────

    def register(self, provider: LLMProvider) -> None:
        """Register a provider.  The first registered provider becomes active."""
        with self._lock:
            self._providers[provider.name] = provider
            if self._active_name is None:
                self._active_name = provider.name
                logger.info("LLMManager: active provider set to '%s'", provider.name)

    @property
    def active_provider(self) -> Optional[LLMProvider]:
        return self._providers.get(self._active_name or "")

    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    @property
    def available_providers(self) -> List[str]:
        return list(self._providers.keys())

    def set_active(self, name: str) -> None:
        """Switch the active provider at runtime."""
        with self._lock:
            if name not in self._providers:
                raise ValueError(
                    f"Unknown provider '{name}'. "
                    f"Available: {list(self._providers.keys())}"
                )
            self._active_name = name
            logger.info("LLMManager: switched active provider to '%s'", name)

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        return self._providers.get(name)

    # ── Unified LLM interface (with automatic fallback) ───────────────

    def call_llm(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Call the active provider.  Falls back to other providers on failure."""
        providers_to_try = self._fallback_order()
        last_error: Optional[Exception] = None

        for provider in providers_to_try:
            try:
                result = provider.call(system, user, temperature=temperature)
                # If we fell back to a different provider, update active
                if provider.name != self._active_name:
                    logger.warning(
                        "LLMManager: primary provider '%s' failed; "
                        "succeeded with fallback '%s'",
                        self._active_name,
                        provider.name,
                    )
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLMManager: provider '%s' failed: %s", provider.name, exc,
                )
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    def stream_llm(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Stream from the active provider.  Falls back on failure."""
        providers_to_try = self._fallback_order()
        last_error: Optional[Exception] = None

        for provider in providers_to_try:
            try:
                yield from provider.stream(system, user, temperature=temperature)
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLMManager: streaming from '%s' failed: %s",
                    provider.name,
                    exc,
                )
                continue

        raise RuntimeError(
            f"All LLM providers failed for streaming. Last error: {last_error}"
        )

    def _fallback_order(self) -> List[LLMProvider]:
        """Active provider first, then all others."""
        result: List[LLMProvider] = []
        if self._active_name and self._active_name in self._providers:
            result.append(self._providers[self._active_name])
        for name, provider in self._providers.items():
            if name != self._active_name:
                result.append(provider)
        return result

    # ── Health ────────────────────────────────────────────────────────

    def provider_status(self) -> Dict[str, bool]:
        """Check health of all registered providers."""
        return {name: p.health_check() for name, p in self._providers.items()}

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "LLMManager":
        """Build an LLMManager from environment variables.

        Registers providers based on available configuration:
        - ``OPENAI_API_KEY`` → OpenAI provider
        - ``LOCAL_LLM_BASE_URL`` or always → Local provider

        The ``LLM_DEFAULT_PROVIDER`` env var sets the default active provider
        (defaults to ``openai`` if an API key is present, else ``local``).
        """
        manager = cls()

        # Register OpenAI if key is present
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                openai_provider = OpenAIProvider(api_key=api_key)
                manager.register(openai_provider)
                logger.info("LLMManager: registered OpenAI provider")
            except Exception as exc:
                logger.warning("LLMManager: failed to register OpenAI: %s", exc)

        # Always register local provider (may not be running but available)
        try:
            local_provider = LocalProvider()
            manager.register(local_provider)
            logger.info("LLMManager: registered local provider")
        except Exception as exc:
            logger.warning("LLMManager: failed to register local: %s", exc)

        # Set default active provider
        default = os.environ.get("LLM_DEFAULT_PROVIDER", "")
        if default and default in manager._providers:
            manager.set_active(default)
        elif "openai" in manager._providers:
            manager.set_active("openai")
        elif "local" in manager._providers:
            manager.set_active("local")

        return manager


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════════

_manager: Optional[LLMManager] = None
_manager_lock = threading.Lock()


def get_llm_manager() -> LLMManager:
    """Return or create the module-level LLMManager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = LLMManager.from_env()
    return _manager


def reset_llm_manager() -> None:
    """Reset the singleton (useful for testing)."""
    global _manager
    with _manager_lock:
        _manager = None
