"""Tests for multi-provider LLM switching and streaming."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


class FakeProvider:
    """Simple in-memory LLM provider for testing."""

    def __init__(self, name: str, response: str = "OK", fail: bool = False):
        self._name = name
        self._response = response
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    def call(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        if self._fail:
            raise RuntimeError(f"{self._name} provider failed")
        return self._response

    def stream(self, system: str, user: str, *, temperature: float = 0.2) -> Iterator[str]:
        if self._fail:
            raise RuntimeError(f"{self._name} provider stream failed")
        for word in self._response.split():
            yield word + " "

    def health_check(self) -> bool:
        return not self._fail


@pytest.fixture
def fresh_manager():
    """Return a fresh LLMManager with no providers."""
    from xclinvision.agent.llm_manager import LLMManager
    return LLMManager()


@pytest.fixture
def populated_manager():
    """Return an LLMManager with two test providers."""
    from xclinvision.agent.llm_manager import LLMManager
    m = LLMManager()
    m.register(FakeProvider("openai", "OpenAI response"))
    m.register(FakeProvider("local", "Local response"))
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Provider Interface
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMProviderInterface:
    """Tests for the abstract LLMProvider base class."""

    def test_fake_provider_call(self):
        p = FakeProvider("test", "hello world")
        assert p.call("sys", "user") == "hello world"

    def test_fake_provider_stream(self):
        p = FakeProvider("test", "hello world")
        chunks = list(p.stream("sys", "user"))
        assert "".join(chunks).strip() == "hello world"

    def test_fake_provider_health(self):
        p = FakeProvider("test", "OK")
        assert p.health_check() is True

        p_fail = FakeProvider("test", "OK", fail=True)
        assert p_fail.health_check() is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LLMManager — Registration & Switching
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMManagerRegistration:
    """Tests for provider registration and listing."""

    def test_empty_manager(self, fresh_manager):
        assert fresh_manager.available_providers == []
        assert fresh_manager.active_provider is None
        assert fresh_manager.active_name is None

    def test_register_sets_first_as_active(self, fresh_manager):
        fresh_manager.register(FakeProvider("openai"))
        assert fresh_manager.active_name == "openai"
        assert fresh_manager.active_provider is not None

    def test_register_multiple(self, populated_manager):
        assert set(populated_manager.available_providers) == {"openai", "local"}
        # First registered becomes active
        assert populated_manager.active_name == "openai"

    def test_get_provider_by_name(self, populated_manager):
        p = populated_manager.get_provider("local")
        assert p is not None
        assert p.name == "local"

    def test_get_unknown_provider(self, populated_manager):
        assert populated_manager.get_provider("nonexistent") is None


class TestLLMManagerSwitching:
    """Tests for runtime provider switching."""

    def test_switch_to_local(self, populated_manager):
        populated_manager.set_active("local")
        assert populated_manager.active_name == "local"

    def test_switch_back_to_openai(self, populated_manager):
        populated_manager.set_active("local")
        populated_manager.set_active("openai")
        assert populated_manager.active_name == "openai"

    def test_switch_unknown_raises(self, populated_manager):
        with pytest.raises(ValueError, match="Unknown provider"):
            populated_manager.set_active("nonexistent")

    def test_switch_updates_call_target(self, populated_manager):
        result1 = populated_manager.call_llm("sys", "user")
        assert result1 == "OpenAI response"

        populated_manager.set_active("local")
        result2 = populated_manager.call_llm("sys", "user")
        assert result2 == "Local response"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLMManager — Fallback
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMManagerFallback:
    """Tests for automatic fallback when primary provider fails."""

    def test_fallback_on_call_failure(self):
        from xclinvision.agent.llm_manager import LLMManager

        m = LLMManager()
        m.register(FakeProvider("openai", "OpenAI response", fail=True))
        m.register(FakeProvider("local", "Local response", fail=False))

        # openai is active but fails → should fallback to local
        result = m.call_llm("sys", "user")
        assert result == "Local response"

    def test_all_providers_fail_raises(self):
        from xclinvision.agent.llm_manager import LLMManager

        m = LLMManager()
        m.register(FakeProvider("openai", "x", fail=True))
        m.register(FakeProvider("local", "x", fail=True))

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            m.call_llm("sys", "user")

    def test_fallback_on_stream_failure(self):
        from xclinvision.agent.llm_manager import LLMManager

        m = LLMManager()
        m.register(FakeProvider("openai", "OpenAI streamed", fail=True))
        m.register(FakeProvider("local", "Local streamed", fail=False))

        chunks = list(m.stream_llm("sys", "user"))
        assert "".join(chunks).strip() == "Local streamed"

    def test_stream_all_fail_raises(self):
        from xclinvision.agent.llm_manager import LLMManager

        m = LLMManager()
        m.register(FakeProvider("a", "x", fail=True))
        m.register(FakeProvider("b", "x", fail=True))

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            list(m.stream_llm("sys", "user"))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLMManager — Health
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMManagerHealth:
    """Tests for provider health checking."""

    def test_health_all_ok(self, populated_manager):
        status = populated_manager.provider_status()
        assert status == {"openai": True, "local": True}

    def test_health_partial_failure(self):
        from xclinvision.agent.llm_manager import LLMManager

        m = LLMManager()
        m.register(FakeProvider("openai", "OK", fail=False))
        m.register(FakeProvider("local", "OK", fail=True))

        status = m.provider_status()
        assert status["openai"] is True
        assert status["local"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Streaming Generator
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamingGenerator:
    """Tests for the streaming interface."""

    def test_stream_yields_chunks(self, populated_manager):
        chunks = list(populated_manager.stream_llm("sys", "user"))
        assert len(chunks) > 0
        full_text = "".join(chunks).strip()
        assert full_text == "OpenAI response"

    def test_stream_after_switch(self, populated_manager):
        populated_manager.set_active("local")
        chunks = list(populated_manager.stream_llm("sys", "user"))
        full_text = "".join(chunks).strip()
        assert full_text == "Local response"

    def test_stream_is_iterator(self, populated_manager):
        gen = populated_manager.stream_llm("sys", "user")
        # Should be an iterator/generator
        first = next(gen)
        assert isinstance(first, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Singleton / Factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMManagerSingleton:
    """Tests for the module-level singleton."""

    def test_get_llm_manager_returns_manager(self):
        from xclinvision.agent.llm_manager import get_llm_manager, reset_llm_manager

        reset_llm_manager()
        m = get_llm_manager()
        assert m is not None
        assert isinstance(m.available_providers, list)

    def test_singleton_returns_same_instance(self):
        from xclinvision.agent.llm_manager import get_llm_manager, reset_llm_manager

        reset_llm_manager()
        m1 = get_llm_manager()
        m2 = get_llm_manager()
        assert m1 is m2

    def test_reset_clears_singleton(self):
        from xclinvision.agent.llm_manager import get_llm_manager, reset_llm_manager

        m1 = get_llm_manager()
        reset_llm_manager()
        m2 = get_llm_manager()
        assert m1 is not m2

    def test_from_env_with_api_key(self):
        from xclinvision.agent.llm_manager import LLMManager

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}, clear=False):
            m = LLMManager.from_env()
            assert "openai" in m.available_providers
            assert "local" in m.available_providers
            assert m.active_name == "openai"

    def test_from_env_without_api_key(self):
        from xclinvision.agent.llm_manager import LLMManager

        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            m = LLMManager.from_env()
            assert "local" in m.available_providers
            assert m.active_name == "local"

    def test_from_env_respects_default_provider(self):
        from xclinvision.agent.llm_manager import LLMManager

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key", "LLM_DEFAULT_PROVIDER": "local"},
            clear=False,
        ):
            m = LLMManager.from_env()
            assert m.active_name == "local"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Integration with Agent Factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentFactoryIntegration:
    """Tests that agent factory functions use LLMManager."""

    def test_create_reasoning_agent_with_manager(self):
        """Verify create_reasoning_agent uses LLMManager."""
        from xclinvision.agent.llm_manager import reset_llm_manager
        reset_llm_manager()

        from xclinvision.agent import create_reasoning_agent
        agent = create_reasoning_agent()
        assert agent is not None
        # Even without an API key, the agent should initialise (rule-based)
        assert agent.tools is not None
