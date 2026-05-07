"""Regression test for issue #10.

Previously xai.py captured ``DEFAULT_CLASS_NAMES = get_class_names()`` at
import time and reused it as a default-arg value, so the snapshot persisted
even after _reset_class_names_cache() was called between tests. The fix
replaces those defaults with ``Optional[List[str]] = None`` and resolves
via ``class_names or get_class_names()`` per call. This test verifies the
live cache is honoured.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import xclinvision.xai as xai
from xclinvision.config import _reset_class_names_cache


def test_default_class_names_symbol_is_gone():
    """The frozen snapshot must no longer exist as a public symbol."""
    assert not hasattr(xai, "DEFAULT_CLASS_NAMES")


def test_explainability_engine_uses_live_class_names():
    """ExplainabilityEngine() with no class_names= must call get_class_names()
    each construction, not a frozen snapshot."""
    fake_model = MagicMock(spec=["modules", "parameters"])
    fake_model.modules.return_value = iter([])

    with patch("xclinvision.xai.get_class_names", return_value=["A", "B"]) as gcn1:
        eng1 = xai.ExplainabilityEngine(fake_model, architecture="test")
        assert eng1.class_names == ["A", "B"]
        assert gcn1.called

    # Cache reset between calls -> different class names returned.
    _reset_class_names_cache()
    with patch("xclinvision.xai.get_class_names", return_value=["X", "Y", "Z"]) as gcn2:
        eng2 = xai.ExplainabilityEngine(fake_model, architecture="test")
        assert eng2.class_names == ["X", "Y", "Z"]
        assert gcn2.called


def test_explainability_engine_explicit_class_names_override():
    """An explicit class_names= argument must win over the live config."""
    fake_model = MagicMock(spec=["modules", "parameters"])
    fake_model.modules.return_value = iter([])

    with patch("xclinvision.xai.get_class_names", return_value=["LIVE"]) as gcn:
        eng = xai.ExplainabilityEngine(
            fake_model, class_names=["EXPLICIT"], architecture="test"
        )
        assert eng.class_names == ["EXPLICIT"]
        # Live config must NOT be consulted when caller passed names.
        assert not gcn.called
