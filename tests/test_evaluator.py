"""Regression tests for MetricsComputer routing.

Pinned by Issue #1: routing was previously controlled by
``is_multilabel()`` (read from configs/system.yaml at construction time)
and ignored the actual input shape, causing multiclass arrays to be sent
into the multilabel sample-averaged code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from xclinvision.evaluator import MetricsComputer


class TestRoutingByShape:
    def test_multiclass_arrays_route_to_multiclass_even_when_multilabel_true(self):
        """1-D y_true must always use multiclass metrics, regardless of
        ``self.multilabel`` or the global config."""
        mc = MetricsComputer(class_names=["A", "B", "C"], multilabel=True)
        # Caller forced multilabel=True, but inputs are clearly multiclass.
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 1])
        y_probs = np.random.rand(4, 3)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)

        # Multiclass-only keys present
        assert "accuracy" in metrics
        assert "weighted_f1" in metrics
        # Multilabel-only keys absent
        assert "subset_accuracy" not in metrics
        assert "sample_f1" not in metrics

    def test_multilabel_arrays_route_to_multilabel_even_when_multilabel_false(self):
        """2-D y_true must always use multilabel metrics, regardless of
        ``self.multilabel`` or the global config."""
        mc = MetricsComputer(class_names=["A", "B", "C"], multilabel=False)
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        y_probs = np.random.rand(3, 3)

        metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)

        assert "subset_accuracy" in metrics
        assert "sample_f1" in metrics
        assert "A_sensitivity" in metrics

    def test_mismatched_input_dims_raise(self):
        mc = MetricsComputer(class_names=["A", "B"])
        y_true = np.array([0, 1, 0])         # 1-D
        y_pred = np.array([[1, 0], [0, 1], [1, 0]])  # 2-D
        y_probs = np.random.rand(3, 2)

        with pytest.raises(ValueError, match="ndim"):
            mc.compute_all_metrics(y_true, y_pred, y_probs)

    def test_explicit_multilabel_constructor_arg_is_honoured_for_ambiguous_2d(self):
        """If the caller wires multilabel=False but passes 2-D arrays, we
        still route by shape (multilabel) — input is authoritative."""
        mc = MetricsComputer(class_names=["A", "B"], multilabel=False)
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [0, 1]])
        y_probs = np.random.rand(2, 2)
        metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)
        assert "subset_accuracy" in metrics
