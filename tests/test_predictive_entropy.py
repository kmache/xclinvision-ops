"""Tests for InferencePipeline._predictive_entropy (issue #2).

The previous implementation applied categorical Shannon entropy
``-Σ p·log p`` to multilabel sigmoid outputs that don't sum to 1,
producing meaningless values. The fix branches on shape: multiclass
keeps the categorical formula; multilabel uses mean per-label
binary entropy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from xclinvision.inference import InferencePipeline


_H = InferencePipeline._predictive_entropy


# ── Multiclass (softmax) ───────────────────────────────────────────────────────

def test_multiclass_uniform_distribution_equals_log_k():
    """For a uniform K-way categorical, H = log K."""
    p = np.array([0.25, 0.25, 0.25, 0.25])
    h = _H(p, multilabel=False)
    assert h == pytest.approx(math.log(4), abs=1e-6)


def test_multiclass_one_hot_distribution_is_zero():
    """Degenerate one-hot has zero entropy (modulo eps slack)."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    h = _H(p, multilabel=False)
    assert h == pytest.approx(0.0, abs=1e-3)


def test_multiclass_accepts_2d_with_axis_reduction():
    """Shape (1, K) — as produced by compute_uncertainty."""
    p = np.array([[0.5, 0.5]])
    h = _H(p, multilabel=False)
    assert h == pytest.approx(math.log(2), abs=1e-6)


# ── Multilabel (sigmoid) ───────────────────────────────────────────────────────

def test_multilabel_all_half_equals_log_2():
    """Mean per-label binary entropy at p=0.5 is log 2 (max uncertainty)."""
    p = np.array([0.5, 0.5, 0.5, 0.5])
    h = _H(p, multilabel=True)
    assert h == pytest.approx(math.log(2), abs=1e-6)


def test_multilabel_all_certain_is_zero():
    """All labels at extremes -> zero binary entropy."""
    p = np.array([0.0, 1.0, 0.0, 1.0])
    h = _H(p, multilabel=True)
    assert h == pytest.approx(0.0, abs=1e-3)


def test_multilabel_does_not_treat_as_categorical():
    """Sigmoid outputs that exceed sum=1 must NOT collapse via -Σp log p.

    With p=[0.9, 0.9, 0.9] the categorical formula gives
    -3 * 0.9 * log(0.9) ≈ 0.284, whereas mean per-label binary entropy
    gives -(0.9 log 0.9 + 0.1 log 0.1) ≈ 0.325. They must differ.
    """
    p = np.array([0.9, 0.9, 0.9])
    multilabel_h = _H(p, multilabel=True)
    multiclass_h = _H(p, multilabel=False)
    assert not math.isclose(multilabel_h, multiclass_h, abs_tol=1e-3)
    # Multilabel value is bounded in [0, log 2] for any per-label p.
    assert 0.0 <= multilabel_h <= math.log(2) + 1e-6


def test_multilabel_bounded_by_log_2():
    """For any per-label p in [0,1], mean binary entropy <= log 2."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(0.0, 1.0, size=10)
        h = _H(p, multilabel=True)
        assert 0.0 <= h <= math.log(2) + 1e-6
