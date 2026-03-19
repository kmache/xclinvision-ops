"""Centralised configuration loader for class names and class map.

The single source of truth is ``configs/system.yaml`` under the key
``model.class_names``.  Every module that needs class names or the
class→index mapping should call :func:`get_class_names` /
:func:`get_class_map` instead of hardcoding values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SYSTEM_CONFIG = _REPO_ROOT / "configs" / "system.yaml"

_FALLBACK_CLASS_NAMES: List[str] = ["Normal", "Pneumonia", "Cardiomegaly"]

_cached_class_names: List[str] | None = None


def get_class_names() -> List[str]:
    """Return the ordered list of class names from ``configs/system.yaml``.

    Falls back to ``["Normal", "Pneumonia", "Cardiomegaly"]`` if the config
    file is missing or malformed (e.g. during unit tests).
    """
    global _cached_class_names
    if _cached_class_names is not None:
        return list(_cached_class_names)

    if _SYSTEM_CONFIG.exists():
        try:
            with open(_SYSTEM_CONFIG, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
            names = cfg.get("model", {}).get("class_names")
            if isinstance(names, list) and len(names) >= 2:
                _cached_class_names = names
                return list(_cached_class_names)
        except Exception as exc:
            logger.warning("Failed to read class_names from %s: %s", _SYSTEM_CONFIG, exc)

    logger.warning(
        "class_names not found in %s — using fallback %s. "
        "This should only happen in unit tests. In production, ensure "
        "configs/system.yaml exists and contains model.class_names.",
        _SYSTEM_CONFIG,
        _FALLBACK_CLASS_NAMES,
    )
    _cached_class_names = _FALLBACK_CLASS_NAMES
    return list(_cached_class_names)


def get_class_map() -> Dict[str, int]:
    """Return ``{lowercase_name: index}`` derived from :func:`get_class_names`."""
    return {name.lower(): idx for idx, name in enumerate(get_class_names())}


def get_num_classes() -> int:
    """Return the number of configured classes."""
    return len(get_class_names())


_cached_classification_mode: str | None = None


def get_classification_mode() -> str:
    """Return the classification mode from ``configs/system.yaml``.

    Supported values: ``"multiclass"`` (default) or ``"multilabel"``.
    """
    global _cached_classification_mode
    if _cached_classification_mode is not None:
        return _cached_classification_mode

    if _SYSTEM_CONFIG.exists():
        try:
            with open(_SYSTEM_CONFIG, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
            mode = cfg.get("model", {}).get("classification_mode", "multiclass")
            if mode in ("multiclass", "multilabel"):
                _cached_classification_mode = mode
                return _cached_classification_mode
            logger.warning(
                "Invalid classification_mode '%s' in %s — defaulting to 'multiclass'.",
                mode, _SYSTEM_CONFIG,
            )
        except Exception as exc:
            logger.warning("Failed to read classification_mode from %s: %s", _SYSTEM_CONFIG, exc)

    _cached_classification_mode = "multiclass"
    return _cached_classification_mode


def is_multilabel() -> bool:
    """Convenience check: ``True`` when classification mode is multilabel."""
    return get_classification_mode() == "multilabel"


# ---------------------------------------------------------------------------
# Clinical rules
# ---------------------------------------------------------------------------

_cached_clinical_rules: Dict[str, Dict] | None = None


def get_clinical_rules() -> Dict[str, Dict]:
    """Return clinical plausibility rules from ``configs/system.yaml``.

    Keys are **lowercased** class names.  Returns an empty dict when no
    rules are defined — callers fall back to a generic heuristic.
    """
    global _cached_clinical_rules
    if _cached_clinical_rules is not None:
        return dict(_cached_clinical_rules)

    if _SYSTEM_CONFIG.exists():
        try:
            with open(_SYSTEM_CONFIG, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
            raw = cfg.get("clinical_rules")
            if isinstance(raw, dict):
                _cached_clinical_rules = {k.lower(): v for k, v in raw.items()}
                return dict(_cached_clinical_rules)
        except Exception as exc:
            logger.warning("Failed to read clinical_rules from %s: %s", _SYSTEM_CONFIG, exc)

    _cached_clinical_rules = {}
    return {}


def _reset_class_names_cache() -> None:
    """Invalidate the module-level class-names cache.

    Intended for **test teardown only**.  Call this when ``system.yaml`` is
    swapped between test cases so that the next call to ``get_class_names()``
    re-reads the config from disk.
    """
    global _cached_class_names, _cached_clinical_rules, _cached_classification_mode
    _cached_class_names = None
    _cached_clinical_rules = None
    _cached_classification_mode = None
