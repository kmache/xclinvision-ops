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

# Walk up from src/xclinvision/config.py → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SYSTEM_CONFIG = _REPO_ROOT / "configs" / "system.yaml"

_FALLBACK_CLASS_NAMES: List[str] = ["Normal", "Pneumonia", "Cardiomegaly"]

# Module-level cache so the file is read at most once per process.
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
        "class_names not found in %s — using fallback %s",
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
