"""LLM provider admin endpoints.

Extracted from the original ``main.py`` god-module. These endpoints
are stateless w.r.t. the analysis/feedback storage and only depend on
``xclinvision.agent.llm_manager``, so they're a clean first slice of
the planned router split (Issue #8).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, HTTPException

try:
    from ..auth import require_auth  # type: ignore[import-not-found]
except ImportError:
    from auth import require_auth  # type: ignore[import-not-found,no-redef]


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/llm", tags=["llm"])


def _get_llm_manager():
    """Return the module-level LLMManager singleton."""
    from xclinvision.agent.llm_manager import get_llm_manager
    return get_llm_manager()


@router.get("/providers")
async def list_llm_providers():
    """List available LLM providers and the current active provider."""
    try:
        manager = _get_llm_manager()
        return {
            "providers": manager.available_providers,
            "active": manager.active_name,
        }
    except Exception as e:  # noqa: BLE001 — exposed via the response payload
        logger.warning("Failed to list LLM providers: %s", e)
        return {"providers": [], "active": None, "error": str(e)}


@router.post("/switch", dependencies=[Depends(require_auth)])
async def switch_llm_provider(provider: str = Form(...)):
    """Switch the active LLM provider at runtime (admin-only)."""
    try:
        manager = _get_llm_manager()
        manager.set_active(provider)
        return {"active": manager.active_name, "status": "switched"}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to switch LLM provider: %s", e)
        raise HTTPException(500, detail=f"Provider switch failed: {e}")


@router.get("/health")
async def llm_provider_health():
    """Health check for all registered LLM providers."""
    try:
        manager = _get_llm_manager()
        return {
            "active": manager.active_name,
            "status": manager.provider_status(),
        }
    except Exception as e:  # noqa: BLE001
        return {"active": None, "status": {}, "error": str(e)}


__all__ = ["router"]
