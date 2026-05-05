"""Bearer-token authentication for PII-touching endpoints.

Single shared token sourced from the ``XCLINVISION_API_TOKEN`` environment
variable. This is intentionally minimal — clinical deployments need RBAC,
audit logging, and identity tied to clinicians, none of which are in scope
here. The goal is to stop unauthenticated traffic to anything that reads
or writes patient analyses, feedback, or images.

Security posture:
  * If ``XCLINVISION_API_TOKEN`` is unset, protected endpoints return
    503 (fail-secure: a missing token must NOT silently disable auth).
  * Missing/invalid Authorization header returns 401.
  * Comparison uses ``hmac.compare_digest`` to avoid timing leaks.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

_ENV_TOKEN = "XCLINVISION_API_TOKEN"


def _get_configured_token() -> str | None:
    raw = os.environ.get(_ENV_TOKEN, "").strip()
    return raw or None


def require_auth(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforces ``Authorization: Bearer <token>``.

    Use as ``dependencies=[Depends(require_auth)]`` on routes that touch
    patient data. Returns nothing on success; raises HTTPException
    otherwise.
    """
    expected = _get_configured_token()
    if expected is None:
        # Fail-secure: refuse to serve protected endpoints without a token.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server auth is not configured: set XCLINVISION_API_TOKEN "
                "to enable protected endpoints."
            ),
        )

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    presented = authorization.split(" ", 1)[1].strip()
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["require_auth"]
