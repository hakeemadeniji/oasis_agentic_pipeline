"""
Rate limiting middleware for OASIS Agentic Pipeline API.

Enforces tiered, role-based rate limits to prevent API abuse. Enforcement is a
single global HTTP middleware backed by an in-process, sliding-window counter, so
it works with zero external dependencies (no Redis/slowapi required) and applies
uniformly to every API endpoint.

Notes / limitations:
* The counter is per-process. Behind multiple uvicorn workers the effective limit
  is ``workers * limit``; for a hard global limit across workers use a shared
  store (Redis). This is fine for single-node edge / kiosk deployments.
* The client role is taken from the *validated* JWT (never a client header in
  production); see :func:`get_user_role`.
"""

import time
import logging
from typing import Dict, Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.exceptions import RateLimitError
from src.config import get_settings


logger = logging.getLogger(__name__)


# Per-minute / per-hour request budgets by role. Only the per-minute window is
# enforced by the in-process limiter below; the hourly figure documents intent.
RATE_LIMITS: Dict[str, tuple] = {
    "admin": ("1000/minute", "10000/hour"),
    "clinician": ("100/minute", "1000/hour"),
    "researcher": ("50/minute", "500/hour"),
    "viewer": ("30/minute", "300/hour"),
    "default": ("20/minute", "200/hour"),
}

# Paths exempt from rate limiting: liveness/health probes (polled frequently),
# API docs, and the static SPA console. Static mounts (/app) are sub-apps and are
# not affected by this middleware anyway, but we exempt the prefix for clarity.
_EXEMPT_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon",
    "/app",
    "/console",
)


def _client_ip(request: Request) -> str:
    """Best-effort client IP (no proxy header trust)."""
    client = request.client
    return client.host if client else "unknown"


def get_user_role(request: Request) -> str:
    """
    Resolve the rate-limit tier for a request.

    Production path: decode the *validated* JWT from the Authorization header and
    read its ``role`` claim. The client-supplied ``X-User-Role`` header is honored
    ONLY in dev mode (otherwise a client could send ``X-User-Role: admin`` to
    escape the strict default tier). Unknown/absent role -> "default".
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            # Lazy import avoids any import-order coupling with the auth module.
            from src.api.middleware.auth import decode_token

            role = decode_token(token).get("role")
            if role in RATE_LIMITS:
                return role
        except Exception:
            # Invalid/expired token -> fall through to default (auth middleware
            # will reject protected routes separately).
            pass

    if get_settings().auth_dev_mode:
        role_header = request.headers.get("X-User-Role", "")
        if role_header in RATE_LIMITS:
            return role_header

    return "default"


def get_rate_limit_key(request: Request) -> str:
    """Rate-limit bucket key: role + client IP."""
    return f"{get_user_role(request)}:{_client_ip(request)}"


class RateLimiter:
    """In-process sliding-window rate limiter with role tiers."""

    def __init__(self) -> None:
        self.request_counts: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        # Driven by config, NOT by whether an optional package is installed.
        return get_settings().enable_rate_limiting

    def check_rate_limit(self, request: Request, role: str = "default") -> bool:
        """
        Record a request and enforce the per-minute budget for ``role``.

        Returns True if allowed; raises :class:`RateLimitError` (HTTP 429) if the
        budget is exceeded.
        """
        if not self.enabled:
            return True

        limits = RATE_LIMITS.get(role, RATE_LIMITS["default"])
        limit_per_minute = int(limits[0].split("/")[0])

        key = f"{role}:{_client_ip(request)}"
        now = time.time()
        self._cleanup_old_entries(now)

        entry = self.request_counts.get(key)
        if entry is None or now - entry["window_start"] > 60:
            entry = {"count": 0, "window_start": now}
            self.request_counts[key] = entry

        if entry["count"] >= limit_per_minute:
            retry_after = max(1, int(60 - (now - entry["window_start"])))
            raise RateLimitError(
                message=f"Rate limit exceeded ({limit_per_minute}/min for role '{role}')",
                retry_after=retry_after,
                limit=limit_per_minute,
                window=60,
            )

        entry["count"] += 1
        return True

    def _cleanup_old_entries(self, now: float) -> None:
        """Drop windows older than 5 minutes to bound memory."""
        cutoff = now - 300
        stale = [k for k, e in self.request_counts.items() if e["window_start"] < cutoff]
        for k in stale:
            del self.request_counts[k]


# Global limiter instance.
rate_limiter = RateLimiter()


def _is_exempt(path: str) -> bool:
    return path == "/" or any(path.startswith(p) for p in _EXEMPT_PREFIXES)


async def rate_limit_middleware(request: Request, call_next):
    """
    Global HTTP middleware that enforces the rate limit on every non-exempt path.

    Exceptions raised in middleware bypass FastAPI's exception handlers, so we
    catch :class:`RateLimitError` here and build the 429 response directly
    (mirroring the error-handler envelope) — including a CORS allow-origin header,
    because this middleware short-circuits before the CORS middleware runs.
    """
    if _is_exempt(request.url.path):
        return await call_next(request)

    try:
        rate_limiter.check_rate_limit(request, get_user_role(request))
    except RateLimitError as exc:
        retry_after = exc.details.get("retry_after", 60)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {"code": exc.error_code, "message": exc.message, **exc.details},
            },
            headers={
                "Retry-After": str(retry_after),
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            },
        )

    return await call_next(request)


# Backwards-compatible dependency form (e.g. to limit a single route explicitly).
async def check_rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces the rate limit for one route."""
    rate_limiter.check_rate_limit(request, get_user_role(request))


def rate_limit(role: str = "default") -> Callable:
    """
    Decorator to enforce a fixed-role rate limit on a single endpoint.

    Most deployments should rely on the global middleware; use this only when an
    endpoint needs a hardcoded tier regardless of the caller's role.
    """

    def decorator(func: Callable) -> Callable:
        import asyncio
        from functools import wraps

        def _find_request(args) -> "Request | None":
            return next((a for a in args if isinstance(a, Request)), None)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request = _find_request(args)
            if request:
                rate_limiter.check_rate_limit(request, role)
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request = _find_request(args)
            if request:
                rate_limiter.check_rate_limit(request, role)
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def setup_rate_limiting(app) -> None:
    """Install the global rate-limiting middleware (if enabled in config)."""
    if not get_settings().enable_rate_limiting:
        logger.info("Rate limiting disabled (ENABLE_RATE_LIMITING=false)")
        return

    app.middleware("http")(rate_limit_middleware)
    logger.info("Rate limiting enabled (in-process, role-tiered, per-minute windows)")


if __name__ == "__main__":
    from unittest.mock import Mock

    os_env_note = "Set ENABLE_RATE_LIMITING=true (default) to exercise enforcement."
    print(os_env_note)
    mock_request = Mock(spec=Request)
    mock_request.headers = {"X-User-Role": "clinician"}
    mock_request.client = Mock(host="127.0.0.1")

    print("Testing rate limiting (clinician = 100/min)...")
    allowed = 0
    for i in range(105):
        try:
            rate_limiter.check_rate_limit(mock_request, "clinician")
            allowed += 1
        except RateLimitError as e:
            print(f"Request {i}: rate limited - {e.message}")
            break
    print(f"Allowed {allowed} requests before limiting.")
