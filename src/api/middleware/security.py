"""
Security headers middleware for OASIS Agentic Pipeline API.

Adds comprehensive security headers to all HTTP responses to protect against
common web vulnerabilities and enforce secure communication practices.
"""

import logging
from typing import Optional, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware


logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.

    Adds headers for:
    - XSS Protection
    - Content Type Options
    - Frame Options (clickjacking protection)
    - Strict Transport Security (HSTS)
    - Content Security Policy (CSP)
    - Referrer Policy
    - Permissions Policy
    """

    def __init__(
        self,
        app,
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = True,
        csp_enabled: bool = True,
        csp_policy: Optional[str] = None,
        allowed_hosts: Optional[List[str]] = None,
    ):
        """
        Initialize security headers middleware.

        Args:
            app: ASGI application
            hsts_max_age: HSTS max-age in seconds
            hsts_include_subdomains: Include subdomains in HSTS
            hsts_preload: Add preload flag to HSTS
            csp_enabled: Enable Content Security Policy
            csp_policy: Custom CSP policy (uses default if not provided)
            allowed_hosts: List of allowed hosts for Host header validation
        """
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.csp_enabled = csp_enabled
        self.csp_policy = csp_policy or self._default_csp_policy()
        self.allowed_hosts = allowed_hosts or []

    def _default_csp_policy(self) -> str:
        """
        Generate default Content Security Policy.

        Returns:
            Default CSP string
        """
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )

    async def dispatch(self, request: Request, call_next):
        """
        Process request and add security headers to response.

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint in chain

        Returns:
            Response with security headers added
        """
        # Validate Host header if allowed hosts are configured
        if self.allowed_hosts:
            host = request.headers.get("host", "")
            if host and host not in self.allowed_hosts:
                logger.warning(f"Rejected request with invalid Host header: {host}")
                return Response(
                    content="Invalid Host header", status_code=400, media_type="text/plain"
                )

        # Process the request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response, request)

        return response

    def _add_security_headers(self, response: Response, request: Request):
        """Add security headers to the response."""

        # X-Content-Type-Options: Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options: Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # X-XSS-Protection: Enable XSS filtering
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Strict-Transport-Security (HSTS): Force HTTPS
        # Only add HSTS header for HTTPS requests
        if request.url.scheme == "https":
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Content-Security-Policy (CSP): Control resource loading
        if self.csp_enabled:
            response.headers["Content-Security-Policy"] = self.csp_policy

        # Referrer-Policy: Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy: Control browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )

        # Remove server information
        response.headers.pop("Server", None)

        # Additional security headers
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"


def configure_cors(
    app,
    allowed_origins: Optional[List[str]] = None,
    allowed_methods: Optional[List[str]] = None,
    allowed_headers: Optional[List[str]] = None,
    allow_credentials: bool = False,
    max_age: int = 600,
):
    """
    Configure CORS middleware with security best practices.

    Args:
        app: FastAPI application
        allowed_origins: List of allowed origins (uses env var if not provided)
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
        allow_credentials: Allow cookies in CORS requests
        max_age: CORS preflight cache duration in seconds
    """
    import os

    # Get allowed origins from environment if not provided
    if allowed_origins is None:
        allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
        if allowed_origins_str == "*":
            allowed_origins = ["*"]
        else:
            allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

    # Default allowed methods
    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    # Default allowed headers
    if allowed_headers is None:
        allowed_headers = ["Content-Type", "Authorization", "X-Request-ID", "X-User-Role"]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        max_age=max_age,
    )

    logger.info(
        f"CORS configured with {len(allowed_origins)} origins, "
        f"credentials={'enabled' if allow_credentials else 'disabled'}"
    )


def setup_security_headers(
    app,
    hsts_enabled: bool = True,
    csp_enabled: bool = True,
    allowed_hosts: Optional[List[str]] = None,
):
    """
    Set up security headers middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        hsts_enabled: Enable HSTS headers
        csp_enabled: Enable CSP headers
        allowed_hosts: List of allowed hosts for validation
    """
    import os

    # Check if we're in development mode
    is_development = os.getenv("DEBUG", "false").lower() == "true"

    # Disable HSTS in development (usually HTTP, not HTTPS)
    hsts_enabled = hsts_enabled and not is_development

    # Add security headers middleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts_max_age=31536000 if hsts_enabled else 0,
        hsts_include_subdomains=hsts_enabled,
        hsts_preload=hsts_enabled,
        csp_enabled=csp_enabled,
        allowed_hosts=allowed_hosts,
    )

    logger.info(
        f"Security headers configured - HSTS: {'enabled' if hsts_enabled else 'disabled'}, "
        f"CSP: {'enabled' if csp_enabled else 'disabled'}, "
        f"Development mode: {is_development}"
    )


if __name__ == "__main__":
    # Test security headers middleware
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    app = FastAPI()
    setup_security_headers(app)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    client = TestClient(app)
    response = client.get("/test")

    print("Security headers test:")
    for header, value in response.headers.items():
        if any(
            keyword in header.lower()
            for keyword in [
                "x-content",
                "x-frame",
                "x-xss",
                "strict-transport",
                "content-security",
                "referrer",
                "permissions",
            ]
        ):
            print(f"  {header}: {value}")
