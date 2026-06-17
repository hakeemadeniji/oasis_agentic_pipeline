"""
Middleware package for OASIS Agentic Pipeline API.

Contains middleware for authentication, rate limiting, security headers,
and error handling.
"""

from .error_handler import (
    setup_error_handlers,
    add_request_id_middleware,
    oasis_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)

from .rate_limit import (
    RateLimiter,
    rate_limit,
    check_rate_limit_dependency,
    setup_rate_limiting,
    rate_limiter,
)

from .security import SecurityHeadersMiddleware, configure_cors, setup_security_headers

from .auth import (
    setup_authentication,
    get_current_user,
    get_current_active_user,
    require_role,
    require_admin,
    require_clinician,
    require_researcher,
    require_viewer,
    get_current_user_optional,
    dev_mode_bypass,
    User,
    auth_config,
)

__all__ = [
    # Error handling
    "setup_error_handlers",
    "add_request_id_middleware",
    "oasis_exception_handler",
    "http_exception_handler",
    "validation_exception_handler",
    "general_exception_handler",
    # Rate limiting
    "RateLimiter",
    "rate_limit",
    "check_rate_limit_dependency",
    "setup_rate_limiting",
    "rate_limiter",
    # Security headers
    "SecurityHeadersMiddleware",
    "configure_cors",
    "setup_security_headers",
    # Authentication
    "setup_authentication",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "require_admin",
    "require_clinician",
    "require_researcher",
    "require_viewer",
    "get_current_user_optional",
    "dev_mode_bypass",
    "User",
    "auth_config",
]
