"""
Global error handler middleware for OASIS Agentic Pipeline API.

Provides consistent error responses across all endpoints with proper HTTP status codes,
request correlation, and detailed error logging for debugging.
"""

import uuid
import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.exceptions import OasisBaseError


logger = logging.getLogger(__name__)


async def oasis_exception_handler(request: Request, exc: OasisBaseError) -> JSONResponse:
    """
    Handle custom OASIS exceptions with consistent error responses.

    Args:
        request: FastAPI request object
        exc: OASIS base exception

    Returns:
        JSON response with error details
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Log the error with context
    logger.error(
        f"Request {request_id} failed with {exc.error_code}: {exc.message}",
        extra={
            "request_id": request_id,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details,
        },
    )

    # Build error response
    error_response = {
        "success": False,
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "request_id": request_id,
            **exc.details,
        },
    }

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle standard HTTP exceptions.

    Args:
        request: FastAPI request object
        exc: Starlette HTTP exception

    Returns:
        JSON response with error details
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.warning(
        f"Request {request_id} HTTP error: {exc.status_code} - {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
        },
    )

    error_response = {
        "success": False,
        "error": {
            "code": f"HTTP_{exc.status_code}",
            "message": str(exc.detail),
            "request_id": request_id,
        },
    }

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Args:
        request: FastAPI request object
        exc: Request validation exception

    Returns:
        JSON response with validation error details
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        validation_errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(
        f"Request {request_id} validation error",
        extra={
            "request_id": request_id,
            "validation_errors": validation_errors,
            "path": request.url.path,
            "method": request.method,
        },
    )

    error_response = {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "request_id": request_id,
            "validation_errors": validation_errors,
        },
    }

    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions as a last resort.

    Args:
        request: FastAPI request object
        exc: Unexpected exception

    Returns:
        JSON response with generic error details
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Log the full traceback for unexpected errors
    logger.error(
        f"Request {request_id} unexpected error: {type(exc).__name__} - {str(exc)}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
        },
        exc_info=True,
    )

    error_response = {
        "success": False,
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "request_id": request_id,
        },
    }

    # Include stack trace in development mode
    import os

    if os.getenv("DEBUG", "false").lower() == "true":
        error_response["error"]["debug"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response)


async def add_request_id_middleware(request: Request, call_next):
    """
    Middleware to add unique request ID to each request for tracing.

    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint in chain

    Returns:
        Response with X-Request-ID header
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


def setup_error_handlers(app):
    """
    Register all exception handlers with the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Custom OASIS exceptions
    app.add_exception_handler(OasisBaseError, oasis_exception_handler)

    # Standard HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Catch-all for unexpected errors
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Error handlers registered successfully")
