"""
Custom exception classes for OASIS Agentic Pipeline.

Domain-specific exceptions for better error handling, logging, and user feedback.
Follows hybrid approach: custom exceptions for critical errors, standard for general cases.
"""

from typing import Optional, Dict, Any


class OasisBaseError(Exception):
    """Base exception for all OASIS pipeline-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "OASIS_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {"error_code": self.error_code, "message": self.message, "details": self.details}


# ============================================================================
# Medical Imaging Exceptions
# ============================================================================


class MedicalImagingError(OasisBaseError):
    """Exception raised for MRI image processing failures."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500
    ):
        super().__init__(
            message=message,
            error_code="MEDICAL_IMAGING_ERROR",
            details=details,
            status_code=status_code,
        )


class ImageValidationError(MedicalImagingError):
    """Exception raised for invalid image uploads or processing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=400)


class ModelInferenceError(MedicalImagingError):
    """Exception raised during model inference failures."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message=message, details=details, status_code=500)


# ============================================================================
# Biomarker Analysis Exceptions
# ============================================================================


class BiomarkerAnalysisError(OasisBaseError):
    """Exception raised for clinical biomarker processing errors."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500
    ):
        super().__init__(
            message=message,
            error_code="BIOMARKER_ANALYSIS_ERROR",
            details=details,
            status_code=status_code,
        )


class DataValidationError(BiomarkerAnalysisError):
    """Exception raised for invalid clinical data."""

    def __init__(
        self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        if field:
            details["field"] = field
        super().__init__(message=message, details=details, status_code=400)


class PatientNotFoundError(BiomarkerAnalysisError):
    """Exception raised when patient data is not found."""

    def __init__(self, patient_id: str, details: Optional[Dict[str, Any]] = None):
        if details is None:
            details = {}
        details["patient_id"] = patient_id
        super().__init__(
            message=f"Patient {patient_id} not found", details=details, status_code=404
        )


# ============================================================================
# LLM Provider Exceptions
# ============================================================================


class LLMProviderError(OasisBaseError):
    """Exception raised for LLM service failures."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 503,
    ):
        if details is None:
            details = {}
        if provider:
            details["provider"] = provider
        super().__init__(
            message=message,
            error_code="LLM_PROVIDER_ERROR",
            details=details,
            status_code=status_code,
        )


class LLMTimeoutError(LLMProviderError):
    """Exception raised when LLM request times out."""

    def __init__(
        self, provider: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(
            message=f"LLM request to {provider} timed out after {timeout_seconds}s",
            provider=provider,
            details=details,
            status_code=504,
        )


class LLMRateLimitError(LLMProviderError):
    """Exception raised when LLM provider rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message=f"LLM provider {provider} rate limit exceeded",
            provider=provider,
            details=details,
            status_code=429,
        )


# ============================================================================
# Authentication & Authorization Exceptions
# ============================================================================


class AuthenticationError(OasisBaseError):
    """Exception raised for authentication failures."""

    def __init__(
        self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message, error_code="AUTHENTICATION_ERROR", details=details, status_code=401
        )


class AuthorizationError(OasisBaseError):
    """Exception raised for authorization failures."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_role: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if required_role:
            details["required_role"] = required_role
        super().__init__(
            message=message, error_code="AUTHORIZATION_ERROR", details=details, status_code=403
        )


class TokenError(AuthenticationError):
    """Exception raised for token validation failures."""

    def __init__(
        self, message: str = "Invalid or expired token", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message=message, details=details)


# ============================================================================
# Rate Limiting Exceptions
# ============================================================================


class RateLimitError(OasisBaseError):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if retry_after:
            details["retry_after"] = retry_after
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = window
        super().__init__(
            message=message, error_code="RATE_LIMIT_ERROR", details=details, status_code=429
        )


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationError(OasisBaseError):
    """Exception raised for input validation failures."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["provided_value"] = str(value)
        super().__init__(
            message=message, error_code="VALIDATION_ERROR", details=details, status_code=400
        )


# ============================================================================
# System Exceptions
# ============================================================================


class SystemInitializationError(OasisBaseError):
    """Exception raised when system components fail to initialize."""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if component:
            details["component"] = component
        super().__init__(
            message=message,
            error_code="SYSTEM_INITIALIZATION_ERROR",
            details=details,
            status_code=503,
        )


class ConfigurationError(OasisBaseError):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message=message, error_code="CONFIGURATION_ERROR", details=details, status_code=500
        )
