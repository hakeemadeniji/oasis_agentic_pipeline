"""
Input validation and sanitization for OASIS Agentic Pipeline API.

Provides comprehensive validation for medical data, patient information,
and file uploads to ensure data integrity and security.
"""

import re
import logging
from typing import Optional, Any, Dict, List
from pathlib import Path
from pydantic import BaseModel, validator, Field

from src.api.exceptions import ValidationError


logger = logging.getLogger(__name__)


# ============================================================================
# Sanitization Functions
# ============================================================================


def sanitize_string(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize string input to prevent injection attacks.

    Args:
        input_string: Input string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        ValidationError: If string is too long or contains invalid characters
    """
    if not isinstance(input_string, str):
        raise ValidationError(
            message="Input must be a string",
            field="sanitize_string",
            value=type(input_string).__name__,
        )

    if len(input_string) > max_length:
        raise ValidationError(
            message=f"String exceeds maximum length of {max_length}",
            field="sanitize_string",
            value=len(input_string),
        )

    # Remove potentially dangerous characters
    # Allow alphanumeric, spaces, basic punctuation, and medical symbols
    sanitized = re.sub(r'[<>"\'\x00-\x1f\x7f-\x9f]', "", input_string)

    return sanitized.strip()


def sanitize_patient_id(patient_id: str) -> str:
    """
    Sanitize patient ID to ensure it meets format requirements.

    Args:
        patient_id: Patient ID to sanitize

    Returns:
        Sanitized patient ID

    Raises:
        ValidationError: If patient ID format is invalid
    """
    if not patient_id:
        raise ValidationError(message="Patient ID cannot be empty", field="patient_id")

    # Remove any whitespace
    patient_id = patient_id.strip()

    # Check format (alphanumeric, underscores, hyphens)
    if not re.match(r"^[A-Za-z0-9_-]+$", patient_id):
        raise ValidationError(
            message="Patient ID contains invalid characters. Only alphanumeric, underscores, and hyphens allowed.",
            field="patient_id",
            value=patient_id,
        )

    # Check length
    if len(patient_id) < 3 or len(patient_id) > 50:
        raise ValidationError(
            message="Patient ID must be between 3 and 50 characters",
            field="patient_id",
            value=len(patient_id),
        )

    return patient_id


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename

    Raises:
        ValidationError: If filename is invalid
    """
    if not filename:
        raise ValidationError(message="Filename cannot be empty", field="filename")

    # Reject path separators, traversal sequences, and null bytes in the RAW input.
    # (Checking after Path(...).name would be dead code — a basename never contains
    # them.) Callers that legitimately receive a path should pass os.path.basename().
    if ".." in filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise ValidationError(
            message="Filename contains path traversal attempts", field="filename", value=filename
        )

    # Reduce to basename defensively, then allow only safe characters.
    filename = Path(filename).name
    if not re.match(r"^[A-Za-z0-9._-]+$", filename):
        raise ValidationError(
            message="Filename contains invalid characters", field="filename", value=filename
        )

    return filename


def validate_email(email: str) -> str:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        Validated email

    Raises:
        ValidationError: If email format is invalid
    """
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    # Also reject consecutive dots (e.g. "user@example..com"), which the pattern
    # above would otherwise accept.
    if not re.match(email_pattern, email) or ".." in email:
        raise ValidationError(message="Invalid email format", field="email", value=email)
    return email.lower().strip()


# ============================================================================
# Medical Data Validators
# ============================================================================


def validate_age(age: Any) -> int:
    """
    Validate patient age.

    Args:
        age: Age value to validate

    Returns:
        Validated age as integer

    Raises:
        ValidationError: If age is invalid
    """
    try:
        age_int = int(float(age))
    except (ValueError, TypeError):
        raise ValidationError(message="Age must be a number", field="age", value=age)

    if age_int < 0 or age_int > 150:
        raise ValidationError(message="Age must be between 0 and 150", field="age", value=age_int)

    return age_int


def validate_mmse(mmse: Any) -> float:
    """
    Validate MMSE (Mini-Mental State Examination) score.

    Args:
        mmse: MMSE score to validate

    Returns:
        Validated MMSE score as float

    Raises:
        ValidationError: If MMSE score is invalid
    """
    try:
        mmse_float = float(mmse)
    except (ValueError, TypeError):
        raise ValidationError(message="MMSE must be a number", field="mmse", value=mmse)

    if mmse_float < 0 or mmse_float > 30:
        raise ValidationError(
            message="MMSE score must be between 0 and 30", field="mmse", value=mmse_float
        )

    return round(mmse_float, 1)


def validate_education_years(education: Any) -> Optional[int]:
    """
    Validate years of education.

    Args:
        education: Education years to validate

    Returns:
        Validated education years as integer, or None if not provided

    Raises:
        ValidationError: If education years are invalid
    """
    if education is None or education == "":
        return None

    try:
        education_int = int(float(education))
    except (ValueError, TypeError):
        raise ValidationError(
            message="Education years must be a number", field="education", value=education
        )

    if education_int < 0 or education_int > 30:
        raise ValidationError(
            message="Education years must be between 0 and 30",
            field="education",
            value=education_int,
        )

    return education_int


def validate_gender(gender: str) -> str:
    """
    Validate gender value.

    Args:
        gender: Gender value to validate

    Returns:
        Validated gender value

    Raises:
        ValidationError: If gender is invalid
    """
    if not gender:
        raise ValidationError(message="Gender cannot be empty", field="gender")

    gender = gender.upper().strip()
    if gender not in ["M", "F", "OTHER"]:
        raise ValidationError(
            message="Gender must be 'M', 'F', or 'OTHER'", field="gender", value=gender
        )

    return gender


# ============================================================================
# File Upload Validators
# ============================================================================


def validate_image_upload(filename: str, file_size: int, max_size_mb: int = 10) -> Dict[str, Any]:
    """
    Validate uploaded medical image file.

    Args:
        filename: Name of uploaded file
        file_size: Size of file in bytes
        max_size_mb: Maximum allowed file size in megabytes

    Returns:
        Dictionary with validation results

    Raises:
        ValidationError: If file validation fails
    """
    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    file_ext = Path(safe_filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise ValidationError(
            message=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(allowed_extensions)}",
            field="file_extension",
            value=file_ext,
        )

    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise ValidationError(
            message=f"File size exceeds maximum of {max_size_mb}MB",
            field="file_size",
            value=f"{file_size / (1024 * 1024):.2f}MB",
        )

    if file_size == 0:
        raise ValidationError(message="File is empty", field="file_size")

    return {
        "safe_filename": safe_filename,
        "file_extension": file_ext,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "valid": True,
    }


# ============================================================================
# Pydantic Models for Request Validation
# ============================================================================


class PatientDataValidation(BaseModel):
    """Pydantic model for patient data validation."""

    patient_id: str = Field(..., min_length=3, max_length=50)
    age: int = Field(..., ge=0, le=150)
    mmse: float = Field(..., ge=0, le=30)
    gender: Optional[str] = Field(None, pattern="^[MFO]$")
    education: Optional[int] = Field(None, ge=0, le=30)

    @validator("patient_id")
    def validate_patient_id_format(cls, v):
        return sanitize_patient_id(v)

    @validator("mmse")
    def validate_mmse_precision(cls, v):
        return round(v, 1)

    @validator("gender")
    def validate_gender_format(cls, v):
        if v:
            return v.upper()
        return v

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "OAS2_0001",
                "age": 75,
                "mmse": 24.0,
                "gender": "F",
                "education": 12,
            }
        }


class DiagnosisRequestValidation(BaseModel):
    """Pydantic model for diagnosis request validation."""

    patient_data: PatientDataValidation
    image_base64: Optional[str] = Field(None, max_length=10_000_000)  # ~10MB base64
    longitudinal_id: Optional[str] = Field(None, min_length=3, max_length=50)

    @validator("longitudinal_id")
    def validate_longitudinal_id_format(cls, v):
        if v:
            return sanitize_patient_id(v)
        return v

    @validator("image_base64")
    def validate_base64_image(cls, v):
        if v:
            # Basic base64 validation
            if not re.match(r"^[A-Za-z0-9+/]+=*$", v):
                raise ValueError("Invalid base64 format")
        return v

    class Config:
        schema_extra = {
            "example": {
                "patient_data": {"patient_id": "OAS2_0001", "age": 75, "mmse": 24.0, "gender": "F"},
                "image_base64": "base64encodedimage...",
                "longitudinal_id": "OAS2_0001",
            }
        }


# ============================================================================
# Batch Validation
# ============================================================================


def validate_batch_request(
    requests: List[Dict[str, Any]], max_batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Validate batch diagnosis request.

    Args:
        requests: List of diagnosis requests
        max_batch_size: Maximum number of requests per batch

    Returns:
        Validated list of requests

    Raises:
        ValidationError: If batch validation fails
    """
    if len(requests) > max_batch_size:
        raise ValidationError(
            message=f"Batch size exceeds maximum of {max_batch_size}",
            field="batch_size",
            value=len(requests),
        )

    if len(requests) == 0:
        raise ValidationError(message="Batch cannot be empty", field="batch_size")

    validated_requests = []
    for i, req in enumerate(requests):
        try:
            validated = DiagnosisRequestValidation(**req)
            validated_requests.append(validated.model_dump())
        except Exception as e:
            raise ValidationError(
                message=f"Request at index {i} failed validation: {str(e)}", field=f"request[{i}]"
            )

    return validated_requests


# ============================================================================
# Security Validators
# ============================================================================


def validate_sql_input(input_string: str) -> str:
    """
    Validate input for SQL injection prevention.

    Args:
        input_string: Input string to validate

    Returns:
        Sanitized string

    Raises:
        ValidationError: If SQL injection patterns are detected
    """
    # Common SQL injection patterns
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC|ALTER)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b|\bAND\b).*=",
        r"\b(XOR|LIKE|BETWEEN)\b",
    ]

    for pattern in sql_patterns:
        if re.search(pattern, input_string, re.IGNORECASE):
            raise ValidationError(
                message="Input contains potentially malicious SQL patterns",
                field="sql_input",
                value=input_string[:50] + "..." if len(input_string) > 50 else input_string,
            )

    return sanitize_string(input_string)


def validate_path_traversal(input_path: str) -> str:
    """
    Validate path to prevent directory traversal attacks.

    Args:
        input_path: Path string to validate

    Returns:
        Sanitized path

    Raises:
        ValidationError: If path traversal patterns are detected
    """
    # Check for path traversal patterns
    if ".." in input_path or input_path.startswith("/"):
        raise ValidationError(
            message="Path contains traversal sequences", field="path", value=input_path
        )

    # Remove null bytes
    input_path = input_path.replace("\x00", "")

    return sanitize_string(input_path, max_length=500)


if __name__ == "__main__":
    # Test validation functions
    print("Testing input validation...")

    # Test patient ID validation
    try:
        valid_id = sanitize_patient_id("OAS2_0001")
        print(f"Valid patient ID: {valid_id}")
    except ValidationError as e:
        print(f"Validation error: {e.message}")

    # Test age validation
    try:
        valid_age = validate_age(75)
        print(f"Valid age: {valid_age}")
    except ValidationError as e:
        print(f"Validation error: {e.message}")

    # Test MMSE validation
    try:
        valid_mmse = validate_mmse(24.5)
        print(f"Valid MMSE: {valid_mmse}")
    except ValidationError as e:
        print(f"Validation error: {e.message}")

    # Test invalid data
    try:
        validate_age(200)
    except ValidationError as e:
        print(f"Expected validation error: {e.message}")
