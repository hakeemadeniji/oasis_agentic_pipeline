"""
Data validation tests for OASIS Agentic Pipeline.

Tests input validation, sanitization, and data integrity checks to ensure
robustness against malicious or malformed inputs.
"""

import pytest
import sys
import os
from io import BytesIO
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.api.validators import (
    sanitize_string,
    sanitize_patient_id,
    sanitize_filename,
    validate_email,
    validate_age,
    validate_mmse,
    validate_education_years,
    validate_gender,
    validate_image_upload,
    validate_sql_input,
    validate_path_traversal,
    PatientDataValidation,
    DiagnosisRequestValidation,
    validate_batch_request,
)
from src.api.exceptions import ValidationError


class TestStringSanitization:
    """Test string sanitization functions"""

    def test_basic_string_sanitization(self):
        """Test basic string sanitization"""
        # Valid strings
        assert sanitize_string("Hello World") == "Hello World"
        assert sanitize_string("Test123") == "Test123"
        assert sanitize_string("test@example.com") == "test@example.com"

        # String with special characters should be sanitized
        result = sanitize_string("Test<script>alert('xss')</script>")
        # Angle brackets and quotes are stripped, which neutralizes the markup.
        # (The sanitizer strips dangerous characters; it is not a word filter, so
        # the literal text "alert" can remain — it is inert without "<>".)
        assert "<script>" not in result
        assert "<" not in result and ">" not in result
        assert "'" not in result and '"' not in result

    def test_string_length_validation(self):
        """Test string length validation"""
        # Valid length
        assert sanitize_string("A" * 100) == "A" * 100

        # Too long
        with pytest.raises(ValidationError):
            sanitize_string("A" * 1001)

    def test_string_type_validation(self):
        """Test string type validation"""
        # Valid type
        assert sanitize_string("test") == "test"

        # Invalid type
        with pytest.raises(ValidationError):
            sanitize_string(123)

        with pytest.raises(ValidationError):
            sanitize_string(None)


class TestPatientIdValidation:
    """Test patient ID validation"""

    def test_valid_patient_ids(self):
        """Test valid patient ID formats"""
        valid_ids = ["OAS2_0001", "TEST-PATIENT-123", "Patient_456", "ABC123XYZ", "12345"]

        for patient_id in valid_ids:
            result = sanitize_patient_id(patient_id)
            assert result == patient_id  # Should remain unchanged if valid

    def test_invalid_patient_ids(self):
        """Test invalid patient ID formats"""
        invalid_ids = [
            "OAS2 0001",  # Contains space
            "OAS2/0001",  # Contains slash
            "OAS2\\0001",  # Contains backslash
            "OAS2@0001",  # Contains @
            "OAS2#0001",  # Contains #
            "A" * 51,  # Too long
            "A",  # Too short
            "",  # Empty
        ]

        for patient_id in invalid_ids:
            with pytest.raises(ValidationError):
                sanitize_patient_id(patient_id)

    def test_patient_id_whitespace(self):
        """Test patient ID whitespace handling"""
        # Should trim whitespace
        assert sanitize_patient_id("  OAS2_0001  ") == "OAS2_0001"
        assert sanitize_patient_id("\tOAS2_0001\n") == "OAS2_0001"


class TestFilenameSanitization:
    """Test filename sanitization"""

    def test_valid_filenames(self):
        """Test valid filename formats"""
        valid_filenames = [
            "image.jpg",
            "scan_123.png",
            "patient-data.tiff",
            "MRI_scan.png",
            "test.file.jpeg",
        ]

        for filename in valid_filenames:
            result = sanitize_filename(filename)
            assert result == filename

    def test_invalid_filenames(self):
        """Test invalid filename formats"""
        invalid_filenames = [
            "../../../etc/passwd",  # Path traversal
            "/etc/passwd",  # Absolute path
            "\\windows\\system32\\config",  # Windows path traversal
            "file\x00null",  # Null byte
            "",  # Empty
        ]

        for filename in invalid_filenames:
            with pytest.raises(ValidationError):
                sanitize_filename(filename)

    def test_filename_path_components(self):
        """Path-bearing filenames are rejected (defense-in-depth)."""
        # A filename containing path separators is rejected outright; callers that
        # legitimately receive a path should pass os.path.basename(...) first.
        with pytest.raises(ValidationError):
            sanitize_filename("/path/to/image.jpg")
        with pytest.raises(ValidationError):
            sanitize_filename("./relative/path/scan.png")


class TestEmailValidation:
    """Test email validation"""

    def test_valid_emails(self):
        """Test valid email formats"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.com",
            "user+tag@example.co.uk",
            "user_name@test-domain.com",
        ]

        for email in valid_emails:
            result = validate_email(email)
            assert "@" in result
            assert result == result.lower()  # Should be lowercase

    def test_invalid_emails(self):
        """Test invalid email formats"""
        invalid_emails = [
            "plainaddress",
            "@example.com",
            "user@",
            "user name@example.com",  # Contains space
            "user@example..com",
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError):
                validate_email(email)


class TestMedicalDataValidation:
    """Test medical data validation functions"""

    def test_age_validation(self):
        """Test age validation"""
        # Valid ages
        assert validate_age(0) == 0
        assert validate_age(75) == 75
        assert validate_age(150) == 150
        assert validate_age(75.5) == 75  # Should convert to int

        # Invalid ages
        with pytest.raises(ValidationError):
            validate_age(-1)

        with pytest.raises(ValidationError):
            validate_age(151)

        with pytest.raises(ValidationError):
            validate_age("invalid")

        with pytest.raises(ValidationError):
            validate_age(None)

    def test_mmse_validation(self):
        """Test MMSE score validation"""
        # Valid MMSE scores
        assert validate_mmse(0) == 0.0
        assert validate_mmse(24.5) == 24.5
        assert validate_mmse(30) == 30.0
        assert validate_mmse(15.345) == 15.3  # Should round to 1 decimal

        # Invalid MMSE scores
        with pytest.raises(ValidationError):
            validate_mmse(-1)

        with pytest.raises(ValidationError):
            validate_mmse(31)

        with pytest.raises(ValidationError):
            validate_mmse("invalid")

    def test_education_validation(self):
        """Test education years validation"""
        # Valid education years
        assert validate_education_years(12) == 12
        assert validate_education_years(0) == 0
        assert validate_education_years(20) == 20

        # None should be allowed
        assert validate_education_years(None) is None
        assert validate_education_years("") is None

        # Invalid education years
        with pytest.raises(ValidationError):
            validate_education_years(-1)

        with pytest.raises(ValidationError):
            validate_education_years(31)

        with pytest.raises(ValidationError):
            validate_education_years("invalid")

    def test_gender_validation(self):
        """Test gender validation"""
        # Valid genders
        assert validate_gender("M") == "M"
        assert validate_gender("F") == "F"
        assert validate_gender("m") == "M"  # Should uppercase
        assert validate_gender("f") == "F"  # Should uppercase
        assert validate_gender("OTHER") == "OTHER"
        assert validate_gender("other") == "OTHER"  # Should uppercase

        # Invalid genders
        with pytest.raises(ValidationError):
            validate_gender("")

        with pytest.raises(ValidationError):
            validate_gender("X")

        with pytest.raises(ValidationError):
            validate_gender("Male")  # Should be M/F/OTHER


class TestImageUploadValidation:
    """Test image upload validation"""

    def test_valid_image_upload(self):
        """Test valid image upload"""
        # Create a small test image
        img = Image.new("L", (100, 100), color=128)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        file_data = buffered.getvalue()

        # Validate upload
        result = validate_image_upload("test.png", len(file_data))
        assert result["valid"] is True
        assert result["file_extension"] == ".png"
        assert result["safe_filename"] == "test.png"

    def test_invalid_file_type(self):
        """Test invalid file type rejection"""
        # Create a file with invalid extension
        file_data = b"fake file content"

        with pytest.raises(ValidationError):
            validate_image_upload("test.exe", len(file_data))

        with pytest.raises(ValidationError):
            validate_image_upload("test.sh", len(file_data))

    def test_file_size_validation(self):
        """Test file size validation"""
        # Create a small test image
        img = Image.new("L", (100, 100), color=128)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        file_data = buffered.getvalue()

        # Valid size
        result = validate_image_upload("test.png", len(file_data), max_size_mb=10)
        assert result["valid"] is True

        # Too large (simulate with size parameter)
        large_size = 11 * 1024 * 1024  # 11MB
        with pytest.raises(ValidationError):
            validate_image_upload("test.png", large_size, max_size_mb=10)

    def test_empty_file_rejection(self):
        """Test empty file rejection"""
        with pytest.raises(ValidationError):
            validate_image_upload("test.png", 0)


class TestSecurityValidation:
    """Test security-related validation"""

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        # Valid input
        assert validate_sql_input("OAS2_0001") == "OAS2_0001"
        assert validate_sql_input("patient_name") == "patient_name"

        # SQL injection attempts
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--",
            "'; EXEC xp_cmdshell('dir') --",
            "1'; DELETE FROM users WHERE 1=1--",
        ]

        for malicious_input in malicious_inputs:
            with pytest.raises(ValidationError):
                validate_sql_input(malicious_input)

    def test_path_traversal_prevention(self):
        """Test path traversal prevention"""
        # Valid paths
        assert validate_path_traversal("images") == "images"
        assert validate_path_traversal("patient_data") == "patient_data"

        # Path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "/etc/passwd",
            "..\\..\\..\\windows\\system32",
            "./../../sensitive_file",
            "/absolute/path/to/file",
            "normal/../../../etc/passwd",
        ]

        for malicious_path in malicious_paths:
            with pytest.raises(ValidationError):
                validate_path_traversal(malicious_path)


class TestPydanticModels:
    """Test Pydantic validation models"""

    def test_patient_data_validation(self):
        """Test PatientDataValidation model"""
        # Valid data
        valid_data = {
            "patient_id": "OAS2_0001",
            "age": 75,
            "mmse": 24.0,
            "gender": "F",
            "education": 12,
        }

        model = PatientDataValidation(**valid_data)
        assert model.patient_id == "OAS2_0001"
        assert model.age == 75
        assert model.mmse == 24.0
        assert model.gender == "F"
        assert model.education == 12

    def test_patient_data_validation_invalid(self):
        """Test PatientDataValidation with invalid data"""
        # Invalid age
        with pytest.raises(Exception):  # Pydantic validation error
            PatientDataValidation(
                patient_id="OAS2_0001",
                age=200,  # Invalid age
                mmse=24.0,
            )

        # Invalid MMSE
        with pytest.raises(Exception):
            PatientDataValidation(
                patient_id="OAS2_0001",
                age=75,
                mmse=35,  # Invalid MMSE
            )

    def test_diagnosis_request_validation(self):
        """Test DiagnosisRequestValidation model"""
        # Valid request
        valid_request = {
            "patient_data": {"patient_id": "OAS2_0001", "age": 75, "mmse": 24.0},
            "image_base64": "base64encodedstring",
            "longitudinal_id": "OAS2_0001",
        }

        model = DiagnosisRequestValidation(**valid_request)
        assert model.patient_data.patient_id == "OAS2_0001"
        assert model.longitudinal_id == "OAS2_0001"

    def test_diagnosis_request_validation_invalid_base64(self):
        """Test DiagnosisRequestValidation with invalid base64"""
        valid_patient_data = {"patient_id": "OAS2_0001", "age": 75, "mmse": 24.0}

        # Invalid base64
        with pytest.raises(Exception):
            DiagnosisRequestValidation(
                patient_data=valid_patient_data, image_base64="not@valid#base64!"
            )


class TestBatchValidation:
    """Test batch request validation"""

    def test_valid_batch_request(self):
        """Test valid batch request"""
        valid_batch = [
            {"patient_data": {"patient_id": f"P{i:03d}", "age": 70 + i, "mmse": 25.0}}
            for i in range(5)
        ]

        result = validate_batch_request(valid_batch, max_batch_size=100)
        assert len(result) == 5

    def test_batch_size_exceeded(self):
        """Test batch size validation"""
        # Create batch that exceeds limit
        large_batch = [
            {"patient_data": {"patient_id": f"P{i:03d}", "age": 70, "mmse": 25.0}}
            for i in range(150)  # Exceeds limit of 100
        ]

        with pytest.raises(ValidationError):
            validate_batch_request(large_batch, max_batch_size=100)

    def test_empty_batch_request(self):
        """Test empty batch request"""
        with pytest.raises(ValidationError):
            validate_batch_request([], max_batch_size=100)

    def test_batch_request_invalid_item(self):
        """Test batch request with invalid item"""
        invalid_batch = [
            {
                "patient_data": {
                    "patient_id": "P001",
                    "age": 200,  # Invalid age
                    "mmse": 25.0,
                }
            },
            {"patient_data": {"patient_id": "P002", "age": 70, "mmse": 25.0}},
        ]

        with pytest.raises(ValidationError):
            validate_batch_request(invalid_batch, max_batch_size=100)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_boundary_values(self):
        """Test boundary value validation"""
        # Age boundaries
        assert validate_age(0) == 0  # Minimum
        assert validate_age(150) == 150  # Maximum

        # MMSE boundaries
        assert validate_mmse(0.0) == 0.0  # Minimum
        assert validate_mmse(30.0) == 30.0  # Maximum

        # Education boundaries
        assert validate_education_years(0) == 0  # Minimum
        assert validate_education_years(30) == 30  # Maximum

    def test_unicode_handling(self):
        """Test Unicode character handling"""
        # Unicode characters should be handled properly
        unicode_string = "Tëst Pätïënt_123"
        result = sanitize_string(unicode_string, max_length=50)
        assert len(result) <= 50

    def test_special_characters(self):
        """Test special character handling"""
        # Special characters that should be allowed
        valid_special = "Patient-Name_123.Test"
        result = sanitize_string(valid_special)
        assert result == valid_special

        # Control characters (e.g. a null byte) are stripped by the sanitizer, not
        # rejected — sanitize_* cleans input; validate_* is what raises.
        result = sanitize_string("Test\x00Null")
        assert "\x00" not in result
        assert result == "TestNull"

    def test_whitespace_variations(self):
        """Test various whitespace variations"""
        assert sanitize_patient_id("  OAS2_0001  ") == "OAS2_0001"
        assert sanitize_patient_id("\tOAS2_0001\n") == "OAS2_0001"
        # An internal tab is an invalid ID character and is rejected, not converted.
        with pytest.raises(ValidationError):
            sanitize_patient_id("OAS2\t0001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
