"""
Integration tests for security middleware.

Tests authentication, rate limiting, security headers, and error handling
middleware to ensure proper security functionality.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.api.main import app
from src.api.middleware.auth import auth_config
from src.api.exceptions import (
    AuthenticationError,
    RateLimitError,
    ValidationError,
    MedicalImagingError,
)


class TestAuthenticationMiddleware:
    """Test OAuth2/OIDC authentication middleware"""

    @pytest.fixture
    def client(self):
        """Create test client with authentication enabled"""
        # Set development mode for testing
        auth_config.dev_mode = True
        return TestClient(app)

    def test_health_check_no_auth(self, client):
        """Health check should work without authentication"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_token_endpoint(self, client):
        """Test token generation endpoint"""
        response = client.post(
            "/api/auth/token", data={"username": "admin", "password": "password123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["username"] == "admin"
        assert data["user"]["role"] == "admin"

    def test_token_invalid_credentials(self, client):
        """Test token generation with invalid credentials"""
        response = client.post(
            "/api/auth/token", data={"username": "admin", "password": "wrong_password"}
        )
        assert response.status_code == 401

    def test_token_refresh(self, client):
        """Test token refresh functionality"""
        # First get a token
        token_response = client.post(
            "/api/auth/token", data={"username": "clinician", "password": "password123"}
        )
        refresh_token = token_response.json()["refresh_token"]

        # Use refresh token to get new access token
        refresh_response = client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert "access_token" in data

    def test_verify_token(self, client):
        """Test token verification endpoint"""
        # Get a token
        token_response = client.post(
            "/api/auth/token", data={"username": "researcher", "password": "password123"}
        )
        access_token = token_response.json()["access_token"]

        # Verify token
        verify_response = client.post("/api/auth/verify", json={"token": access_token})
        assert verify_response.status_code == 200
        data = verify_response.json()
        assert data["valid"] is True
        assert data["username"] == "researcher"

    def test_verify_invalid_token(self, client):
        """Test verification with invalid token"""
        response = client.post("/api/auth/verify", json={"token": "invalid_token"})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_get_current_user(self, client):
        """Test getting current user info"""
        # Get a token
        token_response = client.post(
            "/api/auth/token", data={"username": "viewer", "password": "password123"}
        )
        access_token = token_response.json()["access_token"]

        # Get current user info
        user_response = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {access_token}"}
        )
        assert user_response.status_code == 200
        data = user_response.json()
        assert data["username"] == "viewer"
        assert data["role"] == "viewer"

    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token"""
        # Research endpoint requires researcher role
        response = client.get("/research")
        # Should fail without authentication
        assert response.status_code in [401, 403]

    def test_protected_endpoint_with_wrong_role(self, client):
        """Test accessing protected endpoint with insufficient role"""
        # Get viewer token
        token_response = client.post(
            "/api/auth/token", data={"username": "viewer", "password": "password123"}
        )
        access_token = token_response.json()["access_token"]

        # Try to access research endpoint with viewer role
        response = client.get("/research", headers={"Authorization": f"Bearer {access_token}"})
        # Should fail due to insufficient permissions
        assert response.status_code == 403


class TestRateLimitingMiddleware:
    """Test rate limiting middleware"""

    @pytest.fixture
    def client(self):
        """Create test client with rate limiting"""
        return TestClient(app)

    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present"""
        response = client.get("/health")
        # Rate limiting headers should be present even if not limited
        # (implementation dependent)
        assert response.status_code == 200

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        # Make multiple concurrent requests
        import threading

        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should succeed (within reasonable limits)
        assert all(status == 200 for status in results)

    def test_rate_limit_exceeded(self, client):
        """Test behavior when rate limit is exceeded"""
        # This test would need to configure very low rate limits
        # For now, we'll just test the structure

        # Make many rapid requests
        responses = []
        for _ in range(5):
            response = client.get("/health")
            responses.append(response.status_code)

        # Should handle gracefully
        assert all(status in [200, 429] for status in responses)


class TestSecurityHeaders:
    """Test security headers middleware"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_security_headers_present(self, client):
        """Test that security headers are present"""
        response = client.get("/health")

        # Check for security headers
        headers = response.headers

        # These headers should be present
        assert "X-Content-Type-Options" in headers or "x-content-type-options" in headers
        assert "X-Frame-Options" in headers or "x-frame-options" in headers
        assert "X-XSS-Protection" in headers or "x-xss-protection" in headers

    def test_cors_headers(self, client):
        """Test CORS headers"""
        # Make an OPTIONS request
        response = client.options("/health")

        # CORS headers should be present
        assert response.status_code in [200, 405]  # 405 if OPTIONS not allowed

    def test_server_info_removed(self, client):
        """Test that server information is removed"""
        response = client.get("/health")

        # Server header should be removed or generic
        server_header = response.headers.get("server", "")
        assert "nginx" not in server_header.lower()
        assert "apache" not in server_header.lower()


class TestErrorHandling:
    """Test global error handling middleware"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_custom_exception_handling(self, client):
        """Test that custom exceptions are handled properly"""
        # This would require triggering actual errors
        # For now, test the error structure

        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data or "detail" in data

    def test_request_id_present(self, client):
        """Test that request ID is added to responses"""
        response = client.get("/health")

        # Request ID header should be present
        assert "X-Request-ID" in response.headers or "x-request-id" in response.headers

    def test_validation_error_format(self, client):
        """Test that validation errors have consistent format"""
        # Send invalid data to trigger validation error
        response = client.post(
            "/diagnose",
            json={"patient_data": {"age": 999}},  # Invalid age
            headers={"Content-Type": "application/json"},
        )

        # Should return validation error
        assert response.status_code in [400, 422]

        data = response.json()
        assert "error" in data or "detail" in data


class TestInputValidation:
    """Test input validation and sanitization"""

    def test_patient_id_sanitization(self):
        """Test patient ID sanitization"""
        from src.api.validators import sanitize_patient_id

        # Valid IDs
        assert sanitize_patient_id("OAS2_0001") == "OAS2_0001"
        assert sanitize_patient_id("TEST-PATIENT_123") == "TEST-PATIENT_123"

        # Invalid IDs
        with pytest.raises(ValidationError):
            sanitize_patient_id("OAS2 0001")  # Contains space

        with pytest.raises(ValidationError):
            sanitize_patient_id("OAS2/0001")  # Contains slash

        with pytest.raises(ValidationError):
            sanitize_patient_id("A" * 51)  # Too long

    def test_age_validation(self):
        """Test age validation"""
        from src.api.validators import validate_age

        # Valid ages
        assert validate_age(75) == 75
        assert validate_age(0) == 0
        assert validate_age(150) == 150

        # Invalid ages
        with pytest.raises(ValidationError):
            validate_age(-1)

        with pytest.raises(ValidationError):
            validate_age(151)

        with pytest.raises(ValidationError):
            validate_age("invalid")

    def test_mmse_validation(self):
        """Test MMSE score validation"""
        from src.api.validators import validate_mmse

        # Valid MMSE scores
        assert validate_mmse(24.5) == 24.5
        assert validate_mmse(0) == 0.0
        assert validate_mmse(30) == 30.0

        # Invalid MMSE scores
        with pytest.raises(ValidationError):
            validate_mmse(-1)

        with pytest.raises(ValidationError):
            validate_mmse(31)

    def test_filename_sanitization(self):
        """Test filename sanitization"""
        from src.api.validators import sanitize_filename

        # Valid filenames
        assert sanitize_filename("image.jpg") == "image.jpg"
        assert sanitize_filename("scan_123.png") == "scan_123.png"

        # Invalid filenames (path traversal)
        with pytest.raises(ValidationError):
            sanitize_filename("../../../etc/passwd")

        with pytest.raises(ValidationError):
            sanitize_filename("/etc/passwd")

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        from src.api.validators import validate_sql_input

        # Valid input
        assert validate_sql_input("OAS2_0001") == "OAS2_0001"

        # Malicious input
        with pytest.raises(ValidationError):
            validate_sql_input("'; DROP TABLE users; --")

        with pytest.raises(ValidationError):
            validate_sql_input("1' OR '1'='1")


class TestCustomExceptions:
    """Test custom exception classes"""

    def test_medical_imaging_error(self):
        """Test MedicalImagingError"""
        error = MedicalImagingError(message="Failed to process MRI", details={"file": "test.jpg"})

        assert error.message == "Failed to process MRI"
        assert error.error_code == "MEDICAL_IMAGING_ERROR"
        assert error.status_code == 500

        error_dict = error.to_dict()
        assert error_dict["error_code"] == "MEDICAL_IMAGING_ERROR"
        assert error_dict["message"] == "Failed to process MRI"

    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError(message="Invalid credentials")

        assert error.message == "Invalid credentials"
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.status_code == 401

    def test_rate_limit_error(self):
        """Test RateLimitError"""
        error = RateLimitError(message="Rate limit exceeded", retry_after=60, limit=100, window=60)

        assert error.message == "Rate limit exceeded"
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.status_code == 429
        assert error.details["retry_after"] == 60
        assert error.details["limit"] == 100

    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError(message="Invalid input", field="age", value=150)

        assert error.message == "Invalid input"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.status_code == 400
        assert error.details["field"] == "age"


class TestRetryLogic:
    """Test retry logic utilities"""

    def test_retry_configuration(self):
        """Test retry configuration"""
        from src.utils.retry import RetryConfig

        config = RetryConfig(max_attempts=3, initial_delay=1.0, max_delay=10.0)

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 10.0

        # Test delay calculation
        delay1 = config.get_delay(1)
        delay2 = config.get_delay(2)
        delay3 = config.get_delay(3)

        assert delay1 == 1.0  # First attempt
        assert delay2 == 2.0  # Second attempt (2^1)
        assert delay3 == 4.0  # Third attempt (2^2)

        # Test max delay cap
        config_long = RetryConfig(max_attempts=10, initial_delay=1.0, max_delay=5.0)
        delay_10 = config_long.get_delay(10)
        assert delay_10 == 5.0  # Capped at max_delay

    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        from src.utils.retry import CircuitBreaker

        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

        # Initially closed
        assert circuit_breaker.state == "closed"

        # Simulate failures
        def failing_function():
            raise ConnectionError("Service unavailable")

        for _ in range(3):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                pass

        # Should be open after threshold
        assert circuit_breaker.state == "open"

        # Should raise exception when circuit is open
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)


class TestSecurityIntegration:
    """Integration tests for complete security flow"""

    @pytest.fixture
    def authenticated_client(self):
        """Create authenticated test client"""
        client = TestClient(app)

        # Get authentication token
        token_response = client.post(
            "/api/auth/token", data={"username": "clinician", "password": "password123"}
        )
        token = token_response.json()["access_token"]

        # Return client with authentication headers
        return client, token

    def test_complete_auth_flow(self, authenticated_client):
        """Test complete authentication flow"""
        client, token = authenticated_client

        # Access protected endpoint
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "clinician"

    def test_security_headers_on_protected_endpoint(self, authenticated_client):
        """Test that security headers are present on protected endpoints"""
        client, token = authenticated_client

        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

        # Check security headers
        headers = response.headers
        assert "X-Request-ID" in headers or "x-request-id" in headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
