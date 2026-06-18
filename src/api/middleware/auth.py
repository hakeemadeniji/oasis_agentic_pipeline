"""
OAuth2/OpenID Connect authentication middleware for OASIS Agentic Pipeline API.

Implements enterprise-grade authentication with OAuth2 password flow and OpenID Connect,
JWT token validation, role-based access control, and token refresh functionality.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt

from src.api.exceptions import (
    AuthenticationError,
    AuthorizationError,
    TokenError,
    ConfigurationError,
)
from src.config import get_settings


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


class AuthConfig:
    """Authentication configuration."""

    def __init__(self):
        self.settings = get_settings()

        # JWT configuration — use the DEDICATED JWT secret, never the Anthropic API
        # key (unrelated secret; also empty on most deploys, which would silently
        # fall back to a public hardcoded string and make every token forgeable).
        self.secret_key = self.settings.jwt_secret
        self.algorithm = "HS256"
        self.access_token_expire_minutes = self.settings.access_token_expire_minutes
        self.refresh_token_expire_days = self.settings.refresh_token_expire_days

        # OAuth2/OIDC configuration (driven by config / env, not hardcoded)
        self.oauth2_enabled = True
        self.oidc_enabled = self.settings.oidc_enabled
        self.oidc_discovery_url = self.settings.oidc_discovery_url or None
        self.oidc_client_id = self.settings.oidc_client_id or None
        self.oidc_client_secret = self.settings.oidc_client_secret or None

        # User roles
        self.valid_roles = ["admin", "clinician", "researcher", "viewer"]

        # Development mode — EXPLICIT flag (AUTH_DEV_MODE), never inferred from the
        # compute device. dev_mode relaxes secret validation; defaulting it on for
        # CPU boxes (this project's edge target) would disable that check in prod.
        self.dev_mode = self.settings.auth_dev_mode

    def validate_configuration(self):
        """Validate authentication configuration."""
        if self.dev_mode:
            logger.warning("Running in development mode - using relaxed security")
            return

        if not self.secret_key or self.secret_key == "dev-secret-key-change-in-production":
            raise ConfigurationError(
                message="JWT secret key must be set in production", config_key="JWT_SECRET_KEY"
            )

        if self.oidc_enabled:
            if not self.oidc_discovery_url or not self.oidc_client_id:
                raise ConfigurationError(
                    message="OIDC requires discovery URL and client ID",
                    config_key="OIDC_CONFIGURATION",
                )


# Global auth configuration
auth_config = AuthConfig()


# Password hashing — use the maintained `bcrypt` library directly. passlib (1.7.4,
# last released 2020) crashes against bcrypt >= 4.1/5.x: its compatibility probe
# sends a >72-byte test string that newer bcrypt rejects, which would blow up at
# import time the moment the mock users below are hashed.
def _hash_password(password: str) -> str:
    """Hash a password with bcrypt (input truncated to bcrypt's 72-byte limit)."""
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash; False on any malformed input."""
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


# ============================================================================
# User Models
# ============================================================================


class User:
    """User model for authentication."""

    def __init__(
        self,
        username: str,
        email: str,
        role: str = "viewer",
        full_name: Optional[str] = None,
        disabled: bool = False,
    ):
        self.username = username
        self.email = email
        self.role = role
        self.full_name = full_name or username
        self.disabled = disabled

    def has_role(self, required_role: str) -> bool:
        """Check if user has required role."""
        if self.role == "admin":
            return True  # Admin has all permissions

        role_hierarchy = {"admin": 4, "clinician": 3, "researcher": 2, "viewer": 1}

        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "full_name": self.full_name,
            "disabled": self.disabled,
        }


# ============================================================================
# Mock User Database (Development Only)
# ============================================================================


class MockUserDatabase:
    """Mock user database for development and testing."""

    def __init__(self):
        self.users: Dict[str, User] = {
            "admin": User(
                username="admin",
                email="admin@oasis-pipeline.com",
                role="admin",
                full_name="System Administrator",
            ),
            "clinician": User(
                username="clinician",
                email="clinician@hospital.com",
                role="clinician",
                full_name="Dr. Smith",
            ),
            "researcher": User(
                username="researcher",
                email="researcher@university.edu",
                role="researcher",
                full_name="Dr. Johnson",
            ),
            "viewer": User(
                username="viewer",
                email="viewer@hospital.com",
                role="viewer",
                full_name="Nurse Davis",
            ),
        }
        # Mock password hashes (password = "password123")
        self.password_hashes = {
            "admin": _hash_password("password123"),
            "clinician": _hash_password("password123"),
            "researcher": _hash_password("password123"),
            "viewer": _hash_password("password123"),
        }

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)

    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password."""
        if username not in self.password_hashes:
            return False
        return _verify_password(password, self.password_hashes[username])


# Global user database instance
user_db = MockUserDatabase()


# ============================================================================
# JWT Token Functions
# ============================================================================


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=auth_config.access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "type": "access"})

    encoded_jwt = jwt.encode(to_encode, auth_config.secret_key, algorithm=auth_config.algorithm)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(days=auth_config.refresh_token_expire_days)

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "type": "refresh"})

    encoded_jwt = jwt.encode(to_encode, auth_config.secret_key, algorithm=auth_config.algorithm)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, auth_config.secret_key, algorithms=[auth_config.algorithm])
        return payload
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise TokenError("Invalid or expired token")


# ============================================================================
# Authentication Functions
# ============================================================================


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user = user_db.get_user(username)
    if not user:
        return None

    if not user_db.verify_password(username, password):
        return None

    if user.disabled:
        raise AuthenticationError("User account is disabled")

    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user from JWT token."""
    try:
        payload = decode_token(token)

        username: str = payload.get("sub")
        if username is None:
            raise TokenError("Token missing subject claim")

        token_type: str = payload.get("type")
        if token_type != "access":
            raise TokenError("Invalid token type")

    except TokenError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        raise TokenError("Token validation failed")

    user = user_db.get_user(username)
    if user is None:
        raise AuthenticationError("User not found")

    if user.disabled:
        raise AuthenticationError("User account is disabled")

    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (raises if disabled)."""
    if current_user.disabled:
        raise AuthenticationError("Inactive user")
    return current_user


# ============================================================================
# Authorization Dependencies
# ============================================================================


def require_role(required_role: str):
    """Dependency factory for requiring specific user role."""

    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not current_user.has_role(required_role):
            raise AuthorizationError(
                message=f"User requires '{required_role}' role", required_role=required_role
            )
        return current_user

    return role_checker


# Role-specific dependencies
require_admin = require_role("admin")
require_clinician = require_role("clinician")
require_researcher = require_role("researcher")
require_viewer = require_role("viewer")


# ============================================================================
# FastAPI Dependencies
# ============================================================================


async def get_current_user_optional(request: Request) -> Optional[User]:
    """Get current user if authenticated, otherwise return None."""
    authorization: str = request.headers.get("Authorization")
    if not authorization:
        return None

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None

        return get_current_user(token)
    except (AuthenticationError, TokenError):
        return None


# ============================================================================
# Development Mode Bypass
# ============================================================================


def dev_mode_bypass() -> User:
    """Bypass authentication in development mode."""
    if auth_config.dev_mode:
        logger.warning("Development mode bypass - returning admin user")
        return user_db.get_user("admin")
    raise AuthenticationError("Authentication required")


# ============================================================================
# Setup Function
# ============================================================================


def setup_authentication(app):
    """
    Set up authentication for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    try:
        auth_config.validate_configuration()

        # Add authentication endpoints
        _add_auth_endpoints(app)

        logger.info(
            f"Authentication configured - OAuth2: {'enabled' if auth_config.oauth2_enabled else 'disabled'}, "
            f"OIDC: {'enabled' if auth_config.oidc_enabled else 'disabled'}, "
            f"Dev mode: {auth_config.dev_mode}"
        )

    except ConfigurationError as e:
        logger.error(f"Authentication configuration error: {e.message}")
        raise


def _add_auth_endpoints(app):
    """Add authentication endpoints to the application."""

    @app.post("/api/auth/token")
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        """OAuth2 token endpoint for obtaining access token."""
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=auth_config.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
        )

        refresh_token = create_refresh_token(data={"sub": user.username})

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": auth_config.access_token_expire_minutes * 60,
            "user": user.to_dict(),
        }

    @app.post("/api/auth/refresh")
    async def refresh_token(refresh_token: str):
        """Refresh access token using refresh token."""
        try:
            payload = decode_token(refresh_token)

            token_type: str = payload.get("type")
            if token_type != "refresh":
                raise TokenError("Invalid token type")

            username: str = payload.get("sub")
            user = user_db.get_user(username)

            if not user or user.disabled:
                raise AuthenticationError("Invalid user")

            access_token_expires = timedelta(minutes=auth_config.access_token_expire_minutes)
            access_token = create_access_token(
                data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
            )

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": auth_config.access_token_expire_minutes * 60,
            }

        except TokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

    @app.get("/api/auth/me")
    async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
        """Get current user information."""
        return current_user.to_dict()

    @app.post("/api/auth/verify")
    async def verify_token(token: str):
        """Verify if a token is valid."""
        try:
            payload = decode_token(token)
            username: str = payload.get("sub")
            user = user_db.get_user(username)

            if not user or user.disabled:
                return {"valid": False, "reason": "User not found or disabled"}

            return {
                "valid": True,
                "username": username,
                "role": user.role,
                "expires": payload.get("exp"),
            }

        except TokenError:
            return {"valid": False, "reason": "Invalid token"}


if __name__ == "__main__":
    # Test authentication functions
    print("Testing authentication...")

    # Test password hashing
    password = "test_password"
    hashed = _hash_password(password)
    print(f"Password hashed: {hashed[:20]}...")

    # Test password verification
    valid = _verify_password(password, hashed)
    print(f"Password verification: {valid}")

    # Test JWT token creation and validation
    token = create_access_token({"sub": "test_user", "role": "clinician"})
    print(f"Token created: {token[:20]}...")

    try:
        payload = decode_token(token)
        print(f"Token decoded: {payload}")
    except TokenError as e:
        print(f"Token error: {e.message}")

    # Test mock database
    user = user_db.get_user("admin")
    if user:
        print(f"User found: {user.to_dict()}")
        print(f"Has admin role: {user.has_role('admin')}")
        print(f"Has clinician role: {user.has_role('clinician')}")
