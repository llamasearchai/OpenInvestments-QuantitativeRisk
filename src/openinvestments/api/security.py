"""
Security helpers for the API.

Provides JWT token validation and authentication for the OpenInvestments platform.
Supports both development mode (accepts any token) and production mode (validates JWTs).
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "openinvestments-dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))


class SecurityManager:
    """Manages authentication and authorization for the API."""

    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS

    def create_token(self, user_id: str, role: str = "user") -> str:
        """Create a JWT token for a user."""
        payload = {
            "sub": user_id,
            "role": role,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.expiration_hours)
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token.

        Returns:
            Dict containing user info if valid, None if invalid
        """
        if not token:
            return None

        try:
            # For development/testing, accept any non-empty token
            if os.getenv("ENVIRONMENT", "development") == "development":
                return {"sub": "dev-user", "role": "admin"}

            # Production validation
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            if datetime.utcfromtimestamp(payload["exp"]) < datetime.utcnow():
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None

    def get_user_role(self, token: Optional[str]) -> Optional[str]:
        """Extract user role from token."""
        payload = self.verify_token(token)
        return payload.get("role") if payload else None

    def has_permission(self, token: Optional[str], required_role: str) -> bool:
        """Check if user has required role."""
        user_role = self.get_user_role(token)
        if not user_role:
            return False

        # Role hierarchy: admin > analyst > user
        role_hierarchy = {"user": 1, "analyst": 2, "admin": 3}
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 999)

        return user_level >= required_level


# Global security manager instance
security_manager = SecurityManager()


def verify_token(token: Optional[str]) -> bool:
    """Return True if token is valid."""
    return security_manager.verify_token(token) is not None


def require_auth(token: Optional[str], required_role: str = "user") -> bool:
    """Check if request has valid authentication and required role."""
    return security_manager.has_permission(token, required_role)

