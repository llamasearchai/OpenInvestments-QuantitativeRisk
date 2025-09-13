"""
Lightweight security helpers for the API.

In production, this would validate JWTs/roles. For smoke tests and
development, we accept any non-empty token by default.
"""

from typing import Optional


def verify_token(token: Optional[str]) -> bool:
    """Return True if token is acceptable. Here we allow any token or none."""
    # Minimal stub: always allow for local usage and smoke tests.
    return True

