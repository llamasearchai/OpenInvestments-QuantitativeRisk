"""
Minimal observability middleware setup.

In production, this would configure OpenTelemetry/Prometheus, etc.
For smoke tests and local usage, we keep it as a no-op to avoid heavy deps.
"""

from typing import Any


def setup_observability(app: Any) -> None:
    """Attach observability middlewares if available (no-op here)."""
    # Intentionally minimal to keep smoke tests lightweight.
    return None

