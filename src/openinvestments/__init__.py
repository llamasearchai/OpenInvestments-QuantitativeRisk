"""
OpenInvestments Core Package

Main entry point for the quantitative risk analytics platform.
"""

from .core.config import Config
from .core.logging import setup_logging

__all__ = ['Config', 'setup_logging']
