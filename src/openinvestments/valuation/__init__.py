"""
Valuation Library

Provides advanced pricing models for financial instruments including:
- Monte Carlo simulation
- Binomial and trinomial trees

Note:
- Heavy optional dependencies (e.g., numba) are used only within specific
  modules and are intentionally not imported at package import time to keep
  lightweight environments functional.
"""

from .monte_carlo import MonteCarloPricer
from .trees import BinomialTree, TrinomialTree

__all__ = [
    'MonteCarloPricer',
    'BinomialTree',
    'TrinomialTree',
]
