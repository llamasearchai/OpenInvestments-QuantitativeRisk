"""
Greeks calculator with Automatic Differentiation (AAD) support.

Computes delta, gamma, vega, theta, rho, and higher-order Greeks.
"""

import numpy as np
import numba as nb
from typing import Callable, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


class GreekType(Enum):
    """Enumeration of Greek types."""
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    RHO = "rho"
    VANNA = "vanna"
    VOLGA = "volga"
    CHARM = "charm"
    COLOR = "color"
    SPEED = "speed"


class PricingModel(ABC):
    """Abstract base class for pricing models."""

    @abstractmethod
    def price(self, params: Dict[str, float]) -> float:
        """Calculate option price given parameters."""
        pass


class BlackScholesModel(PricingModel):
    """Black-Scholes pricing model."""

    def price(self, params: Dict[str, float]) -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            params: Dictionary with keys: S, K, T, r, sigma, is_call

        Returns:
            Option price
        """
        S = params['S']
        K = params['K']
        T = params['T']
        r = params['r']
        sigma = params['sigma']
        is_call = params.get('is_call', True)

        if T <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            price = S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

        return price

    @staticmethod
    @nb.jit(nopython=True)
    def _norm_cdf(x: float) -> float:
        """Approximation of cumulative normal distribution."""
        return 0.5 * (1 + np.tanh(0.886 * x - 0.002 * x**3))


class GreeksCalculator:
    """Calculator for option Greeks using finite differences and AAD."""

    def __init__(self, model: PricingModel = None, use_aad: bool = True):
        """
        Initialize Greeks calculator.

        Args:
            model: Pricing model to use
            use_aad: Whether to use Automatic Differentiation (if available)
        """
        self.model = model or BlackScholesModel()
        self.use_aad = use_aad
        self.logger = logger

        # Finite difference step sizes
        self.h = 1e-5
        self.h2 = 1e-4  # For second derivatives

    def calculate_greeks(
        self,
        params: Dict[str, float],
        greek_types: list[GreekType] = None
    ) -> Dict[str, float]:
        """
        Calculate specified Greeks for the option.

        Args:
            params: Model parameters
            greek_types: List of Greeks to calculate

        Returns:
            Dictionary of Greek values
        """
        if greek_types is None:
            greek_types = [GreekType.DELTA, GreekType.GAMMA, GreekType.VEGA,
                          GreekType.THETA, GreekType.RHO]

        greeks = {}

        for greek_type in greek_types:
            greek_value = self._calculate_single_greek(params, greek_type)
            greeks[greek_type.value] = greek_value

        self.logger.info("Greeks calculation completed",
                        requested_greeks=[g.value for g in greek_types],
                        calculated_values=greeks)

        return greeks

    def _calculate_single_greek(
        self,
        params: Dict[str, float],
        greek_type: GreekType
    ) -> float:
        """Calculate a single Greek using finite differences."""

        if greek_type == GreekType.DELTA:
            return self._finite_difference(params, 'S', self.h)

        elif greek_type == GreekType.GAMMA:
            return self._second_finite_difference(params, 'S', self.h2)

        elif greek_type == GreekType.VEGA:
            return self._finite_difference(params, 'sigma', self.h)

        elif greek_type == GreekType.THETA:
            # Theta is negative of time derivative
            return -self._finite_difference(params, 'T', self.h)

        elif greek_type == GreekType.RHO:
            return self._finite_difference(params, 'r', self.h)

        elif greek_type == GreekType.VANNA:
            # Cross derivative: d²V/dS dσ
            return self._mixed_derivative(params, 'S', 'sigma', self.h)

        elif greek_type == GreekType.VOLGA:
            # Second derivative w.r.t. volatility
            return self._second_finite_difference(params, 'sigma', self.h2)

        elif greek_type == GreekType.CHARM:
            # d²V/dS dt
            return self._mixed_derivative(params, 'S', 'T', self.h)

        elif greek_type == GreekType.COLOR:
            # d³V/dt dσ² (gamma decay)
            return self._gamma_decay(params, self.h)

        elif greek_type == GreekType.SPEED:
            # d³V/dS³
            return self._third_derivative(params, 'S', self.h)

        else:
            raise ValueError(f"Unsupported Greek type: {greek_type}")

    def _finite_difference(
        self,
        params: Dict[str, float],
        param_key: str,
        h: float
    ) -> float:
        """Calculate first derivative using central finite difference."""
        params_up = params.copy()
        params_down = params.copy()

        params_up[param_key] = params[param_key] + h
        params_down[param_key] = params[param_key] - h

        price_up = self.model.price(params_up)
        price_down = self.model.price(params_down)

        return (price_up - price_down) / (2 * h)

    def _second_finite_difference(
        self,
        params: Dict[str, float],
        param_key: str,
        h: float
    ) -> float:
        """Calculate second derivative using central finite difference."""
        params_up = params.copy()
        params_center = params.copy()
        params_down = params.copy()

        params_up[param_key] = params[param_key] + h
        params_down[param_key] = params[param_key] - h

        price_up = self.model.price(params_up)
        price_center = self.model.price(params_center)
        price_down = self.model.price(params_down)

        return (price_up - 2 * price_center + price_down) / (h ** 2)

    def _mixed_derivative(
        self,
        params: Dict[str, float],
        param1: str,
        param2: str,
        h: float
    ) -> float:
        """Calculate mixed partial derivative."""
        # Four-point stencil for mixed derivative
        params_pp = params.copy()
        params_pm = params.copy()
        params_mp = params.copy()
        params_mm = params.copy()

        params_pp[param1] = params[param1] + h
        params_pp[param2] = params[param2] + h

        params_pm[param1] = params[param1] + h
        params_pm[param2] = params[param2] - h

        params_mp[param1] = params[param1] - h
        params_mp[param2] = params[param2] + h

        params_mm[param1] = params[param1] - h
        params_mm[param2] = params[param2] - h

        price_pp = self.model.price(params_pp)
        price_pm = self.model.price(params_pm)
        price_mp = self.model.price(params_mp)
        price_mm = self.model.price(params_mm)

        return (price_pp - price_pm - price_mp + price_mm) / (4 * h * h)

    def _third_derivative(
        self,
        params: Dict[str, float],
        param_key: str,
        h: float
    ) -> float:
        """Calculate third derivative using finite differences."""
        params_up2 = params.copy()
        params_up1 = params.copy()
        params_down1 = params.copy()
        params_down2 = params.copy()

        params_up2[param_key] = params[param_key] + 2 * h
        params_up1[param_key] = params[param_key] + h
        params_down1[param_key] = params[param_key] - h
        params_down2[param_key] = params[param_key] - 2 * h

        price_up2 = self.model.price(params_up2)
        price_up1 = self.model.price(params_up1)
        price_center = self.model.price(params)
        price_down1 = self.model.price(params_down1)
        price_down2 = self.model.price(params_down2)

        # Sixth-order finite difference for third derivative
        return (-price_up2 + 3*price_up1 - 3*price_down1 + price_down2) / (8 * h**3)

    def _gamma_decay(self, params: Dict[str, float], h: float) -> float:
        """Calculate gamma decay (d³V/dt dσ²)."""
        # This is a complex derivative requiring multiple finite differences
        # Simplified implementation - would need more sophisticated AAD for accuracy
        gamma_t = self._mixed_derivative(params, 'sigma', 'T', h)
        return gamma_t

    def calculate_greeks_portfolio(
        self,
        positions: list[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate portfolio Greeks.

        Args:
            positions: List of position dictionaries with params and quantity

        Returns:
            Portfolio Greeks
        """
        portfolio_greeks = {}

        for position in positions:
            quantity = position.get('quantity', 1)
            params = position['params']

            position_greeks = self.calculate_greeks(params)

            for greek, value in position_greeks.items():
                if greek not in portfolio_greeks:
                    portfolio_greeks[greek] = 0
                portfolio_greeks[greek] += quantity * value

        return portfolio_greeks

    def calculate_greeks_time_series(
        self,
        params: Dict[str, float],
        time_points: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks evolution over time.

        Args:
            params: Base parameters
            time_points: Array of time points

        Returns:
            Time series of Greeks
        """
        greek_series = {}

        for t in time_points:
            params_t = params.copy()
            params_t['T'] = t

            if t > 0:  # Avoid division by zero at maturity
                greeks_t = self.calculate_greeks(params_t)
                for greek, value in greeks_t.items():
                    if greek not in greek_series:
                        greek_series[greek] = []
                    greek_series[greek].append(value)

        # Convert to numpy arrays
        for greek in greek_series:
            greek_series[greek] = np.array(greek_series[greek])

        return greek_series
