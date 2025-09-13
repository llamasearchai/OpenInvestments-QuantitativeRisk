"""
Value at Risk (VaR) and Expected Shortfall (ES) calculators.

Supports multiple calculation methods: historical simulation, parametric, Monte Carlo.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    EVT = "extreme_value_theory"


class ESDistribution(Enum):
    """Expected Shortfall distribution assumptions."""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    HISTORICAL = "historical"


@dataclass
class RiskConfig:
    """Configuration for risk calculations."""
    confidence_level: float = 0.95
    horizon_days: int = 1
    portfolio_value: float = 1_000_000
    method: VaRMethod = VaRMethod.HISTORICAL
    use_volatility_scaling: bool = True
    lambda_decay: float = 0.94  # For EWMA


class VaRCalculator:
    """Value at Risk calculator with multiple methodologies."""

    def __init__(self, config: RiskConfig = None):
        """
        Initialize VaR calculator.

        Args:
            config: Risk calculation configuration
        """
        self.config = config or RiskConfig()
        self.logger = logger

    def calculate_var(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        method: VaRMethod = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk.

        Args:
            returns: Historical returns data
            method: VaR calculation method
            **kwargs: Additional method-specific parameters

        Returns:
            Dictionary with VaR results
        """
        method = method or self.config.method

        if isinstance(returns, pd.DataFrame):
            # Portfolio VaR
            return self._calculate_portfolio_var(returns, method, **kwargs)
        else:
            # Single asset VaR
            return self._calculate_single_asset_var(returns, method, **kwargs)

    def _calculate_single_asset_var(
        self,
        returns: Union[np.ndarray, pd.Series],
        method: VaRMethod,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR for single asset."""
        returns_array = np.asarray(returns)

        if method == VaRMethod.HISTORICAL:
            return self._historical_var(returns_array, **kwargs)
        elif method == VaRMethod.PARAMETRIC:
            return self._parametric_var(returns_array, **kwargs)
        elif method == VaRMethod.MONTE_CARLO:
            return self._monte_carlo_var(returns_array, **kwargs)
        elif method == VaRMethod.EVT:
            return self._evt_var(returns_array, **kwargs)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    def _calculate_portfolio_var(
        self,
        returns: pd.DataFrame,
        method: VaRMethod,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR for portfolio."""
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(returns, **kwargs)

        # Calculate VaR on portfolio returns
        portfolio_var = self._calculate_single_asset_var(portfolio_returns, method, **kwargs)

        # Add portfolio-specific information
        portfolio_var.update({
            "portfolio_weights": kwargs.get("weights", self._equal_weights(returns.shape[1])),
            "assets": list(returns.columns),
            "correlation_matrix": returns.corr().values
        })

        return portfolio_var

    def _historical_var(
        self,
        returns: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR using historical simulation."""
        confidence_level = kwargs.get('confidence_level', self.config.confidence_level)

        # Sort returns in ascending order (worst to best)
        sorted_returns = np.sort(returns)

        # Find the quantile
        quantile_index = int((1 - confidence_level) * len(sorted_returns))
        var_value = -sorted_returns[quantile_index]  # Make VaR positive

        # Scale for horizon if requested
        if self.config.use_volatility_scaling and self.config.horizon_days > 1:
            var_value *= np.sqrt(self.config.horizon_days)

        result = {
            "var": var_value,
            "method": "historical",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "var_amount": var_value * self.config.portfolio_value,
            "expected_shortfall": self._calculate_es_from_historical(sorted_returns, confidence_level)
        }

        self.logger.info("Historical VaR calculated",
                        var=var_value,
                        confidence_level=confidence_level)

        return result

    def _parametric_var(
        self,
        returns: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR using parametric (normal distribution) assumption."""
        confidence_level = kwargs.get('confidence_level', self.config.confidence_level)

        # Calculate mean and volatility
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var_value = -(mu + z_score * sigma)

        # Scale for horizon if requested
        if self.config.use_volatility_scaling and self.config.horizon_days > 1:
            var_value *= np.sqrt(self.config.horizon_days)

        result = {
            "var": var_value,
            "method": "parametric",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "var_amount": var_value * self.config.portfolio_value,
            "mean_return": mu,
            "volatility": sigma,
            "expected_shortfall": self._calculate_es_parametric(mu, sigma, confidence_level)
        }

        return result

    def _monte_carlo_var(
        self,
        returns: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR using Monte Carlo simulation."""
        confidence_level = kwargs.get('confidence_level', self.config.confidence_level)
        n_simulations = kwargs.get('n_simulations', 10000)

        # Fit distribution to historical returns
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Generate simulated returns
        simulated_returns = np.random.normal(mu, sigma, n_simulations)

        # Scale for horizon
        if self.config.horizon_days > 1:
            simulated_returns *= np.sqrt(self.config.horizon_days)

        # Calculate VaR from simulated distribution
        sorted_simulated = np.sort(simulated_returns)
        quantile_index = int((1 - confidence_level) * n_simulations)
        var_value = -sorted_simulated[quantile_index]

        result = {
            "var": var_value,
            "method": "monte_carlo",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "var_amount": var_value * self.config.portfolio_value,
            "n_simulations": n_simulations,
            "expected_shortfall": self._calculate_es_from_historical(sorted_simulated, confidence_level)
        }

        return result

    def _evt_var(
        self,
        returns: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR using Extreme Value Theory."""
        confidence_level = kwargs.get('confidence_level', self.config.confidence_level)

        # Identify tail losses (negative returns)
        losses = -returns[returns < 0]

        if len(losses) < 10:
            # Fallback to historical if insufficient tail data
            return self._historical_var(returns, **kwargs)

        # Fit Generalized Pareto Distribution to tail
        sorted_losses = np.sort(losses)
        threshold_index = int(0.9 * len(sorted_losses))  # Use top 10% as tail
        threshold = sorted_losses[threshold_index]
        tail_data = sorted_losses[threshold_index:]

        # Fit GPD parameters
        try:
            from scipy.stats import genpareto
            params = genpareto.fit(tail_data)
            xi, mu, sigma = params

            # Calculate VaR using GPD
            p_tail = len(tail_data) / len(losses)
            p_var = (1 - confidence_level) / p_tail

            if xi != 0:
                var_value = threshold + (sigma / xi) * (p_var**xi - 1)
            else:
                var_value = threshold - sigma * np.log(p_var)

        except Exception as e:
            self.logger.warning(f"EVT fitting failed, falling back to historical: {e}")
            return self._historical_var(returns, **kwargs)

        result = {
            "var": var_value,
            "method": "evt",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "var_amount": var_value * self.config.portfolio_value,
            "threshold": threshold,
            "tail_size": len(tail_data),
            "gpd_params": {"xi": xi, "mu": mu, "sigma": sigma}
        }

        return result

    def _calculate_portfolio_returns(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """Calculate portfolio returns from individual asset returns."""
        weights = kwargs.get('weights', self._equal_weights(returns.shape[1]))

        if len(weights) != returns.shape[1]:
            raise ValueError("Weights length must match number of assets")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)

        return portfolio_returns.values

    def _equal_weights(self, n_assets: int) -> np.ndarray:
        """Generate equal weights for portfolio."""
        return np.ones(n_assets) / n_assets

    def _calculate_es_from_historical(
        self,
        sorted_returns: np.ndarray,
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall from historical data."""
        quantile_index = int((1 - confidence_level) * len(sorted_returns))
        tail_losses = sorted_returns[:quantile_index]
        return -np.mean(tail_losses)

    def _calculate_es_parametric(
        self,
        mu: float,
        sigma: float,
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall using parametric assumption."""
        z_alpha = stats.norm.ppf(1 - confidence_level)
        z_es = stats.norm.pdf(z_alpha) / (1 - confidence_level)
        es = mu + sigma * z_es
        return -es  # Make positive


class ESCalculator:
    """Expected Shortfall (ES) calculator."""

    def __init__(self, config: RiskConfig = None):
        """
        Initialize ES calculator.

        Args:
            config: Risk calculation configuration
        """
        self.config = config or RiskConfig()
        self.var_calculator = VaRCalculator(config)
        self.logger = logger

    def calculate_es(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        method: ESDistribution = ESDistribution.HISTORICAL,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate Expected Shortfall.

        Args:
            returns: Historical returns data
            method: ES calculation method
            **kwargs: Additional parameters

        Returns:
            Dictionary with ES results
        """
        confidence_level = kwargs.get('confidence_level', self.config.confidence_level)

        if method == ESDistribution.HISTORICAL:
            return self._historical_es(returns, confidence_level, **kwargs)
        elif method == ESDistribution.NORMAL:
            return self._parametric_es(returns, confidence_level, **kwargs)
        elif method == ESDistribution.STUDENT_T:
            return self._student_t_es(returns, confidence_level, **kwargs)
        else:
            raise ValueError(f"Unsupported ES method: {method}")

    def _historical_es(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        confidence_level: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate ES using historical simulation."""
        if isinstance(returns, pd.DataFrame):
            # Portfolio ES
            portfolio_returns = self.var_calculator._calculate_portfolio_returns(returns, **kwargs)
            returns_array = portfolio_returns
        else:
            returns_array = np.asarray(returns)

        # Sort returns in ascending order
        sorted_returns = np.sort(returns_array)

        # Find tail losses beyond VaR
        quantile_index = int((1 - confidence_level) * len(sorted_returns))
        tail_losses = sorted_returns[:quantile_index]

        es_value = -np.mean(tail_losses)

        # Scale for horizon if requested
        if self.config.use_volatility_scaling and self.config.horizon_days > 1:
            es_value *= np.sqrt(self.config.horizon_days)

        result = {
            "es": es_value,
            "method": "historical",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "es_amount": es_value * self.config.portfolio_value,
            "tail_size": len(tail_losses),
            "var": -sorted_returns[quantile_index]  # Corresponding VaR
        }

        return result

    def _parametric_es(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        confidence_level: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate ES assuming normal distribution."""
        if isinstance(returns, pd.DataFrame):
            # Portfolio ES
            portfolio_returns = self.var_calculator._calculate_portfolio_returns(returns, **kwargs)
            returns_array = portfolio_returns
        else:
            returns_array = np.asarray(returns)

        mu = np.mean(returns_array)
        sigma = np.std(returns_array, ddof=1)

        # ES formula for normal distribution
        z_alpha = stats.norm.ppf(1 - confidence_level)
        es_z = stats.norm.pdf(z_alpha) / (1 - confidence_level)
        es_value = -(mu + sigma * es_z)

        # Scale for horizon
        if self.config.use_volatility_scaling and self.config.horizon_days > 1:
            es_value *= np.sqrt(self.config.horizon_days)

        result = {
            "es": es_value,
            "method": "parametric",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "es_amount": es_value * self.config.portfolio_value,
            "mean_return": mu,
            "volatility": sigma,
            "var": -(mu + z_alpha * sigma)  # Corresponding VaR
        }

        return result

    def _student_t_es(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        confidence_level: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate ES assuming Student-t distribution."""
        if isinstance(returns, pd.DataFrame):
            # Portfolio ES
            portfolio_returns = self.var_calculator._calculate_portfolio_returns(returns, **kwargs)
            returns_array = portfolio_returns
        else:
            returns_array = np.asarray(returns)

        # Fit Student-t distribution
        params = stats.t.fit(returns_array)
        df, loc, scale = params

        # Calculate ES for Student-t
        z_alpha = stats.t.ppf(1 - confidence_level, df)
        es_z = (stats.t.pdf(z_alpha, df) / (1 - confidence_level)) * \
               ((df + z_alpha**2) / (df - 1))
        es_value = -(loc + scale * es_z)

        # Scale for horizon
        if self.config.use_volatility_scaling and self.config.horizon_days > 1:
            es_value *= np.sqrt(self.config.horizon_days)

        result = {
            "es": es_value,
            "method": "student_t",
            "confidence_level": confidence_level,
            "horizon_days": self.config.horizon_days,
            "portfolio_value": self.config.portfolio_value,
            "es_amount": es_value * self.config.portfolio_value,
            "degrees_of_freedom": df,
            "location": loc,
            "scale": scale,
            "var": -(loc + scale * z_alpha)  # Corresponding VaR
        }

        return result

    def calculate_es_spectrum(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        confidence_levels: List[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate ES across multiple confidence levels.

        Args:
            returns: Historical returns data
            confidence_levels: List of confidence levels
            **kwargs: Additional parameters

        Returns:
            Dictionary with ES spectrum results
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]

        es_values = []
        var_values = []

        for conf_level in confidence_levels:
            kwargs['confidence_level'] = conf_level
            es_result = self.calculate_es(returns, **kwargs)

            es_values.append(es_result['es'])
            var_values.append(es_result.get('var', 0))

        return {
            "confidence_levels": confidence_levels,
            "es_values": es_values,
            "var_values": var_values,
            "es_spectrum": dict(zip(confidence_levels, es_values)),
            "var_spectrum": dict(zip(confidence_levels, var_values))
        }
