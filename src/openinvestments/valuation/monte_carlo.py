"""
Monte Carlo pricing engine for financial instruments.

Supports European and American options, exotic options, and complex derivatives.
"""

import numpy as np
from typing import Callable, Optional, Union, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    paths: int = config.default_mc_paths
    steps: int = config.default_mc_steps
    timestep: float = config.default_mc_timestep
    random_seed: Optional[int] = None
    use_antithetic: bool = True
    use_control_variates: bool = False
    parallel: bool = True


class StochasticProcess(ABC):
    """Abstract base class for stochastic processes."""

    @abstractmethod
    def simulate(self, S0: float, T: float, paths: int, steps: int,
                 random_seed: Optional[int] = None) -> np.ndarray:
        """Simulate paths of the stochastic process."""
        pass


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion process."""

    def __init__(self, mu: float, sigma: float):
        """
        Initialize GBM process.

        Args:
            mu: Drift parameter
            sigma: Volatility parameter
        """
        self.mu = mu
        self.sigma = sigma

    def simulate(self, S0: float, T: float, paths: int, steps: int,
                 random_seed: Optional[int] = None) -> np.ndarray:
        """Simulate GBM paths."""
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = T / steps
        Z = np.random.normal(0, 1, (paths, steps))

        # Pre-compute drift and diffusion terms
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Initialize price paths
        S = np.zeros((paths, steps + 1))
        S[:, 0] = S0

        # Simulate paths
        for t in range(1, steps + 1):
            S[:, t] = S[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])

        return S


class HestonProcess(StochasticProcess):
    """Heston stochastic volatility process."""

    def __init__(self, mu: float, kappa: float, theta: float, sigma: float, rho: float):
        """
        Initialize Heston process.

        Args:
            mu: Drift of asset
            kappa: Mean reversion speed of variance
            theta: Long-term variance
            sigma: Volatility of variance
            rho: Correlation between asset and variance
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def simulate(self, S0: float, V0: float, T: float, paths: int, steps: int,
                 random_seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Heston paths. Returns (asset_prices, variances)."""
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = T / steps

        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, (paths, steps))
        Z2 = np.random.normal(0, 1, (paths, steps))
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        # Initialize arrays
        S = np.zeros((paths, steps + 1))
        V = np.zeros((paths, steps + 1))
        S[:, 0] = S0
        V[:, 0] = V0

        for t in range(1, steps + 1):
            # Variance process (CIR)
            V[:, t] = V[:, t-1] + self.kappa * (self.theta - V[:, t-1]) * dt + \
                     self.sigma * np.sqrt(np.maximum(V[:, t-1], 0)) * np.sqrt(dt) * Z2[:, t-1]
            V[:, t] = np.maximum(V[:, t], 0)  # Ensure positive variance

            # Asset process
            S[:, t] = S[:, t-1] * np.exp((self.mu - 0.5 * V[:, t]) * dt + \
                                        np.sqrt(V[:, t]) * np.sqrt(dt) * Z1[:, t-1])

        return S, V


class Payoff(ABC):
    """Abstract base class for option payoffs."""

    @abstractmethod
    def calculate(self, paths: np.ndarray) -> np.ndarray:
        """Calculate payoff for given price paths."""
        pass


class EuropeanCallPayoff(Payoff):
    """European call option payoff."""

    def __init__(self, strike: float):
        self.strike = strike

    def calculate(self, paths: np.ndarray) -> np.ndarray:
        """Calculate European call payoff."""
        return np.maximum(paths[:, -1] - self.strike, 0)


class EuropeanPutPayoff(Payoff):
    """European put option payoff."""

    def __init__(self, strike: float):
        self.strike = strike

    def calculate(self, paths: np.ndarray) -> np.ndarray:
        """Calculate European put payoff."""
        return np.maximum(self.strike - paths[:, -1], 0)


class AsianCallPayoff(Payoff):
    """Asian call option payoff (average price)."""

    def __init__(self, strike: float):
        self.strike = strike

    def calculate(self, paths: np.ndarray) -> np.ndarray:
        """Calculate Asian call payoff."""
        avg_price = np.mean(paths, axis=1)
        return np.maximum(avg_price - self.strike, 0)


class BarrierCallPayoff(Payoff):
    """Barrier call option payoff."""

    def __init__(self, strike: float, barrier: float, barrier_type: str = "up-and-out"):
        """
        Initialize barrier option.

        Args:
            strike: Strike price
            barrier: Barrier level
            barrier_type: Type of barrier ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
        """
        self.strike = strike
        self.barrier = barrier
        self.barrier_type = barrier_type

    def calculate(self, paths: np.ndarray) -> np.ndarray:
        """Calculate barrier call payoff."""
        final_price = paths[:, -1]
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)

        # Determine if barrier is breached
        if "up" in self.barrier_type:
            breached = max_price >= self.barrier
        else:
            breached = min_price <= self.barrier

        # Calculate payoff based on barrier type
        if "out" in self.barrier_type:
            payoff = np.where(breached, 0, np.maximum(final_price - self.strike, 0))
        else:  # "in" barrier
            payoff = np.where(breached, np.maximum(final_price - self.strike, 0), 0)

        return payoff


class MonteCarloPricer:
    """Monte Carlo pricing engine."""

    def __init__(self, config: MonteCarloConfig = None):
        """
        Initialize Monte Carlo pricer.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config or MonteCarloConfig()
        self.logger = logger

    def price_option(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        payoff: Payoff,
        process: StochasticProcess = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Price an option using Monte Carlo simulation.

        Args:
            S0: Initial asset price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            payoff: Option payoff function
            process: Stochastic process (defaults to GBM)
            **kwargs: Additional parameters

        Returns:
            Dictionary with price, standard error, and confidence intervals
        """
        if process is None:
            process = GeometricBrownianMotion(r, sigma)

        # Simulate paths
        paths = process.simulate(S0, T, self.config.paths, self.config.steps,
                               self.config.random_seed)

        # Calculate payoffs
        payoffs = payoff.calculate(paths)

        # Apply discounting
        discounted_payoffs = payoffs * np.exp(-r * T)

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.config.paths)

        # Confidence intervals (95%)
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        result = {
            "price": price,
            "standard_error": std_error,
            "confidence_interval": (ci_lower, ci_upper),
            "paths_simulated": self.config.paths,
            "simulation_time": T,
            "discount_rate": r
        }

        self.logger.info("Monte Carlo pricing completed",
                        price=price,
                        std_error=std_error,
                        paths=self.config.paths)

        return result

    def price_portfolio(
        self,
        instruments: list,
        correlation_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Price a portfolio of instruments using Monte Carlo with correlations.
        Args:
            instruments: List of instrument specifications, each with 'S0', 'mu', 'sigma', 'notional', 'payoff'
            correlation_matrix: Correlation matrix for multi-asset pricing
            **kwargs: Additional parameters
        Returns:
            Portfolio pricing results
        """
        n_instruments = len(instruments)
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_instruments)

        # Extract parameters
        S0s = np.array([inst.get('S0', 100.0) for inst in instruments])
        mus = np.array([inst.get('mu', 0.05) for inst in instruments])
        sigmas = np.array([inst.get('sigma', 0.2) for inst in instruments])
        notionals = np.array([inst.get('notional', 1.0) for inst in instruments])
        payoffs = [inst.get('payoff', EuropeanCallPayoff(100.0)) for inst in instruments]

        T = kwargs.get('T', 1.0)
        r = kwargs.get('r', 0.05)
        paths = self.config.paths
        steps = self.config.steps

        # Generate correlated random shocks
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)
        Z = np.random.multivariate_normal(np.zeros(n_instruments), correlation_matrix, (paths, steps))

        # Simulate paths for each instrument
        portfolio_payoffs = np.zeros(paths)
        for i in range(n_instruments):
            # GBM simulation for instrument i
            dt = T / steps
            drift = (mus[i] - 0.5 * sigmas[i]**2) * dt
            diffusion = sigmas[i] * np.sqrt(dt)
            log_S = np.cumsum(drift + diffusion * Z[:, :, i], axis=1)
            S_paths = S0s[i] * np.exp(log_S)
            S_terminal = S_paths[:, -1]

            # Calculate payoff
            inst_payoff = payoffs[i].calculate(S_terminal)
            portfolio_payoffs += notionals[i] * inst_payoff

        # Discount and average
        discounted_payoffs = portfolio_payoffs * np.exp(-r * T)
        portfolio_value = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(paths)

        # Diversification ratio (simplified)
        diversification_ratio = 1.0 / np.sqrt(np.trace(correlation_matrix) / n_instruments)

        return {
            "portfolio_value": portfolio_value,
            "standard_error": std_error,
            "instruments_count": n_instruments,
            "diversification_ratio": diversification_ratio,
            "paths_simulated": paths,
            "correlation_matrix": correlation_matrix.tolist()
        }
