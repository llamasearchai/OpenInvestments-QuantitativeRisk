"""
Binomial and trinomial tree pricing models for options.

Supports American and European options with various payoff structures.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


@dataclass
class TreeConfig:
    """Configuration for tree-based pricing."""
    steps: int = 100
    american_exercise: bool = True
    early_exercise_boundary: bool = False
    convergence_check: bool = True
    max_iterations: int = 1000


class PayoffFunction(ABC):
    """Abstract base class for payoff functions."""

    @abstractmethod
    def calculate(self, S: float, K: float) -> float:
        """Calculate payoff for given spot and strike."""
        pass


class CallPayoff(PayoffFunction):
    """European/American call option payoff."""

    def calculate(self, S: float, K: float) -> float:
        return max(S - K, 0)


class PutPayoff(PayoffFunction):
    """European/American put option payoff."""

    def calculate(self, S: float, K: float) -> float:
        return max(K - S, 0)


class CashOrNothingPayoff(PayoffFunction):
    """Cash-or-nothing option payoff."""

    def __init__(self, cash_amount: float, is_call: bool = True):
        self.cash_amount = cash_amount
        self.is_call = is_call

    def calculate(self, S: float, K: float) -> float:
        if self.is_call:
            return self.cash_amount if S >= K else 0
        else:
            return self.cash_amount if S <= K else 0


class AssetOrNothingPayoff(PayoffFunction):
    """Asset-or-nothing option payoff."""

    def __init__(self, is_call: bool = True):
        self.is_call = is_call

    def calculate(self, S: float, K: float) -> float:
        if self.is_call:
            return S if S >= K else 0
        else:
            return S if S <= K else 0


class BinomialTree:
    """Binomial tree option pricing model."""

    def __init__(self, config: TreeConfig = None):
        """
        Initialize binomial tree model.

        Args:
            config: Tree configuration
        """
        self.config = config or TreeConfig()
        self.logger = logger

    def price_european(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        payoff_func: PayoffFunction,
        dividend_yield: float = 0.0
    ) -> Dict[str, Any]:
        """
        Price European option using binomial tree.

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            payoff_func: Payoff function
            dividend_yield: Continuous dividend yield

        Returns:
            Pricing results dictionary
        """
        dt = T / self.config.steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = np.exp(-sigma * np.sqrt(dt))  # Down factor
        p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability

        # Build stock price tree
        stock_tree = self._build_stock_tree(S0, u, d, self.config.steps)

        # Build option value tree (European - backward induction)
        option_tree = self._build_european_option_tree(
            stock_tree, K, p, dt, r, payoff_func
        )

        result = {
            "price": option_tree[0, 0],
            "stock_tree": stock_tree,
            "option_tree": option_tree,
            "parameters": {
                "steps": self.config.steps,
                "up_factor": u,
                "down_factor": d,
                "risk_neutral_prob": p,
                "delta_t": dt
            }
        }

        self.logger.info("European binomial tree pricing completed",
                        price=result["price"],
                        steps=self.config.steps)

        return result

    def price_american(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        payoff_func: PayoffFunction,
        dividend_yield: float = 0.0
    ) -> Dict[str, Any]:
        """
        Price American option using binomial tree.

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            payoff_func: Payoff function
            dividend_yield: Continuous dividend yield

        Returns:
            Pricing results dictionary
        """
        dt = T / self.config.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)

        # Build stock price tree
        stock_tree = self._build_stock_tree(S0, u, d, self.config.steps)

        # Build option value tree (American - check early exercise)
        option_tree = self._build_american_option_tree(
            stock_tree, K, p, dt, r, payoff_func
        )

        result = {
            "price": option_tree[0, 0],
            "stock_tree": stock_tree,
            "option_tree": option_tree,
            "early_exercise_boundary": self._extract_early_exercise_boundary(
                stock_tree, option_tree
            ) if self.config.early_exercise_boundary else None,
            "parameters": {
                "steps": self.config.steps,
                "up_factor": u,
                "down_factor": d,
                "risk_neutral_prob": p,
                "delta_t": dt
            }
        }

        self.logger.info("American binomial tree pricing completed",
                        price=result["price"],
                        steps=self.config.steps)

        return result

    def _build_stock_tree(self, S0: float, u: float, d: float, steps: int) -> np.ndarray:
        """Build binomial stock price tree."""
        tree = np.zeros((steps + 1, steps + 1))

        for i in range(steps + 1):
            for j in range(i + 1):
                tree[j, i] = S0 * (u ** j) * (d ** (i - j))

        return tree

    def _build_european_option_tree(
        self,
        stock_tree: np.ndarray,
        K: float,
        p: float,
        dt: float,
        r: float,
        payoff_func: PayoffFunction
    ) -> np.ndarray:
        """Build European option value tree using backward induction."""
        steps = stock_tree.shape[1] - 1
        option_tree = np.zeros((steps + 1, steps + 1))

        # Terminal payoffs
        for j in range(steps + 1):
            S = stock_tree[j, steps]
            option_tree[j, steps] = payoff_func.calculate(S, K)

        # Backward induction
        discount = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                continuation = discount * (
                    p * option_tree[j + 1, i + 1] +
                    (1 - p) * option_tree[j, i + 1]
                )
                option_tree[j, i] = continuation

        return option_tree

    def _build_american_option_tree(
        self,
        stock_tree: np.ndarray,
        K: float,
        p: float,
        dt: float,
        r: float,
        payoff_func: PayoffFunction
    ) -> np.ndarray:
        """Build American option value tree using backward induction with early exercise."""
        steps = stock_tree.shape[1] - 1
        option_tree = np.zeros((steps + 1, steps + 1))

        # Terminal payoffs
        for j in range(steps + 1):
            S = stock_tree[j, steps]
            option_tree[j, steps] = payoff_func.calculate(S, K)

        # Backward induction with early exercise check
        discount = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                S = stock_tree[j, i]
                continuation = discount * (
                    p * option_tree[j + 1, i + 1] +
                    (1 - p) * option_tree[j, i + 1]
                )
                exercise = payoff_func.calculate(S, K)

                # American option: take maximum of continuation and exercise
                option_tree[j, i] = max(continuation, exercise)

        return option_tree

    def _extract_early_exercise_boundary(
        self,
        stock_tree: np.ndarray,
        option_tree: np.ndarray
    ) -> np.ndarray:
        """Extract early exercise boundary from American option tree."""
        steps = stock_tree.shape[1] - 1
        boundary = []

        for i in range(steps):
            exercised = False
            for j in range(i + 1):
                S = stock_tree[j, i]
                continuation = np.exp(-0.05 * (steps - i) / steps) * (
                    0.5 * option_tree[j + 1, i + 1] +
                    0.5 * option_tree[j, i + 1]
                )
                exercise = max(S - 100, 0)  # Assuming call option for boundary

                if exercise > continuation:
                    boundary.append(S)
                    exercised = True
                    break

            if not exercised:
                boundary.append(np.nan)

        return np.array(boundary)


class TrinomialTree:
    """Trinomial tree option pricing model."""

    def __init__(self, config: TreeConfig = None):
        """
        Initialize trinomial tree model.

        Args:
            config: Tree configuration
        """
        self.config = config or TreeConfig()
        self.logger = logger

    def price_european(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        payoff_func: PayoffFunction,
        dividend_yield: float = 0.0
    ) -> Dict[str, Any]:
        """
        Price European option using trinomial tree.

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            payoff_func: Payoff function
            dividend_yield: Continuous dividend yield

        Returns:
            Pricing results dictionary
        """
        dt = T / self.config.steps
        dx = sigma * np.sqrt(3 * dt)
        u = np.exp(dx)
        m = 1.0  # Middle factor
        d = np.exp(-dx)

        # Risk-neutral probabilities
        nu = (r - dividend_yield - 0.5 * sigma**2) * dt / dx
        pu = 0.5 * (sigma**2 * dt / dx**2 + nu)
        pm = 1 - sigma**2 * dt / dx**2
        pd = 0.5 * (sigma**2 * dt / dx**2 - nu)

        # Build stock price tree
        stock_tree = self._build_stock_tree(S0, u, m, d, self.config.steps)

        # Build option value tree
        option_tree = self._build_european_option_tree(
            stock_tree, K, pu, pm, pd, dt, r, payoff_func
        )

        result = {
            "price": option_tree[0, self.config.steps],
            "stock_tree": stock_tree,
            "option_tree": option_tree,
            "parameters": {
                "steps": self.config.steps,
                "up_factor": u,
                "middle_factor": m,
                "down_factor": d,
                "prob_up": pu,
                "prob_middle": pm,
                "prob_down": pd,
                "delta_t": dt,
                "delta_x": dx
            }
        }

        self.logger.info("European trinomial tree pricing completed",
                        price=result["price"],
                        steps=self.config.steps)

        return result

    def _build_stock_tree(
        self,
        S0: float,
        u: float,
        m: float,
        d: float,
        steps: int
    ) -> np.ndarray:
        """Build trinomial stock price tree."""
        tree = np.zeros((2 * steps + 1, steps + 1))

        for i in range(steps + 1):
            for j in range(2 * i + 1):
                k = j - i  # From -i to +i
                tree[j, i] = S0 * (u ** max(k, 0)) * (d ** max(-k, 0))

        return tree

    def _build_european_option_tree(
        self,
        stock_tree: np.ndarray,
        K: float,
        pu: float,
        pm: float,
        pd: float,
        dt: float,
        r: float,
        payoff_func: PayoffFunction
    ) -> np.ndarray:
        """Build European option value tree using backward induction."""
        steps = stock_tree.shape[1] - 1
        option_tree = np.zeros((2 * steps + 1, steps + 1))

        # Terminal payoffs
        for j in range(2 * steps + 1):
            S = stock_tree[j, steps]
            option_tree[j, steps] = payoff_func.calculate(S, K)

        # Backward induction
        discount = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            for j in range(2 * i + 1):
                if j > 0 and j < 2 * i:
                    continuation = discount * (
                        pu * option_tree[j + 1, i + 1] +
                        pm * option_tree[j, i + 1] +
                        pd * option_tree[j - 1, i + 1]
                    )
                elif j == 0:
                    continuation = discount * (
                        pu * option_tree[j + 1, i + 1] +
                        pm * option_tree[j, i + 1] +
                        pd * option_tree[j, i + 1]  # Reflect at boundary
                    )
                else:  # j == 2*i
                    continuation = discount * (
                        pu * option_tree[j, i + 1] +  # Reflect at boundary
                        pm * option_tree[j, i + 1] +
                        pd * option_tree[j - 1, i + 1]
                    )

                option_tree[j, i] = continuation

        return option_tree
