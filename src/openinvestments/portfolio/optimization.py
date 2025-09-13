"""
Advanced portfolio optimization algorithms.

Implements sophisticated optimization methods including:
- Black-Litterman model
- Conditional Value at Risk (CVaR) optimization
- Risk parity portfolios
- Maximum diversification portfolios
- Minimum variance portfolios
- Mean-CVaR optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.stats import norm
import cvxpy as cp

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimization_status: str
    convergence_info: Dict[str, Any]
    constraints_satisfied: bool


@dataclass
class BlackLittermanInputs:
    """Inputs for Black-Litterman model."""
    market_cap_weights: np.ndarray
    market_risk_premium: float
    risk_aversion: float
    investor_views: Dict[int, float]  # Asset index -> expected return view
    view_confidences: Dict[int, float]  # Asset index -> confidence level
    tau: float = 0.025  # Uncertainty in prior


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""

    @abstractmethod
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> OptimizationResult:
        """Optimize portfolio weights."""
        pass


class MeanVarianceOptimizer(PortfolioOptimizer):
    """Classical mean-variance portfolio optimization."""

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize mean-variance portfolio.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            target_return: Target portfolio return (optional)
            risk_free_rate: Risk-free rate for Sharpe ratio

        Returns:
            Optimization result
        """
        n_assets = len(expected_returns)

        # Objective function (minimize portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        if target_return is not None:
            # Target return constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })

        # Bounds (no short selling)
        bounds = Bounds(0, 1, keep_feasible=True)

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                optimization_status="success" if result.success else "failed",
                convergence_info={
                    "success": result.success,
                    "message": result.message,
                    "iterations": result.nit,
                    "function_evaluations": result.nfev
                },
                constraints_satisfied=self._check_constraints(weights, expected_returns, target_return)
            )

        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=np.mean(expected_returns),
                volatility=np.sqrt(np.mean(np.diag(covariance_matrix))),
                sharpe_ratio=0,
                optimization_status="error",
                convergence_info={"error": str(e)},
                constraints_satisfied=False
            )

    def _check_constraints(self, weights: np.ndarray, expected_returns: np.ndarray,
                          target_return: Optional[float]) -> bool:
        """Check if optimization constraints are satisfied."""
        # Check weight sum
        weight_sum_ok = abs(np.sum(weights) - 1) < 1e-6

        # Check target return if specified
        return_ok = True
        if target_return is not None:
            actual_return = np.dot(weights, expected_returns)
            return_ok = abs(actual_return - target_return) < 1e-6

        # Check bounds
        bounds_ok = np.all(weights >= -1e-6) and np.all(weights <= 1 + 1e-6)

        return weight_sum_ok and return_ok and bounds_ok


class CVaROptimizer(PortfolioOptimizer):
    """Conditional Value at Risk (CVaR) portfolio optimization."""

    def __init__(self, confidence_level: float = 0.95, num_simulations: int = 10000):
        """
        Initialize CVaR optimizer.

        Args:
            confidence_level: Confidence level for VaR/CVaR calculation
            num_simulations: Number of Monte Carlo simulations
        """
        self.confidence_level = confidence_level
        self.num_simulations = num_simulations

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize portfolio for minimum CVaR.

        Uses Monte Carlo simulation to estimate CVaR and optimize portfolio weights.
        """
        n_assets = len(expected_returns)

        # Generate Monte Carlo scenarios
        np.random.seed(42)
        scenarios = np.random.multivariate_normal(expected_returns, covariance_matrix, self.num_simulations)

        def objective(weights):
            """Minimize CVaR of portfolio returns."""
            portfolio_returns = scenarios.dot(weights)

            # Calculate VaR
            sorted_returns = np.sort(portfolio_returns)
            var_index = int((1 - self.confidence_level) * len(sorted_returns))
            var = sorted_returns[var_index]

            # Calculate CVaR (Expected Shortfall)
            tail_losses = sorted_returns[:var_index]
            cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else 0

            return cvar

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: np.dot(w, expected_returns) - 0.02}  # Minimum return constraint
        ]

        # Bounds
        bounds = Bounds(0, 1, keep_feasible=True)

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            portfolio_cvar = objective(weights)

            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
                optimization_status="success" if result.success else "failed",
                convergence_info={
                    "success": result.success,
                    "message": result.message,
                    "cvar_value": portfolio_cvar,
                    "simulations": self.num_simulations
                },
                constraints_satisfied=self._check_constraints(weights, expected_returns)
            )

        except Exception as e:
            logger.error(f"CVaR optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=np.mean(expected_returns),
                volatility=np.sqrt(np.mean(np.diag(covariance_matrix))),
                sharpe_ratio=0,
                optimization_status="error",
                convergence_info={"error": str(e)},
                constraints_satisfied=False
            )

    def _check_constraints(self, weights: np.ndarray, expected_returns: np.ndarray) -> bool:
        """Check CVaR optimization constraints."""
        weight_sum_ok = abs(np.sum(weights) - 1) < 1e-6
        return_ok = np.dot(weights, expected_returns) >= 0.02
        bounds_ok = np.all(weights >= -1e-6) and np.all(weights <= 1 + 1e-6)

        return weight_sum_ok and return_ok and bounds_ok


class BlackLittermanOptimizer(PortfolioOptimizer):
    """Black-Litterman portfolio optimization model."""

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        bl_inputs: BlackLittermanInputs = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize portfolio using Black-Litterman model.

        Combines market equilibrium returns with investor views.
        """
        if bl_inputs is None:
            # Fall back to mean-variance optimization
            optimizer = MeanVarianceOptimizer()
            return optimizer.optimize(expected_returns, covariance_matrix, **kwargs)

        n_assets = len(expected_returns)

        # Extract Black-Litterman parameters
        pi = bl_inputs.market_cap_weights  # Market capitalization weights
        lambda_param = bl_inputs.risk_aversion
        tau = bl_inputs.tau

        # Calculate market equilibrium returns
        market_returns = lambda_param * covariance_matrix.dot(pi)

        # Process investor views
        if bl_inputs.investor_views:
            P = np.zeros((len(bl_inputs.investor_views), n_assets))
            Q = np.zeros(len(bl_inputs.investor_views))

            for i, (asset_idx, view_return) in enumerate(bl_inputs.investor_views.items()):
                P[i, asset_idx] = 1.0
                Q[i] = view_return

            # View confidence matrix
            omega = np.zeros((len(bl_inputs.investor_views), len(bl_inputs.investor_views)))
            for i, (asset_idx, confidence) in enumerate(bl_inputs.view_confidences.items()):
                omega[i, i] = (1 / confidence - 1) * np.dot(np.dot(P[i:i+1], covariance_matrix), P[i:i+1].T)[0, 0]

            # Black-Litterman formula
            tau_sigma = tau * covariance_matrix

            # Posterior expected returns
            temp1 = np.linalg.inv(tau_sigma)
            temp2 = np.dot(np.dot(P.T, np.linalg.inv(omega)), P)
            temp3 = np.linalg.inv(temp1 + temp2)

            temp4 = np.dot(temp1, market_returns) + np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)
            posterior_returns = np.dot(temp3, temp4)

            # Posterior covariance
            posterior_cov = covariance_matrix + temp3

        else:
            # No views, use market equilibrium
            posterior_returns = market_returns
            posterior_cov = covariance_matrix

        # Optimize with posterior estimates
        optimizer = MeanVarianceOptimizer()
        return optimizer.optimize(posterior_returns, posterior_cov, **kwargs)


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity portfolio optimization."""

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize risk parity portfolio.

        Equalizes risk contribution of each asset to total portfolio risk.
        """
        n_assets = len(expected_returns)

        def objective(weights):
            """Minimize the variance of risk contributions."""
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

            # Marginal risk contributions
            marginal_risk = np.dot(covariance_matrix, weights) / portfolio_volatility

            # Total risk contributions
            risk_contributions = weights * marginal_risk

            # Risk contribution volatility (want this to be zero)
            risk_volatility = np.var(risk_contributions)

            return risk_volatility

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = Bounds(0.01, 1, keep_feasible=True)  # Minimum 1% allocation

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                optimization_status="success" if result.success else "failed",
                convergence_info={
                    "success": result.success,
                    "message": result.message,
                    "objective_value": result.fun
                },
                constraints_satisfied=self._check_constraints(weights)
            )

        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=np.mean(expected_returns),
                volatility=np.sqrt(np.mean(np.diag(covariance_matrix))),
                sharpe_ratio=0,
                optimization_status="error",
                convergence_info={"error": str(e)},
                constraints_satisfied=False
            )

    def _check_constraints(self, weights: np.ndarray) -> bool:
        """Check risk parity constraints."""
        weight_sum_ok = abs(np.sum(weights) - 1) < 1e-6
        bounds_ok = np.all(weights >= 0.01) and np.all(weights <= 1)

        return weight_sum_ok and bounds_ok


class MaximumDiversificationOptimizer(PortfolioOptimizer):
    """Maximum diversification portfolio optimization."""

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize maximum diversification portfolio.

        Maximizes the diversification ratio: weighted average volatility / portfolio volatility.
        """
        n_assets = len(expected_returns)

        def objective(weights):
            """Maximize diversification ratio (negative for minimization)."""
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

            # Weighted average volatility
            individual_volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_avg_volatility = np.dot(weights, individual_volatilities)

            # Diversification ratio
            diversification_ratio = weighted_avg_volatility / portfolio_volatility

            # Return negative for maximization
            return -diversification_ratio

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = Bounds(0, 1, keep_feasible=True)

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

            # Calculate actual diversification ratio
            individual_volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_avg_volatility = np.dot(weights, individual_volatilities)
            diversification_ratio = weighted_avg_volatility / portfolio_volatility

            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                optimization_status="success" if result.success else "failed",
                convergence_info={
                    "success": result.success,
                    "message": result.message,
                    "diversification_ratio": diversification_ratio,
                    "objective_value": result.fun
                },
                constraints_satisfied=self._check_constraints(weights)
            )

        except Exception as e:
            logger.error(f"Maximum diversification optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=np.mean(expected_returns),
                volatility=np.sqrt(np.mean(np.diag(covariance_matrix))),
                sharpe_ratio=0,
                optimization_status="error",
                convergence_info={"error": str(e)},
                constraints_satisfied=False
            )

    def _check_constraints(self, weights: np.ndarray) -> bool:
        """Check maximum diversification constraints."""
        weight_sum_ok = abs(np.sum(weights) - 1) < 1e-6
        bounds_ok = np.all(weights >= -1e-6) and np.all(weights <= 1 + 1e-6)

        return weight_sum_ok and bounds_ok


class PortfolioOptimizationManager:
    """
    Manager class for portfolio optimization with multiple strategies.
    """

    def __init__(self):
        self.optimizers = {
            'mean_variance': MeanVarianceOptimizer(),
            'cvar': CVaROptimizer(),
            'black_litterman': BlackLittermanOptimizer(),
            'risk_parity': RiskParityOptimizer(),
            'max_diversification': MaximumDiversificationOptimizer()
        }
        self.logger = logger

    def optimize_portfolio(
        self,
        strategy: str,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified strategy.

        Args:
            strategy: Optimization strategy name
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            **kwargs: Strategy-specific parameters

        Returns:
            Optimization result
        """
        if strategy not in self.optimizers:
            available_strategies = list(self.optimizers.keys())
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {available_strategies}")

        optimizer = self.optimizers[strategy]

        self.logger.info(f"Running {strategy} portfolio optimization")
        result = optimizer.optimize(expected_returns, covariance_matrix, **kwargs)

        self.logger.info(f"{strategy} optimization completed: {result.optimization_status}")
        return result

    def compare_strategies(
        self,
        strategies: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> Dict[str, OptimizationResult]:
        """
        Compare multiple optimization strategies.

        Args:
            strategies: List of strategy names to compare
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            **kwargs: Common parameters for all strategies

        Returns:
            Dictionary of optimization results by strategy
        """
        results = {}

        for strategy in strategies:
            try:
                result = self.optimize_portfolio(strategy, expected_returns, covariance_matrix, **kwargs)
                results[strategy] = result
            except Exception as e:
                self.logger.error(f"Failed to optimize with {strategy}: {e}")
                results[strategy] = None

        return results

    def get_strategy_info(self) -> Dict[str, str]:
        """Get information about available optimization strategies."""
        return {
            'mean_variance': 'Classical Markowitz mean-variance optimization',
            'cvar': 'Conditional Value at Risk optimization for tail risk control',
            'black_litterman': 'Combines market equilibrium with investor views',
            'risk_parity': 'Equalizes risk contribution across assets',
            'max_diversification': 'Maximizes diversification ratio'
        }


# Global optimization manager instance
optimization_manager = PortfolioOptimizationManager()


def get_optimization_manager() -> PortfolioOptimizationManager:
    """Get the global portfolio optimization manager instance."""
    return optimization_manager
