"""
FastAPI routes for portfolio analysis endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class PortfolioWeights(BaseModel):
    """Portfolio weights specification."""
    weights: List[float] = Field(..., description="Portfolio weights for each asset")
    asset_names: Optional[List[str]] = Field(None, description="Asset names")


class PortfolioAnalysisRequest(BaseModel):
    """Request for portfolio analysis."""
    returns_data: List[List[float]] = Field(..., description="Historical returns matrix")
    weights: Optional[List[float]] = Field(None, description="Portfolio weights")
    asset_names: Optional[List[str]] = Field(None, description="Asset names")
    risk_free_rate: float = Field(0.02, description="Risk-free rate for Sharpe ratio")


class OptimizationRequest(BaseModel):
    """Request for portfolio optimization."""
    returns_data: List[List[float]] = Field(..., description="Historical returns matrix")
    asset_names: Optional[List[str]] = Field(None, description="Asset names")
    method: str = Field("equal_weight", description="Optimization method")
    target_return: Optional[float] = Field(None, description="Target portfolio return")
    target_volatility: Optional[float] = Field(None, description="Target portfolio volatility")


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    Analyze portfolio performance and risk characteristics.

    Returns comprehensive portfolio statistics including returns, risk metrics,
    and diversification measures.
    """
    try:
        logger.info("Received portfolio analysis request",
                   num_assets=len(request.returns_data),
                   has_weights=request.weights is not None)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        # Use provided weights or equal weights
        if request.weights:
            weights = np.array(request.weights)
            weights = weights / np.sum(weights)  # Normalize
        else:
            weights = np.ones(len(request.returns_data)) / len(request.returns_data)

        # Calculate portfolio returns
        portfolio_returns = returns_df.dot(weights)

        # Calculate basic statistics
        mean_return = float(np.mean(portfolio_returns))
        volatility = float(np.std(portfolio_returns))
        skewness = float(pd.Series(portfolio_returns).skew())
        kurtosis = float(pd.Series(portfolio_returns).kurtosis())

        # Sharpe ratio
        sharpe_ratio = (mean_return - request.risk_free_rate / 252) / volatility * np.sqrt(252)

        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        # Value at Risk (simple historical)
        confidence_levels = [0.90, 0.95, 0.99]
        var_values = {}
        for conf in confidence_levels:
            var_values[conf] = float(-np.percentile(portfolio_returns, (1 - conf) * 100))

        # Correlation matrix
        corr_matrix = returns_df.corr()

        # Diversification metrics
        avg_correlation = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean())
        diversification_ratio = float(1 / np.sqrt(np.sum(corr_matrix ** 2).sum() / len(corr_matrix)))

        # Individual asset contributions
        asset_contributions = {}
        for i, asset in enumerate(returns_df.columns):
            asset_contributions[asset] = {
                "weight": float(weights[i]),
                "expected_return": float(np.mean(returns_df[asset])),
                "volatility": float(np.std(returns_df[asset])),
                "sharpe_ratio": float((np.mean(returns_df[asset]) - request.risk_free_rate / 252) /
                                    np.std(returns_df[asset]) * np.sqrt(252))
            }

        result = {
            "portfolio_summary": {
                "num_assets": len(request.returns_data),
                "asset_names": list(returns_df.columns),
                "weights": weights.tolist(),
                "data_points": len(returns_df)
            },
            "performance_metrics": {
                "mean_daily_return": mean_return,
                "annual_volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "maximum_drawdown": max_drawdown,
                "skewness": skewness,
                "kurtosis": kurtosis
            },
            "risk_metrics": {
                "value_at_risk": var_values,
                "average_correlation": avg_correlation,
                "diversification_ratio": diversification_ratio
            },
            "asset_contributions": asset_contributions,
            "correlation_matrix": corr_matrix.values.tolist(),
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Portfolio analysis completed",
                   num_assets=len(request.returns_data),
                   sharpe_ratio=sharpe_ratio)

        return result

    except Exception as e:
        logger.error("Portfolio analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio weights using various methods.

    Supports equal weighting, risk parity, and minimum variance optimization.
    """
    try:
        logger.info("Received portfolio optimization request",
                   num_assets=len(request.returns_data),
                   method=request.method)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        returns = returns_df.values
        n_assets = returns.shape[1]

        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        weights = None
        optimization_result = {}

        if request.method == "equal_weight":
            weights = np.ones(n_assets) / n_assets
            optimization_result = {"method": "equal_weight", "message": "Equal weights assigned to all assets"}

        elif request.method == "risk_parity":
            # Risk parity - equal risk contribution
            volatilities = np.sqrt(np.diag(cov_matrix))

            # Inverse volatility weighting
            inv_vol_weights = 1 / volatilities
            weights = inv_vol_weights / np.sum(inv_vol_weights)

            optimization_result = {
                "method": "risk_parity",
                "message": "Weights optimized for equal risk contribution",
                "volatilities": volatilities.tolist()
            }

        elif request.method == "min_variance":
            # Minimum variance portfolio
            ones = np.ones(n_assets)
            try:
                weights = np.linalg.solve(cov_matrix, ones)
                weights = weights / np.sum(weights)

                # Ensure no negative weights (long-only constraint)
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)

                optimization_result = {
                    "method": "min_variance",
                    "message": "Weights optimized for minimum variance",
                    "convergence": "successful"
                }
            except np.linalg.LinAlgError:
                # Fallback to equal weights if matrix is singular
                weights = np.ones(n_assets) / n_assets
                optimization_result = {
                    "method": "min_variance",
                    "message": "Matrix singular, using equal weights as fallback",
                    "convergence": "failed"
                }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported optimization method: {request.method}")

        # Calculate portfolio metrics with optimized weights
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Individual asset contributions to risk
        marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contributions = weights * marginal_contributions

        result = {
            "optimization": optimization_result,
            "optimized_weights": weights.tolist(),
            "asset_names": list(returns_df.columns),
            "portfolio_metrics": {
                "expected_return": float(portfolio_return),
                "volatility": float(portfolio_volatility),
                "sharpe_ratio": float(portfolio_return / portfolio_volatility * np.sqrt(252))
            },
            "asset_analysis": {
                "expected_returns": expected_returns.tolist(),
                "volatilities": np.sqrt(np.diag(cov_matrix)).tolist(),
                "risk_contributions": risk_contributions.tolist(),
                "marginal_contributions": marginal_contributions.tolist()
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Portfolio optimization completed",
                   method=request.method,
                   portfolio_return=portfolio_return,
                   portfolio_volatility=portfolio_volatility)

        return result

    except Exception as e:
        logger.error("Portfolio optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")


@router.post("/backtest", response_model=Dict[str, Any])
async def backtest_portfolio(
    returns_data: List[List[float]],
    weights: Optional[List[float]] = None,
    asset_names: Optional[List[str]] = None,
    rebalance_frequency: str = "monthly",
    initial_investment: float = 100000
):
    """
    Backtest portfolio performance with periodic rebalancing.

    Query parameters:
    - rebalance_frequency: Rebalancing frequency ("daily", "weekly", "monthly")
    - initial_investment: Initial portfolio value
    """
    try:
        logger.info("Received portfolio backtest request",
                   num_assets=len(returns_data),
                   rebalance_frequency=rebalance_frequency)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(returns_data).T,
            columns=asset_names or [f"Asset_{i}" for i in range(len(returns_data))]
        )

        returns = returns_df.values
        n_periods, n_assets = returns.shape

        # Use provided weights or equal weights
        if weights:
            portfolio_weights = np.array(weights)
            portfolio_weights = portfolio_weights / np.sum(portfolio_weights)
        else:
            portfolio_weights = np.ones(n_assets) / n_assets

        # Simulate portfolio returns
        portfolio_returns = np.zeros(n_periods)
        portfolio_values = np.zeros(n_periods + 1)
        portfolio_values[0] = initial_investment

        # Asset values (assuming equal initial allocation)
        asset_values = np.ones((n_periods + 1, n_assets)) * (initial_investment / n_assets)

        for t in range(n_periods):
            # Calculate portfolio return for this period
            period_return = np.sum(portfolio_weights * returns[t])
            portfolio_returns[t] = period_return

            # Update portfolio value
            portfolio_values[t + 1] = portfolio_values[t] * (1 + period_return)

            # Rebalance to target weights (simplified - assuming monthly rebalancing for now)
            if rebalance_frequency == "daily" or (rebalance_frequency == "monthly" and (t + 1) % 21 == 0):
                for i in range(n_assets):
                    asset_values[t + 1, i] = portfolio_values[t + 1] * portfolio_weights[i]
            else:
                # No rebalancing - maintain current asset values
                for i in range(n_assets):
                    asset_values[t + 1, i] = asset_values[t, i] * (1 + returns[t, i])

        # Calculate performance metrics
        total_return = (portfolio_values[-1] - initial_investment) / initial_investment
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility

        # Maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        result = {
            "backtest_summary": {
                "initial_investment": initial_investment,
                "final_value": float(portfolio_values[-1]),
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(annualized_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "maximum_drawdown": float(max_drawdown),
                "rebalance_frequency": rebalance_frequency
            },
            "performance_data": {
                "portfolio_returns": portfolio_returns.tolist(),
                "portfolio_values": portfolio_values.tolist(),
                "drawdown_series": drawdown.tolist()
            },
            "asset_data": {
                "weights": portfolio_weights.tolist(),
                "asset_names": list(returns_df.columns),
                "final_asset_values": asset_values[-1].tolist()
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Portfolio backtest completed",
                   total_return=total_return,
                   annualized_return=annualized_return,
                   sharpe_ratio=sharpe_ratio)

        return result

    except Exception as e:
        logger.error("Portfolio backtest failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio backtest failed: {str(e)}")


@router.post("/monte-carlo", response_model=Dict[str, Any])
async def monte_carlo_simulation(
    returns_data: List[List[float]],
    num_simulations: int = 1000,
    time_horizon: int = 252,
    confidence_level: float = 0.95,
    initial_investment: float = 100000
):
    """
    Run Monte Carlo simulation for portfolio projection.

    Query parameters:
    - num_simulations: Number of Monte Carlo simulations
    - time_horizon: Time horizon in days
    - confidence_level: Confidence level for projections
    - initial_investment: Initial portfolio value
    """
    try:
        logger.info("Received Monte Carlo simulation request",
                   num_simulations=num_simulations,
                   time_horizon=time_horizon)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(np.array(returns_data).T)
        returns = returns_df.values

        # Calculate parameters from historical data
        mu = np.mean(returns, axis=0)  # Expected returns
        cov = np.cov(returns.T)        # Covariance matrix

        # Equal weight portfolio
        n_assets = returns.shape[1]
        weights = np.ones(n_assets) / n_assets

        # Portfolio parameters
        portfolio_mu = np.sum(weights * mu)
        portfolio_sigma = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

        # Run Monte Carlo simulation
        simulated_portfolio_values = np.zeros((num_simulations, time_horizon + 1))
        simulated_portfolio_values[:, 0] = initial_investment

        np.random.seed(42)  # For reproducibility

        for sim in range(num_simulations):
            for t in range(1, time_horizon + 1):
                # Generate random return from multivariate normal
                shock = np.random.multivariate_normal(mu, cov)
                portfolio_return = np.sum(weights * shock)

                simulated_portfolio_values[sim, t] = \
                    simulated_portfolio_values[sim, t-1] * (1 + portfolio_return)

        # Calculate statistics
        final_values = simulated_portfolio_values[:, -1]
        mean_final_value = float(np.mean(final_values))
        median_final_value = float(np.median(final_values))

        # Confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100

        ci_lower = float(np.percentile(final_values, lower_percentile))
        ci_upper = float(np.percentile(final_values, upper_percentile))

        # Probability of loss
        prob_loss = float(np.mean(final_values < initial_investment))
        prob_gain = float(np.mean(final_values > initial_investment))

        # Risk metrics
        worst_case = float(np.min(final_values))
        best_case = float(np.max(final_values))
        var_95 = float(np.percentile(final_values, 5))

        result = {
            "simulation_summary": {
                "num_simulations": num_simulations,
                "time_horizon": time_horizon,
                "confidence_level": confidence_level,
                "initial_investment": initial_investment
            },
            "projection_results": {
                "mean_final_value": mean_final_value,
                "median_final_value": median_final_value,
                "confidence_interval": [ci_lower, ci_upper],
                "probability_of_loss": prob_loss,
                "probability_of_gain": prob_gain,
                "worst_case": worst_case,
                "best_case": best_case,
                "var_95": var_95
            },
            "simulation_data": {
                "final_values": final_values.tolist()[:100],  # Sample of final values
                "sample_paths": simulated_portfolio_values[:10].tolist()  # Sample of simulation paths
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Monte Carlo simulation completed",
                   mean_final_value=mean_final_value,
                   prob_loss=prob_loss)

        return result

    except Exception as e:
        logger.error("Monte Carlo simulation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {str(e)}")
