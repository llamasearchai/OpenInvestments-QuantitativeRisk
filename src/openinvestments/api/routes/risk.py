"""
FastAPI routes for risk analysis endpoints.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

from ...risk import VaRCalculator, ESCalculator, RiskConfig, VaRMethod, ESDistribution
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class RiskCalculationRequest(BaseModel):
    """Request for risk calculations."""
    confidence_level: float = Field(0.95, gt=0, lt=1, description="Confidence level")
    method: str = Field("historical", description="Calculation method")
    portfolio_value: float = Field(1000000, gt=0, description="Portfolio value in dollars")
    horizon_days: int = Field(1, gt=0, description="Risk horizon in days")


class PortfolioRiskRequest(BaseModel):
    """Request for portfolio risk analysis."""
    returns_data: List[List[float]] = Field(..., description="Historical returns matrix (assets x time)")
    asset_names: Optional[List[str]] = Field(None, description="Asset names")
    weights: Optional[List[float]] = Field(None, description="Portfolio weights")
    risk_params: RiskCalculationRequest


class StressTestRequest(BaseModel):
    """Request for stress testing."""
    returns_data: List[List[float]] = Field(..., description="Historical returns matrix")
    asset_names: Optional[List[str]] = Field(None, description="Asset names")
    num_scenarios: int = Field(100, gt=0, description="Number of stress scenarios")
    severity_levels: List[float] = Field([0.90, 0.95, 0.99], description="Severity levels")
    portfolio_value: float = Field(1000000, gt=0, description="Portfolio value")


class RiskReportRequest(BaseModel):
    """Request for comprehensive risk report."""
    returns_data: List[List[float]] = Field(..., description="Historical returns matrix")
    asset_names: Optional[List[str]] = Field(None, description="Asset names")
    weights: Optional[List[float]] = Field(None, description="Portfolio weights")
    confidence_levels: List[float] = Field([0.90, 0.95, 0.99], description="Confidence levels for analysis")
    portfolio_value: float = Field(1000000, gt=0, description="Portfolio value")


@router.post("/var/single-asset", response_model=Dict[str, Any])
async def calculate_var_single_asset(
    returns: List[float],
    request: RiskCalculationRequest
):
    """
    Calculate Value at Risk for single asset.

    Body parameters:
    - returns: List of historical returns
    - request: Risk calculation parameters
    """
    try:
        logger.info("Received single asset VaR request",
                   num_returns=len(returns),
                   confidence=request.confidence_level,
                   method=request.method)

        # Configure risk calculator
        risk_config = RiskConfig(
            confidence_level=request.confidence_level,
            portfolio_value=request.portfolio_value,
            horizon_days=request.horizon_days
        )

        calculator = VaRCalculator(risk_config)

        # Map method string to enum
        method_map = {
            'historical': VaRMethod.HISTORICAL,
            'parametric': VaRMethod.PARAMETRIC,
            'monte_carlo': VaRMethod.MONTE_CARLO,
            'evt': VaRMethod.EVT
        }

        if request.method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")

        # Convert returns to numpy array
        returns_array = np.array(returns)

        # Calculate VaR
        result = calculator.calculate_var(returns_array, method_map[request.method])

        # Add metadata
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "asset_type": "single_asset",
            "calculation_method": request.method,
            "data_points": len(returns)
        })

        logger.info("Single asset VaR calculation completed",
                   var=result["var"],
                   method=request.method)

        return result

    except Exception as e:
        logger.error("Single asset VaR calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"VaR calculation failed: {str(e)}")


@router.post("/var/portfolio", response_model=Dict[str, Any])
async def calculate_var_portfolio(request: PortfolioRiskRequest):
    """
    Calculate Value at Risk for multi-asset portfolio.

    Returns VaR with asset-level contributions and risk decomposition.
    """
    try:
        logger.info("Received portfolio VaR request",
                   num_assets=len(request.returns_data),
                   confidence=request.risk_params.confidence_level,
                   method=request.risk_params.method)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,  # Transpose to get time x assets
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        # Configure risk calculator
        risk_config = RiskConfig(
            confidence_level=request.risk_params.confidence_level,
            portfolio_value=request.risk_params.portfolio_value,
            horizon_days=request.risk_params.horizon_days
        )

        calculator = VaRCalculator(risk_config)

        # Map method string to enum
        method_map = {
            'historical': VaRMethod.HISTORICAL,
            'parametric': VaRMethod.PARAMETRIC,
            'monte_carlo': VaRMethod.MONTE_CARLO,
            'evt': VaRMethod.EVT
        }

        if request.risk_params.method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.risk_params.method}")

        # Calculate portfolio VaR
        result = calculator.calculate_var(returns_df, method_map[request.risk_params.method])

        # Add portfolio-specific information
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "asset_type": "portfolio",
            "num_assets": len(request.returns_data),
            "asset_names": list(returns_df.columns),
            "portfolio_weights": request.weights or [1.0/len(request.returns_data)] * len(request.returns_data),
            "correlation_matrix": returns_df.corr().values.tolist(),
            "calculation_method": request.risk_params.method,
            "data_points": len(returns_df)
        })

        logger.info("Portfolio VaR calculation completed",
                   var=result["var"],
                   num_assets=len(request.returns_data),
                   method=request.risk_params.method)

        return result

    except Exception as e:
        logger.error("Portfolio VaR calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio VaR calculation failed: {str(e)}")


@router.post("/es/single-asset", response_model=Dict[str, Any])
async def calculate_es_single_asset(
    returns: List[float],
    request: RiskCalculationRequest
):
    """
    Calculate Expected Shortfall for single asset.
    """
    try:
        logger.info("Received single asset ES request",
                   num_returns=len(returns),
                   confidence=request.confidence_level,
                   method=request.method)

        # Configure risk calculator
        risk_config = RiskConfig(
            confidence_level=request.confidence_level,
            portfolio_value=request.portfolio_value,
            horizon_days=request.horizon_days
        )

        calculator = ESCalculator(risk_config)

        # Map method string to enum
        method_map = {
            'historical': ESDistribution.HISTORICAL,
            'parametric': ESDistribution.NORMAL,
            'student_t': ESDistribution.STUDENT_T
        }

        if request.method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")

        # Convert returns to numpy array
        returns_array = np.array(returns)

        # Calculate ES
        result = calculator.calculate_es(returns_array, method_map[request.method])

        # Add metadata
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "asset_type": "single_asset",
            "calculation_method": request.method,
            "data_points": len(returns)
        })

        logger.info("Single asset ES calculation completed",
                   es=result["es"],
                   method=request.method)

        return result

    except Exception as e:
        logger.error("Single asset ES calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"ES calculation failed: {str(e)}")


@router.post("/es/portfolio", response_model=Dict[str, Any])
async def calculate_es_portfolio(request: PortfolioRiskRequest):
    """
    Calculate Expected Shortfall for multi-asset portfolio.
    """
    try:
        logger.info("Received portfolio ES request",
                   num_assets=len(request.returns_data),
                   confidence=request.risk_params.confidence_level,
                   method=request.risk_params.method)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        # Configure risk calculator
        risk_config = RiskConfig(
            confidence_level=request.risk_params.confidence_level,
            portfolio_value=request.risk_params.portfolio_value,
            horizon_days=request.risk_params.horizon_days
        )

        calculator = ESCalculator(risk_config)

        # Map method string to enum
        method_map = {
            'historical': ESDistribution.HISTORICAL,
            'parametric': ESDistribution.NORMAL,
            'student_t': ESDistribution.STUDENT_T
        }

        if request.risk_params.method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.risk_params.method}")

        # Calculate portfolio ES
        result = calculator.calculate_es(returns_df, method_map[request.risk_params.method])

        # Add portfolio-specific information
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "asset_type": "portfolio",
            "num_assets": len(request.returns_data),
            "asset_names": list(returns_df.columns),
            "portfolio_weights": request.weights or [1.0/len(request.returns_data)] * len(request.returns_data),
            "calculation_method": request.risk_params.method,
            "data_points": len(returns_df)
        })

        logger.info("Portfolio ES calculation completed",
                   es=result["es"],
                   num_assets=len(request.returns_data),
                   method=request.risk_params.method)

        return result

    except Exception as e:
        logger.error("Portfolio ES calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio ES calculation failed: {str(e)}")


@router.post("/es/spectrum", response_model=Dict[str, Any])
async def calculate_es_spectrum(request: PortfolioRiskRequest):
    """
    Calculate Expected Shortfall across multiple confidence levels.

    Returns ES spectrum showing tail risk at different confidence levels.
    """
    try:
        logger.info("Received ES spectrum request",
                   num_assets=len(request.returns_data),
                   method=request.risk_params.method)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        # Configure risk calculator
        risk_config = RiskConfig(
            portfolio_value=request.risk_params.portfolio_value,
            horizon_days=request.risk_params.horizon_days
        )

        calculator = ESCalculator(risk_config)

        # Map method string to enum
        method_map = {
            'historical': ESDistribution.HISTORICAL,
            'parametric': ESDistribution.NORMAL,
            'student_t': ESDistribution.STUDENT_T
        }

        if request.risk_params.method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.risk_params.method}")

        # Define confidence levels for spectrum
        confidence_levels = [0.90, 0.95, 0.99, 0.995, 0.999]

        # Calculate ES spectrum
        spectrum_result = calculator.calculate_es_spectrum(
            returns_df, confidence_levels
        )

        # Add metadata
        result = {
            "es_spectrum": {
                "confidence_levels": spectrum_result["confidence_levels"],
                "es_values": spectrum_result["es_values"],
                "var_values": spectrum_result["var_values"],
                "es_var_ratios": [
                    es/var if var != 0 else 0
                    for es, var in zip(spectrum_result["es_values"], spectrum_result["var_values"])
                ]
            },
            "timestamp": datetime.utcnow().isoformat(),
            "asset_type": "portfolio" if len(request.returns_data) > 1 else "single_asset",
            "num_assets": len(request.returns_data),
            "calculation_method": request.risk_params.method,
            "data_points": len(returns_df)
        }

        logger.info("ES spectrum calculation completed",
                   num_levels=len(confidence_levels),
                   method=request.risk_params.method)

        return result

    except Exception as e:
        logger.error("ES spectrum calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"ES spectrum calculation failed: {str(e)}")


@router.post("/stress-test", response_model=Dict[str, Any])
async def perform_stress_test(request: StressTestRequest):
    """
    Perform stress testing on portfolio using historical scenarios.

    Generates stress scenarios and calculates potential losses.
    """
    try:
        logger.info("Received stress test request",
                   num_assets=len(request.returns_data),
                   num_scenarios=request.num_scenarios,
                   severity_levels=request.severity_levels)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        stress_results = {}

        for severity in request.severity_levels:
            # Calculate portfolio returns
            portfolio_returns = returns_df.mean(axis=1).values

            # Find historical scenarios beyond severity threshold
            sorted_returns = np.sort(portfolio_returns)
            quantile_index = int((1 - severity) * len(sorted_returns))

            # Generate stress scenarios by resampling from tail
            tail_returns = sorted_returns[:quantile_index]

            if len(tail_returns) == 0:
                # If no tail data, use the worst historical return
                tail_returns = np.array([sorted_returns[0]])

            stress_scenarios = np.random.choice(
                tail_returns,
                min(request.num_scenarios, len(tail_returns)),
                replace=True
            )

            # Calculate stress test metrics
            avg_stress_loss = -np.mean(stress_scenarios)
            max_stress_loss = -np.min(stress_scenarios)
            var_95_stress = -np.percentile(stress_scenarios, 5)  # 95% VaR from stress scenarios

            stress_results[severity] = {
                "average_loss": avg_stress_loss,
                "maximum_loss": max_stress_loss,
                "var_95": var_95_stress,
                "scenarios_generated": len(stress_scenarios),
                "portfolio_value": request.portfolio_value,
                "average_loss_dollars": avg_stress_loss * request.portfolio_value,
                "maximum_loss_dollars": max_stress_loss * request.portfolio_value,
                "var_95_dollars": var_95_stress * request.portfolio_value
            }

        result = {
            "stress_test_results": stress_results,
            "timestamp": datetime.utcnow().isoformat(),
            "num_assets": len(request.returns_data),
            "portfolio_value": request.portfolio_value,
            "method": "historical_simulation",
            "total_scenarios": sum(r["scenarios_generated"] for r in stress_results.values())
        }

        logger.info("Stress test completed",
                   scenarios_generated=result["total_scenarios"],
                   severity_levels=request.severity_levels)

        return result

    except Exception as e:
        logger.error("Stress test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")


@router.post("/risk-report", response_model=Dict[str, Any])
async def generate_risk_report(request: RiskReportRequest):
    """
    Generate comprehensive risk report for portfolio.

    Includes VaR, ES, stress testing, and risk decomposition.
    """
    try:
        logger.info("Received risk report request",
                   num_assets=len(request.returns_data),
                   confidence_levels=request.confidence_levels)

        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(
            np.array(request.returns_data).T,
            columns=request.asset_names or [f"Asset_{i}" for i in range(len(request.returns_data))]
        )

        # Configure calculators
        risk_config = RiskConfig(portfolio_value=request.portfolio_value)
        var_calculator = VaRCalculator(risk_config)
        es_calculator = ESCalculator(risk_config)

        report = {
            "portfolio_summary": {
                "num_assets": len(request.returns_data),
                "asset_names": list(returns_df.columns),
                "portfolio_value": request.portfolio_value,
                "data_points": len(returns_df),
                "analysis_period": "historical_data"
            },
            "risk_measures": {},
            "stress_testing": {},
            "risk_decomposition": {},
            "recommendations": []
        }

        # Calculate VaR at different confidence levels
        var_measures = {}
        for conf in request.confidence_levels:
            var_result = var_calculator.calculate_var(
                returns_df, VaRMethod.HISTORICAL, confidence_level=conf
            )
            var_measures[conf] = {
                "var_percent": var_result["var"],
                "var_dollars": var_result["var_amount"],
                "expected_shortfall_percent": var_result.get("expected_shortfall", 0),
                "expected_shortfall_dollars": var_result.get("expected_shortfall", 0) * request.portfolio_value
            }

        report["risk_measures"]["value_at_risk"] = var_measures

        # Calculate ES spectrum
        es_spectrum = es_calculator.calculate_es_spectrum(
            returns_df, request.confidence_levels
        )

        report["risk_measures"]["expected_shortfall"] = {
            "spectrum": {
                "confidence_levels": es_spectrum["confidence_levels"],
                "es_values": es_spectrum["es_values"],
                "var_values": es_spectrum["var_values"]
            }
        }

        # Portfolio statistics
        portfolio_returns = returns_df.mean(axis=1)
        report["portfolio_statistics"] = {
            "mean_return": float(np.mean(portfolio_returns)),
            "volatility": float(np.std(portfolio_returns)),
            "skewness": float(pd.Series(portfolio_returns).skew()),
            "kurtosis": float(pd.Series(portfolio_returns).kurtosis()),
            "maximum_drawdown": float(self._calculate_max_drawdown(portfolio_returns)),
            "sharpe_ratio": float(self._calculate_sharpe_ratio(portfolio_returns))
        }

        # Risk decomposition (simplified)
        asset_contributions = {}
        portfolio_var = var_measures[0.95]["var_percent"]  # Use 95% VaR

        for asset in returns_df.columns:
            asset_returns = returns_df[asset].values
            asset_var = var_calculator.calculate_var(
                asset_returns, VaRMethod.HISTORICAL, confidence_level=0.95
            )["var"]

            # Simplified contribution calculation
            correlation = returns_df[asset].corr(portfolio_returns)
            weight = request.weights[i] if request.weights else 1.0 / len(request.returns_data)

            contribution = weight * asset_var * correlation
            asset_contributions[asset] = {
                "contribution_percent": contribution,
                "contribution_dollars": contribution * request.portfolio_value,
                "weight": weight,
                "correlation": correlation
            }

        report["risk_decomposition"]["asset_contributions"] = asset_contributions

        # Generate recommendations
        recommendations = []

        # Check diversification
        max_contribution = max([c["contribution_percent"] for c in asset_contributions.values()])
        if max_contribution > 0.5:
            recommendations.append({
                "type": "diversification",
                "severity": "high",
                "message": f"Portfolio is concentrated - largest asset contributes {max_contribution:.1%} to VaR"
            })

        # Check tail risk
        es_99 = es_spectrum["es_values"][-1] if len(es_spectrum["es_values"]) > 2 else 0
        var_99 = es_spectrum["var_values"][-1] if len(es_spectrum["var_values"]) > 2 else 1
        tail_risk_ratio = es_99 / var_99 if var_99 != 0 else 0

        if tail_risk_ratio > 2.0:
            recommendations.append({
                "type": "tail_risk",
                "severity": "medium",
                "message": f"High tail risk - ES/VaR ratio at 99% is {tail_risk_ratio:.2f}"
            })

        report["recommendations"] = recommendations

        # Add metadata
        report.update({
            "timestamp": datetime.utcnow().isoformat(),
            "generated_by": "OpenInvestments Risk Platform",
            "version": "1.0.0"
        })

        logger.info("Comprehensive risk report generated",
                   num_assets=len(request.returns_data),
                   recommendations=len(recommendations))

        return report

    except Exception as e:
        logger.error("Risk report generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Risk report generation failed: {str(e)}")

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        return drawdown.min()

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio."""
        excess_returns = pd.Series(returns) - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
