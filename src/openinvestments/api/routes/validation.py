"""
FastAPI routes for model validation endpoints.
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


class ValidationRequest(BaseModel):
    """Request for model validation."""
    predictions: List[float] = Field(..., description="Model predictions")
    actuals: List[float] = Field(..., description="Actual values")
    model_name: Optional[str] = Field(None, description="Model name or identifier")


class ComparisonRequest(BaseModel):
    """Request for model comparison."""
    model_predictions: List[float] = Field(..., description="Primary model predictions")
    benchmark_predictions: List[float] = Field(..., description="Benchmark model predictions")
    actuals: List[float] = Field(..., description="Actual values")
    model_name: Optional[str] = Field(None, description="Primary model name")
    benchmark_name: Optional[str] = Field(None, description="Benchmark model name")


class StabilityRequest(BaseModel):
    """Request for model stability analysis."""
    time_series_predictions: List[List[float]] = Field(..., description="Time series of predictions")
    time_series_actuals: List[List[float]] = Field(..., description="Time series of actuals")
    window_size: int = Field(30, description="Rolling window size for stability analysis")


@router.post("/validate", response_model=Dict[str, Any])
async def validate_model(request: ValidationRequest):
    """
    Validate model predictions against actual values.

    Returns comprehensive validation metrics including accuracy measures,
    error statistics, and model performance indicators.
    """
    try:
        logger.info("Received model validation request",
                   num_predictions=len(request.predictions),
                   model_name=request.model_name)

        if len(request.predictions) != len(request.actuals):
            raise HTTPException(status_code=400, detail="Predictions and actuals must have the same length")

        predictions = np.array(request.predictions)
        actuals = np.array(request.actuals)

        # Calculate errors
        errors = predictions - actuals
        abs_errors = np.abs(errors)

        # Basic metrics
        mse = float(np.mean(errors**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_errors))
        mape = float(np.mean(abs_errors / np.abs(actuals)) * 100)

        # Statistical tests
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors, ddof=1))

        # Diebold-Mariano test (simplified)
        dm_stat = mean_error / (std_error / np.sqrt(len(errors)))

        # Model accuracy metrics
        accuracy_1pct = float(np.mean(abs_errors / np.abs(actuals) <= 0.01) * 100)
        accuracy_5pct = float(np.mean(abs_errors / np.abs(actuals) <= 0.05) * 100)

        # Error distribution analysis
        error_skewness = float(pd.Series(errors).skew())
        error_kurtosis = float(pd.Series(errors).kurtosis())

        # Model assessment
        if mape < 5:
            model_quality = "excellent"
        elif mape < 10:
            model_quality = "good"
        elif mape < 20:
            model_quality = "fair"
        else:
            model_quality = "poor"

        # Statistical significance
        if abs(dm_stat) > 1.96:
            bias_significance = "statistically_significant"
        else:
            bias_significance = "not_statistically_significant"

        result = {
            "validation_summary": {
                "model_name": request.model_name,
                "num_predictions": len(request.predictions),
                "model_quality": model_quality,
                "bias_significance": bias_significance
            },
            "accuracy_metrics": {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "accuracy_1pct": accuracy_1pct,
                "accuracy_5pct": accuracy_5pct
            },
            "error_statistics": {
                "mean_error": mean_error,
                "std_error": std_error,
                "dm_statistic": dm_stat,
                "min_error": float(np.min(errors)),
                "max_error": float(np.max(errors)),
                "error_skewness": error_skewness,
                "error_kurtosis": error_kurtosis
            },
            "error_distribution": {
                "errors": errors.tolist()[:1000],  # Sample of errors
                "percentiles": {
                    "1": float(np.percentile(abs_errors, 1)),
                    "5": float(np.percentile(abs_errors, 5)),
                    "25": float(np.percentile(abs_errors, 25)),
                    "50": float(np.percentile(abs_errors, 50)),
                    "75": float(np.percentile(abs_errors, 75)),
                    "95": float(np.percentile(abs_errors, 95)),
                    "99": float(np.percentile(abs_errors, 99))
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Model validation completed",
                   mape=mape,
                   model_quality=model_quality)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model validation failed: {str(e)}")


@router.post("/compare", response_model=Dict[str, Any])
async def compare_models(request: ComparisonRequest):
    """
    Compare two models against a benchmark.

    Returns comparative statistics and significance tests.
    """
    try:
        logger.info("Received model comparison request",
                   num_predictions=len(request.model_predictions),
                   model_name=request.model_name,
                   benchmark_name=request.benchmark_name)

        if not (len(request.model_predictions) == len(request.benchmark_predictions) == len(request.actuals)):
            raise HTTPException(status_code=400, detail="All prediction arrays must have the same length")

        model_pred = np.array(request.model_predictions)
        benchmark_pred = np.array(request.benchmark_predictions)
        actuals = np.array(request.actuals)

        # Calculate errors for both models
        model_errors = model_pred - actuals
        benchmark_errors = benchmark_pred - actuals

        # Calculate metrics for both models
        def calculate_metrics(errors, predictions):
            mse = float(np.mean(errors**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(errors)))
            mape = float(np.mean(np.abs(errors) / np.abs(actuals)) * 100)
            return mse, rmse, mae, mape

        model_mse, model_rmse, model_mae, model_mape = calculate_metrics(model_errors, model_pred)
        bench_mse, bench_rmse, bench_mae, bench_mape = calculate_metrics(benchmark_errors, benchmark_pred)

        # Diebold-Mariano test for predictive accuracy comparison
        error_diff = model_errors**2 - benchmark_errors**2
        dm_stat = float(np.mean(error_diff) / (np.std(error_diff, ddof=1) / np.sqrt(len(error_diff))))

        # Model comparison metrics
        mse_improvement = float((bench_mse - model_mse) / bench_mse * 100)
        rmse_improvement = float((bench_rmse - model_rmse) / bench_rmse * 100)
        mae_improvement = float((bench_mae - model_mae) / bench_mae * 100)

        # Statistical significance of improvement
        if abs(dm_stat) > 1.96:
            dm_significance = "statistically_significant"
            if dm_stat < 0:
                winner = "model_better"
            else:
                winner = "benchmark_better"
        else:
            dm_significance = "not_statistically_significant"
            winner = "no_significant_difference"

        # Model quality assessment
        def assess_quality(mape):
            if mape < 5:
                return "excellent"
            elif mape < 10:
                return "good"
            elif mape < 20:
                return "fair"
            else:
                return "poor"

        model_quality = assess_quality(model_mape)
        benchmark_quality = assess_quality(bench_mape)

        result = {
            "comparison_summary": {
                "model_name": request.model_name,
                "benchmark_name": request.benchmark_name,
                "num_predictions": len(request.model_predictions),
                "winner": winner,
                "dm_significance": dm_significance
            },
            "model_metrics": {
                "mse": model_mse,
                "rmse": model_rmse,
                "mae": model_mae,
                "mape": model_mape,
                "quality": model_quality
            },
            "benchmark_metrics": {
                "mse": bench_mse,
                "rmse": bench_rmse,
                "mae": bench_mae,
                "mape": bench_mape,
                "quality": benchmark_quality
            },
            "improvement_analysis": {
                "mse_improvement": mse_improvement,
                "rmse_improvement": rmse_improvement,
                "mae_improvement": mae_improvement,
                "dm_statistic": dm_stat
            },
            "error_analysis": {
                "model_error_std": float(np.std(model_errors)),
                "benchmark_error_std": float(np.std(benchmark_errors)),
                "model_error_skewness": float(pd.Series(model_errors).skew()),
                "benchmark_error_skewness": float(pd.Series(benchmark_errors).skew())
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Model comparison completed",
                   winner=winner,
                   dm_significance=dm_significance)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")


@router.post("/stability", response_model=Dict[str, Any])
async def analyze_stability(request: StabilityRequest):
    """
    Analyze model stability over time using rolling windows.

    Returns stability metrics and trend analysis.
    """
    try:
        logger.info("Received stability analysis request",
                   num_periods=len(request.time_series_predictions),
                   window_size=request.window_size)

        predictions_series = np.array(request.time_series_predictions)
        actuals_series = np.array(request.time_series_actuals)

        if predictions_series.shape != actuals_series.shape:
            raise HTTPException(status_code=400, detail="Prediction and actual series must have the same shape")

        n_periods = predictions_series.shape[0]

        if n_periods < request.window_size:
            raise HTTPException(status_code=400, detail=f"Need at least {request.window_size} periods for stability analysis")

        # Rolling window analysis
        mse_series = []
        rmse_series = []
        mae_series = []
        mape_series = []

        for i in range(request.window_size, n_periods + 1):
            window_pred = predictions_series[i-request.window_size:i]
            window_actual = actuals_series[i-request.window_size:i]

            errors = window_pred - window_actual
            mse = float(np.mean(errors**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(errors)))
            mape = float(np.mean(np.abs(errors) / np.abs(window_actual)) * 100)

            mse_series.append(mse)
            rmse_series.append(rmse)
            mae_series.append(mae)
            mape_series.append(mape)

        # Calculate stability metrics
        mse_volatility = float(np.std(mse_series))
        rmse_volatility = float(np.std(rmse_series))
        mae_volatility = float(np.std(mae_series))
        mape_volatility = float(np.std(mape_series))

        # Trend analysis using linear regression
        x = np.arange(len(mse_series))
        mse_trend = float(np.polyfit(x, mse_series, 1)[0])
        rmse_trend = float(np.polyfit(x, rmse_series, 1)[0])

        # Calculate R-squared for trends
        mse_y_pred = mse_trend * x + np.polyfit(x, mse_series, 1)[1]
        mse_ss_res = np.sum((np.array(mse_series) - mse_y_pred) ** 2)
        mse_ss_tot = np.sum((np.array(mse_series) - np.mean(mse_series)) ** 2)
        mse_r_squared = float(1 - (mse_ss_res / mse_ss_tot))

        # Stability score (1 - coefficient of variation)
        stability_score = float(1 - (mape_volatility / np.mean(mape_series)))

        # Stability rating
        if stability_score > 0.8:
            stability_rating = "very_stable"
        elif stability_score > 0.6:
            stability_rating = "stable"
        elif stability_score > 0.4:
            stability_rating = "moderately_stable"
        else:
            stability_rating = "unstable"

        # Trend direction
        if abs(mse_trend) < 0.001:
            trend_direction = "stable"
        elif mse_trend > 0:
            trend_direction = "degrading"
        else:
            trend_direction = "improving"

        result = {
            "stability_summary": {
                "num_periods": n_periods,
                "window_size": request.window_size,
                "analysis_periods": len(mse_series),
                "stability_score": stability_score,
                "stability_rating": stability_rating,
                "trend_direction": trend_direction
            },
            "rolling_metrics": {
                "mse": {
                    "values": mse_series,
                    "volatility": mse_volatility,
                    "trend": mse_trend,
                    "r_squared": mse_r_squared
                },
                "rmse": {
                    "values": rmse_series,
                    "volatility": rmse_volatility,
                    "trend": rmse_trend
                },
                "mae": {
                    "values": mae_series,
                    "volatility": mae_volatility
                },
                "mape": {
                    "values": mape_series,
                    "volatility": mape_volatility
                }
            },
            "stability_assessment": {
                "mean_mape": float(np.mean(mape_series)),
                "mape_volatility": mape_volatility,
                "min_mape": float(np.min(mape_series)),
                "max_mape": float(np.max(mape_series)),
                "mape_range": float(np.max(mape_series) - np.min(mape_series))
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Stability analysis completed",
                   stability_score=stability_score,
                   stability_rating=stability_rating)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Stability analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stability analysis failed: {str(e)}")
