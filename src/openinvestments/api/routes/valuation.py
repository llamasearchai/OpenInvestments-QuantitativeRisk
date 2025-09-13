"""
FastAPI routes for valuation endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

from ...valuation import MonteCarloPricer, GreeksCalculator, BinomialTree, EuropeanCallPayoff
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class OptionParameters(BaseModel):
    """Parameters for option pricing."""
    S0: float = Field(..., gt=0, description="Initial stock price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to maturity in years")
    r: float = Field(0.0, ge=0, description="Risk-free rate")
    sigma: float = Field(..., gt=0, description="Volatility")
    is_call: bool = Field(True, description="Call option (True) or put (False)")
    dividend_yield: float = Field(0.0, ge=0, description="Continuous dividend yield")


class MonteCarloRequest(BaseModel):
    """Request for Monte Carlo option pricing."""
    option_params: OptionParameters
    paths: int = Field(10000, gt=0, description="Number of simulation paths")
    steps: int = Field(252, gt=0, description="Number of time steps")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GreeksRequest(BaseModel):
    """Request for Greeks calculation."""
    option_params: OptionParameters
    greeks: List[str] = Field(
        ["delta", "gamma", "vega", "theta", "rho"],
        description="List of Greeks to calculate"
    )


class BatchPricingRequest(BaseModel):
    """Request for batch option pricing."""
    options: List[OptionParameters]
    method: str = Field("monte_carlo", description="Pricing method")


class ExoticOptionRequest(BaseModel):
    """Request for exotic option pricing."""
    option_params: OptionParameters
    option_type: str = Field(..., description="Type of exotic option")
    barrier_level: Optional[float] = Field(None, description="Barrier level for barrier options")
    barrier_type: Optional[str] = Field("up-and-out", description="Barrier option type")


@router.post("/price/monte-carlo", response_model=Dict[str, Any])
async def price_option_monte_carlo(request: MonteCarloRequest):
    """
    Price European option using Monte Carlo simulation.

    Returns pricing results with confidence intervals and Greeks.
    """
    try:
        logger.info("Received Monte Carlo pricing request",
                   S0=request.option_params.S0,
                   K=request.option_params.K,
                   T=request.option_params.T)

        # Create payoff function
        if request.option_params.is_call:
            payoff = EuropeanCallPayoff(request.option_params.K)
        else:
            from ...valuation.monte_carlo import EuropeanPutPayoff
            payoff = EuropeanPutPayoff(request.option_params.K)

        # Create and configure pricer
        pricer = MonteCarloPricer()

        # Price the option
        result = pricer.price_option(
            S0=request.option_params.S0,
            T=request.option_params.T,
            r=request.option_params.r,
            sigma=request.option_params.sigma,
            payoff=payoff
        )

        # Add metadata
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "method": "monte_carlo",
            "option_type": "call" if request.option_params.is_call else "put",
            "parameters": {
                "S0": request.option_params.S0,
                "K": request.option_params.K,
                "T": request.option_params.T,
                "r": request.option_params.r,
                "sigma": request.option_params.sigma,
                "dividend_yield": request.option_params.dividend_yield
            }
        })

        logger.info("Monte Carlo pricing completed",
                   price=result["price"],
                   std_error=result["standard_error"])

        return result

    except Exception as e:
        logger.error("Monte Carlo pricing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pricing failed: {str(e)}")


@router.post("/greeks", response_model=Dict[str, Any])
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate option Greeks using finite differences.

    Returns all requested Greeks with their values and sensitivities.
    """
    try:
        logger.info("Received Greeks calculation request",
                   S0=request.option_params.S0,
                   K=request.option_params.K,
                   greeks=request.greeks)

        # Create Greeks calculator
        calculator = GreeksCalculator()

        # Prepare parameters
        params = {
            'S': request.option_params.S0,
            'K': request.option_params.K,
            'T': request.option_params.T,
            'r': request.option_params.r,
            'sigma': request.option_params.sigma,
            'is_call': request.option_params.is_call
        }

        # Calculate Greeks
        greeks_result = calculator.calculate_greeks(params, request.greeks)

        # Add metadata
        result = {
            "greeks": greeks_result,
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": params,
            "method": "finite_differences"
        }

        logger.info("Greeks calculation completed",
                   greeks_calculated=list(greeks_result.keys()))

        return result

    except Exception as e:
        logger.error("Greeks calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Greeks calculation failed: {str(e)}")


@router.post("/price/binomial-tree", response_model=Dict[str, Any])
async def price_option_binomial_tree(
    S0: float,
    K: float,
    T: float,
    r: float = 0.05,
    sigma: float = 0.2,
    is_call: bool = True,
    steps: int = 100
):
    """
    Price European option using binomial tree.

    Query parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - is_call: Call option flag
    - steps: Number of tree steps
    """
    try:
        logger.info("Received binomial tree pricing request",
                   S0=S0, K=K, T=T, steps=steps)

        # Create payoff function
        if is_call:
            payoff_func = EuropeanCallPayoff(K)
        else:
            from ...valuation.monte_carlo import EuropeanPutPayoff
            payoff_func = EuropeanPutPayoff(K)

        # Create and configure tree
        tree = BinomialTree()

        # Price the option
        result = tree.price_european(
            S0=S0, K=K, T=T, r=r, sigma=sigma,
            payoff_func=payoff_func
        )

        # Add metadata
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "method": "binomial_tree",
            "option_type": "call" if is_call else "put"
        })

        logger.info("Binomial tree pricing completed",
                   price=result["price"],
                   steps=steps)

        return result

    except Exception as e:
        logger.error("Binomial tree pricing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pricing failed: {str(e)}")


@router.post("/price/batch", response_model=Dict[str, Any])
async def batch_price_options(request: BatchPricingRequest, background_tasks: BackgroundTasks):
    """
    Price multiple options in batch.

    Supports Monte Carlo and Black-Scholes pricing methods.
    """
    try:
        logger.info("Received batch pricing request",
                   num_options=len(request.options),
                   method=request.method)

        results = []

        for i, option_params in enumerate(request.options):
            try:
                if request.method == "monte_carlo":
                    # Monte Carlo pricing
                    pricer = MonteCarloPricer()

                    if option_params.is_call:
                        payoff = EuropeanCallPayoff(option_params.K)
                    else:
                        from ...valuation.monte_carlo import EuropeanPutPayoff
                        payoff = EuropeanPutPayoff(option_params.K)

                    result = pricer.price_option(
                        S0=option_params.S0,
                        T=option_params.T,
                        r=option_params.r,
                        sigma=option_params.sigma,
                        payoff=payoff
                    )

                elif request.method == "black_scholes":
                    # Black-Scholes pricing
                    from ...valuation.greeks import BlackScholesModel
                    model = BlackScholesModel()

                    params = {
                        'S': option_params.S0,
                        'K': option_params.K,
                        'T': option_params.T,
                        'r': option_params.r,
                        'sigma': option_params.sigma,
                        'is_call': option_params.is_call
                    }

                    price = model.price(params)
                    result = {
                        "price": price,
                        "method": "black_scholes"
                    }

                else:
                    raise ValueError(f"Unsupported pricing method: {request.method}")

                # Add option parameters to result
                result.update({
                    "option_index": i,
                    "S0": option_params.S0,
                    "K": option_params.K,
                    "T": option_params.T,
                    "r": option_params.r,
                    "sigma": option_params.sigma,
                    "is_call": option_params.is_call
                })

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to price option {i}", error=str(e))
                results.append({
                    "option_index": i,
                    "error": str(e),
                    "S0": option_params.S0,
                    "K": option_params.K,
                    "T": option_params.T
                })

        batch_result = {
            "method": request.method,
            "total_options": len(request.options),
            "successful_pricings": len([r for r in results if "error" not in r]),
            "failed_pricings": len([r for r in results if "error" in r]),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info("Batch pricing completed",
                   successful=batch_result["successful_pricings"],
                   failed=batch_result["failed_pricings"])

        return batch_result

    except Exception as e:
        logger.error("Batch pricing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch pricing failed: {str(e)}")


@router.post("/price/exotic", response_model=Dict[str, Any])
async def price_exotic_option(request: ExoticOptionRequest):
    """
    Price exotic options (Asian, Barrier, etc.).

    Supports various exotic option types with Monte Carlo simulation.
    """
    try:
        logger.info("Received exotic option pricing request",
                   option_type=request.option_type,
                   S0=request.option_params.S0,
                   K=request.option_params.K)

        pricer = MonteCarloPricer()

        # Create appropriate payoff function
        if request.option_type.lower() == "asian":
            from ...valuation.monte_carlo import AsianCallPayoff
            payoff = AsianCallPayoff(request.option_params.K)

        elif request.option_type.lower() == "barrier":
            from ...valuation.monte_carlo import BarrierCallPayoff
            if request.barrier_level is None:
                raise HTTPException(status_code=400, detail="Barrier level required for barrier options")

            payoff = BarrierCallPayoff(
                request.option_params.K,
                request.barrier_level,
                request.barrier_type or "up-and-out"
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported exotic option type: {request.option_type}")

        # Price the option
        result = pricer.price_option(
            S0=request.option_params.S0,
            T=request.option_params.T,
            r=request.option_params.r,
            sigma=request.option_params.sigma,
            payoff=payoff
        )

        # Add metadata
        result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "method": "monte_carlo",
            "option_type": request.option_type,
            "exotic_parameters": {
                "barrier_level": request.barrier_level,
                "barrier_type": request.barrier_type
            }
        })

        logger.info("Exotic option pricing completed",
                   option_type=request.option_type,
                   price=result["price"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Exotic option pricing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Exotic pricing failed: {str(e)}")
