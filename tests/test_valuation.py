import pytest
import numpy as np
from openinvestments.valuation import MonteCarloPricer, GeometricBrownianMotion, EuropeanCallPayoff, GreeksCalculator, BlackScholesModel
from openinvestments.valuation.trees import BinomialTree, CallPayoff

def test_monte_carlo_pricing():
    pricer = MonteCarloPricer()
    process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    payoff = EuropeanCallPayoff(strike=100)
    
    result = pricer.price_option(
        S0=100, T=1.0, r=0.05, sigma=0.2, payoff=payoff, process=process
    )
    
    assert result["price"] > 0
    assert "standard_error" in result
    assert len(result["confidence_interval"]) == 2

def test_greeks_calculation():
    calculator = GreeksCalculator()
    params = {
        'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'is_call': True
    }
    
    greeks = calculator.calculate_greeks(params)
    
    assert "delta" in greeks
    assert "gamma" in greeks
    assert "vega" in greeks
    assert abs(greeks["delta"]) <= 1  # Delta should be between -1 and 1
    assert greeks["gamma"] > 0  # Gamma should be positive for vanilla options

def test_binomial_tree_pricing():
    tree = BinomialTree()
    payoff = CallPayoff()
    
    result = tree.price_european(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.2, payoff_func=payoff
    )
    
    assert "price" in result
    assert result["price"] > 0
    assert "parameters" in result

def test_black_scholes_model():
    model = BlackScholesModel()
    params = {
        'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'is_call': True
    }
    
    price = model.price(params)
    
    assert price > 0
    assert price < 100  # Option price should be less than intrinsic value

if __name__ == "__main__":
    pytest.main([__file__, "-v"])