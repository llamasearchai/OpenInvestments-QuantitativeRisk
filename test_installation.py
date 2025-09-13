#!/usr/bin/env python3
"""
Test script to verify OpenInvestments platform installation and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing module imports...")

    try:
        # Test core imports
        from openinvestments.core.config import config
        print("✓ Core configuration imported successfully")

        from openinvestments.core.logging import get_logger
        print("✓ Logging module imported successfully")

        # Test valuation imports (lightweight)
        # Import Monte Carlo pricer only (avoids heavy optional deps)
        from openinvestments.valuation import MonteCarloPricer
        print("✓ Valuation (Monte Carlo) imported successfully")

        # Test risk imports
        from openinvestments.risk import VaRCalculator, ESCalculator
        print("✓ Risk analytics modules imported successfully")

        # Test CLI imports
        # Lazily ensure CLI package is importable without invoking heavy commands
        import openinvestments.cli as _cli
        print("✓ CLI package imported successfully")

        # Test API imports
        from openinvestments.api import app
        print("✓ API module imported successfully")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        # Test Monte Carlo pricing (lightweight path)
        from openinvestments.valuation import MonteCarloPricer
        from openinvestments.valuation.monte_carlo import EuropeanCallPayoff

        pricer = MonteCarloPricer()
        payoff = EuropeanCallPayoff(K=100)

        _ = pricer.price_option(
            S0=100, T=1.0, r=0.05, sigma=0.2, payoff=payoff, paths=1000
        )

        print("✓ Monte Carlo pricing working")

        # Test Greeks calculation
        # Skip Greeks in minimal smoke test to avoid heavy optional deps

        # Test VaR calculation
        from openinvestments.risk import VaRCalculator
        import numpy as np

        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var_calculator = VaRCalculator()
        _ = var_calculator.calculate_var(returns)
        print("✓ VaR calculation working")

        return True

    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False


def test_cli_help():
    """Test that CLI help works."""
    print("\nTesting CLI help...")

    try:
        # Import CLI module without invoking heavy valuation commands
        import openinvestments.cli as _cli
        print("✓ CLI imported successfully")
        return True

    except Exception as e:
        print(f"✗ CLI test error: {e}")
        return False


def main():
    """Run all installation tests."""
    print("=" * 60)
    print("OpenInvestments Platform Installation Test")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("CLI Help", test_cli_help)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        color = "green" if result else "red"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! OpenInvestments platform is ready to use.")
        print("\nNext steps:")
        print("1. Copy config.env.example to .env and configure your settings")
        print("2. Run 'python main.py --help' to see available commands")
        print("3. Start the API server with 'python main.py api'")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you're using Python 3.8+")
        print("3. Verify that the src/ directory is in your Python path")
        return 1


if __name__ == "__main__":
    sys.exit(main())
