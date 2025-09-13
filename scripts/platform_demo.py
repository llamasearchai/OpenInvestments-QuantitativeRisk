#!/usr/bin/env python3
"""
OpenInvestments Platform Demonstration Script

This script showcases the comprehensive functionality of the OpenInvestments
Quantitative Risk Platform by running a series of demonstrations covering:

- Option pricing and valuation
- Risk analysis and VaR calculation
- Portfolio optimization
- Machine learning model training
- Data quality validation
- Automated alerting
- Real-time market data simulation

Usage:
    python scripts/platform_demo.py [--full] [--quick]

Arguments:
    --full: Run complete demonstration (takes longer)
    --quick: Run quick demonstration (default)
"""

import sys
import time
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Import platform modules
    from openinvestments.core.config import config
    from openinvestments.core.logging import get_logger
    from openinvestments.valuation import MonteCarloPricer, EuropeanCallPayoff
    from openinvestments.risk import VaRCalculator, ESCalculator
    from openinvestments.portfolio.optimization import PortfolioOptimizationManager
    from openinvestments.ml.models import EnsemblePricePredictor
    from openinvestments.data_quality.validator import DataQualityManager
    from openinvestments.monitoring.alerts import AlertManager, RiskAlertRules
    from openinvestments.core.market_data import SimulatedMarketDataFeed

    logger = get_logger(__name__)

except ImportError as e:
    print(f"❌ Error importing platform modules: {e}")
    print("Please ensure the platform is properly installed:")
    print("  pip install -r requirements.txt")
    print("  pip install -e .")
    sys.exit(1)


class PlatformDemonstrator:
    """Comprehensive platform demonstration class."""

    def __init__(self):
        self.logger = logger
        self.results = {}

    def print_header(self, title: str):
        """Print formatted section header."""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")

    def print_subheader(self, title: str):
        """Print formatted subsection header."""
        print(f"\n📍 {title}")
        print(f"{'─'*40}")

    def print_result(self, label: str, value: Any, unit: str = ""):
        """Print formatted result."""
        if isinstance(value, float):
            print(f"   {label}: {value:.4f}{unit}")
        else:
            print(f"   {label}: {value}{unit}")

    async def demo_option_pricing(self):
        """Demonstrate option pricing capabilities."""
        self.print_subheader("European Call Option Pricing")

        # Create option parameters
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

        # Monte Carlo pricing
        pricer = MonteCarloPricer()
        payoff = EuropeanCallPayoff(K)

        start_time = time.time()
        result = pricer.price_option(
            S0=S0, T=T, r=r, sigma=sigma, payoff=payoff, paths=50000
        )
        mc_time = time.time() - start_time

        self.print_result("Spot Price", S0, " USD")
        self.print_result("Strike Price", K, " USD")
        self.print_result("Time to Maturity", T, " years")
        self.print_result("Risk-free Rate", r, "%")
        self.print_result("Volatility", sigma, "%")
        self.print_result("Monte Carlo Price", result['price'], " USD")
        self.print_result("Standard Error", result['standard_error'], " USD")
        self.print_result("95% CI Lower", result['confidence_interval'][0], " USD")
        self.print_result("95% CI Upper", result['confidence_interval'][1], " USD")
        self.print_result("Computation Time", mc_time, " seconds")

        self.results['option_pricing'] = {
            'price': result['price'],
            'computation_time': mc_time
        }

    async def demo_risk_analysis(self):
        """Demonstrate risk analysis capabilities."""
        self.print_subheader("Portfolio Risk Analysis")

        # Generate sample portfolio returns
        np.random.seed(42)
        n_assets = 5
        n_periods = 1000

        # Simulate asset returns with correlations
        mu = np.array([0.08, 0.06, 0.10, 0.04, 0.12]) / 252  # Daily returns
        cov_matrix = np.array([
            [0.04, 0.02, 0.015, 0.01, 0.005],
            [0.02, 0.03, 0.012, 0.008, 0.003],
            [0.015, 0.012, 0.05, 0.006, 0.02],
            [0.01, 0.008, 0.006, 0.025, 0.002],
            [0.005, 0.003, 0.02, 0.002, 0.06]
        ]) / 252  # Daily covariance

        # Generate returns
        returns = np.random.multivariate_normal(mu, cov_matrix, n_periods)
        returns_df = pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(n_assets)])

        # Calculate portfolio VaR
        weights = np.ones(n_assets) / n_assets
        portfolio_returns = returns_df.dot(weights)

        var_calculator = VaRCalculator()
        var_result = var_calculator.calculate_var(portfolio_returns, confidence_level=0.95)

        es_calculator = ESCalculator()
        es_result = es_calculator.calculate_es(portfolio_returns, confidence_level=0.95)

        self.print_result("Portfolio Assets", n_assets)
        self.print_result("Historical Periods", n_periods)
        self.print_result("Portfolio VaR (95%)", var_result['var'], "%")
        self.print_result("Portfolio VaR Amount", var_result['var_amount'], " USD")
        self.print_result("Expected Shortfall (95%)", es_result['es'], "%")
        self.print_result("Portfolio Volatility", np.std(portfolio_returns) * np.sqrt(252), "%")

        self.results['risk_analysis'] = {
            'var_95': var_result['var'],
            'es_95': es_result['es'],
            'volatility': np.std(portfolio_returns) * np.sqrt(252)
        }

    async def demo_portfolio_optimization(self):
        """Demonstrate portfolio optimization capabilities."""
        self.print_subheader("Portfolio Optimization")

        # Generate sample data
        np.random.seed(42)
        n_assets = 4

        # Expected returns and covariance matrix
        expected_returns = np.array([0.08, 0.06, 0.10, 0.04])
        cov_matrix = np.array([
            [0.04, 0.02, 0.015, 0.01],
            [0.02, 0.03, 0.012, 0.008],
            [0.015, 0.012, 0.05, 0.006],
            [0.01, 0.008, 0.006, 0.025]
        ])

        # Optimize portfolio
        optimizer = PortfolioOptimizationManager()
        result = optimizer.optimize_portfolio(
            'mean_variance',
            expected_returns,
            cov_matrix
        )

        self.print_result("Optimization Method", "Mean-Variance")
        self.print_result("Expected Return", result.expected_return, "%")
        self.print_result("Portfolio Volatility", result.volatility, "%")
        self.print_result("Sharpe Ratio", result.sharpe_ratio)
        self.print_result("Optimization Status", result.optimization_status)

        print("\n   Optimized Weights:")
        for i, weight in enumerate(result.weights):
            print(f"      Asset {i+1}: {weight:.4f}")

        self.results['portfolio_optimization'] = {
            'expected_return': result.expected_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'weights': result.weights.tolist()
        }

    async def demo_machine_learning(self):
        """Demonstrate machine learning capabilities."""
        self.print_subheader("Machine Learning Price Prediction")

        # Generate synthetic price data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Create features (technical indicators)
        X = np.random.randn(n_samples, n_features)

        # Create target (next day return)
        noise = np.random.randn(n_samples) * 0.1
        y = X[:, 0] * 0.3 + X[:, 1] * 0.2 + X[:, 2] * 0.1 + noise

        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = EnsemblePricePredictor()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        performance = model.evaluate(X_test, y_test)

        self.print_result("Training Samples", len(X_train))
        self.print_result("Test Samples", len(X_test))
        self.print_result("Model Type", "Ensemble (RF + GB + XGB)")
        self.print_result("R² Score", performance.r2_score)
        self.print_result("RMSE", performance.rmse)
        self.print_result("MAE", performance.mae)
        self.print_result("MAPE", performance.mape, "%")

        self.results['machine_learning'] = {
            'r2_score': performance.r2_score,
            'rmse': performance.rmse,
            'mae': performance.mae,
            'mape': performance.mape
        }

    async def demo_data_quality(self):
        """Demonstrate data quality validation."""
        self.print_subheader("Data Quality Validation")

        # Create sample data with quality issues
        np.random.seed(42)
        n_samples = 1000

        # Generate base data
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        # Introduce quality issues
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.randint(1000, 10000, n_samples),
            'returns': np.random.randn(n_samples) * 0.02
        })

        # Add missing values
        missing_indices = np.random.choice(n_samples, size=50, replace=False)
        df.loc[missing_indices, 'volume'] = np.nan

        # Add outliers
        outlier_indices = np.random.choice(n_samples, size=10, replace=False)
        df.loc[outlier_indices, 'price'] = df.loc[outlier_indices, 'price'] * (1 + np.random.randn(10) * 0.5)

        # Validate data quality
        quality_manager = DataQualityManager()
        report = quality_manager.validate_data(df)

        self.print_result("Total Records", report.data_statistics['total_rows'])
        self.print_result("Data Quality Score", report.overall_score, "/100")
        self.print_result("Issues Found", report.total_issues)
        self.print_result("Missing Values", len([i for i in report.detailed_issues if i['type'].value == 'missing_values']))
        self.print_result("Outliers Detected", len([i for i in report.detailed_issues if i['type'].value == 'outliers']))

        self.results['data_quality'] = {
            'overall_score': report.overall_score,
            'total_issues': report.total_issues,
            'missing_values': len([i for i in report.detailed_issues if i['type'].value == 'missing_values']),
            'outliers': len([i for i in report.detailed_issues if i['type'].value == 'outliers'])
        }

    async def demo_market_data(self):
        """Demonstrate market data capabilities."""
        self.print_subheader("Real-time Market Data Simulation")

        # Initialize market data feed
        feed = SimulatedMarketDataFeed()

        # Connect and subscribe
        await feed.connect()
        await feed.subscribe(['AAPL', 'MSFT', 'GOOGL'])

        # Get current prices
        prices = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            data = await feed.get_current_price(symbol)
            if data:
                prices[symbol] = data.price

        # Get historical data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        hist_data = await feed.get_historical_data('AAPL', start_date, end_date, '1D')

        await feed.disconnect()

        self.print_result("Connected Symbols", len(prices))
        for symbol, price in prices.items():
            self.print_result(f"{symbol} Price", price, " USD")

        self.print_result("Historical Days", len(hist_data))
        self.print_result("Data Frequency", "Daily (1D)")

        self.results['market_data'] = {
            'connected_symbols': len(prices),
            'current_prices': prices,
            'historical_days': len(hist_data)
        }

    async def demo_alerting(self):
        """Demonstrate alerting system."""
        self.print_subheader("Automated Alerting System")

        # Set up alert manager
        alert_manager = AlertManager()

        # Add risk threshold rules
        var_rule = RiskAlertRules.var_breach_threshold(threshold=0.05, confidence_level=0.95)
        drawdown_rule = RiskAlertRules.drawdown_limit(drawdown_limit=0.10)

        alert_manager.add_rule(var_rule)
        alert_manager.add_rule(drawdown_rule)

        # Simulate risk data that triggers alerts
        risk_data = {'var': 0.08, 'portfolio_value': 1000000}  # VaR above threshold
        drawdown_data = {'drawdown': 0.12, 'portfolio_value': 1000000}  # Drawdown above limit

        # Check for alerts
        var_alert = await alert_manager.check_condition(var_rule, risk_data)
        drawdown_alert = await alert_manager.check_condition(drawdown_rule, drawdown_data)

        alerts_triggered = [a for a in [var_alert, drawdown_alert] if a is not None]

        self.print_result("Alert Rules Configured", 2)
        self.print_result("Alerts Triggered", len(alerts_triggered))

        if alerts_triggered:
            for i, alert in enumerate(alerts_triggered, 1):
                self.print_result(f"Alert {i} Type", alert.alert_type.value)
                self.print_result(f"Alert {i} Severity", alert.severity.value)
                self.print_result(f"Alert {i} Title", alert.title)

        self.results['alerting'] = {
            'rules_configured': 2,
            'alerts_triggered': len(alerts_triggered),
            'alert_types': [a.alert_type.value for a in alerts_triggered]
        }

    async def run_full_demo(self):
        """Run complete platform demonstration."""
        self.print_header("OpenInvestments Quantitative Risk Platform Demo")

        print("🚀 Demonstrating comprehensive quantitative finance capabilities...")
        print("This demo showcases the platform's core functionality including:")
        print("  • Option pricing and valuation")
        print("  • Risk analysis and portfolio management")
        print("  • Machine learning and predictive modeling")
        print("  • Data quality validation and cleansing")
        print("  • Real-time market data simulation")
        print("  • Automated alerting and monitoring")

        # Run all demonstrations
        await self.demo_option_pricing()
        await self.demo_risk_analysis()
        await self.demo_portfolio_optimization()
        await self.demo_machine_learning()
        await self.demo_data_quality()
        await self.demo_market_data()
        await self.demo_alerting()

        # Print summary
        self.print_header("Demo Summary")

        print("✅ All demonstrations completed successfully!")
        print("\n📊 Key Results:")

        if 'option_pricing' in self.results:
            print(".4f")

        if 'risk_analysis' in self.results:
            print(".2f")

        if 'portfolio_optimization' in self.results:
            print(".3f")

        if 'machine_learning' in self.results:
            print(".3f")

        if 'data_quality' in self.results:
            print(".1f")

        print(f"\n🎯 Platform Status: {'✅ FULLY OPERATIONAL' if len(self.results) >= 6 else '⚠️ PARTIAL FUNCTIONALITY'}")
        print(f"📈 Demonstrated Capabilities: {len(self.results)}/7 modules")

        # Performance metrics
        total_computation_time = sum([
            self.results.get('option_pricing', {}).get('computation_time', 0)
        ])

        if total_computation_time > 0:
            print(".2f")

        print("
💡 Next Steps:"        print("  1. Run 'python main.py --help' to explore CLI commands")
        print("  2. Start API server with 'python main.py api'")
        print("  3. Visit http://localhost:8000/docs for interactive API documentation")
        print("  4. Explore the comprehensive documentation in the docs/ directory")

        print("
🎉 OpenInvestments Platform is ready for production use!"        # Save results
        results_file = Path(__file__).parent.parent / "demo_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results,
                'summary': {
                    'modules_demonstrated': len(self.results),
                    'platform_status': 'operational' if len(self.results) >= 6 else 'partial',
                    'computation_time': total_computation_time
                }
            }, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

    async def run_quick_demo(self):
        """Run quick demonstration of core features."""
        self.print_header("Quick OpenInvestments Demo")

        print("⚡ Running accelerated demonstration...")

        # Quick option pricing
        await self.demo_option_pricing()

        # Quick risk analysis
        await self.demo_risk_analysis()

        print("
✅ Quick demo completed!"        print(".4f"        print(".2f"        print("
💡 Use --full flag for complete demonstration"    def save_demo_report(self):
        """Save demonstration report."""
        report = {
            "demo_timestamp": datetime.now().isoformat(),
            "platform_version": "1.0.0",
            "demonstrated_modules": list(self.results.keys()),
            "results_summary": self.results,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }

        report_path = Path(__file__).parent.parent / "demo_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report_path


async def main():
    """Main demonstration function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        demo = PlatformDemonstrator()
        await demo.run_full_demo()
    else:
        demo = PlatformDemonstrator()
        await demo.run_quick_demo()

    # Save report
    report_path = demo.save_demo_report()
    print(f"\n📄 Demo report saved to: {report_path}")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
