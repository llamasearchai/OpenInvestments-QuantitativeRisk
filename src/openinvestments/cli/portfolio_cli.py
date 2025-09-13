"""
Portfolio management CLI commands.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import numpy as np
import pandas as pd
from typing import Optional

from ..core.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def portfolio_group():
    """Portfolio management and optimization commands."""
    pass


@portfolio_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--output-file', type=click.Path(), help='Output file for portfolio statistics')
@click.option('--risk-free-rate', type=float, default=0.02, help='Risk-free rate for Sharpe ratio')
def analyze_portfolio(input_file, output_file, risk_free_rate):
    """Analyze portfolio performance and risk characteristics."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing portfolio...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            if df.shape[1] < 2:
                console.print("[red]Error: Need at least 2 assets for portfolio analysis[/red]")
                return

            # Calculate portfolio statistics
            returns = df.values
            portfolio_returns = np.mean(returns, axis=1)

            # Basic statistics
            mean_return = np.mean(portfolio_returns)
            volatility = np.std(portfolio_returns)
            skewness = pd.Series(portfolio_returns).skew()
            kurtosis = pd.Series(portfolio_returns).kurtosis()

            # Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate / 252) / volatility * np.sqrt(252)

            # Maximum drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Value at Risk (simple historical)
            confidence_levels = [0.90, 0.95, 0.99]
            var_values = {}
            for conf in confidence_levels:
                var_values[conf] = -np.percentile(portfolio_returns, (1 - conf) * 100)

            # Correlation matrix
            corr_matrix = df.corr()

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Portfolio Analysis Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            table.add_row("Mean Daily Return", ".6f", "Average daily return")
            table.add_row("Annual Volatility", ".4f", "Annualized volatility")
            table.add_row("Sharpe Ratio", ".4f", "Risk-adjusted return measure")
            table.add_row("Maximum Drawdown", ".4f", "Largest peak-to-trough decline")
            table.add_row("Skewness", ".4f", "Return distribution asymmetry")
            table.add_row("Kurtosis", ".4f", "Return distribution tail thickness")

            for conf, var in var_values.items():
                table.add_row(".1%", ".4f", f"Historical VaR at {conf:.1%} confidence")

            console.print(table)

            # Correlation heatmap summary
            console.print("\n[bold]Correlation Matrix Summary:[/bold]")
            corr_table = Table(title="Asset Correlations")
            corr_table.add_column("Asset Pair", style="cyan")
            corr_table.add_column("Correlation", style="green")

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    asset1 = corr_matrix.columns[i]
                    asset2 = corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    corr_table.add_row(f"{asset1} vs {asset2}", ".4f")

            console.print(corr_table)

            # Risk assessment
            diversification_ratio = 1 / np.sqrt(np.sum(corr_matrix ** 2).sum() / len(corr_matrix))
            risk_assessment = Panel(
                f"[bold]Portfolio Risk Assessment[/bold]\n\n"
                f"Diversification Ratio: [green]{diversification_ratio:.3f}[/green]\n"
                f"Portfolio Volatility: [yellow]{volatility:.1%}[/yellow]\n"
                f"VaR (95%): [red]{var_values[0.95]:.1%}[/red]\n\n"
                f"[dim]Higher diversification ratio indicates better risk spreading[/dim]",
                title="Risk Summary",
                border_style="blue"
            )

            console.print("\n", risk_assessment)

            # Save results if requested
            if output_file:
                results = {
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'diversification_ratio': diversification_ratio,
                    'var_90': var_values[0.90],
                    'var_95': var_values[0.95],
                    'var_99': var_values[0.99]
                }

                results_df = pd.DataFrame([results])
                results_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error analyzing portfolio: {e}[/red]")
            logger.error("Portfolio analysis failed", error=str(e))


@portfolio_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--target-return', type=float, help='Target portfolio return')
@click.option('--target-volatility', type=float, help='Target portfolio volatility')
@click.option('--method', type=click.Choice(['equal_weight', 'risk_parity', 'min_variance']),
              default='equal_weight', help='Portfolio optimization method')
@click.option('--output-file', type=click.Path(), help='Output file for optimized weights')
def optimize_portfolio(input_file, target_return, target_volatility, method, output_file):
    """Optimize portfolio weights using various methods."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing portfolio...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            if df.shape[1] < 2:
                console.print("[red]Error: Need at least 2 assets for portfolio optimization[/red]")
                return

            returns = df.values
            n_assets = returns.shape[1]
            asset_names = df.columns.tolist()

            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(returns, axis=0)
            cov_matrix = np.cov(returns.T)

            weights = None

            if method == 'equal_weight':
                # Equal weight portfolio
                weights = np.ones(n_assets) / n_assets

            elif method == 'risk_parity':
                # Risk parity - equal risk contribution
                volatilities = np.sqrt(np.diag(cov_matrix))

                # Inverse volatility weighting
                inv_vol_weights = 1 / volatilities
                weights = inv_vol_weights / np.sum(inv_vol_weights)

            elif method == 'min_variance':
                # Minimum variance portfolio
                ones = np.ones(n_assets)
                weights = np.linalg.solve(cov_matrix, ones)
                weights = weights / np.sum(weights)

            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Individual asset contributions to risk
            marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
            risk_contributions = weights * marginal_contributions

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Portfolio Optimization Results")
            table.add_column("Asset", style="cyan")
            table.add_column("Weight", style="green")
            table.add_column("Expected Return", style="yellow")
            table.add_column("Volatility", style="red")
            table.add_column("Risk Contribution", style="blue")

            for i, asset in enumerate(asset_names):
                table.add_row(
                    asset,
                    ".4f",
                    ".6f",
                    ".6f",
                    ".4f"
                )

            console.print(table)

            # Portfolio summary
            summary_table = Table(title="Portfolio Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")

            summary_table.add_row("Optimization Method", method.replace('_', ' ').title())
            summary_table.add_row("Expected Portfolio Return", ".6f")
            summary_table.add_row("Portfolio Volatility", ".6f")
            summary_table.add_row("Sharpe Ratio (RF=2%)", ".4f")

            console.print("\n", summary_table)

            # Risk concentration analysis
            max_risk_contribution = np.max(risk_contributions)
            risk_concentration = max_risk_contribution / portfolio_volatility

            if risk_concentration > 0.3:
                risk_level = "[red]High[/red]"
            elif risk_concentration > 0.2:
                risk_level = "[yellow]Medium[/yellow]"
            else:
                risk_level = "[green]Low[/green]"

            console.print(f"\n[bold]Risk Concentration:[/bold] {risk_level}")
            console.print(f"Maximum risk contribution: {max_risk_contribution:.1%}")
            console.print(f"Risk concentration ratio: {risk_concentration:.1%}")

            # Save results if requested
            if output_file:
                results_df = pd.DataFrame({
                    'asset': asset_names,
                    'weight': weights,
                    'expected_return': expected_returns,
                    'volatility': np.sqrt(np.diag(cov_matrix)),
                    'risk_contribution': risk_contributions
                })

                # Add portfolio summary as additional row
                summary_row = pd.DataFrame({
                    'asset': ['PORTFOLIO'],
                    'weight': [1.0],
                    'expected_return': [portfolio_return],
                    'volatility': [portfolio_volatility],
                    'risk_contribution': [portfolio_volatility]
                })

                final_df = pd.concat([results_df, summary_row], ignore_index=True)
                final_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error optimizing portfolio: {e}[/red]")
            logger.error("Portfolio optimization failed", error=str(e))


@portfolio_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--rebalance-frequency', type=click.Choice(['daily', 'weekly', 'monthly', 'quarterly']),
              default='monthly', help='Rebalancing frequency')
@click.option('--initial-investment', type=float, default=100000, help='Initial investment amount')
@click.option('--output-file', type=click.Path(), help='Output file for backtest results')
def backtest_portfolio(input_file, rebalance_frequency, initial_investment, output_file):
    """Backtest portfolio performance with periodic rebalancing."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Backtesting portfolio...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            if df.shape[1] < 2:
                console.print("[red]Error: Need at least 2 assets for backtesting[/red]")
                return

            returns = df.values
            n_periods, n_assets = returns.shape
            asset_names = df.columns.tolist()

            # Simple equal-weight strategy
            weights = np.ones(n_assets) / n_assets

            # Simulate portfolio returns
            portfolio_returns = np.zeros(n_periods)
            portfolio_values = np.zeros(n_periods + 1)
            portfolio_values[0] = initial_investment

            # Asset values (assuming equal initial allocation)
            asset_values = np.ones((n_periods + 1, n_assets)) * (initial_investment / n_assets)

            for t in range(n_periods):
                # Calculate portfolio return for this period
                period_return = np.sum(weights * returns[t])
                portfolio_returns[t] = period_return

                # Update portfolio value
                portfolio_values[t + 1] = portfolio_values[t] * (1 + period_return)

                # Rebalance to target weights (simplified - assuming daily rebalancing for now)
                if rebalance_frequency == 'daily' or (rebalance_frequency == 'monthly' and (t + 1) % 21 == 0):
                    for i in range(n_assets):
                        asset_values[t + 1, i] = portfolio_values[t + 1] * weights[i]
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

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Portfolio Backtest Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            table.add_row("Initial Investment", ".0f", "Starting portfolio value")
            table.add_row("Final Value", ".0f", "Ending portfolio value")
            table.add_row("Total Return", ".4f", "Total return over period")
            table.add_row("Annualized Return", ".4f", "Annualized return")
            table.add_row("Annualized Volatility", ".4f", "Annualized volatility")
            table.add_row("Sharpe Ratio", ".4f", "Risk-adjusted return")
            table.add_row("Maximum Drawdown", ".4f", "Largest peak-to-trough decline")
            table.add_row("Rebalancing Frequency", rebalance_frequency, "Portfolio rebalancing schedule")

            console.print(table)

            # Performance summary
            if total_return > 0:
                performance_color = "green"
                performance_status = "Positive"
            else:
                performance_color = "red"
                performance_status = "Negative"

            performance_panel = Panel(
                f"[bold]Performance Summary[/bold]\n\n"
                f"Status: [{performance_color}]{performance_status}[/{performance_color}]\n"
                f"Total Return: [{performance_color}]{total_return:.1%}[/{performance_color}]\n"
                f"Annualized Return: [{performance_color}]{annualized_return:.1%}[/{performance_color}]\n"
                f"Best Period: [green]{np.max(portfolio_returns):.1%}[/green]\n"
                f"Worst Period: [red]{np.min(portfolio_returns):.1%}[/red]\n\n"
                f"[dim]Backtest based on historical data with {rebalance_frequency} rebalancing[/dim]",
                title="Backtest Summary",
                border_style="blue"
            )

            console.print("\n", performance_panel)

            # Save results if requested
            if output_file:
                backtest_df = pd.DataFrame({
                    'period': range(1, n_periods + 1),
                    'portfolio_return': portfolio_returns,
                    'portfolio_value': portfolio_values[1:],
                    'drawdown': drawdown
                })

                # Add summary statistics
                summary_df = pd.DataFrame({
                    'metric': ['total_return', 'annualized_return', 'annualized_volatility',
                              'sharpe_ratio', 'max_drawdown'],
                    'value': [total_return, annualized_return, annualized_volatility,
                             sharpe_ratio, max_drawdown]
                })

                with pd.ExcelWriter(output_file) as writer:
                    backtest_df.to_excel(writer, sheet_name='Returns', index=False)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error backtesting portfolio: {e}[/red]")
            logger.error("Portfolio backtest failed", error=str(e))


@portfolio_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--num-simulations', type=int, default=1000, help='Number of Monte Carlo simulations')
@click.option('--time-horizon', type=int, default=252, help='Time horizon in days')
@click.option('--confidence-level', type=float, default=0.95, help='Confidence level for projections')
@click.option('--output-file', type=click.Path(), help='Output file for simulation results')
def monte_carlo_simulation(input_file, num_simulations, time_horizon, confidence_level, output_file):
    """Run Monte Carlo simulation for portfolio projection."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running Monte Carlo simulation...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)
            returns = df.values

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
            simulated_portfolio_values[:, 0] = 1.0  # Start with $1

            np.random.seed(42)  # For reproducibility

            for sim in range(num_simulations):
                for t in range(1, time_horizon + 1):
                    # Generate random return from normal distribution
                    shock = np.random.multivariate_normal(mu, cov)
                    portfolio_return = np.sum(weights * shock)

                    simulated_portfolio_values[sim, t] = \
                        simulated_portfolio_values[sim, t-1] * (1 + portfolio_return)

            # Calculate statistics
            final_values = simulated_portfolio_values[:, -1]
            mean_final_value = np.mean(final_values)
            median_final_value = np.median(final_values)

            # Confidence intervals
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100

            ci_lower = np.percentile(final_values, lower_percentile)
            ci_upper = np.percentile(final_values, upper_percentile)

            # Probability of loss
            prob_loss = np.mean(final_values < 1.0)
            prob_gain = np.mean(final_values > 1.0)

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Monte Carlo Simulation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            table.add_row("Number of Simulations", f"{num_simulations:,}", "Total Monte Carlo paths")
            table.add_row("Time Horizon", f"{time_horizon} days", "Simulation period")
            table.add_row("Mean Final Value", ".4f", "Average portfolio value at end")
            table.add_row("Median Final Value", ".4f", "Median portfolio value at end")
            table.add_row(".1%", ".4f", f"Lower {confidence_level:.1%} confidence bound")
            table.add_row(".1%", ".4f", f"Upper {confidence_level:.1%} confidence bound")
            table.add_row("Probability of Loss", ".1%", "Chance of ending below starting value")
            table.add_row("Probability of Gain", ".1%", "Chance of ending above starting value")

            console.print(table)

            # Risk analysis
            worst_case = np.min(final_values)
            best_case = np.max(final_values)
            var_95 = np.percentile(final_values, 5)  # 95% VaR

            risk_table = Table(title="Risk Analysis")
            risk_table.add_column("Risk Metric", style="cyan")
            risk_table.add_column("Value", style="red")

            risk_table.add_row("Best Case Outcome", ".4f")
            risk_table.add_row("Worst Case Outcome", ".4f")
            risk_table.add_row("VaR (95%)", ".4f")
            risk_table.add_row("Expected Shortfall", ".4f")

            console.print("\n", risk_table)

            # Distribution analysis
            returns_distribution = (final_values - 1) * 100  # Convert to percentage

            dist_table = Table(title="Return Distribution")
            dist_table.add_column("Percentile", style="cyan")
            dist_table.add_column("Return (%)", style="green")

            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                dist_table.add_row(".0f", ".2f")

            console.print("\n", dist_table)

            # Save results if requested
            if output_file:
                # Save simulation paths (sample of them to avoid huge files)
                sample_size = min(100, num_simulations)
                sample_indices = np.random.choice(num_simulations, sample_size, replace=False)

                simulation_df = pd.DataFrame()
                for i, idx in enumerate(sample_indices):
                    simulation_df[f'path_{i+1}'] = simulated_portfolio_values[idx]

                simulation_df.to_csv(output_file, index=False)

                # Also save summary statistics
                summary_file = output_file.replace('.csv', '_summary.csv')
                summary_df = pd.DataFrame({
                    'metric': ['mean_final_value', 'median_final_value', 'ci_lower', 'ci_upper',
                              'prob_loss', 'prob_gain', 'worst_case', 'best_case', 'var_95'],
                    'value': [mean_final_value, median_final_value, ci_lower, ci_upper,
                             prob_loss, prob_gain, worst_case, best_case, var_95]
                })
                summary_df.to_csv(summary_file, index=False)

                console.print(f"[green]Simulation results saved to {output_file}[/green]")
                console.print(f"[green]Summary statistics saved to {summary_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error running Monte Carlo simulation: {e}[/red]")
            logger.error("Monte Carlo simulation failed", error=str(e))
