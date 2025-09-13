"""
Risk analysis CLI commands for VaR, ES, and portfolio risk management.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import numpy as np
import pandas as pd
from typing import Optional, List

from ..risk import VaRCalculator, ESCalculator, RiskConfig, VaRMethod, ESDistribution
from ..core.config import config
from ..core.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def risk_group():
    """Risk analysis and portfolio risk management commands."""
    pass


@risk_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--confidence', type=float, default=0.95, help='Confidence level (0.90, 0.95, 0.99)')
@click.option('--method', type=click.Choice(['historical', 'parametric', 'monte_carlo', 'evt']),
              default='historical', help='VaR calculation method')
@click.option('--portfolio-value', type=float, default=1000000, help='Portfolio value in dollars')
@click.option('--horizon', type=int, default=1, help='Risk horizon in days')
@click.option('--output-file', type=click.Path(), help='Output file for VaR results')
def calculate_var(input_file, confidence, method, portfolio_value, horizon, output_file):
    """Calculate Value at Risk (VaR) for portfolio or single asset."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Calculating VaR...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            # Configure risk calculator
            risk_config = RiskConfig(
                confidence_level=confidence,
                portfolio_value=portfolio_value,
                horizon_days=horizon
            )

            calculator = VaRCalculator(risk_config)

            # Map method string to enum
            method_map = {
                'historical': VaRMethod.HISTORICAL,
                'parametric': VaRMethod.PARAMETRIC,
                'monte_carlo': VaRMethod.MONTE_CARLO,
                'evt': VaRMethod.EVT
            }

            # Calculate VaR
            if df.shape[1] > 1:
                # Multi-asset portfolio
                result = calculator.calculate_var(df, method_map[method])
            else:
                # Single asset
                returns = df.iloc[:, 0].values
                result = calculator.calculate_var(returns, method_map[method])

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Value at Risk (VaR) Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            table.add_row("VaR (%)", ".4f", "Value at Risk as percentage")
            table.add_row("VaR ($)", ".2f", "Value at Risk in dollars")
            table.add_row("Expected Shortfall (%)", ".4f", "Expected Shortfall (ES) as percentage")
            table.add_row("Confidence Level", ".1%", "Confidence level")
            table.add_row("Horizon", f"{horizon} days", "Risk measurement horizon")
            table.add_row("Method", method.title(), "Calculation method")

            if 'mean_return' in result:
                table.add_row("Mean Return", ".6f", "Average historical return")

            if 'volatility' in result:
                table.add_row("Volatility", ".6f", "Historical volatility")

            console.print(table)

            # Save results if requested
            if output_file:
                result_df = pd.DataFrame([result])
                result_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error calculating VaR: {e}[/red]")
            logger.error("VaR calculation failed", error=str(e))


@risk_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--confidence', type=float, default=0.95, help='Confidence level (0.90, 0.95, 0.99)')
@click.option('--method', type=click.Choice(['historical', 'parametric', 'student_t']),
              default='historical', help='ES calculation method')
@click.option('--portfolio-value', type=float, default=1000000, help='Portfolio value in dollars')
@click.option('--horizon', type=int, default=1, help='Risk horizon in days')
@click.option('--output-file', type=click.Path(), help='Output file for ES results')
def calculate_es(input_file, confidence, method, portfolio_value, horizon, output_file):
    """Calculate Expected Shortfall (ES) for portfolio or single asset."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Calculating Expected Shortfall...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            # Configure risk calculator
            risk_config = RiskConfig(
                confidence_level=confidence,
                portfolio_value=portfolio_value,
                horizon_days=horizon
            )

            calculator = ESCalculator(risk_config)

            # Map method string to enum
            method_map = {
                'historical': ESDistribution.HISTORICAL,
                'parametric': ESDistribution.NORMAL,
                'student_t': ESDistribution.STUDENT_T
            }

            # Calculate ES
            if df.shape[1] > 1:
                # Multi-asset portfolio
                result = calculator.calculate_es(df, method_map[method])
            else:
                # Single asset
                returns = df.iloc[:, 0].values
                result = calculator.calculate_es(returns, method_map[method])

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Expected Shortfall (ES) Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            table.add_row("Expected Shortfall (%)", ".4f", "Expected Shortfall as percentage")
            table.add_row("Expected Shortfall ($)", ".2f", "Expected Shortfall in dollars")
            table.add_row("VaR (%)", ".4f", "Corresponding Value at Risk")
            table.add_row("Confidence Level", ".1%", "Confidence level")
            table.add_row("Horizon", f"{horizon} days", "Risk measurement horizon")
            table.add_row("Method", method.title(), "Calculation method")

            if 'mean_return' in result:
                table.add_row("Mean Return", ".6f", "Average historical return")

            if 'volatility' in result:
                table.add_row("Volatility", ".6f", "Historical volatility")

            if 'degrees_of_freedom' in result:
                table.add_row("Degrees of Freedom", ".2f", "Student-t degrees of freedom")

            console.print(table)

            # Save results if requested
            if output_file:
                result_df = pd.DataFrame([result])
                result_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error calculating ES: {e}[/red]")
            logger.error("ES calculation failed", error=str(e))


@risk_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--confidences', multiple=True, type=float, default=[0.90, 0.95, 0.99, 0.995],
              help='Confidence levels for ES spectrum')
@click.option('--portfolio-value', type=float, default=1000000, help='Portfolio value in dollars')
@click.option('--horizon', type=int, default=1, help='Risk horizon in days')
@click.option('--output-file', type=click.Path(), help='Output file for ES spectrum results')
def es_spectrum(input_file, confidences, portfolio_value, horizon, output_file):
    """Calculate Expected Shortfall across multiple confidence levels."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Calculating ES spectrum...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            # Configure risk calculator
            risk_config = RiskConfig(
                portfolio_value=portfolio_value,
                horizon_days=horizon
            )

            calculator = ESCalculator(risk_config)

            # Calculate ES spectrum
            if df.shape[1] > 1:
                # Multi-asset portfolio
                result = calculator.calculate_es_spectrum(df, list(confidences))
            else:
                # Single asset
                returns = df.iloc[:, 0].values
                result = calculator.calculate_es_spectrum(returns, list(confidences))

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Expected Shortfall Spectrum")
            table.add_column("Confidence Level", style="cyan")
            table.add_column("ES (%)", style="green")
            table.add_column("VaR (%)", style="yellow")
            table.add_column("ES/VaR Ratio", style="blue")

            for i, conf in enumerate(result['confidence_levels']):
                es_pct = result['es_values'][i]
                var_pct = result['var_values'][i]
                ratio = es_pct / var_pct if var_pct != 0 else 0

                table.add_row(".1%", ".4f", ".4f", ".3f")

            console.print(table)

            # Additional summary statistics
            summary_table = Table(title="ES Spectrum Summary")
            summary_table.add_column("Statistic", style="cyan")
            summary_table.add_column("Value", style="green")

            es_values = np.array(result['es_values'])
            summary_table.add_row("Average ES", ".4f")
            summary_table.add_row("Min ES", ".4f")
            summary_table.add_row("Max ES", ".4f")
            summary_table.add_row("ES Range", ".4f")

            console.print("\n", summary_table)

            # Save results if requested
            if output_file:
                spectrum_df = pd.DataFrame({
                    'confidence_level': result['confidence_levels'],
                    'es_value': result['es_values'],
                    'var_value': result['var_values']
                })
                spectrum_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error calculating ES spectrum: {e}[/red]")
            logger.error("ES spectrum calculation failed", error=str(e))


@risk_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--portfolio-value', type=float, default=1000000, help='Portfolio value in dollars')
@click.option('--rebalance-frequency', type=click.Choice(['daily', 'weekly', 'monthly']),
              default='monthly', help='Portfolio rebalancing frequency')
@click.option('--output-file', type=click.Path(), help='Output file for risk decomposition results')
def risk_decomposition(input_file, portfolio_value, rebalance_frequency, output_file):
    """Perform risk decomposition analysis for multi-asset portfolio."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Performing risk decomposition...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            if df.shape[1] < 2:
                console.print("[red]Error: Need at least 2 assets for risk decomposition[/red]")
                return

            # Configure risk calculator
            risk_config = RiskConfig(portfolio_value=portfolio_value)
            calculator = VaRCalculator(risk_config)

            # Calculate portfolio VaR
            portfolio_var = calculator.calculate_var(df, VaRMethod.HISTORICAL)

            # Calculate individual asset contributions
            asset_contributions = {}
            total_var = portfolio_var['var']

            for asset in df.columns:
                # Calculate VaR contribution (simplified - using marginal VaR)
                asset_returns = df[asset].values
                asset_var = calculator.calculate_var(asset_returns, VaRMethod.HISTORICAL)

                # Simplified marginal contribution
                correlation = df.corr().loc[asset].mean()
                weight = 1.0 / df.shape[1]  # Equal weight assumption

                contribution = weight * asset_var['var'] * correlation
                asset_contributions[asset] = contribution

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Portfolio Risk Decomposition")
            table.add_column("Asset", style="cyan")
            table.add_column("VaR Contribution (%)", style="green")
            table.add_column("VaR Contribution ($)", style="yellow")
            table.add_column("Weight (%)", style="blue")

            total_contribution = sum(asset_contributions.values())

            for asset, contribution in asset_contributions.items():
                weight_pct = (1.0 / df.shape[1]) * 100
                contribution_pct = (contribution / total_var) * 100 if total_var != 0 else 0
                contribution_dollar = contribution * portfolio_value

                table.add_row(
                    asset,
                    ".2f",
                    ".2f",
                    ".1f"
                )

            # Summary row
            table.add_row(
                "[bold]Total Portfolio[/bold]",
                "[bold]100.00[/bold]",
                "[bold]0.2f[/bold]",
                "[bold]100.0[/bold]"
            )

            console.print(table)

            # Risk concentration analysis
            max_contribution = max(asset_contributions.values())
            concentration_ratio = max_contribution / total_var if total_var != 0 else 0

            risk_analysis = Panel(
                f"[bold]Risk Concentration Analysis[/bold]\n\n"
                f"Largest single asset contribution: [red]{concentration_ratio:.1%}[/red] of total VaR\n"
                f"Risk diversification ratio: [green]{1-concentration_ratio:.1%}[/green]\n\n"
                f"[dim]Note: High concentration indicates portfolio is sensitive to individual asset movements[/dim]",
                title="Risk Analysis",
                border_style="blue"
            )

            console.print("\n", risk_analysis)

            # Save results if requested
            if output_file:
                decomposition_df = pd.DataFrame({
                    'asset': list(asset_contributions.keys()),
                    'var_contribution': list(asset_contributions.values()),
                    'weight': [1.0/df.shape[1]] * df.shape[1]
                })
                decomposition_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error in risk decomposition: {e}[/red]")
            logger.error("Risk decomposition failed", error=str(e))


@risk_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with historical returns data')
@click.option('--stress-scenarios', type=int, default=100, help='Number of stress scenarios to generate')
@click.option('--severity-levels', multiple=True, type=float, default=[0.90, 0.95, 0.99],
              help='Severity levels for stress scenarios')
@click.option('--output-file', type=click.Path(), help='Output file for stress test results')
def stress_test(input_file, stress_scenarios, severity_levels, output_file):
    """Perform stress testing on portfolio."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running stress tests...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            # Generate stress scenarios using historical simulation
            stress_results = {}

            for severity in severity_levels:
                # Find historical worst-case scenarios
                portfolio_returns = df.mean(axis=1).values  # Simplified portfolio return

                # Sort returns to find worst cases
                sorted_returns = np.sort(portfolio_returns)
                quantile_index = int((1 - severity) * len(sorted_returns))

                # Generate stress scenarios by resampling from tail
                tail_returns = sorted_returns[:quantile_index]
                stress_scenarios_data = np.random.choice(tail_returns, stress_scenarios, replace=True)

                # Calculate stress test metrics
                avg_stress_loss = -np.mean(stress_scenarios_data)
                max_stress_loss = -np.min(stress_scenarios_data)
                var_stress = -np.percentile(stress_scenarios_data, (1-severity)*100)

                stress_results[severity] = {
                    'average_loss': avg_stress_loss,
                    'maximum_loss': max_stress_loss,
                    'var_loss': var_stress,
                    'scenarios_generated': stress_scenarios
                }

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Portfolio Stress Test Results")
            table.add_column("Severity Level", style="cyan")
            table.add_column("Average Loss (%)", style="red")
            table.add_column("Maximum Loss (%)", style="red", style="bold red")
            table.add_column("VaR Loss (%)", style="yellow")
            table.add_column("Scenarios", style="green")

            for severity, results in stress_results.items():
                table.add_row(
                    ".1%",
                    ".4f",
                    ".4f",
                    ".4f",
                    str(results['scenarios_generated'])
                )

            console.print(table)

            # Stress test summary
            worst_case = max([r['maximum_loss'] for r in stress_results.values()])

            summary_panel = Panel(
                f"[bold]Stress Test Summary[/bold]\n\n"
                f"Worst-case scenario loss: [bold red]{worst_case:.2%}[/bold red]\n"
                f"Stress scenarios generated: [green]{stress_scenarios}[/green]\n"
                f"Severity levels tested: [cyan]{', '.join([f'{s:.1%}' for s in severity_levels])}[/cyan]\n\n"
                f"[dim]Note: Results based on historical simulation from available data[/dim]",
                title="Summary",
                border_style="yellow"
            )

            console.print("\n", summary_panel)

            # Save results if requested
            if output_file:
                stress_df = pd.DataFrame([
                    {
                        'severity_level': severity,
                        'average_loss': results['average_loss'],
                        'maximum_loss': results['maximum_loss'],
                        'var_loss': results['var_loss'],
                        'scenarios': results['scenarios_generated']
                    }
                    for severity, results in stress_results.items()
                ])
                stress_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error in stress testing: {e}[/red]")
            logger.error("Stress testing failed", error=str(e))
