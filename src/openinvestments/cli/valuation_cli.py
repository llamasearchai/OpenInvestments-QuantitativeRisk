"""
Valuation CLI commands for option pricing and derivatives analysis.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import numpy as np
from typing import Optional

from ..core.config import config
from ..core.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def valuation_group():
    """Option valuation and derivatives pricing commands."""
    pass


@valuation_group.command()
@click.option('--s0', type=float, default=100.0, help='Initial stock price')
@click.option('--k', type=float, default=100.0, help='Strike price')
@click.option('--t', type=float, default=1.0, help='Time to maturity (years)')
@click.option('--r', type=float, default=0.05, help='Risk-free rate')
@click.option('--sigma', type=float, default=0.2, help='Volatility')
@click.option('--paths', type=int, default=10000, help='Number of Monte Carlo paths')
@click.option('--is-call', type=bool, default=True, help='Call option (True) or put (False)')
@click.option('--method', type=click.Choice(['mc', 'bs', 'tree']), default='mc',
              help='Pricing method: mc (Monte Carlo), bs (Black-Scholes), tree (Binomial tree)')
def price_option(s0, k, t, r, sigma, paths, is_call, method):
    """Price European options using various methods."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Pricing option...", total=1)

        try:
            if method == 'mc':
                # Monte Carlo pricing
                from ..valuation import MonteCarloPricer
                from ..valuation.monte_carlo import EuropeanCallPayoff, EuropeanPutPayoff

                pricer = MonteCarloPricer()
                payoff = EuropeanCallPayoff(k) if is_call else EuropeanPutPayoff(k)

                result = pricer.price_option(
                    S0=s0, T=t, r=r, sigma=sigma, payoff=payoff
                )

                progress.update(task, completed=1)

                # Display results
                table = Table(title="Monte Carlo Option Pricing Results")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Option Price", ".6f")
                table.add_row("Standard Error", ".6f")
                table.add_row("95% CI Lower", ".6f")
                table.add_row("95% CI Upper", ".6f")
                table.add_row("Simulated Paths", f"{paths:,}")
                table.add_row("Pricing Method", "Monte Carlo")

            elif method == 'bs':
                # Black-Scholes pricing (simplified implementation)
                from ..valuation.greeks import BlackScholesModel

                model = BlackScholesModel()
                params = {
                    'S': s0, 'K': k, 'T': t, 'r': r, 'sigma': sigma, 'is_call': is_call
                }

                price = model.price(params)
                progress.update(task, completed=1)

                table = Table(title="Black-Scholes Option Pricing Results")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Option Price", ".6f")
                table.add_row("Pricing Method", "Black-Scholes")

            elif method == 'tree':
                # Binomial tree pricing
                from ..valuation import BinomialTree
                from ..valuation.monte_carlo import EuropeanCallPayoff, EuropeanPutPayoff

                tree = BinomialTree()
                payoff_func = EuropeanCallPayoff(k) if is_call else EuropeanPutPayoff(k)

                result = tree.price_european(
                    S0=s0, K=k, T=t, r=r, sigma=sigma, payoff_func=payoff_func
                )

                progress.update(task, completed=1)

                table = Table(title="Binomial Tree Option Pricing Results")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Option Price", ".6f")
                table.add_row("Tree Steps", str(100))
                table.add_row("Pricing Method", "Binomial Tree")

            console.print(table)

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error pricing option: {e}[/red]")
            logger.error("Option pricing failed", error=str(e))


@valuation_group.command()
@click.option('--s0', type=float, default=100.0, help='Initial stock price')
@click.option('--k', type=float, default=100.0, help='Strike price')
@click.option('--t', type=float, default=1.0, help='Time to maturity (years)')
@click.option('--r', type=float, default=0.05, help='Risk-free rate')
@click.option('--sigma', type=float, default=0.2, help='Volatility')
@click.option('--is-call', type=bool, default=True, help='Call option (True) or put (False)')
@click.option('--greeks', multiple=True,
              type=click.Choice(['delta', 'gamma', 'vega', 'theta', 'rho']),
              default=['delta', 'gamma', 'vega', 'theta', 'rho'],
              help='Greeks to calculate')
def calculate_greeks(s0, k, t, r, sigma, is_call, greeks):
    """Calculate option Greeks."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Calculating Greeks...", total=1)

        try:
            from ..valuation.greeks import GreeksCalculator

            calculator = GreeksCalculator()
            params = {
                'S': s0, 'K': k, 'T': t, 'r': r, 'sigma': sigma, 'is_call': is_call
            }

            greeks_result = calculator.calculate_greeks(params, list(greeks))
            progress.update(task, completed=1)

            # Display results
            table = Table(title="Option Greeks Results")
            table.add_column("Greek", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            greek_descriptions = {
                'delta': 'Rate of change of option price w.r.t. underlying',
                'gamma': 'Rate of change of delta w.r.t. underlying',
                'vega': 'Rate of change of option price w.r.t. volatility',
                'theta': 'Rate of change of option price w.r.t. time',
                'rho': 'Rate of change of option price w.r.t. interest rate'
            }

            for greek, value in greeks_result.items():
                description = greek_descriptions.get(greek, '')
                table.add_row(greek.title(), ".6f", description)

            console.print(table)

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error calculating Greeks: {e}[/red]")
            logger.error("Greeks calculation failed", error=str(e))


@valuation_group.command()
@click.option('--s0', type=float, default=100.0, help='Initial stock price')
@click.option('--k', type=float, default=100.0, help='Strike price')
@click.option('--t', type=float, default=1.0, help='Time to maturity (years)')
@click.option('--r', type=float, default=0.05, help='Risk-free rate')
@click.option('--sigma', type=float, default=0.2, help='Volatility')
@click.option('--paths', type=int, default=10000, help='Number of Monte Carlo paths')
@click.option('--asian/--european', default=False, help='Asian or European option')
@click.option('--barrier', type=float, help='Barrier level for barrier options')
@click.option('--barrier-type', type=click.Choice(['up-and-out', 'up-and-in', 'down-and-out', 'down-and-in']),
              default='up-and-out', help='Barrier option type')
def price_exotic(s0, k, t, r, sigma, paths, asian, barrier, barrier_type):
    """Price exotic options (Asian, Barrier, etc.)."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Pricing exotic option...", total=1)

        try:
            from ..valuation import MonteCarloPricer

            if asian:
                from ..valuation.monte_carlo import AsianCallPayoff
                payoff = AsianCallPayoff(k)
                option_type = "Asian Call"
            elif barrier is not None:
                from ..valuation.monte_carlo import BarrierCallPayoff
                payoff = BarrierCallPayoff(k, barrier, barrier_type)
                option_type = f"Barrier Call ({barrier_type})"
            else:
                payoff = EuropeanCallPayoff(k)
                option_type = "European Call"

            result = pricer.price_option(
                S0=s0, T=t, r=r, sigma=sigma, payoff=payoff
            )

            progress.update(task, completed=1)

            # Display results
            table = Table(title=f"Exotic Option Pricing Results - {option_type}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Option Price", ".6f")
            table.add_row("Standard Error", ".6f")
            table.add_row("95% CI Lower", ".6f")
            table.add_row("95% CI Upper", ".6f")
            table.add_row("Simulated Paths", f"{paths:,}")
            table.add_row("Option Type", option_type)

            if barrier is not None:
                table.add_row("Barrier Level", ".2f")
                table.add_row("Barrier Type", barrier_type)

            console.print(table)

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error pricing exotic option: {e}[/red]")
            logger.error("Exotic option pricing failed", error=str(e))


@valuation_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with option parameters')
@click.option('--output-file', type=click.Path(), help='Output file for results')
@click.option('--method', type=click.Choice(['mc', 'bs', 'tree']), default='mc',
              help='Pricing method')
@click.option('--batch-size', type=int, default=100, help='Batch size for processing')
def batch_price(input_file, output_file, method, batch_size):
    """Price multiple options from CSV file."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Batch pricing options...", total=1)

        try:
            import pandas as pd

            # Read input file
            df = pd.read_csv(input_file)
            total_options = len(df)

            console.print(f"[green]Processing {total_options} options...[/green]")

            results = []

            # Process in batches
            for i in range(0, total_options, batch_size):
                batch = df.iloc[i:i+batch_size]

                for _, row in batch.iterrows():
                    try:
                        # Extract parameters
                        s0 = row.get('S0', 100.0)
                        k = row['K']
                        t = row['T']
                        r = row.get('r', 0.05)
                        sigma = row.get('sigma', 0.2)
                        is_call = row.get('is_call', True)

                        # Price option
                        if method == 'mc':
                            pricer = MonteCarloPricer()
                            payoff = EuropeanCallPayoff(k) if is_call else EuropeanPutPayoff(k)
                            result = pricer.price_option(S0=s0, T=t, r=r, sigma=sigma, payoff=payoff)
                            price = result['price']
                        elif method == 'bs':
                            from ..valuation.greeks import BlackScholesModel
                            model = BlackScholesModel()
                            params = {'S': s0, 'K': k, 'T': t, 'r': r, 'sigma': sigma, 'is_call': is_call}
                            price = model.price(params)
                        else:
                            price = 0.0  # Placeholder for tree method

                        results.append({
                            'K': k, 'T': t, 'price': price,
                            'S0': s0, 'r': r, 'sigma': sigma, 'is_call': is_call
                        })

                    except Exception as e:
                        console.print(f"[red]Error pricing option K={k}, T={t}: {e}[/red]")
                        results.append({
                            'K': k, 'T': t, 'price': None, 'error': str(e),
                            'S0': s0, 'r': r, 'sigma': sigma, 'is_call': is_call
                        })

            # Save results
            results_df = pd.DataFrame(results)

            if output_file:
                results_df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")
            else:
                # Display summary
                successful = results_df['price'].notna().sum()
                failed = total_options - successful

                table = Table(title="Batch Pricing Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Total Options", str(total_options))
                table.add_row("Successfully Priced", str(successful))
                table.add_row("Failed", str(failed))
                table.add_row("Success Rate", ".1%")

                console.print(table)

                if successful > 0:
                    avg_price = results_df['price'].mean()
                    console.print(".6f")

            progress.update(task, completed=1)

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error in batch pricing: {e}[/red]")
            logger.error("Batch pricing failed", error=str(e))


# Import missing classes for EuropeanPutPayoff
from ..valuation.monte_carlo import EuropeanPutPayoff
