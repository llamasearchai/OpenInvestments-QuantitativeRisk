"""
Model validation CLI commands.
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
def validation_group():
    """Model validation and testing commands."""
    pass


@validation_group.command()
@click.option('--model-file', type=click.Path(exists=True), required=True,
              help='File containing model predictions')
@click.option('--actual-file', type=click.Path(exists=True), required=True,
              help='File containing actual values')
@click.option('--output-file', type=click.Path(), help='Output file for validation results')
@click.option('--confidence-level', type=float, default=0.95, help='Confidence level for tests')
def validate_model(model_file, actual_file, output_file, confidence_level):
    """Validate model predictions against actual values."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Validating model...", total=1)

        try:
            # Load data
            model_df = pd.read_csv(model_file)
            actual_df = pd.read_csv(actual_file)

            if len(model_df) != len(actual_df):
                console.print("[red]Error: Model and actual data must have the same length[/red]")
                return

            # Extract prediction and actual columns (assuming first column)
            predictions = model_df.iloc[:, 0].values
            actuals = actual_df.iloc[:, 0].values

            # Calculate validation metrics
            errors = predictions - actuals
            abs_errors = np.abs(errors)

            # Basic metrics
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            mae = np.mean(abs_errors)
            mape = np.mean(abs_errors / np.abs(actuals)) * 100

            # Statistical tests
            mean_error = np.mean(errors)
            std_error = np.std(errors, ddof=1)

            # Diebold-Mariano test (simplified)
            dm_stat = mean_error / (std_error / np.sqrt(len(errors)))

            # Coverage tests
            z_score = 1.96  # 95% confidence
            ci_lower = mean_error - z_score * std_error
            ci_upper = mean_error + z_score * std_error

            # Model accuracy metrics
            accuracy_1pct = np.mean(abs_errors / np.abs(actuals) <= 0.01) * 100
            accuracy_5pct = np.mean(abs_errors / np.abs(actuals) <= 0.05) * 100

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Model Validation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description")

            table.add_row("Mean Squared Error", ".6f", "Average squared prediction error")
            table.add_row("Root Mean Squared Error", ".6f", "Square root of MSE")
            table.add_row("Mean Absolute Error", ".6f", "Average absolute prediction error")
            table.add_row("Mean Absolute Percentage Error", ".2f", "MAPE as percentage")
            table.add_row("Mean Prediction Error", ".6f", "Average prediction bias")
            table.add_row("Prediction Std Dev", ".6f", "Standard deviation of errors")
            table.add_row("Diebold-Mariano Statistic", ".4f", "Test for predictive accuracy")

            table.add_row("Accuracy (1%)", ".1f", "Predictions within 1% of actual")
            table.add_row("Accuracy (5%)", ".1f", "Predictions within 5% of actual")

            console.print(table)

            # Statistical significance
            if abs(dm_stat) > 1.96:
                significance = "[red]Statistically Significant[/red]"
            else:
                significance = "[green]Not Statistically Significant[/green]"

            # Model assessment
            if mape < 5:
                model_quality = "[green]Excellent[/green]"
            elif mape < 10:
                model_quality = "[yellow]Good[/yellow]"
            elif mape < 20:
                model_quality = "[orange]Fair[/orange]"
            else:
                model_quality = "[red]Poor[/red]"

            assessment_panel = Panel(
                f"[bold]Model Assessment[/bold]\n\n"
                f"Overall Quality: {model_quality}\n"
                f"MAPE: {mape:.2f}%\n"
                f"Bias Significance: {significance}\n\n"
                f"[dim]Model validation based on {len(predictions):,} predictions[/dim]",
                title="Validation Summary",
                border_style="blue"
            )

            console.print("\n", assessment_panel)

            # Error distribution analysis
            error_table = Table(title="Error Distribution")
            error_table.add_column("Statistic", style="cyan")
            error_table.add_column("Value", style="green")

            error_table.add_row("Minimum Error", ".6f")
            error_table.add_row("Maximum Error", ".6f")
            error_table.add_row("Error Skewness", ".4f")
            error_table.add_row("Error Kurtosis", ".4f")

            console.print("\n", error_table)

            # Save results if requested
            if output_file:
                validation_results = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'mean_error': mean_error,
                    'std_error': std_error,
                    'dm_statistic': dm_stat,
                    'accuracy_1pct': accuracy_1pct,
                    'accuracy_5pct': accuracy_5pct,
                    'min_error': np.min(errors),
                    'max_error': np.max(errors),
                    'error_skewness': pd.Series(errors).skew(),
                    'error_kurtosis': pd.Series(errors).kurtosis(),
                    'n_predictions': len(predictions)
                }

                results_df = pd.DataFrame([validation_results])
                results_df.to_csv(output_file, index=False)
                console.print(f"[green]Validation results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error validating model: {e}[/red]")
            logger.error("Model validation failed", error=str(e))


@validation_group.command()
@click.option('--model-file', type=click.Path(exists=True), required=True,
              help='File containing model predictions')
@click.option('--benchmark-file', type=click.Path(exists=True), required=True,
              help='File containing benchmark predictions')
@click.option('--actual-file', type=click.Path(exists=True), required=True,
              help='File containing actual values')
@click.option('--output-file', type=click.Path(), help='Output file for comparison results')
def compare_models(model_file, benchmark_file, actual_file, output_file):
    """Compare two models against a benchmark."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Comparing models...", total=1)

        try:
            # Load data
            model_df = pd.read_csv(model_file)
            benchmark_df = pd.read_csv(benchmark_file)
            actual_df = pd.read_csv(actual_file)

            if not (len(model_df) == len(benchmark_df) == len(actual_df)):
                console.print("[red]Error: All files must have the same length[/red]")
                return

            # Extract predictions and actuals
            model_pred = model_df.iloc[:, 0].values
            benchmark_pred = benchmark_df.iloc[:, 0].values
            actuals = actual_df.iloc[:, 0].values

            # Calculate errors for both models
            model_errors = model_pred - actuals
            benchmark_errors = benchmark_pred - actuals

            # Calculate metrics for both models
            def calculate_metrics(errors, predictions):
                mse = np.mean(errors**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(errors))
                mape = np.mean(np.abs(errors) / np.abs(actuals)) * 100
                return mse, rmse, mae, mape

            model_mse, model_rmse, model_mae, model_mape = calculate_metrics(model_errors, model_pred)
            bench_mse, bench_rmse, bench_mae, bench_mape = calculate_metrics(benchmark_errors, benchmark_pred)

            # Diebold-Mariano test for predictive accuracy comparison
            error_diff = model_errors**2 - benchmark_errors**2
            dm_stat = np.mean(error_diff) / (np.std(error_diff, ddof=1) / np.sqrt(len(error_diff)))

            # Model comparison metrics
            mse_improvement = (bench_mse - model_mse) / bench_mse * 100
            rmse_improvement = (bench_rmse - model_rmse) / bench_rmse * 100
            mae_improvement = (bench_mae - model_mae) / bench_mae * 100

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Model Comparison Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Model", style="green")
            table.add_column("Benchmark", style="yellow")
            table.add_column("Improvement (%)", style="blue")

            table.add_row("MSE", ".6f", ".6f", ".2f")
            table.add_row("RMSE", ".6f", ".6f", ".2f")
            table.add_row("MAE", ".6f", ".6f", ".2f")
            table.add_row("MAPE (%)", ".2f", ".2f", ".2f")

            console.print(table)

            # Statistical significance
            if abs(dm_stat) > 1.96:
                dm_result = "[green]Statistically Significant[/green]"
                if dm_stat < 0:
                    winner = "Model outperforms benchmark"
                else:
                    winner = "Benchmark outperforms model"
            else:
                dm_result = "[yellow]Not Statistically Significant[/yellow]"
                winner = "No significant difference"

            comparison_panel = Panel(
                f"[bold]Model Comparison Summary[/bold]\n\n"
                f"Diebold-Mariano Test: {dm_result}\n"
                f"DM Statistic: {dm_stat:.4f}\n"
                f"Winner: {winner}\n\n"
                f"[dim]Comparison based on {len(model_pred):,} predictions[/dim]",
                title="Statistical Comparison",
                border_style="blue"
            )

            console.print("\n", comparison_panel)

            # Performance assessment
            if model_mape < bench_mape:
                model_status = "[green]Better[/green]"
                bench_status = "[red]Worse[/red]"
            else:
                model_status = "[red]Worse[/red]"
                bench_status = "[green]Better[/green]"

            assessment_table = Table(title="Performance Assessment")
            assessment_table.add_column("Model", style="cyan")
            assessment_table.add_column("MAPE (%)", style="green")
            assessment_table.add_column("Status", style="yellow")

            assessment_table.add_row("Primary Model", ".2f", model_status)
            assessment_table.add_row("Benchmark", ".2f", bench_status)

            console.print("\n", assessment_table)

            # Save results if requested
            if output_file:
                comparison_results = {
                    'model_mse': model_mse,
                    'model_rmse': model_rmse,
                    'model_mae': model_mae,
                    'model_mape': model_mape,
                    'benchmark_mse': bench_mse,
                    'benchmark_rmse': bench_rmse,
                    'benchmark_mae': bench_mae,
                    'benchmark_mape': bench_mape,
                    'dm_statistic': dm_stat,
                    'mse_improvement': mse_improvement,
                    'rmse_improvement': rmse_improvement,
                    'mae_improvement': mae_improvement,
                    'n_predictions': len(model_pred)
                }

                results_df = pd.DataFrame([comparison_results])
                results_df.to_csv(output_file, index=False)
                console.print(f"[green]Comparison results saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error comparing models: {e}[/red]")
            logger.error("Model comparison failed", error=str(e))


@validation_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with time series data')
@click.option('--output-file', type=click.Path(), help='Output file for stability analysis')
@click.option('--window-size', type=int, default=252, help='Rolling window size for stability analysis')
def analyze_stability(input_file, output_file, window_size):
    """Analyze model stability over time using rolling windows."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing model stability...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            if df.shape[1] < 2:
                console.print("[red]Error: Need at least 2 columns (predictions and actuals)[/red]")
                return

            # Assume first column is predictions, second is actuals
            predictions = df.iloc[:, 0].values
            actuals = df.iloc[:, 1].values

            n_periods = len(predictions)

            if n_periods < window_size:
                console.print(f"[red]Error: Need at least {window_size} periods for stability analysis[/red]")
                return

            # Rolling window analysis
            mse_series = []
            rmse_series = []
            mae_series = []
            mape_series = []

            for i in range(window_size, n_periods + 1):
                window_pred = predictions[i-window_size:i]
                window_actual = actuals[i-window_size:i]

                errors = window_pred - window_actual
                mse = np.mean(errors**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(errors))
                mape = np.mean(np.abs(errors) / np.abs(window_actual)) * 100

                mse_series.append(mse)
                rmse_series.append(rmse)
                mae_series.append(mae)
                mape_series.append(mape)

            # Calculate stability metrics
            mse_volatility = np.std(mse_series)
            rmse_volatility = np.std(rmse_series)
            mae_volatility = np.std(mae_series)
            mape_volatility = np.std(mape_series)

            # Trend analysis
            mse_trend = np.polyfit(range(len(mse_series)), mse_series, 1)[0]
            rmse_trend = np.polyfit(range(len(rmse_series)), rmse_series, 1)[0]

            progress.update(task, completed=1)

            # Display results
            table = Table(title="Model Stability Analysis")
            table.add_column("Metric", style="cyan")
            table.add_column("Mean", style="green")
            table.add_column("Volatility", style="yellow")
            table.add_column("Trend", style="blue")

            table.add_row("MSE", ".6f", ".6f", ".8f")
            table.add_row("RMSE", ".6f", ".6f", ".8f")
            table.add_row("MAE", ".6f", ".6f", ".6f")
            table.add_row("MAPE (%)", ".2f", ".2f", ".4f")

            console.print(table)

            # Stability assessment
            stability_score = 1 - (mape_volatility / np.mean(mape_series))

            if stability_score > 0.8:
                stability_rating = "[green]Very Stable[/green]"
            elif stability_score > 0.6:
                stability_rating = "[yellow]Stable[/yellow]"
            elif stability_score > 0.4:
                stability_rating = "[orange]Moderately Stable[/orange]"
            else:
                stability_rating = "[red]Unstable[/red]"

            stability_panel = Panel(
                f"[bold]Stability Assessment[/bold]\n\n"
                f"Stability Rating: {stability_rating}\n"
                f"Stability Score: {stability_score:.3f}\n"
                f"Window Size: {window_size} periods\n"
                f"Analysis Periods: {len(mse_series)}\n\n"
                f"[dim]Higher stability score indicates more consistent performance[/dim]",
                title="Stability Summary",
                border_style="blue"
            )

            console.print("\n", stability_panel)

            # Trend analysis
            if abs(mse_trend) < 0.001:
                trend_desc = "Stable (no significant trend)"
            elif mse_trend > 0:
                trend_desc = "Degrading (increasing errors)"
            else:
                trend_desc = "Improving (decreasing errors)"

            trend_table = Table(title="Trend Analysis")
            trend_table.add_column("Metric", style="cyan")
            trend_table.add_column("Trend Direction", style="green")

            trend_table.add_row("MSE Trend", trend_desc)
            trend_table.add_row("RMSE Trend", "Improving" if rmse_trend < 0 else "Degrading")

            console.print("\n", trend_table)

            # Save results if requested
            if output_file:
                stability_df = pd.DataFrame({
                    'window_start': range(len(mse_series)),
                    'mse': mse_series,
                    'rmse': rmse_series,
                    'mae': mae_series,
                    'mape': mape_series
                })

                stability_df.to_csv(output_file, index=False)

                # Save summary statistics
                summary_file = output_file.replace('.csv', '_summary.csv')
                summary_df = pd.DataFrame({
                    'metric': ['stability_score', 'mse_volatility', 'rmse_volatility',
                              'mae_volatility', 'mape_volatility', 'mse_trend', 'rmse_trend'],
                    'value': [stability_score, mse_volatility, rmse_volatility,
                             mae_volatility, mape_volatility, mse_trend, rmse_trend]
                })
                summary_df.to_csv(summary_file, index=False)

                console.print(f"[green]Stability analysis saved to {output_file}[/green]")
                console.print(f"[green]Summary statistics saved to {summary_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error analyzing stability: {e}[/red]")
            logger.error("Stability analysis failed", error=str(e))
