"""
Reporting and visualization CLI commands.
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
from pathlib import Path
import json

from ..core.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def reporting_group():
    """Reporting and visualization commands."""
    pass


@reporting_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with analysis results')
@click.option('--report-type', type=click.Choice(['valuation', 'risk', 'portfolio', 'validation']),
              default='valuation', help='Type of report to generate')
@click.option('--output-file', type=click.Path(), help='Output file for the report')
@click.option('--format', type=click.Choice(['text', 'json', 'html']), default='text',
              help='Report output format')
def generate_report(input_file, report_type, output_file, format):
    """Generate comprehensive analysis reports."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating report...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            if format == 'text':
                report_content = generate_text_report(df, report_type)
                if output_file:
                    with open(output_file, 'w') as f:
                        f.write(report_content)
                    console.print(f"[green]Report saved to {output_file}[/green]")
                else:
                    console.print(report_content)

            elif format == 'json':
                report_data = generate_json_report(df, report_type)
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(report_data, f, indent=2)
                    console.print(f"[green]JSON report saved to {output_file}[/green]")
                else:
                    console.print(json.dumps(report_data, indent=2))

            elif format == 'html':
                html_content = generate_html_report(df, report_type)
                if output_file:
                    with open(output_file, 'w') as f:
                        f.write(html_content)
                    console.print(f"[green]HTML report saved to {output_file}[/green]")
                else:
                    console.print("[yellow]HTML reports must be saved to file[/yellow]")

            progress.update(task, completed=1)

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error generating report: {e}[/red]")
            logger.error("Report generation failed", error=str(e))


def generate_text_report(df, report_type):
    """Generate text format report."""
    report_lines = []

    if report_type == 'valuation':
        report_lines.append("OPTION VALUATION REPORT")
        report_lines.append("=" * 50)

        if 'price' in df.columns:
            avg_price = df['price'].mean()
            std_price = df['price'].std()
            report_lines.append(".6f")
            report_lines.append(".6f")

        if 'delta' in df.columns:
            report_lines.append(".4f")

    elif report_type == 'risk':
        report_lines.append("RISK ANALYSIS REPORT")
        report_lines.append("=" * 50)

        if 'var' in df.columns:
            avg_var = df['var'].mean()
            report_lines.append(".4f")

        if 'es' in df.columns:
            avg_es = df['es'].mean()
            report_lines.append(".4f")

    elif report_type == 'portfolio':
        report_lines.append("PORTFOLIO ANALYSIS REPORT")
        report_lines.append("=" * 50)

        if 'return' in df.columns:
            total_return = df['return'].sum()
            report_lines.append(".4f")

        if 'weight' in df.columns:
            report_lines.append(f"Assets in portfolio: {len(df)}")

    elif report_type == 'validation':
        report_lines.append("MODEL VALIDATION REPORT")
        report_lines.append("=" * 50)

        if 'mape' in df.columns:
            avg_mape = df['mape'].mean()
            report_lines.append(".2f")

        if 'rmse' in df.columns:
            avg_rmse = df['rmse'].mean()
            report_lines.append(".6f")

    report_lines.append("")
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data points: {len(df)}")

    return "\n".join(report_lines)


def generate_json_report(df, report_type):
    """Generate JSON format report."""
    report_data = {
        "report_type": report_type,
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_points": len(df),
        "summary": {}
    }

    # Calculate summary statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        report_data["summary"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }

    report_data["data"] = df.to_dict('records')

    return report_data


def generate_html_report(df, report_type):
    """Generate HTML format report."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenInvestments {report_type.title()} Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>OpenInvestments {report_type.title()} Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Data points: {len(df)}</p>
        </div>

        <div class="summary">
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
    """

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        html_template += ".6f"".6f"".6f"".6f""

    html_template += """
            </table>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <tr>
    """

    # Add table headers
    for col in df.columns:
        html_template += f"<th>{col}</th>"

    html_template += "</tr>"

    # Add table rows
    for _, row in df.iterrows():
        html_template += "<tr>"
        for value in row:
            html_template += ".6f"
        html_template += "</tr>"

    html_template += """
        </table>
    </body>
    </html>
    """

    return html_template


@reporting_group.command()
@click.option('--results-dir', type=click.Path(exists=True), default='.',
              help='Directory containing analysis results')
@click.option('--output-file', type=click.Path(), required=True,
              help='Output file for the dashboard')
def create_dashboard(results_dir, output_file):
    """Create an interactive dashboard from analysis results."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating dashboard...", total=1)

        try:
            results_path = Path(results_dir)
            dashboard_data = {
                "title": "OpenInvestments Risk Analytics Dashboard",
                "generated_at": pd.Timestamp.now().isoformat(),
                "sections": []
            }

            # Look for different types of result files
            csv_files = list(results_path.glob("*.csv"))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)

                    section = {
                        "title": csv_file.stem.replace('_', ' ').title(),
                        "filename": csv_file.name,
                        "data_points": len(df),
                        "columns": list(df.columns),
                        "summary": {}
                    }

                    # Calculate summary for numeric columns
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        section["summary"][col] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max())
                        }

                    dashboard_data["sections"].append(section)

                except Exception as e:
                    logger.warning(f"Could not process {csv_file}: {e}")
                    continue

            # Save dashboard configuration
            with open(output_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)

            progress.update(task, completed=1)

            # Display dashboard summary
            table = Table(title="Dashboard Configuration")
            table.add_column("Section", style="cyan")
            table.add_column("Data Points", style="green")
            table.add_column("Columns", style="yellow")

            for section in dashboard_data["sections"]:
                table.add_row(
                    section["title"],
                    str(section["data_points"]),
                    str(len(section["columns"]))
                )

            console.print(table)
            console.print(f"[green]Dashboard configuration saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error creating dashboard: {e}[/red]")
            logger.error("Dashboard creation failed", error=str(e))


@reporting_group.command()
@click.option('--input-files', multiple=True, type=click.Path(exists=True), required=True,
              help='CSV files to combine for comparison report')
@click.option('--output-file', type=click.Path(), required=True,
              help='Output file for the comparison report')
@click.option('--key-column', type=str, default='date',
              help='Column to use as key for merging')
def compare_scenarios(input_files, output_file, key_column):
    """Compare results from different scenarios or time periods."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Comparing scenarios...", total=1)

        try:
            # Load all input files
            dfs = []
            scenario_names = []

            for i, file_path in enumerate(input_files):
                df = pd.read_csv(file_path)
                scenario_name = f"scenario_{i+1}"
                df['scenario'] = scenario_name
                dfs.append(df)
                scenario_names.append(scenario_name)

            # Merge dataframes
            if key_column in dfs[0].columns:
                merged_df = dfs[0]
                for df in dfs[1:]:
                    merged_df = pd.merge(merged_df, df, on=key_column,
                                       suffixes=('', f'_{df["scenario"].iloc[0]}'))
            else:
                # If no common key, just concatenate
                merged_df = pd.concat(dfs, ignore_index=True)

            # Calculate comparison metrics
            comparison_results = {}

            # Get numeric columns for comparison
            numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'scenario']

            for col in numeric_columns:
                if col in merged_df.columns:
                    values = merged_df[col].dropna()

                    if len(values) > 0:
                        comparison_results[col] = {
                            "mean": float(values.mean()),
                            "std": float(values.std()),
                            "min": float(values.min()),
                            "max": float(values.max()),
                            "range": float(values.max() - values.min())
                        }

            # Save comparison report
            merged_df.to_csv(output_file, index=False)

            # Create summary report
            summary_file = output_file.replace('.csv', '_summary.csv')
            summary_df = pd.DataFrame.from_dict(comparison_results, orient='index')
            summary_df.index.name = 'metric'
            summary_df.reset_index(inplace=True)
            summary_df.to_csv(summary_file, index=False)

            progress.update(task, completed=1)

            # Display comparison summary
            table = Table(title="Scenario Comparison Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Mean", style="green")
            table.add_column("Std Dev", style="yellow")
            table.add_column("Range", style="blue")

            for metric, stats in comparison_results.items():
                table.add_row(
                    metric,
                    ".4f",
                    ".4f",
                    ".4f"
                )

            console.print(table)
            console.print(f"[green]Comparison results saved to {output_file}[/green]")
            console.print(f"[green]Summary saved to {summary_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error comparing scenarios: {e}[/red]")
            logger.error("Scenario comparison failed", error=str(e))


@reporting_group.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='CSV file with time series data')
@click.option('--output-file', type=click.Path(), required=True,
              help='Output file for the trend analysis')
@click.option('--window-size', type=int, default=30, help='Window size for rolling statistics')
def analyze_trends(input_file, output_file, window_size):
    """Analyze trends and patterns in time series data."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing trends...", total=1)

        try:
            # Load data
            df = pd.read_csv(input_file)

            # Assume first column is time/date and others are values
            if len(df.columns) < 2:
                console.print("[red]Error: Need at least 2 columns for trend analysis[/red]")
                return

            time_col = df.columns[0]
            value_columns = df.columns[1:]

            trend_results = {}

            for col in value_columns:
                series = df[col].dropna()

                if len(series) < window_size:
                    console.print(f"[yellow]Warning: Series {col} has fewer than {window_size} points[/yellow]")
                    continue

                # Rolling statistics
                rolling_mean = series.rolling(window=window_size).mean()
                rolling_std = series.rolling(window=window_size).std()
                rolling_min = series.rolling(window=window_size).min()
                rolling_max = series.rolling(window=window_size).max()

                # Trend analysis using linear regression
                x = np.arange(len(series))
                slope, intercept = np.polyfit(x, series.values, 1)

                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((series.values - y_pred) ** 2)
                ss_tot = np.sum((series.values - np.mean(series.values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                trend_results[col] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_squared),
                    "trend_direction": "increasing" if slope > 0 else "decreasing",
                    "trend_strength": "strong" if abs(r_squared) > 0.7 else "weak" if abs(r_squared) > 0.3 else "very_weak",
                    "rolling_mean_final": float(rolling_mean.iloc[-1]),
                    "rolling_std_final": float(rolling_std.iloc[-1]),
                    "volatility_trend": "increasing" if rolling_std.iloc[-1] > rolling_std.iloc[0] else "decreasing"
                }

            # Save trend analysis
            trend_df = pd.DataFrame.from_dict(trend_results, orient='index')
            trend_df.index.name = 'series'
            trend_df.reset_index(inplace=True)
            trend_df.to_csv(output_file, index=False)

            progress.update(task, completed=1)

            # Display trend analysis
            table = Table(title="Trend Analysis Results")
            table.add_column("Series", style="cyan")
            table.add_column("Direction", style="green")
            table.add_column("Strength", style="yellow")
            table.add_column("R²", style="blue")
            table.add_column("Volatility Trend", style="magenta")

            for series, results in trend_results.items():
                direction_color = "green" if results["trend_direction"] == "increasing" else "red"
                strength_color = {
                    "strong": "green",
                    "weak": "yellow",
                    "very_weak": "red"
                }.get(results["trend_strength"], "white")

                table.add_row(
                    series,
                    f"[{direction_color}]{results['trend_direction']}[/{direction_color}]",
                    f"[{strength_color}]{results['trend_strength']}[/{strength_color}]",
                    ".3f",
                    results["volatility_trend"]
                )

            console.print(table)

            # Trend summary
            increasing_trends = sum(1 for r in trend_results.values() if r["trend_direction"] == "increasing")
            strong_trends = sum(1 for r in trend_results.values() if r["trend_strength"] == "strong")

            summary_panel = Panel(
                f"[bold]Trend Analysis Summary[/bold]\n\n"
                f"Increasing Trends: {increasing_trends}/{len(trend_results)}\n"
                f"Strong Trends: {strong_trends}/{len(trend_results)}\n"
                f"Window Size: {window_size} periods\n\n"
                f"[dim]Analysis based on linear regression and rolling statistics[/dim]",
                title="Summary",
                border_style="blue"
            )

            console.print("\n", summary_panel)

            console.print(f"[green]Trend analysis saved to {output_file}[/green]")

        except Exception as e:
            progress.update(task, completed=1)
            console.print(f"[red]Error analyzing trends: {e}[/red]")
            logger.error("Trend analysis failed", error=str(e))
