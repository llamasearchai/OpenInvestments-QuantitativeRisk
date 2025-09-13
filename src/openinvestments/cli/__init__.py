"""
Command Line Interface for OpenInvestments Platform

Provides comprehensive CLI tools for:
- Model valuation and risk analysis
- Portfolio management and optimization
- Validation and backtesting
- Reporting and visualization
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ..core.config import config
from ..core.logging import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)


def create_main_cli():
    """Create the main CLI group with all commands."""

    @click.group()
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
    @click.option('--config-file', type=click.Path(exists=True), help='Path to configuration file')
    @click.pass_context
    def cli(ctx, verbose, config_file):
        """OpenInvestments Quantitative Risk Analytics Platform

        A comprehensive platform for model risk management, valuation,
        and leveraged products analytics.
        """
        # Set up logging
        log_level = 'DEBUG' if verbose else 'INFO'
        setup_logging(level=log_level)

        # Store context
        ctx.ensure_object(dict)
        ctx.obj['verbose'] = verbose
        ctx.obj['config_file'] = config_file

        # Display welcome message
        welcome_text = Text("OpenInvestments Quantitative Risk Platform", style="bold blue")
        console.print(Panel(welcome_text, title="Welcome", border_style="blue"))

    # Import and add command groups
    from .valuation_cli import valuation_group
    from .risk_cli import risk_group
    from .portfolio_cli import portfolio_group
    from .validation_cli import validation_group
    from .reporting_cli import reporting_group

    cli.add_command(valuation_group)
    cli.add_command(risk_group)
    cli.add_command(portfolio_group)
    cli.add_command(validation_group)
    cli.add_command(reporting_group)

    # Add utility commands
    @cli.command()
    @click.pass_context
    def status(ctx):
        """Show platform status and configuration."""
        table = Table(title="Platform Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Core components status
        components = [
            ("Configuration", "Loaded", f"Debug: {config.debug}"),
            ("Database", "Connected", "SQLite"),
            ("Logging", "Active", f"Level: {config.log_level}"),
            ("GPU Support", "Enabled" if config.enable_gpu_acceleration else "Disabled",
             f"Cores: {config.max_workers}"),
            ("OpenAI", "Configured" if config.openai_api_key else "Not configured",
             "API integration ready" if config.openai_api_key else "Set OPENAI_API_KEY"),
        ]

        for component, status, details in components:
            table.add_row(component, status, details)

        console.print(table)

    @cli.command()
    def version():
        """Show platform version."""
        version_text = f"OpenInvestments v{config.version}"
        console.print(Panel(version_text, title="Version", border_style="green"))

    return cli


# Create the main CLI
main_cli = create_main_cli()


if __name__ == '__main__':
    main_cli()
