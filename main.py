#!/usr/bin/env python3
"""
Main entry point for OpenInvestments Quantitative Risk Platform.

This script provides command-line access to all platform features including:
- Option valuation and pricing
- Risk analysis and VaR calculations
- Portfolio optimization and analysis
- Model validation and backtesting
- Interactive CLI with rich formatting
- FastAPI web server for API access

Usage:
    python main.py --help                    # Show help
    python main.py valuation price-option   # Price options
    python main.py risk calculate-var       # Calculate VaR
    python main.py portfolio analyze        # Analyze portfolio
    python main.py api                      # Start FastAPI server
    python main.py web                      # Start web interface
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from openinvestments.cli import main_cli
    from openinvestments.api import app
    import uvicorn
    from rich.console import Console
    from rich.panel import Panel
    import click

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("  pip install -e .")
    sys.exit(1)

console = Console()


def run_cli():
    """Run the command-line interface."""
    try:
        main_cli()
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]CLI Error: {e}[/red]")
        sys.exit(1)


def run_api_server(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI server."""
    console.print(Panel.fit(
        "[bold blue]Starting OpenInvestments API Server[/bold blue]\n\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Reload: {reload}\n\n"
        "[dim]Press Ctrl+C to stop the server[/dim]",
        title="FastAPI Server"
    ))

    try:
        uvicorn.run(
            "openinvestments.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Server Error: {e}[/red]")
        sys.exit(1)


def run_web_interface():
    """Run the web interface by opening API docs in browser."""
    console.print(Panel.fit(
        "[bold green]Web Interface[/bold green]\n\n"
        "Launching API documentation in browser...\n\n"
        "[dim]The web interface is accessed via the FastAPI docs at http://localhost:8000/docs[/dim]",
        title="Web Interface"
    ))
    import webbrowser
    import threading
    import time
    def start_server():
        run_api_server(host="127.0.0.1", port=8000, reload=False)
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:8000/docs')


@click.group()
@click.version_option(version="1.0.0")
def main():
    """
    OpenInvestments Quantitative Risk Analytics Platform

    A comprehensive platform for model risk management, valuation, and
    leveraged products analytics with CLI and API interfaces.
    """
    pass


@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', default=8000, type=int, help='Port to bind the server to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def api(host, port, reload):
    """Start the FastAPI server for API access."""
    run_api_server(host=host, port=port, reload=reload)


@main.command()
def web():
    """Start the web interface (future implementation)."""
    run_web_interface()


# Add all CLI commands as subcommands
main.add_command(main_cli, name='cli')

# For backward compatibility, run CLI if no specific command is given
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run CLI
        run_cli()
    else:
        # Check if first argument is a known command
        known_commands = ['api', 'web', 'cli', '--help', '--version']

        # Extract command from arguments
        first_arg = sys.argv[1] if len(sys.argv) > 1 else None

        if first_arg in known_commands or first_arg.startswith('-'):
            # Run main command group
            main()
        else:
            # Run CLI with all arguments
            run_cli()
