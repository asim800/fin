"""Command-line interface for Options Analysis Toolkit."""

import click
import logging
import sys
import json
from pathlib import Path
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .orchestrator import OptionsAnalysisOrchestrator
from .config import Config

console = Console()


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose):
    """Options Analysis Toolkit - Comprehensive options trading analysis.

    Analyze options chains, calculate elasticity, detect arbitrage opportunities,
    and generate visualizations for trading strategies.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@main.command()
@click.argument('ticker')
@click.option('--plots/--no-plots', default=True, help='Generate plots')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
def analyze(ticker, plots, output):
    """Analyze a single ticker's options chain.

    \b
    Examples:
        options-analysis analyze AAPL
        options-analysis analyze MSFT --no-plots
        options-analysis analyze NVDA -o results/
    """
    try:
        console.print(f"\n[bold cyan]Analyzing {ticker}...[/bold cyan]")

        orchestrator = OptionsAnalysisOrchestrator()
        results = orchestrator.run_ticker_analysis(ticker)

        if results:
            quote = results['quote']
            option_chain = results['option_chain']

            # Create results table
            table = Table(title=f"{ticker} Analysis Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Current Price", f"${quote['price']:.2f}")
            table.add_row("Volume", f"{quote['volume']:,}")
            table.add_row("Option Expiries", str(len(option_chain)))

            console.print(table)

            console.print(f"\n[green]✓[/green] Analysis complete!")
            if plots:
                console.print(f"[green]✓[/green] Plots saved to: [blue]plots/[/blue]")
        else:
            console.print(f"[red]✗[/red] Could not retrieve data for {ticker}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument('ticker')
@click.option('--format', type=click.Choice(['table', 'csv', 'json']), default='table')
@click.option('--expiry', help='Specific expiry date (e.g., "Jan.16.2026")')
@click.option('--top', type=int, default=10, help='Show top N options by elasticity')
def elasticity(ticker, format, expiry, top):
    """Show option elasticity for a ticker.

    Elasticity measures leverage: how much the option price changes for a 1% change
    in the underlying stock price.

    \b
    Examples:
        options-analysis elasticity AAPL
        options-analysis elasticity NVDA --format csv
        options-analysis elasticity MSFT --expiry "Mar.20.2026" --top 5
    """
    try:
        console.print(f"\n[bold cyan]Calculating elasticity for {ticker}...[/bold cyan]")

        orchestrator = OptionsAnalysisOrchestrator()
        market_data = orchestrator.fetch_market_data([ticker])

        if not market_data.is_valid():
            console.print(f"[red]✗[/red] Failed to fetch data for {ticker}")
            sys.exit(1)

        # Process with elasticity
        processed = orchestrator.processor.extract_contract_identifiers(
            market_data.option_chains,
            prices=market_data.prices,
            current_time=market_data.timestamp
        )

        if ticker not in processed:
            console.print(f"[red]✗[/red] No options data for {ticker}")
            sys.exit(1)

        ticker_data = processed[ticker]
        current_price = market_data.prices[ticker]

        # Select expiry
        if expiry and expiry in ticker_data:
            expiries_to_show = [expiry]
        elif expiry:
            console.print(f"[red]✗[/red] Expiry '{expiry}' not found")
            console.print(f"Available: {', '.join(list(ticker_data.keys())[:5])}")
            sys.exit(1)
        else:
            expiries_to_show = list(ticker_data.keys())[:1]  # Show first expiry

        for exp in expiries_to_show:
            exp_data = ticker_data[exp]
            call_elast = exp_data.get('call_elasticity')
            calls = exp_data.get('calls')

            if call_elast is not None and not call_elast.empty:
                # Merge and sort
                merged = calls.join(call_elast)
                merged = merged[merged['call_elasticity'].notna()]
                merged = merged.nlargest(top, 'call_elasticity')

                if format == 'table':
                    table = Table(title=f"{ticker} Call Elasticity - {exp} (Price: ${current_price:.2f})")
                    table.add_column("Strike", style="cyan")
                    table.add_column("Price", style="green")
                    table.add_column("IV", style="yellow")
                    table.add_column("Elasticity", style="magenta", justify="right")

                    for idx, row in merged.iterrows():
                        marker = " ← ATM" if abs(row['Strike'] - current_price) / current_price < 0.05 else ""
                        table.add_row(
                            f"${row['Strike']:.2f}{marker}",
                            f"${row['Last']:.2f}",
                            f"{row['IV']:.1%}",
                            f"{row['call_elasticity']:.2f}"
                        )

                    console.print(table)

                elif format == 'csv':
                    print(merged[['Strike', 'Last', 'IV', 'call_elasticity']].to_csv())

                elif format == 'json':
                    print(merged[['Strike', 'Last', 'IV', 'call_elasticity']].to_json(orient='records', indent=2))

        console.print(f"\n[green]✓[/green] Elasticity analysis complete!")

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def elasticity_command():
    """Entry point for options-elasticity command."""
    sys.argv[0] = 'options-elasticity'
    # Remove the command name and call elasticity directly
    sys.argv = ['options-elasticity'] + sys.argv[1:]
    elasticity.main(standalone_mode=False)


@main.command()
@click.option('--min-profit', type=float, default=0.01, help='Minimum profit threshold (e.g., 0.01 = 1%)')
@click.option('--max-results', type=int, default=10, help='Maximum results to show')
@click.option('--file', type=click.Path(exists=True), help='Ticker file to analyze')
def arbitrage(min_profit, max_results, file):
    """Find put-call parity arbitrage opportunities.

    Scans for violations of put-call parity that indicate potential arbitrage.

    \b
    Examples:
        options-analysis arbitrage
        options-analysis arbitrage --min-profit 0.05
        options-analysis arbitrage --file tickers.txt --max-results 20
    """
    try:
        console.print("\n[bold cyan]Scanning for arbitrage opportunities...[/bold cyan]")

        orchestrator = OptionsAnalysisOrchestrator()

        # Run full analysis
        if file:
            results = orchestrator.run_full_analysis(file)
        else:
            results = orchestrator.run_full_analysis()

        arbitrage_df = results.get('arbitrage')

        if arbitrage_df is not None and not arbitrage_df.empty:
            # Filter by min_profit
            filtered = arbitrage_df[arbitrage_df['expected_profit'] >= min_profit * 100]
            filtered = filtered.head(max_results)

            if not filtered.empty:
                table = Table(title=f"Arbitrage Opportunities (≥{min_profit:.1%} profit)")
                table.add_column("Ticker", style="cyan")
                table.add_column("Strike", style="green")
                table.add_column("Expiry", style="yellow")
                table.add_column("Profit", style="magenta", justify="right")

                for _, row in filtered.iterrows():
                    table.add_row(
                        row['ticker'],
                        f"${row['strike']:.2f}",
                        row.get('expiry', 'N/A'),
                        f"${row['expected_profit']:.2f}"
                    )

                console.print(table)
                console.print(f"\n[green]✓[/green] Found {len(filtered)} opportunities!")
            else:
                console.print(f"[yellow]ℹ[/yellow] No opportunities found above {min_profit:.1%} threshold")
        else:
            console.print(f"[yellow]ℹ[/yellow] No arbitrage opportunities detected")

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def arbitrage_command():
    """Entry point for options-arbitrage command."""
    sys.argv[0] = 'options-arbitrage'
    arbitrage.main(standalone_mode=False)


@main.command()
@click.argument('ticker')
@click.option('--format', type=click.Choice(['csv', 'excel', 'json']), default='csv')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file path')
def export(ticker, format, output):
    """Export options data to file.

    \b
    Examples:
        options-analysis export AAPL --format csv -o aapl_options.csv
        options-analysis export NVDA --format excel -o nvda_data.xlsx
        options-analysis export MSFT --format json -o msft.json
    """
    try:
        console.print(f"\n[bold cyan]Exporting {ticker} data to {format.upper()}...[/bold cyan]")

        orchestrator = OptionsAnalysisOrchestrator()
        market_data = orchestrator.fetch_market_data([ticker])

        if not market_data.is_valid():
            console.print(f"[red]✗[/red] Failed to fetch data for {ticker}")
            sys.exit(1)

        # Process with elasticity
        processed = orchestrator.processor.extract_contract_identifiers(
            market_data.option_chains,
            prices=market_data.prices
        )

        # Create tables
        tables = orchestrator.processor.create_option_tables(processed)

        if ticker not in tables:
            console.print(f"[red]✗[/red] No data to export for {ticker}")
            sys.exit(1)

        ticker_tables = tables[ticker]
        output_path = Path(output)

        if format == 'csv':
            # Save each table as separate CSV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            base_name = output_path.stem
            for table_name, df in ticker_tables.items():
                csv_file = output_path.parent / f"{base_name}_{table_name}.csv"
                df.to_csv(csv_file)
                console.print(f"[green]✓[/green] Saved: {csv_file}")

        elif format == 'excel':
            # Save all tables to Excel with separate sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for table_name, df in ticker_tables.items():
                    # Truncate sheet name to 31 chars (Excel limit)
                    sheet_name = table_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name)
            console.print(f"[green]✓[/green] Saved: {output_path}")

        elif format == 'json':
            # Save as JSON
            import json
            json_data = {name: df.to_dict('split') for name, df in ticker_tables.items()}
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            console.print(f"[green]✓[/green] Saved: {output_path}")

        console.print(f"\n[green]✓[/green] Export complete!")

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def export_command():
    """Entry point for options-export command."""
    sys.argv[0] = 'options-export'
    export.main(standalone_mode=False)


@main.command()
def recipes():
    """List available analysis recipes.

    Recipes are pre-built workflows for common analysis patterns.
    """
    try:
        from pathlib import Path
        recipes_dir = Path(__file__).parent.parent.parent / "recipes"

        table = Table(title="Available Recipes")
        table.add_column("Recipe", style="cyan")
        table.add_column("Description", style="green")

        if recipes_dir.exists():
            recipe_files = sorted(recipes_dir.glob("*.py"))
            for recipe_file in recipe_files:
                if recipe_file.stem != "__init__":
                    # Read first docstring
                    with open(recipe_file) as f:
                        lines = f.readlines()
                        desc = "No description"
                        for line in lines:
                            if '"""' in line or "'''" in line:
                                desc = line.strip(' "\n\'')
                                break
                    table.add_row(recipe_file.stem, desc)
        else:
            table.add_row("No recipes found", "Run from recipes/ directory")

        console.print(table)

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")


if __name__ == '__main__':
    main()
