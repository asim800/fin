#!/usr/bin/env python3
"""CLI demonstration of the Option Collar Framework.

This script demonstrates:
1. Constructing a collar position on SPY
2. Displaying pricing summary
3. Running scenario analysis
4. Showing delta profile
5. Finding a zero-cost collar
6. SABR volatility skew analysis
7. Capped gains / opportunity cost analysis
8. Plotly visualizations (saved as HTML files)
"""

from datetime import date, timedelta

from lib.pricing import BSMPricer
from lib.volatility import SABRVolatility, SABRParameters, DEFAULT_SPY_SABR
from src.builder import CollarBuilder
from src.capped_gains import CappedGainsAnalyzer
from src.greeks import GreeksAnalyzer
from src.scenario import NAMED_SCENARIOS, ScenarioAnalyzer
from src.visualizer import CollarVisualizer


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def main() -> None:
    """Run the demonstration."""
    # Configuration
    symbol = "SPY"
    spot = 500.0
    shares = 100
    put_strike = 475.0  # 5% OTM
    call_strike = 525.0  # 5% OTM
    days_to_expiry = 30
    volatility = 0.20
    risk_free_rate = 0.05

    today = date.today()
    expiry = today + timedelta(days=days_to_expiry)
    expiry_years = days_to_expiry / 365.0

    # Build the collar with flat volatility
    builder = CollarBuilder(risk_free_rate=risk_free_rate)
    position, pricing = builder.build_collar(
        symbol=symbol,
        spot=spot,
        shares=shares,
        put_strike=put_strike,
        call_strike=call_strike,
        expiry=expiry,
        volatility=volatility,
        entry_date=today,
    )

    # === COLLAR POSITION ===
    print_section("COLLAR POSITION")
    print(f"Underlying: {symbol} @ ${spot:.2f}")
    print(f"Shares: {shares}")
    print(f"Put Strike: ${put_strike:.2f} ({position.put_strike_pct * 100:.1f}% of spot)")
    print(f"Call Strike: ${call_strike:.2f} ({position.call_strike_pct * 100:.1f}% of spot)")
    print(f"Expiry: {days_to_expiry} days")

    # === PRICING ===
    print_section("PRICING (Flat Volatility)")
    print(f"Put Premium:  ${pricing.put_price:.2f}/share (${pricing.put_price * shares:.2f} total)")
    print(f"Call Premium: ${pricing.call_price:.2f}/share (${pricing.call_price * shares:.2f} total)")
    print(f"Net Premium:  ${pricing.net_premium:.2f}/share (${pricing.net_premium * shares:.2f} {'debit' if pricing.net_premium > 0 else 'credit'})")
    print(f"Net Cost:     {pricing.net_premium_pct * 100:.2f}% of position")

    # === SABR VOLATILITY SKEW ===
    print_section("SABR VOLATILITY SKEW")

    # Create SABR model calibrated to ATM volatility
    sabr_model = SABRVolatility.from_atm_vol(
        atm_vol=volatility,
        forward=spot,
        expiry_years=expiry_years,
        beta=0.5,
        rho=-0.35,  # Negative for equity skew
        nu=0.4,
    )

    # Show implied volatilities at different strikes
    print(f"{'Strike':>10} {'Moneyness':>12} {'Implied Vol':>12}")
    print("-" * 36)

    test_strikes = [450, 475, 500, 525, 550]
    for strike in test_strikes:
        iv = sabr_model.implied_vol(strike, spot, expiry_years)
        moneyness = (strike / spot - 1) * 100
        print(f"${strike:>9.0f} {moneyness:>+11.1f}% {iv * 100:>11.1f}%")

    # Build collar with SABR skew
    sabr_builder = CollarBuilder(
        risk_free_rate=risk_free_rate,
        volatility_model=sabr_model,
    )
    position_sabr, pricing_sabr = sabr_builder.build_collar(
        symbol=symbol,
        spot=spot,
        shares=shares,
        put_strike=put_strike,
        call_strike=call_strike,
        expiry=expiry,
        entry_date=today,
    )

    print(f"\nPricing with SABR Skew:")
    print(f"Put Premium:  ${pricing_sabr.put_price:.2f}/share (vs ${pricing.put_price:.2f} flat)")
    print(f"Call Premium: ${pricing_sabr.call_price:.2f}/share (vs ${pricing.call_price:.2f} flat)")
    print(f"Net Premium:  ${pricing_sabr.net_premium:.2f}/share (vs ${pricing.net_premium:.2f} flat)")

    skew_impact = (pricing_sabr.net_premium - pricing.net_premium) * shares
    print(f"Skew Impact:  ${skew_impact:+.2f} total")

    # === PROTECTION METRICS ===
    print_section("PROTECTION METRICS")
    max_loss = (position.protection_level - pricing.net_premium_pct) * spot * shares
    max_gain = (position.upside_cap - pricing.net_premium_pct) * spot * shares
    print(f"Max Loss: ${max_loss:,.2f} ({(position.protection_level - pricing.net_premium_pct) * 100:.2f}%)")
    print(f"Max Gain: +${max_gain:,.2f} (+{(position.upside_cap - pricing.net_premium_pct) * 100:.2f}%)")

    # === GREEKS ===
    print_section("GREEKS")
    print(f"Delta: {pricing.net_delta:.3f} | Gamma: {pricing.net_gamma:.4f} | Vega: {pricing.net_vega:.2f} | Theta: {pricing.net_theta:.2f}")

    # === SCENARIO ANALYSIS ===
    print_section("SCENARIO ANALYSIS")
    analyzer = ScenarioAnalyzer()
    scenarios = list(NAMED_SCENARIOS.values())

    print(f"{'Scenario':<22} {'Price':>8} {'Vol':>8} {'Collar P&L':>12} {'Unhedged':>12} {'Benefit':>12}")
    print("-" * 76)

    scenario_results = []
    for scenario in scenarios:
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=volatility,
            risk_free_rate=risk_free_rate,
        )
        scenario_results.append(result)

        print(
            f"{result.scenario.name:<22} "
            f"{result.scenario.price_change_pct * 100:>+7.1f}% "
            f"{result.scenario.vol_change_pct * 100:>+7.1f}% "
            f"${result.total_pnl:>+10,.0f} "
            f"${result.unhedged_pnl:>+10,.0f} "
            f"${result.hedge_benefit:>+10,.0f}"
        )

    # === DELTA PROFILE ===
    print_section("DELTA PROFILE")
    greeks_analyzer = GreeksAnalyzer()
    delta_df = greeks_analyzer.delta_profile(
        collar=position,
        spot=spot,
        price_range_pct=(-0.20, 0.20),
        volatility=volatility,
        days_to_expiry=days_to_expiry,
        risk_free_rate=risk_free_rate,
        resolution=5,
    )

    print(f"{'Price':>10} {'% Change':>10} {'Put Delta':>12} {'Call Delta':>12} {'Collar Delta':>14}")
    print("-" * 60)
    for _, row in delta_df.iterrows():
        print(
            f"${row['price']:>9.2f} "
            f"{row['price_pct'] * 100:>+9.1f}% "
            f"{row['put_delta']:>+11.3f} "
            f"{row['call_delta']:>+11.3f} "
            f"{row['collar_delta']:>+13.3f}"
        )

    # === CAPPED GAINS / OPPORTUNITY COST ANALYSIS ===
    print_section("CAPPED GAINS / OPPORTUNITY COST ANALYSIS")

    capped_analyzer = CappedGainsAnalyzer(
        volatility=volatility,
        risk_free_rate=risk_free_rate,
    )

    capped_result = capped_analyzer.analyze(
        collar=position,
        pricing=pricing,
        expiry_years=expiry_years,
    )

    capped_df = capped_analyzer.to_dataframe(capped_result)

    print(f"{'Upside':>8} {'Target':>10} {'Prob':>8} {'Uncapped':>12} {'Capped':>12} {'Opp Cost':>12} {'Exp Cost':>12}")
    print("-" * 80)

    for _, row in capped_df.iterrows():
        print(
            f"{row['upside_level']:>8} "
            f"${row['target_price']:>9.2f} "
            f"{row['probability_pct']:>8} "
            f"${row['uncapped_gain']:>11.2f} "
            f"${row['capped_gain']:>11.2f} "
            f"${row['opportunity_cost']:>11.2f} "
            f"${row['expected_cost']:>11.2f}"
        )

    print(f"\nTotal Expected Opportunity Cost: ${capped_result.total_expected_opportunity_cost:.2f}")
    print(f"Breakeven Upside: {capped_result.breakeven_upside * 100:.2f}%")

    # === ZERO-COST COLLAR ===
    print_section("ZERO-COST COLLAR")
    zc_position, zc_pricing = builder.find_zero_cost_collar(
        symbol=symbol,
        spot=spot,
        shares=shares,
        put_strike=put_strike,
        expiry=expiry,
        volatility=volatility,
        tolerance=0.01,
        entry_date=today,
    )

    print(f"Put @ ${put_strike:.2f}, Call @ ${zc_position.short_call.strike:.2f}")
    print(f"Net Premium: ${zc_pricing.net_premium:.2f} (effectively zero)")
    print(f"New Upside Cap: {zc_position.upside_cap * 100:.1f}%")

    # === PLOTLY VISUALIZATIONS ===
    print_section("PLOTLY VISUALIZATIONS")

    visualizer = CollarVisualizer()

    # Generate all visualizations
    print("Generating visualizations...")

    # 1. Payoff Diagram
    payoff_fig = visualizer.payoff_diagram(
        collar=position,
        pricing=pricing,
    )
    payoff_fig.write_html("output/payoff_diagram.html")
    print("  - Payoff diagram: output/payoff_diagram.html")

    # 2. Scenario Comparison
    scenario_fig = visualizer.scenario_comparison(scenario_results)
    scenario_fig.write_html("output/scenario_comparison.html")
    print("  - Scenario comparison: output/scenario_comparison.html")

    # 3. Greeks Profile
    greeks_fig = visualizer.greeks_profile(delta_df)
    greeks_fig.write_html("output/greeks_profile.html")
    print("  - Greeks profile: output/greeks_profile.html")

    # 4. Greeks Comparison
    greeks_comp_fig = visualizer.greeks_comparison(pricing)
    greeks_comp_fig.write_html("output/greeks_comparison.html")
    print("  - Greeks comparison: output/greeks_comparison.html")

    # 5. Opportunity Cost Analysis
    opp_cost_fig = visualizer.opportunity_cost_analysis(capped_result)
    opp_cost_fig.write_html("output/opportunity_cost.html")
    print("  - Opportunity cost: output/opportunity_cost.html")

    # 6. Volatility Skew
    skew_fig = visualizer.volatility_skew(
        vol_model=sabr_model,
        forward=spot,
        expiry_years=expiry_years,
    )
    skew_fig.write_html("output/volatility_skew.html")
    print("  - Volatility skew: output/volatility_skew.html")

    print("\n" + "=" * 60)
    print(" Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    import os
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    main()
