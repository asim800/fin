"""Plotly visualizations for collar analysis.

Provides interactive charts for:
- Payoff diagrams
- Scenario P&L comparisons
- Greeks profiles
- Opportunity cost analysis
- Volatility skew curves
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from lib.volatility import VolatilityModel
    from src.capped_gains import CappedGainsResult
    from src.scenario import ScenarioResult
    from src.structures import CollarPosition, CollarPricing


class CollarVisualizer:
    """Creates Plotly visualizations for collar analysis.

    All methods return Plotly Figure objects that can be displayed
    in Jupyter notebooks, saved to files, or rendered in Streamlit.

    Attributes:
        color_scheme: Dict mapping component names to colors.
        template: Plotly template for styling.
    """

    # Default color scheme
    DEFAULT_COLORS = {
        "stock": "#2E86AB",  # Blue
        "put": "#A23B72",  # Magenta
        "call": "#F18F01",  # Orange
        "collar_net": "#C73E1D",  # Red
        "profit_zone": "#28A745",  # Green
        "loss_zone": "#DC3545",  # Red
        "neutral": "#6C757D",  # Gray
    }

    def __init__(
        self,
        color_scheme: dict[str, str] | None = None,
        template: str = "plotly_white",
    ) -> None:
        """Initialize visualizer.

        Args:
            color_scheme: Custom color mappings.
            template: Plotly template (default "plotly_white").
        """
        self.colors = {**self.DEFAULT_COLORS, **(color_scheme or {})}
        self.template = template

    def payoff_diagram(
        self,
        collar: CollarPosition,
        pricing: CollarPricing,
        price_range_pct: tuple[float, float] = (-0.30, 0.30),
        resolution: int = 100,
        show_components: bool = True,
    ) -> go.Figure:
        """Create collar payoff diagram at expiration.

        Shows P&L for stock, put, call, and net collar position
        across a range of underlying prices.

        Args:
            collar: The collar position.
            pricing: Pricing information.
            price_range_pct: Price range as (min%, max%) from entry.
            resolution: Number of price points.
            show_components: Whether to show individual leg P&Ls.

        Returns:
            Plotly Figure with payoff diagram.
        """
        entry_price = collar.underlying_entry_price
        put_strike = collar.long_put.strike
        call_strike = collar.short_call.strike

        # Generate price range
        min_price = entry_price * (1 + price_range_pct[0])
        max_price = entry_price * (1 + price_range_pct[1])
        prices = [
            min_price + (max_price - min_price) * i / (resolution - 1)
            for i in range(resolution)
        ]

        # Calculate payoffs at each price
        stock_pnl = []
        put_pnl = []
        call_pnl = []
        collar_pnl = []

        for price in prices:
            # Stock P&L
            s_pnl = price - entry_price
            stock_pnl.append(s_pnl)

            # Put payoff (intrinsic at expiry) minus premium paid
            put_intrinsic = max(put_strike - price, 0)
            p_pnl = put_intrinsic - pricing.put_price
            put_pnl.append(p_pnl)

            # Call payoff (short, so negative intrinsic) plus premium received
            call_intrinsic = max(price - call_strike, 0)
            c_pnl = pricing.call_price - call_intrinsic
            call_pnl.append(c_pnl)

            # Net collar P&L
            collar_pnl.append(s_pnl + p_pnl + c_pnl)

        # Create figure
        fig = go.Figure()

        # Add component traces if requested
        if show_components:
            fig.add_trace(
                go.Scatter(
                    x=prices,
                    y=stock_pnl,
                    name="Stock",
                    line=dict(color=self.colors["stock"], dash="dash"),
                    opacity=0.6,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=prices,
                    y=put_pnl,
                    name="Long Put",
                    line=dict(color=self.colors["put"], dash="dot"),
                    opacity=0.6,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=prices,
                    y=call_pnl,
                    name="Short Call",
                    line=dict(color=self.colors["call"], dash="dot"),
                    opacity=0.6,
                )
            )

        # Add collar net P&L (main line)
        fig.add_trace(
            go.Scatter(
                x=prices,
                y=collar_pnl,
                name="Collar Net",
                line=dict(color=self.colors["collar_net"], width=3),
            )
        )

        # Add horizontal zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

        # Add vertical lines for strikes
        fig.add_vline(
            x=put_strike,
            line_dash="dash",
            line_color=self.colors["put"],
            annotation_text=f"Put ${put_strike:.0f}",
            annotation_position="top",
        )
        fig.add_vline(
            x=call_strike,
            line_dash="dash",
            line_color=self.colors["call"],
            annotation_text=f"Call ${call_strike:.0f}",
            annotation_position="top",
        )
        fig.add_vline(
            x=entry_price,
            line_dash="dot",
            line_color=self.colors["neutral"],
            annotation_text=f"Entry ${entry_price:.0f}",
            annotation_position="bottom",
        )

        # Update layout
        fig.update_layout(
            title="Collar Payoff Diagram at Expiration",
            xaxis_title="Underlying Price ($)",
            yaxis_title="P&L per Share ($)",
            template=self.template,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            hovermode="x unified",
        )

        return fig

    def scenario_comparison(
        self,
        results: list[ScenarioResult],
        show_unhedged: bool = True,
    ) -> go.Figure:
        """Create scenario P&L comparison bar chart.

        Args:
            results: List of ScenarioResult objects.
            show_unhedged: Whether to show unhedged comparison.

        Returns:
            Plotly Figure with grouped bar chart.
        """
        scenarios = [r.scenario.name for r in results]
        collar_pnl = [r.total_pnl for r in results]
        unhedged_pnl = [r.unhedged_pnl for r in results]

        fig = go.Figure()

        # Collar P&L bars
        fig.add_trace(
            go.Bar(
                name="Collar P&L",
                x=scenarios,
                y=collar_pnl,
                marker_color=self.colors["collar_net"],
            )
        )

        if show_unhedged:
            # Unhedged P&L bars
            fig.add_trace(
                go.Bar(
                    name="Unhedged P&L",
                    x=scenarios,
                    y=unhedged_pnl,
                    marker_color=self.colors["stock"],
                    opacity=0.6,
                )
            )

        # Update layout
        fig.update_layout(
            title="Scenario P&L Comparison",
            xaxis_title="Scenario",
            yaxis_title="P&L ($)",
            template=self.template,
            barmode="group",
            hovermode="x unified",
        )

        return fig

    def greeks_profile(
        self,
        delta_df: pd.DataFrame,
        gamma_df: pd.DataFrame | None = None,
        show_components: bool = True,
    ) -> go.Figure:
        """Create Greeks profile visualization.

        Args:
            delta_df: DataFrame from GreeksAnalyzer.delta_profile().
            gamma_df: Optional DataFrame from GreeksAnalyzer.gamma_profile().
            show_components: Whether to show individual leg Greeks.

        Returns:
            Plotly Figure with Greeks profiles.
        """
        if gamma_df is not None:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Delta Profile", "Gamma Profile"),
                vertical_spacing=0.15,
            )
            has_gamma = True
        else:
            fig = go.Figure()
            has_gamma = False

        # Delta traces
        if show_components:
            fig.add_trace(
                go.Scatter(
                    x=delta_df["price"],
                    y=delta_df["put_delta"],
                    name="Put Delta",
                    line=dict(color=self.colors["put"], dash="dot"),
                ),
                row=1 if has_gamma else None,
                col=1 if has_gamma else None,
            )

            fig.add_trace(
                go.Scatter(
                    x=delta_df["price"],
                    y=delta_df["call_delta"],
                    name="Call Delta",
                    line=dict(color=self.colors["call"], dash="dot"),
                ),
                row=1 if has_gamma else None,
                col=1 if has_gamma else None,
            )

        # Stock delta (always 1)
        fig.add_trace(
            go.Scatter(
                x=delta_df["price"],
                y=[1.0] * len(delta_df),
                name="Stock Delta",
                line=dict(color=self.colors["stock"], dash="dash"),
                opacity=0.5,
            ),
            row=1 if has_gamma else None,
            col=1 if has_gamma else None,
        )

        # Collar net delta
        fig.add_trace(
            go.Scatter(
                x=delta_df["price"],
                y=delta_df["collar_delta"],
                name="Collar Delta",
                line=dict(color=self.colors["collar_net"], width=3),
            ),
            row=1 if has_gamma else None,
            col=1 if has_gamma else None,
        )

        if has_gamma and gamma_df is not None:
            # Gamma traces
            if show_components:
                fig.add_trace(
                    go.Scatter(
                        x=gamma_df["price"],
                        y=gamma_df["put_gamma"],
                        name="Put Gamma",
                        line=dict(color=self.colors["put"], dash="dot"),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=gamma_df["price"],
                        y=gamma_df["call_gamma"],
                        name="Call Gamma",
                        line=dict(color=self.colors["call"], dash="dot"),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=gamma_df["price"],
                    y=gamma_df["collar_gamma"],
                    name="Collar Gamma",
                    line=dict(color=self.colors["collar_net"], width=3),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Update layout
        title = "Greeks Profile" if not has_gamma else "Delta and Gamma Profiles"
        fig.update_layout(
            title=title,
            template=self.template,
            hovermode="x unified",
        )

        if has_gamma:
            fig.update_xaxes(title_text="Underlying Price ($)", row=2, col=1)
            fig.update_yaxes(title_text="Delta", row=1, col=1)
            fig.update_yaxes(title_text="Gamma", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Underlying Price ($)")
            fig.update_yaxes(title_text="Delta")

        return fig

    def greeks_comparison(
        self,
        collar_pricing: CollarPricing,
    ) -> go.Figure:
        """Compare collar Greeks to stock-only (delta=1, others=0).

        Args:
            collar_pricing: CollarPricing with Greeks.

        Returns:
            Plotly Figure with Greeks comparison.
        """
        greeks = ["Delta", "Gamma", "Vega", "Theta"]
        stock_values = [1.0, 0.0, 0.0, 0.0]
        collar_values = [
            collar_pricing.net_delta,
            collar_pricing.net_gamma,
            collar_pricing.net_vega,
            collar_pricing.net_theta,
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Stock Only",
                x=greeks,
                y=stock_values,
                marker_color=self.colors["stock"],
                opacity=0.6,
            )
        )

        fig.add_trace(
            go.Bar(
                name="Collar Position",
                x=greeks,
                y=collar_values,
                marker_color=self.colors["collar_net"],
            )
        )

        fig.update_layout(
            title="Greeks: Collar vs Stock-Only Position",
            xaxis_title="Greek",
            yaxis_title="Value",
            template=self.template,
            barmode="group",
        )

        return fig

    def opportunity_cost_analysis(
        self,
        result: CappedGainsResult,
    ) -> go.Figure:
        """Visualize capped gains opportunity cost analysis.

        Args:
            result: CappedGainsResult from CappedGainsAnalyzer.

        Returns:
            Plotly Figure with opportunity cost visualization.
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Capped vs Uncapped Gains",
                "Probability-Weighted Opportunity Cost",
            ),
            vertical_spacing=0.15,
        )

        # Extract data
        levels = [f"{s.level_pct * 100:.0f}%" for s in result.upside_scenarios]
        uncapped = [s.uncapped_gain for s in result.upside_scenarios]
        capped = [s.capped_gain for s in result.upside_scenarios]
        opp_cost = [s.opportunity_cost for s in result.upside_scenarios]
        expected_cost = [s.expected_opportunity_cost for s in result.upside_scenarios]
        probs = [s.probability * 100 for s in result.upside_scenarios]

        # Row 1: Capped vs Uncapped gains
        fig.add_trace(
            go.Bar(
                name="Uncapped Gain",
                x=levels,
                y=uncapped,
                marker_color=self.colors["stock"],
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                name="Capped Gain",
                x=levels,
                y=capped,
                marker_color=self.colors["collar_net"],
            ),
            row=1,
            col=1,
        )

        # Add opportunity cost as scatter overlay
        fig.add_trace(
            go.Scatter(
                name="Opportunity Cost",
                x=levels,
                y=opp_cost,
                mode="lines+markers",
                line=dict(color=self.colors["call"], width=2),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
        )

        # Row 2: Expected opportunity cost with probability
        fig.add_trace(
            go.Bar(
                name="Expected Cost",
                x=levels,
                y=expected_cost,
                marker_color=self.colors["call"],
            ),
            row=2,
            col=1,
        )

        # Add probability as scatter on secondary y-axis
        fig.add_trace(
            go.Scatter(
                name="P(Reach Level) %",
                x=levels,
                y=probs,
                mode="lines+markers",
                line=dict(color=self.colors["neutral"], dash="dot"),
                marker=dict(size=6),
                yaxis="y4",
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=(
                f"Opportunity Cost Analysis<br>"
                f"<sub>Total Expected Cost: ${result.total_expected_opportunity_cost:.2f}/share</sub>"
            ),
            template=self.template,
            barmode="group",
            yaxis4=dict(
                title="Probability (%)",
                overlaying="y3",
                side="right",
                range=[0, 100],
            ),
        )

        fig.update_xaxes(title_text="Upside Level", row=2, col=1)
        fig.update_yaxes(title_text="Gain per Share ($)", row=1, col=1)
        fig.update_yaxes(title_text="Expected Cost ($)", row=2, col=1)

        return fig

    def volatility_skew(
        self,
        vol_model: VolatilityModel,
        forward: float,
        expiry_years: float,
        strike_range_pct: tuple[float, float] = (-0.20, 0.20),
        resolution: int = 50,
    ) -> go.Figure:
        """Visualize implied volatility skew from volatility model.

        Args:
            vol_model: VolatilityModel instance (e.g., SABRVolatility).
            forward: Forward price.
            expiry_years: Time to expiry in years.
            strike_range_pct: Strike range as percentage of forward.
            resolution: Number of strike points.

        Returns:
            Plotly Figure with volatility skew curve.
        """
        min_strike = forward * (1 + strike_range_pct[0])
        max_strike = forward * (1 + strike_range_pct[1])
        strikes = [
            min_strike + (max_strike - min_strike) * i / (resolution - 1)
            for i in range(resolution)
        ]

        vols = [
            vol_model.implied_vol(k, forward, expiry_years) * 100 for k in strikes
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=vols,
                mode="lines",
                name="Implied Vol",
                line=dict(color=self.colors["collar_net"], width=2),
            )
        )

        # Add ATM marker
        atm_vol = vol_model.implied_vol(forward, forward, expiry_years) * 100
        fig.add_trace(
            go.Scatter(
                x=[forward],
                y=[atm_vol],
                mode="markers",
                name="ATM",
                marker=dict(size=12, color=self.colors["stock"]),
            )
        )

        fig.add_vline(
            x=forward,
            line_dash="dash",
            line_color=self.colors["neutral"],
            opacity=0.5,
        )

        fig.update_layout(
            title="Implied Volatility Skew",
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility (%)",
            template=self.template,
        )

        return fig
