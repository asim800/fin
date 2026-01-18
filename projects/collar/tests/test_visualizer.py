"""Tests for the visualizer module."""

import pytest
from datetime import date, timedelta

import plotly.graph_objects as go

from lib.volatility import SABRVolatility, SABRParameters
from src.builder import CollarBuilder
from src.capped_gains import CappedGainsAnalyzer
from src.greeks import GreeksAnalyzer
from src.scenario import ScenarioAnalyzer, NAMED_SCENARIOS
from src.visualizer import CollarVisualizer


@pytest.fixture
def sample_collar():
    """Create a sample collar position for testing."""
    builder = CollarBuilder(risk_free_rate=0.05)
    today = date.today()
    expiry = today + timedelta(days=30)

    position, pricing = builder.build_collar(
        symbol="SPY",
        spot=500.0,
        shares=100,
        put_strike=475.0,
        call_strike=525.0,
        expiry=expiry,
        volatility=0.20,
        entry_date=today,
    )
    return position, pricing


@pytest.fixture
def visualizer():
    """Create a visualizer instance."""
    return CollarVisualizer()


class TestPayoffDiagram:
    """Tests for payoff diagram generation."""

    def test_payoff_diagram_returns_figure(self, visualizer, sample_collar):
        """Test that payoff_diagram returns a Plotly Figure."""
        position, pricing = sample_collar
        fig = visualizer.payoff_diagram(
            collar=position,
            pricing=pricing,
        )
        assert isinstance(fig, go.Figure)

    def test_payoff_diagram_has_traces(self, visualizer, sample_collar):
        """Test that payoff diagram has expected traces."""
        position, pricing = sample_collar
        fig = visualizer.payoff_diagram(
            collar=position,
            pricing=pricing,
        )
        # Should have 4 traces: stock, put, call, collar_net
        assert len(fig.data) == 4

    def test_payoff_diagram_custom_price_range(self, visualizer, sample_collar):
        """Test payoff diagram with custom price range."""
        position, pricing = sample_collar
        fig = visualizer.payoff_diagram(
            collar=position,
            pricing=pricing,
            price_range_pct=(-0.30, 0.30),
        )
        assert isinstance(fig, go.Figure)
        # Check that price range is reflected in x data
        x_data = fig.data[0].x
        assert min(x_data) < 400  # 30% below 500
        assert max(x_data) > 600  # 30% above 500


class TestScenarioComparison:
    """Tests for scenario comparison chart."""

    def test_scenario_comparison_returns_figure(self, visualizer, sample_collar):
        """Test that scenario_comparison returns a Plotly Figure."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        results = []
        for scenario in list(NAMED_SCENARIOS.values())[:3]:
            result = analyzer.analyze_scenario(
                collar=position,
                initial_pricing=pricing,
                scenario=scenario,
                initial_vol=0.20,
                risk_free_rate=0.05,
            )
            results.append(result)

        fig = visualizer.scenario_comparison(results)
        assert isinstance(fig, go.Figure)

    def test_scenario_comparison_has_bars(self, visualizer, sample_collar):
        """Test that scenario comparison has bar traces."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        results = []
        for scenario in list(NAMED_SCENARIOS.values())[:2]:
            result = analyzer.analyze_scenario(
                collar=position,
                initial_pricing=pricing,
                scenario=scenario,
                initial_vol=0.20,
                risk_free_rate=0.05,
            )
            results.append(result)

        fig = visualizer.scenario_comparison(results)
        # Should have 2 bar traces (collar P&L, unhedged P&L)
        assert len(fig.data) == 2
        assert all(isinstance(trace, go.Bar) for trace in fig.data)


class TestGreeksProfile:
    """Tests for Greeks profile chart."""

    def test_greeks_profile_returns_figure(self, visualizer, sample_collar):
        """Test that greeks_profile returns a Plotly Figure."""
        position, _ = sample_collar
        greeks_analyzer = GreeksAnalyzer()

        delta_df = greeks_analyzer.delta_profile(
            collar=position,
            spot=500.0,
            price_range_pct=(-0.15, 0.15),
            volatility=0.20,
            days_to_expiry=30,
            risk_free_rate=0.05,
            resolution=5,
        )

        fig = visualizer.greeks_profile(delta_df)
        assert isinstance(fig, go.Figure)

    def test_greeks_profile_has_traces(self, visualizer, sample_collar):
        """Test that Greeks profile has expected traces."""
        position, _ = sample_collar
        greeks_analyzer = GreeksAnalyzer()

        delta_df = greeks_analyzer.delta_profile(
            collar=position,
            spot=500.0,
            price_range_pct=(-0.15, 0.15),
            volatility=0.20,
            days_to_expiry=30,
            risk_free_rate=0.05,
            resolution=5,
        )

        fig = visualizer.greeks_profile(delta_df)
        # Should have traces for delta components
        assert len(fig.data) >= 1


class TestGreeksComparison:
    """Tests for Greeks comparison chart."""

    def test_greeks_comparison_returns_figure(self, visualizer, sample_collar):
        """Test that greeks_comparison returns a Plotly Figure."""
        _, pricing = sample_collar
        fig = visualizer.greeks_comparison(pricing)
        assert isinstance(fig, go.Figure)

    def test_greeks_comparison_has_bars(self, visualizer, sample_collar):
        """Test that Greeks comparison has bar traces."""
        _, pricing = sample_collar
        fig = visualizer.greeks_comparison(pricing)
        # Should have 2 bar traces (collar, stock)
        assert len(fig.data) == 2


class TestOpportunityCostAnalysis:
    """Tests for opportunity cost analysis chart."""

    def test_opportunity_cost_returns_figure(self, visualizer, sample_collar):
        """Test that opportunity_cost_analysis returns a Plotly Figure."""
        position, pricing = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=30 / 365.0,
        )

        fig = visualizer.opportunity_cost_analysis(result)
        assert isinstance(fig, go.Figure)

    def test_opportunity_cost_has_traces(self, visualizer, sample_collar):
        """Test that opportunity cost chart has expected traces."""
        position, pricing = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=30 / 365.0,
        )

        fig = visualizer.opportunity_cost_analysis(result)
        # Should have traces for uncapped, capped, opportunity cost
        assert len(fig.data) >= 2


class TestVolatilitySkew:
    """Tests for volatility skew chart."""

    def test_volatility_skew_returns_figure(self, visualizer):
        """Test that volatility_skew returns a Plotly Figure."""
        params = SABRParameters(alpha=0.20, beta=0.5, rho=-0.35, nu=0.4)
        vol_model = SABRVolatility(params)

        fig = visualizer.volatility_skew(
            vol_model=vol_model,
            forward=500.0,
            expiry_years=30 / 365.0,
        )
        assert isinstance(fig, go.Figure)

    def test_volatility_skew_has_trace(self, visualizer):
        """Test that volatility skew chart has a trace."""
        params = SABRParameters(alpha=0.20, beta=0.5, rho=-0.35, nu=0.4)
        vol_model = SABRVolatility(params)

        fig = visualizer.volatility_skew(
            vol_model=vol_model,
            forward=500.0,
            expiry_years=30 / 365.0,
        )
        assert len(fig.data) >= 1

    def test_volatility_skew_custom_strike_range(self, visualizer):
        """Test volatility skew with custom strike range."""
        params = SABRParameters(alpha=0.20, beta=0.5, rho=-0.35, nu=0.4)
        vol_model = SABRVolatility(params)

        fig = visualizer.volatility_skew(
            vol_model=vol_model,
            forward=500.0,
            expiry_years=30 / 365.0,
            strike_range_pct=(-0.30, 0.30),
        )
        assert isinstance(fig, go.Figure)


class TestVisualizerCustomization:
    """Tests for visualizer customization options."""

    def test_custom_colors(self, sample_collar):
        """Test visualizer with custom colors."""
        custom_colors = {
            "stock": "#FF0000",
            "put": "#00FF00",
            "call": "#0000FF",
            "collar_net": "#FFFF00",
        }
        vis = CollarVisualizer(color_scheme=custom_colors)

        position, pricing = sample_collar
        fig = vis.payoff_diagram(
            collar=position,
            pricing=pricing,
        )
        assert isinstance(fig, go.Figure)

    def test_show_components_option(self, visualizer, sample_collar):
        """Test that show_components parameter works."""
        position, pricing = sample_collar
        fig = visualizer.payoff_diagram(
            collar=position,
            pricing=pricing,
            show_components=False,
        )
        assert isinstance(fig, go.Figure)
        # Without components, only 1 trace (collar net)
        assert len(fig.data) == 1


class TestFigureProperties:
    """Tests for common figure properties."""

    def test_payoff_diagram_has_title(self, visualizer, sample_collar):
        """Test that payoff diagram has a title."""
        position, pricing = sample_collar
        fig = visualizer.payoff_diagram(
            collar=position,
            pricing=pricing,
        )
        assert fig.layout.title.text is not None

    def test_scenario_comparison_has_title(self, visualizer, sample_collar):
        """Test that scenario comparison has a title."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        results = []
        for scenario in list(NAMED_SCENARIOS.values())[:2]:
            result = analyzer.analyze_scenario(
                collar=position,
                initial_pricing=pricing,
                scenario=scenario,
                initial_vol=0.20,
                risk_free_rate=0.05,
            )
            results.append(result)

        fig = visualizer.scenario_comparison(results)
        assert fig.layout.title.text is not None

    def test_greeks_comparison_has_title(self, visualizer, sample_collar):
        """Test that Greeks comparison has a title."""
        _, pricing = sample_collar
        fig = visualizer.greeks_comparison(pricing)
        assert fig.layout.title.text is not None
