"""Tests for scenario analysis."""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.builder import CollarBuilder
from src.scenario import NAMED_SCENARIOS, Scenario, ScenarioAnalyzer, ScenarioResult


class TestScenario:
    """Test Scenario dataclass."""

    def test_scenario_creation(self):
        """Test creating a scenario."""
        scenario = Scenario(
            name="Test Scenario",
            price_change_pct=-0.10,
            vol_change_pct=0.20,
            days_elapsed=30,
        )

        assert scenario.name == "Test Scenario"
        assert scenario.price_change_pct == -0.10
        assert scenario.vol_change_pct == 0.20
        assert scenario.days_elapsed == 30

    def test_scenario_defaults(self):
        """Test scenario default values."""
        scenario = Scenario(name="Simple", price_change_pct=-0.05)

        assert scenario.vol_change_pct == 0.0
        assert scenario.days_elapsed == 0


class TestNamedScenarios:
    """Test predefined named scenarios."""

    def test_named_scenarios_exist(self):
        """Test all named scenarios are defined."""
        expected_keys = [
            "mild_correction",
            "sharp_correction",
            "crash",
            "gfc_2008",
            "covid_2020",
            "rally",
            "sideways",
        ]

        for key in expected_keys:
            assert key in NAMED_SCENARIOS

    def test_mild_correction_values(self):
        """Test mild correction scenario values."""
        scenario = NAMED_SCENARIOS["mild_correction"]
        assert scenario.name == "Mild Correction"
        assert scenario.price_change_pct == -0.10
        assert scenario.vol_change_pct == 0.20
        assert scenario.days_elapsed == 30

    def test_crash_scenario_values(self):
        """Test market crash scenario values."""
        scenario = NAMED_SCENARIOS["crash"]
        assert scenario.name == "Market Crash"
        assert scenario.price_change_pct == -0.35
        assert scenario.vol_change_pct == 1.00


class TestScenarioAnalyzer:
    """Test ScenarioAnalyzer class."""

    @pytest.fixture
    def sample_collar(self):
        """Create a sample collar for testing."""
        builder = CollarBuilder(risk_free_rate=0.05)
        today = date.today()
        expiry = today + timedelta(days=60)

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

    def test_analyze_scenario_returns_result(self, sample_collar):
        """Test analyze_scenario returns ScenarioResult."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        scenario = Scenario("Test", price_change_pct=-0.10)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        assert isinstance(result, ScenarioResult)
        assert result.scenario == scenario

    def test_downside_scenario_hedge_benefit(self, sample_collar):
        """Test hedge provides benefit in downside scenario."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        scenario = Scenario("Crash", price_change_pct=-0.30, vol_change_pct=0.50)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        # In a crash, hedge should provide benefit
        assert result.hedge_benefit > 0
        # Total P&L should be better than unhedged
        assert result.total_pnl > result.unhedged_pnl

    def test_upside_scenario_capped(self, sample_collar):
        """Test gains are capped in upside scenario."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        scenario = Scenario("Rally", price_change_pct=0.20)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        # In a rally, unhedged would do better
        assert result.unhedged_pnl > result.total_pnl
        # Hedge benefit is negative (opportunity cost)
        assert result.hedge_benefit < 0

    def test_collar_binding_put(self, sample_collar):
        """Test collar binding detection for put."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        # Price drops below put strike
        scenario = Scenario("Deep Drop", price_change_pct=-0.10, days_elapsed=60)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        # At 10% drop from 500, price = 450 < put strike 475
        assert result.collar_binding == "put"

    def test_collar_binding_call(self, sample_collar):
        """Test collar binding detection for call."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        # Price rises above call strike
        scenario = Scenario("Strong Rally", price_change_pct=0.10, days_elapsed=60)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        # At 10% rise from 500, price = 550 > call strike 525
        assert result.collar_binding == "call"

    def test_collar_binding_neither(self, sample_collar):
        """Test collar binding detection for neither strike."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        # Price stays between strikes
        scenario = Scenario("Sideways", price_change_pct=0.02)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        # At 2% rise from 500, price = 510, between 475 and 525
        assert result.collar_binding == "neither"

    def test_analyze_scenarios_returns_dataframe(self, sample_collar):
        """Test analyze_scenarios returns DataFrame."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        scenarios = [
            NAMED_SCENARIOS["mild_correction"],
            NAMED_SCENARIOS["crash"],
            NAMED_SCENARIOS["rally"],
        ]

        df = analyzer.analyze_scenarios(
            collar=position,
            initial_pricing=pricing,
            scenarios=scenarios,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "scenario_name" in df.columns
        assert "total_pnl" in df.columns
        assert "hedge_benefit" in df.columns

    def test_analyze_scenarios_columns(self, sample_collar):
        """Test analyze_scenarios DataFrame has expected columns."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        scenarios = [NAMED_SCENARIOS["crash"]]

        df = analyzer.analyze_scenarios(
            collar=position,
            initial_pricing=pricing,
            scenarios=scenarios,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        expected_columns = [
            "scenario_name",
            "price_change",
            "vol_change",
            "days_elapsed",
            "underlying_pnl",
            "put_pnl",
            "call_pnl",
            "total_pnl",
            "total_pnl_pct",
            "unhedged_pnl",
            "hedge_benefit",
            "delta",
            "gamma",
            "vega",
            "theta",
            "binding",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_pnl_components_sum_correctly(self, sample_collar):
        """Test that P&L components sum to total."""
        position, pricing = sample_collar
        analyzer = ScenarioAnalyzer()

        scenario = Scenario("Test", price_change_pct=-0.15)
        result = analyzer.analyze_scenario(
            collar=position,
            initial_pricing=pricing,
            scenario=scenario,
            initial_vol=0.20,
            risk_free_rate=0.05,
        )

        calculated_total = result.underlying_pnl + result.put_pnl + result.call_pnl
        assert result.total_pnl == pytest.approx(calculated_total, rel=1e-6)
