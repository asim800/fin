"""Tests for capped gains analysis."""

from datetime import date, timedelta

import pytest

from src.builder import CollarBuilder
from src.capped_gains import CappedGainsAnalyzer, CappedGainsResult, UpsideScenario


class TestCappedGainsAnalyzer:
    """Test CappedGainsAnalyzer class."""

    @pytest.fixture
    def sample_collar(self):
        """Create a sample collar for testing."""
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
        return position, pricing, 30 / 365.0

    def test_analyze_returns_result(self, sample_collar):
        """Test analyze returns CappedGainsResult."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        assert isinstance(result, CappedGainsResult)
        assert result.collar == position
        assert result.pricing == pricing

    def test_default_upside_levels(self, sample_collar):
        """Test default upside levels are used."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        levels = [s.level_pct for s in result.upside_scenarios]
        assert levels == [0.05, 0.10, 0.15, 0.20, 0.25]

    def test_custom_upside_levels(self, sample_collar):
        """Test custom upside levels are used."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        custom_levels = [0.03, 0.06, 0.09]
        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
            upside_levels=custom_levels,
        )

        levels = [s.level_pct for s in result.upside_scenarios]
        assert levels == custom_levels

    def test_target_prices_correct(self, sample_collar):
        """Test target prices are calculated correctly."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        entry_price = position.underlying_entry_price
        for scenario in result.upside_scenarios:
            expected_target = entry_price * (1 + scenario.level_pct)
            assert scenario.target_price == pytest.approx(expected_target)

    def test_probability_bounds(self, sample_collar):
        """Test probabilities are between 0 and 1."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        for scenario in result.upside_scenarios:
            assert 0 <= scenario.probability <= 1

    def test_probability_decreases_with_upside(self, sample_collar):
        """Test probabilities decrease for higher upside levels."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        probs = [s.probability for s in result.upside_scenarios]
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1], "Higher upside should have lower probability"

    def test_opportunity_cost_zero_below_call_strike(self, sample_collar):
        """Test opportunity cost is zero when target below call strike."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        # Call strike is 525 (5% above 500)
        # So 5% upside (target = 525) should have zero opportunity cost
        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
            upside_levels=[0.03, 0.05],  # 3% and 5%
        )

        # 3% upside: target = 515 < 525, should have zero opp cost
        assert result.upside_scenarios[0].opportunity_cost == 0.0

    def test_opportunity_cost_positive_above_call_strike(self, sample_collar):
        """Test opportunity cost is positive when target above call strike."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        # Call strike is 525 (5% above 500)
        # 10% upside (target = 550) should have positive opportunity cost
        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
            upside_levels=[0.10, 0.15],
        )

        for scenario in result.upside_scenarios:
            assert scenario.opportunity_cost > 0

    def test_capped_gain_equals_max_at_call_strike(self, sample_collar):
        """Test capped gain is limited at call strike minus premium."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        entry_price = position.underlying_entry_price
        call_strike = position.short_call.strike
        net_premium = pricing.net_premium

        max_capped_gain = (call_strike - entry_price) - net_premium

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
            upside_levels=[0.10, 0.15, 0.20],  # All above call strike
        )

        for scenario in result.upside_scenarios:
            assert scenario.capped_gain == pytest.approx(max_capped_gain)

    def test_expected_cost_is_probability_weighted(self, sample_collar):
        """Test expected cost equals probability times opportunity cost."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        for scenario in result.upside_scenarios:
            expected = scenario.probability * scenario.opportunity_cost
            assert scenario.expected_opportunity_cost == pytest.approx(expected)

    def test_total_expected_cost_sums_correctly(self, sample_collar):
        """Test total expected cost is sum of individual expected costs."""
        position, pricing, expiry_years = sample_collar
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=expiry_years,
        )

        expected_total = sum(s.expected_opportunity_cost for s in result.upside_scenarios)
        assert result.total_expected_opportunity_cost == pytest.approx(expected_total)


class TestBreakevenCalculation:
    """Test breakeven upside calculation."""

    def test_breakeven_positive_when_net_debit(self):
        """Test breakeven is positive when net premium is debit."""
        builder = CollarBuilder(risk_free_rate=0.05)
        today = date.today()
        expiry = today + timedelta(days=30)

        # Wide collar - put more expensive than call = net debit
        position, pricing = builder.build_collar(
            symbol="SPY",
            spot=500.0,
            shares=100,
            put_strike=490.0,  # Closer to ATM
            call_strike=530.0,  # Further OTM
            expiry=expiry,
            volatility=0.20,
            entry_date=today,
        )

        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)
        breakeven = analyzer.calculate_breakeven_upside(position, pricing)

        if pricing.net_premium > 0:
            assert breakeven > 0, "Breakeven should be positive for net debit"

    def test_breakeven_negative_when_net_credit(self):
        """Test breakeven is negative when net premium is credit."""
        builder = CollarBuilder(risk_free_rate=0.05)
        today = date.today()
        expiry = today + timedelta(days=30)

        # Tight collar - call more expensive than put = net credit
        position, pricing = builder.build_collar(
            symbol="SPY",
            spot=500.0,
            shares=100,
            put_strike=470.0,  # Further OTM
            call_strike=510.0,  # Closer to ATM
            expiry=expiry,
            volatility=0.20,
            entry_date=today,
        )

        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)
        breakeven = analyzer.calculate_breakeven_upside(position, pricing)

        if pricing.net_premium < 0:
            assert breakeven < 0, "Breakeven should be negative for net credit"


class TestToDataFrame:
    """Test DataFrame conversion."""

    def test_to_dataframe_columns(self):
        """Test DataFrame has expected columns."""
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

        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)
        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=30 / 365.0,
        )

        df = analyzer.to_dataframe(result)

        expected_columns = [
            "upside_level",
            "target_price",
            "probability",
            "probability_pct",
            "uncapped_gain",
            "capped_gain",
            "opportunity_cost",
            "expected_cost",
        ]
        for col in expected_columns:
            assert col in df.columns

    def test_to_dataframe_row_count(self):
        """Test DataFrame has correct number of rows."""
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

        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)
        result = analyzer.analyze(
            collar=position,
            pricing=pricing,
            expiry_years=30 / 365.0,
            upside_levels=[0.05, 0.10, 0.15],
        )

        df = analyzer.to_dataframe(result)
        assert len(df) == 3


class TestLognormalProbability:
    """Test lognormal probability calculations."""

    def test_atm_probability_near_50_percent(self):
        """Test ATM probability is near 50% for short expiry."""
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.00)

        # With zero drift, P(S_T > S_0) should be ~50%
        prob = analyzer._lognormal_prob_above(
            S0=100,
            K=100,
            T=0.1,
        )

        # Should be close to 50% but slightly less due to vol drag
        assert 0.45 < prob < 0.55

    def test_deep_otm_probability_low(self):
        """Test deep OTM has low probability."""
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        # 50% above current price - very unlikely
        prob = analyzer._lognormal_prob_above(
            S0=100,
            K=150,
            T=0.25,  # 3 months
        )

        assert prob < 0.05  # Less than 5%

    def test_itm_probability_high(self):
        """Test ITM has high probability."""
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        # 10% below current price - very likely to exceed
        prob = analyzer._lognormal_prob_above(
            S0=100,
            K=90,
            T=0.25,
        )

        assert prob > 0.70  # More than 70%

    def test_zero_expiry(self):
        """Test zero expiry returns 0 or 1."""
        analyzer = CappedGainsAnalyzer(volatility=0.20, risk_free_rate=0.05)

        # Above target
        prob_above = analyzer._lognormal_prob_above(S0=100, K=90, T=0.0)
        assert prob_above == 1.0

        # Below target
        prob_below = analyzer._lognormal_prob_above(S0=100, K=110, T=0.0)
        assert prob_below == 0.0
