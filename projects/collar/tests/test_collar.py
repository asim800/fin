"""Tests for collar construction and P&L calculations."""

from datetime import date, timedelta

import pytest

from src.builder import CollarBuilder
from src.structures import CollarLeg, CollarPosition, CollarPricing


class TestCollarPosition:
    """Test CollarPosition dataclass and properties."""

    def test_collar_position_creation(self):
        """Test creating a collar position."""
        today = date.today()
        expiry = today + timedelta(days=30)

        position = CollarPosition(
            underlying_symbol="SPY",
            underlying_shares=100,
            underlying_entry_price=500.0,
            entry_date=today,
            long_put=CollarLeg(
                option_type="put",
                strike=475.0,
                expiry=expiry,
                position="long",
                contracts=1,
            ),
            short_call=CollarLeg(
                option_type="call",
                strike=525.0,
                expiry=expiry,
                position="short",
                contracts=1,
            ),
        )

        assert position.underlying_symbol == "SPY"
        assert position.underlying_shares == 100
        assert position.underlying_entry_price == 500.0

    def test_put_strike_pct(self):
        """Test put strike percentage calculation."""
        today = date.today()
        expiry = today + timedelta(days=30)

        position = CollarPosition(
            underlying_symbol="SPY",
            underlying_shares=100,
            underlying_entry_price=500.0,
            entry_date=today,
            long_put=CollarLeg("put", 475.0, expiry, "long"),
            short_call=CollarLeg("call", 525.0, expiry, "short"),
        )

        assert position.put_strike_pct == 0.95  # 475/500

    def test_call_strike_pct(self):
        """Test call strike percentage calculation."""
        today = date.today()
        expiry = today + timedelta(days=30)

        position = CollarPosition(
            underlying_symbol="SPY",
            underlying_shares=100,
            underlying_entry_price=500.0,
            entry_date=today,
            long_put=CollarLeg("put", 475.0, expiry, "long"),
            short_call=CollarLeg("call", 525.0, expiry, "short"),
        )

        assert position.call_strike_pct == 1.05  # 525/500

    def test_protection_level(self):
        """Test protection level calculation."""
        today = date.today()
        expiry = today + timedelta(days=30)

        position = CollarPosition(
            underlying_symbol="SPY",
            underlying_shares=100,
            underlying_entry_price=500.0,
            entry_date=today,
            long_put=CollarLeg("put", 475.0, expiry, "long"),
            short_call=CollarLeg("call", 525.0, expiry, "short"),
        )

        assert position.protection_level == pytest.approx(-0.05)  # 5% downside

    def test_upside_cap(self):
        """Test upside cap calculation."""
        today = date.today()
        expiry = today + timedelta(days=30)

        position = CollarPosition(
            underlying_symbol="SPY",
            underlying_shares=100,
            underlying_entry_price=500.0,
            entry_date=today,
            long_put=CollarLeg("put", 475.0, expiry, "long"),
            short_call=CollarLeg("call", 525.0, expiry, "short"),
        )

        assert position.upside_cap == pytest.approx(0.05)  # 5% upside


class TestCollarBuilder:
    """Test CollarBuilder methods."""

    def test_build_collar(self):
        """Test building a collar with explicit strikes."""
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

        assert position.underlying_symbol == "SPY"
        assert position.underlying_shares == 100
        assert position.long_put.strike == 475.0
        assert position.short_call.strike == 525.0
        assert pricing.put_price > 0
        assert pricing.call_price > 0

    def test_build_symmetric_collar(self):
        """Test building a symmetric collar."""
        builder = CollarBuilder(risk_free_rate=0.05)
        today = date.today()
        expiry = today + timedelta(days=30)

        position, pricing = builder.build_symmetric_collar(
            symbol="SPY",
            spot=500.0,
            shares=100,
            otm_pct=0.05,
            expiry=expiry,
            volatility=0.20,
            entry_date=today,
        )

        assert position.long_put.strike == 475.0  # 500 * 0.95
        assert position.short_call.strike == 525.0  # 500 * 1.05

    def test_find_zero_cost_collar(self):
        """Test finding a zero-cost collar."""
        builder = CollarBuilder(risk_free_rate=0.05)
        today = date.today()
        expiry = today + timedelta(days=30)

        position, pricing = builder.find_zero_cost_collar(
            symbol="SPY",
            spot=500.0,
            shares=100,
            put_strike=475.0,
            expiry=expiry,
            volatility=0.20,
            tolerance=0.05,
            entry_date=today,
        )

        # Net premium should be approximately zero
        assert abs(pricing.net_premium) < 0.10

    def test_collar_net_premium_calculation(self):
        """Test net premium is put_price - call_price."""
        builder = CollarBuilder(risk_free_rate=0.05)
        today = date.today()
        expiry = today + timedelta(days=30)

        _, pricing = builder.build_collar(
            symbol="SPY",
            spot=500.0,
            shares=100,
            put_strike=475.0,
            call_strike=525.0,
            expiry=expiry,
            volatility=0.20,
            entry_date=today,
        )

        assert pricing.net_premium == pytest.approx(
            pricing.put_price - pricing.call_price
        )


class TestCollarPnL:
    """Test collar P&L at various price points."""

    def test_pnl_below_put_strike(self):
        """Test P&L when price falls below put strike (max loss)."""
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

        # At expiry with price at $400 (below put strike)
        final_price = 400.0
        put_strike = position.long_put.strike
        call_strike = position.short_call.strike
        entry_price = position.underlying_entry_price
        shares = position.underlying_shares

        # Put payoff: max(K_put - S_T, 0) = 475 - 400 = 75
        put_payoff = max(put_strike - final_price, 0)
        # Call payoff: max(S_T - K_call, 0) = 0 (OTM)
        call_payoff = max(final_price - call_strike, 0)

        # Stock loss: 400 - 500 = -100
        stock_pnl = final_price - entry_price

        # Total P&L per share: stock + put - call - net_premium
        # = -100 + 75 - 0 - net_premium
        total_pnl_per_share = stock_pnl + put_payoff - call_payoff - pricing.net_premium

        # Max loss should be (put_strike - entry) - net_premium
        expected_max_loss = (put_strike - entry_price) - pricing.net_premium
        assert total_pnl_per_share == pytest.approx(expected_max_loss)

    def test_pnl_between_strikes(self):
        """Test P&L when price is between strikes (linear region)."""
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

        final_price = 510.0  # Between strikes
        entry_price = position.underlying_entry_price

        # Both options expire worthless
        put_payoff = 0
        call_payoff = 0

        # Stock gain
        stock_pnl = final_price - entry_price

        # Total P&L: stock - net_premium
        total_pnl_per_share = stock_pnl + put_payoff - call_payoff - pricing.net_premium
        expected = (final_price - entry_price) - pricing.net_premium

        assert total_pnl_per_share == pytest.approx(expected)

    def test_pnl_above_call_strike(self):
        """Test P&L when price rises above call strike (max gain)."""
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

        final_price = 600.0  # Above call strike
        put_strike = position.long_put.strike
        call_strike = position.short_call.strike
        entry_price = position.underlying_entry_price

        # Put expires worthless
        put_payoff = 0
        # Call payoff: 600 - 525 = 75 (we owe this)
        call_payoff = max(final_price - call_strike, 0)

        # Stock gain: 600 - 500 = 100
        stock_pnl = final_price - entry_price

        # Total P&L: stock - call_payoff - net_premium
        # Stock gain is capped by call obligation
        total_pnl_per_share = stock_pnl - call_payoff - pricing.net_premium

        # Max gain should be (call_strike - entry) - net_premium
        expected_max_gain = (call_strike - entry_price) - pricing.net_premium
        assert total_pnl_per_share == pytest.approx(expected_max_gain)
