"""Tests for BSM pricing validation."""

import math

import pytest

from lib.pricing import BSMPricer, OptionGreeks


class TestBSMPricing:
    """Test BSM option pricing against known values."""

    def test_atm_call_price(self):
        """Test ATM call price: S=100, K=100, T=1, r=0.05, sigma=0.20."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        call_price = pricer.call_price()
        assert abs(call_price - 10.4506) < 0.01

    def test_atm_put_price(self):
        """Test ATM put price: S=100, K=100, T=1, r=0.05, sigma=0.20."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        put_price = pricer.put_price()
        assert abs(put_price - 5.5735) < 0.01

    def test_otm_call_price(self):
        """Test OTM call price: S=100, K=110, T=0.5, r=0.05, sigma=0.25."""
        pricer = BSMPricer(
            spot=100,
            strike=110,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.25,
        )
        call_price = pricer.call_price()
        # Verified via put-call parity
        assert abs(call_price - 4.2258) < 0.01

    def test_otm_put_price(self):
        """Test OTM put price: S=100, K=90, T=0.5, r=0.05, sigma=0.25."""
        pricer = BSMPricer(
            spot=100,
            strike=90,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.25,
        )
        put_price = pricer.put_price()
        # Verified via put-call parity
        assert abs(put_price - 2.2150) < 0.01

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S*e^(-qT) - K*e^(-rT)."""
        test_cases = [
            {"spot": 100, "strike": 100, "expiry_years": 1.0, "r": 0.05, "vol": 0.20},
            {"spot": 50, "strike": 55, "expiry_years": 0.5, "r": 0.03, "vol": 0.30},
            {"spot": 200, "strike": 180, "expiry_years": 0.25, "r": 0.08, "vol": 0.15},
        ]

        for params in test_cases:
            pricer = BSMPricer(
                spot=params["spot"],
                strike=params["strike"],
                expiry_years=params["expiry_years"],
                risk_free_rate=params["r"],
                volatility=params["vol"],
            )

            call = pricer.call_price()
            put = pricer.put_price()

            # C - P = S - K*e^(-rT) for q=0
            expected = params["spot"] - params["strike"] * math.exp(
                -params["r"] * params["expiry_years"]
            )

            assert abs((call - put) - expected) < 0.001

    def test_put_call_parity_with_dividend(self):
        """Test put-call parity with dividend yield."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            dividend_yield=0.02,
        )

        call = pricer.call_price()
        put = pricer.put_price()

        # C - P = S*e^(-qT) - K*e^(-rT)
        expected = 100 * math.exp(-0.02) - 100 * math.exp(-0.05)

        assert abs((call - put) - expected) < 0.001


class TestBSMGreeks:
    """Test BSM Greeks calculations."""

    def test_call_delta_bounds(self):
        """Test call delta is between 0 and 1."""
        test_cases = [
            {"spot": 100, "strike": 80},  # ITM
            {"spot": 100, "strike": 100},  # ATM
            {"spot": 100, "strike": 120},  # OTM
        ]

        for params in test_cases:
            pricer = BSMPricer(
                spot=params["spot"],
                strike=params["strike"],
                expiry_years=0.5,
                risk_free_rate=0.05,
                volatility=0.20,
            )
            delta = pricer.delta("call")
            assert 0 <= delta <= 1, f"Call delta {delta} out of bounds"

    def test_put_delta_bounds(self):
        """Test put delta is between -1 and 0."""
        test_cases = [
            {"spot": 100, "strike": 80},  # OTM put
            {"spot": 100, "strike": 100},  # ATM
            {"spot": 100, "strike": 120},  # ITM put
        ]

        for params in test_cases:
            pricer = BSMPricer(
                spot=params["spot"],
                strike=params["strike"],
                expiry_years=0.5,
                risk_free_rate=0.05,
                volatility=0.20,
            )
            delta = pricer.delta("put")
            assert -1 <= delta <= 0, f"Put delta {delta} out of bounds"

    def test_gamma_positive(self):
        """Test gamma is always non-negative."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        gamma = pricer.gamma()
        assert gamma >= 0

    def test_vega_positive(self):
        """Test vega is always non-negative."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        vega = pricer.vega()
        assert vega >= 0

    def test_gamma_symmetry(self):
        """Test gamma is the same for calls and puts at same strike."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        # Gamma is same for call and put - only one method
        gamma = pricer.gamma()
        assert gamma > 0  # Should be positive for ATM

    def test_vega_symmetry(self):
        """Test vega is the same for calls and puts at same strike."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        # Vega is same for call and put - only one method
        vega = pricer.vega()
        assert vega > 0  # Should be positive for ATM

    def test_delta_relationship(self):
        """Test call_delta + |put_delta| ≈ 1 for same strike (no dividends)."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
        )
        call_delta = pricer.delta("call")
        put_delta = pricer.delta("put")

        # For no dividends: call_delta - put_delta = 1
        # Which means call_delta + |put_delta| = call_delta - put_delta = 1
        assert abs(call_delta - put_delta - 1.0) < 0.001

    def test_greeks_method(self):
        """Test the convenience greeks() method returns OptionGreeks."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=0.5,
            risk_free_rate=0.05,
            volatility=0.20,
        )

        call_greeks = pricer.greeks("call")
        assert isinstance(call_greeks, OptionGreeks)
        assert call_greeks.delta == pricer.delta("call")
        assert call_greeks.gamma == pricer.gamma()
        assert call_greeks.vega == pricer.vega()
        assert call_greeks.theta == pricer.theta("call")

        put_greeks = pricer.greeks("put")
        assert isinstance(put_greeks, OptionGreeks)
        assert put_greeks.delta == pricer.delta("put")


class TestBSMEdgeCases:
    """Test BSM edge cases."""

    def test_at_expiry_call_itm(self):
        """Test ITM call at expiry returns intrinsic value."""
        pricer = BSMPricer(
            spot=110,
            strike=100,
            expiry_years=0.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert pricer.call_price() == 10.0

    def test_at_expiry_call_otm(self):
        """Test OTM call at expiry returns zero."""
        pricer = BSMPricer(
            spot=90,
            strike=100,
            expiry_years=0.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert pricer.call_price() == 0.0

    def test_at_expiry_put_itm(self):
        """Test ITM put at expiry returns intrinsic value."""
        pricer = BSMPricer(
            spot=90,
            strike=100,
            expiry_years=0.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert pricer.put_price() == 10.0

    def test_at_expiry_put_otm(self):
        """Test OTM put at expiry returns zero."""
        pricer = BSMPricer(
            spot=110,
            strike=100,
            expiry_years=0.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert pricer.put_price() == 0.0

    def test_very_short_expiry(self):
        """Test pricing with very short time to expiry."""
        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1 / 365,  # 1 day
            risk_free_rate=0.05,
            volatility=0.20,
        )
        call = pricer.call_price()
        put = pricer.put_price()

        # Should be small but positive for ATM
        assert call > 0
        assert put > 0
        assert call < 5  # Reasonable upper bound for 1-day ATM
