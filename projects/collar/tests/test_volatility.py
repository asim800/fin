"""Tests for volatility models."""

import math

import pytest

from lib.pricing import BSMPricer
from lib.volatility import (
    DEFAULT_SPY_SABR,
    FlatVolatility,
    SABRParameters,
    SABRVolatility,
)


class TestFlatVolatility:
    """Test FlatVolatility model."""

    def test_constant_vol_across_strikes(self):
        """Test that flat vol returns constant regardless of strike."""
        model = FlatVolatility(volatility=0.20)

        vol1 = model.implied_vol(strike=90, forward=100, expiry_years=0.5)
        vol2 = model.implied_vol(strike=100, forward=100, expiry_years=0.5)
        vol3 = model.implied_vol(strike=110, forward=100, expiry_years=0.5)

        assert vol1 == 0.20
        assert vol2 == 0.20
        assert vol3 == 0.20

    def test_constant_vol_across_expiries(self):
        """Test that flat vol returns constant regardless of expiry."""
        model = FlatVolatility(volatility=0.25)

        vol1 = model.implied_vol(strike=100, forward=100, expiry_years=0.1)
        vol2 = model.implied_vol(strike=100, forward=100, expiry_years=1.0)
        vol3 = model.implied_vol(strike=100, forward=100, expiry_years=2.0)

        assert vol1 == 0.25
        assert vol2 == 0.25
        assert vol3 == 0.25


class TestSABRParameters:
    """Test SABRParameters validation."""

    def test_valid_parameters(self):
        """Test creating valid SABR parameters."""
        params = SABRParameters(alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        assert params.alpha == 0.20
        assert params.beta == 0.5
        assert params.rho == -0.3
        assert params.nu == 0.4

    def test_default_parameters(self):
        """Test default SABR parameters."""
        params = SABRParameters(alpha=0.20)
        assert params.beta == 0.5
        assert params.rho == -0.35
        assert params.nu == 0.4

    def test_invalid_beta_high(self):
        """Test beta > 1 raises error."""
        with pytest.raises(ValueError, match="beta must be in"):
            SABRParameters(alpha=0.20, beta=1.5)

    def test_invalid_beta_low(self):
        """Test beta < 0 raises error."""
        with pytest.raises(ValueError, match="beta must be in"):
            SABRParameters(alpha=0.20, beta=-0.1)

    def test_invalid_rho_high(self):
        """Test rho > 1 raises error."""
        with pytest.raises(ValueError, match="rho must be in"):
            SABRParameters(alpha=0.20, rho=1.5)

    def test_invalid_rho_low(self):
        """Test rho < -1 raises error."""
        with pytest.raises(ValueError, match="rho must be in"):
            SABRParameters(alpha=0.20, rho=-1.5)

    def test_invalid_alpha(self):
        """Test alpha <= 0 raises error."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            SABRParameters(alpha=0.0)

    def test_invalid_nu(self):
        """Test nu < 0 raises error."""
        with pytest.raises(ValueError, match="nu must be non-negative"):
            SABRParameters(alpha=0.20, nu=-0.1)


class TestSABRVolatility:
    """Test SABR volatility model."""

    def test_atm_vol_reasonable(self):
        """Test ATM volatility is close to alpha for beta=1."""
        params = SABRParameters(alpha=0.20, beta=1.0, rho=0.0, nu=0.0)
        model = SABRVolatility(params)

        vol = model.implied_vol(strike=100, forward=100, expiry_years=0.5)

        # For beta=1, rho=0, nu=0: ATM vol should be close to alpha
        assert abs(vol - 0.20) < 0.01

    def test_negative_rho_creates_skew(self):
        """Test negative rho produces skew (OTM puts have higher vol than OTM calls)."""
        # Use calibrated model for proper testing
        forward = 100
        expiry = 0.5

        model = SABRVolatility.from_atm_vol(
            atm_vol=0.20,
            forward=forward,
            expiry_years=expiry,
            beta=0.5,
            rho=-0.4,
            nu=0.4,
        )

        vol_otm_put = model.implied_vol(strike=90, forward=forward, expiry_years=expiry)
        vol_atm = model.implied_vol(strike=100, forward=forward, expiry_years=expiry)
        vol_otm_call = model.implied_vol(strike=110, forward=forward, expiry_years=expiry)

        # With negative rho, OTM puts should have higher vol than OTM calls (skew)
        assert vol_otm_put > vol_otm_call, "OTM put vol should be higher than OTM call"
        # OTM puts should be higher than ATM
        assert vol_otm_put > vol_atm, "OTM put vol should be higher than ATM"

    def test_positive_rho_creates_reverse_skew(self):
        """Test positive rho produces reverse skew."""
        # Use calibrated model for proper testing
        forward = 100
        expiry = 0.5

        model = SABRVolatility.from_atm_vol(
            atm_vol=0.20,
            forward=forward,
            expiry_years=expiry,
            beta=0.5,
            rho=0.4,
            nu=0.4,
        )

        vol_otm_put = model.implied_vol(strike=90, forward=forward, expiry_years=expiry)
        vol_atm = model.implied_vol(strike=100, forward=forward, expiry_years=expiry)
        vol_otm_call = model.implied_vol(strike=110, forward=forward, expiry_years=expiry)

        # With positive rho, OTM calls should have higher vol than OTM puts (reverse skew)
        assert vol_otm_call > vol_otm_put, "OTM call vol should be higher than OTM put"
        # OTM calls should be higher than ATM
        assert vol_otm_call > vol_atm, "OTM call vol should be higher with positive rho"

    def test_vol_always_positive(self):
        """Test implied vol is always positive."""
        params = SABRParameters(alpha=0.20, beta=0.5, rho=-0.5, nu=0.5)
        model = SABRVolatility(params)

        for strike in [50, 75, 100, 125, 150]:
            vol = model.implied_vol(strike=strike, forward=100, expiry_years=0.5)
            assert vol > 0, f"Vol should be positive for strike {strike}"

    def test_from_atm_vol_calibration(self):
        """Test from_atm_vol calibrates correctly."""
        target_atm_vol = 0.25
        forward = 100
        expiry = 0.5

        model = SABRVolatility.from_atm_vol(
            atm_vol=target_atm_vol,
            forward=forward,
            expiry_years=expiry,
            beta=0.5,
            rho=-0.3,
            nu=0.4,
        )

        actual_atm_vol = model.implied_vol(strike=forward, forward=forward, expiry_years=expiry)

        assert abs(actual_atm_vol - target_atm_vol) < 0.001, (
            f"Calibrated ATM vol {actual_atm_vol} should match target {target_atm_vol}"
        )

    def test_zero_expiry_returns_alpha(self):
        """Test zero expiry returns alpha."""
        params = SABRParameters(alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        model = SABRVolatility(params)

        vol = model.implied_vol(strike=100, forward=100, expiry_years=0.0)
        assert vol == 0.20


class TestDefaultSPYSABR:
    """Test DEFAULT_SPY_SABR preset."""

    def test_default_spy_sabr_values(self):
        """Test DEFAULT_SPY_SABR has expected values."""
        assert DEFAULT_SPY_SABR.alpha == 0.20
        assert DEFAULT_SPY_SABR.beta == 0.5
        assert DEFAULT_SPY_SABR.rho == -0.35
        assert DEFAULT_SPY_SABR.nu == 0.40

    def test_default_spy_sabr_produces_skew(self):
        """Test DEFAULT_SPY_SABR produces realistic equity skew."""
        forward = 500
        expiry = 30 / 365  # 30 days

        # Use calibrated model for realistic behavior
        model = SABRVolatility.from_atm_vol(
            atm_vol=0.20,
            forward=forward,
            expiry_years=expiry,
            beta=DEFAULT_SPY_SABR.beta,
            rho=DEFAULT_SPY_SABR.rho,
            nu=DEFAULT_SPY_SABR.nu,
        )

        vol_95_pct = model.implied_vol(strike=475, forward=forward, expiry_years=expiry)
        vol_atm = model.implied_vol(strike=500, forward=forward, expiry_years=expiry)
        vol_105_pct = model.implied_vol(strike=525, forward=forward, expiry_years=expiry)

        # SPY-like skew: puts more expensive than calls (OTM puts > OTM calls)
        assert vol_95_pct > vol_105_pct, "OTM puts should be more expensive than OTM calls"
        # OTM puts should be above ATM
        assert vol_95_pct > vol_atm, "OTM puts should be above ATM"


class TestBSMPricerWithVolModel:
    """Test BSMPricer integration with volatility models."""

    def test_bsm_with_flat_vol_model(self):
        """Test BSMPricer works with FlatVolatility model."""
        vol_model = FlatVolatility(volatility=0.20)

        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility_model=vol_model,
        )

        # Should match explicit volatility pricing
        pricer_explicit = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )

        assert pricer.call_price() == pytest.approx(pricer_explicit.call_price())
        assert pricer.put_price() == pytest.approx(pricer_explicit.put_price())

    def test_bsm_with_sabr_model(self):
        """Test BSMPricer works with SABR model."""
        model = SABRVolatility.from_atm_vol(
            atm_vol=0.20,
            forward=100,
            expiry_years=1.0,
            beta=0.5,
            rho=-0.3,
            nu=0.4,
        )

        # Price OTM put and OTM call
        put_pricer = BSMPricer(
            spot=100,
            strike=90,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility_model=model,
        )

        call_pricer = BSMPricer(
            spot=100,
            strike=110,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility_model=model,
        )

        # Put should use higher vol due to skew
        assert put_pricer.volatility > call_pricer.volatility
        # Both should price successfully
        assert put_pricer.put_price() > 0
        assert call_pricer.call_price() > 0

    def test_bsm_requires_vol_or_model(self):
        """Test BSMPricer raises error if neither vol nor model provided."""
        with pytest.raises(ValueError, match="Either volatility or volatility_model"):
            BSMPricer(
                spot=100,
                strike=100,
                expiry_years=1.0,
                risk_free_rate=0.05,
            )

    def test_bsm_vol_model_takes_precedence(self):
        """Test volatility_model takes precedence over explicit volatility."""
        vol_model = FlatVolatility(volatility=0.30)

        pricer = BSMPricer(
            spot=100,
            strike=100,
            expiry_years=1.0,
            risk_free_rate=0.05,
            volatility=0.20,  # This should be ignored
            volatility_model=vol_model,
        )

        assert pricer.volatility == 0.30  # From model, not explicit


class TestPutCallParityWithSkew:
    """Test put-call parity still holds with skewed volatilities."""

    def test_put_call_parity_with_different_vols(self):
        """Put-call parity holds even when using different vols per strike.

        Note: Put-call parity relates C and P at the SAME strike, so even
        with skew, each strike uses its own consistent volatility.
        """
        model = SABRVolatility.from_atm_vol(
            atm_vol=0.20,
            forward=100,
            expiry_years=0.5,
        )

        for strike in [90, 100, 110]:
            pricer = BSMPricer(
                spot=100,
                strike=strike,
                expiry_years=0.5,
                risk_free_rate=0.05,
                volatility_model=model,
            )

            call = pricer.call_price()
            put = pricer.put_price()

            # Put-call parity: C - P = S - K*e^(-rT)
            expected = 100 - strike * math.exp(-0.05 * 0.5)
            assert abs((call - put) - expected) < 0.001
