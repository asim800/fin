"""Collar position builder with various construction methods."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Type

from scipy.optimize import brentq

from lib.pricing import BSMPricer, PricingEngine
from lib.volatility import FlatVolatility
from src.structures import CollarLeg, CollarPosition, CollarPricing

if TYPE_CHECKING:
    from lib.volatility import VolatilityModel


class CollarBuilder:
    """Builder for constructing collar positions with pricing.

    Provides methods to build collars with explicit strikes, symmetric OTM
    percentages, or zero-cost optimization.

    Args:
        pricing_engine: Pricing engine class to use (default BSMPricer).
        risk_free_rate: Annual risk-free rate (default 0.05).
        dividend_yield: Annual dividend yield (default 0.0).
        volatility_model: Optional volatility model for strike-dependent vol.
    """

    def __init__(
        self,
        pricing_engine: Type[PricingEngine] = BSMPricer,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        volatility_model: VolatilityModel | None = None,
    ) -> None:
        self.pricing_engine = pricing_engine
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.volatility_model = volatility_model

    def build_collar(
        self,
        symbol: str,
        spot: float,
        shares: int,
        put_strike: float,
        call_strike: float,
        expiry: date,
        volatility: float | None = None,
        entry_date: date | None = None,
    ) -> tuple[CollarPosition, CollarPricing]:
        """Build a collar position with explicit strikes.

        Args:
            symbol: Underlying ticker symbol.
            spot: Current underlying price.
            shares: Number of shares in the position.
            put_strike: Strike price for the protective put.
            call_strike: Strike price for the covered call.
            expiry: Option expiration date.
            volatility: Annualized volatility assumption. Optional if
                volatility_model was provided to constructor.
            entry_date: Position entry date (default today).

        Returns:
            Tuple of (CollarPosition, CollarPricing).

        Raises:
            ValueError: If neither volatility nor volatility_model is provided.
        """
        if entry_date is None:
            entry_date = date.today()

        # Determine volatility model to use
        vol_model = self._get_volatility_model(volatility)

        # Calculate time to expiry in years
        days_to_expiry = (expiry - entry_date).days
        expiry_years = days_to_expiry / 365.0

        # Price the put (with strike-specific volatility from model)
        put_pricer = self.pricing_engine(
            spot=spot,
            strike=put_strike,
            expiry_years=expiry_years,
            risk_free_rate=self.risk_free_rate,
            volatility_model=vol_model,
            dividend_yield=self.dividend_yield,
        )

        # Price the call (with strike-specific volatility from model)
        call_pricer = self.pricing_engine(
            spot=spot,
            strike=call_strike,
            expiry_years=expiry_years,
            risk_free_rate=self.risk_free_rate,
            volatility_model=vol_model,
            dividend_yield=self.dividend_yield,
        )

        # Build the position
        position = CollarPosition(
            underlying_symbol=symbol,
            underlying_shares=shares,
            underlying_entry_price=spot,
            entry_date=entry_date,
            long_put=CollarLeg(
                option_type="put",
                strike=put_strike,
                expiry=expiry,
                position="long",
                contracts=shares // 100 or 1,
            ),
            short_call=CollarLeg(
                option_type="call",
                strike=call_strike,
                expiry=expiry,
                position="short",
                contracts=shares // 100 or 1,
            ),
        )

        # Build pricing
        pricing = CollarPricing.from_prices_and_greeks(
            put_price=put_pricer.put_price(),
            call_price=call_pricer.call_price(),
            spot=spot,
            put_greeks=put_pricer.greeks("put"),
            call_greeks=call_pricer.greeks("call"),
        )

        return position, pricing

    def build_symmetric_collar(
        self,
        symbol: str,
        spot: float,
        shares: int,
        otm_pct: float,
        expiry: date,
        volatility: float | None = None,
        entry_date: date | None = None,
    ) -> tuple[CollarPosition, CollarPricing]:
        """Build a symmetric collar with equal OTM percentages.

        Args:
            symbol: Underlying ticker symbol.
            spot: Current underlying price.
            shares: Number of shares in the position.
            otm_pct: Out-of-the-money percentage (e.g., 0.05 for 5%).
            expiry: Option expiration date.
            volatility: Annualized volatility assumption. Optional if
                volatility_model was provided to constructor.
            entry_date: Position entry date (default today).

        Returns:
            Tuple of (CollarPosition, CollarPricing).
        """
        put_strike = spot * (1.0 - otm_pct)
        call_strike = spot * (1.0 + otm_pct)

        return self.build_collar(
            symbol=symbol,
            spot=spot,
            shares=shares,
            put_strike=put_strike,
            call_strike=call_strike,
            expiry=expiry,
            volatility=volatility,
            entry_date=entry_date,
        )

    def find_zero_cost_collar(
        self,
        symbol: str,
        spot: float,
        shares: int,
        put_strike: float,
        expiry: date,
        volatility: float | None = None,
        tolerance: float = 0.01,
        entry_date: date | None = None,
    ) -> tuple[CollarPosition, CollarPricing]:
        """Find a zero-cost collar for a given put strike.

        Uses optimization to find the call strike that results in
        approximately zero net premium.

        Args:
            symbol: Underlying ticker symbol.
            spot: Current underlying price.
            shares: Number of shares in the position.
            put_strike: Fixed put strike price.
            expiry: Option expiration date.
            volatility: Annualized volatility assumption. Optional if
                volatility_model was provided to constructor.
            tolerance: Acceptable net premium tolerance (default $0.01).
            entry_date: Position entry date (default today).

        Returns:
            Tuple of (CollarPosition, CollarPricing).
        """
        if entry_date is None:
            entry_date = date.today()

        # Determine volatility model to use
        vol_model = self._get_volatility_model(volatility)

        days_to_expiry = (expiry - entry_date).days
        expiry_years = days_to_expiry / 365.0

        # Get put price first
        put_pricer = self.pricing_engine(
            spot=spot,
            strike=put_strike,
            expiry_years=expiry_years,
            risk_free_rate=self.risk_free_rate,
            volatility_model=vol_model,
            dividend_yield=self.dividend_yield,
        )
        put_price = put_pricer.put_price()

        def net_premium_at_strike(call_strike: float) -> float:
            """Calculate net premium for a given call strike."""
            call_pricer = self.pricing_engine(
                spot=spot,
                strike=call_strike,
                expiry_years=expiry_years,
                risk_free_rate=self.risk_free_rate,
                volatility_model=vol_model,
                dividend_yield=self.dividend_yield,
            )
            return put_price - call_pricer.call_price()

        # Search for call strike between spot and 2x spot
        # Net premium should go from positive (high call strike, low premium)
        # to negative (low call strike, high premium)
        try:
            optimal_call_strike = brentq(
                net_premium_at_strike,
                spot * 0.9,  # Lower bound (ITM call)
                spot * 1.5,  # Upper bound (deep OTM call)
                xtol=tolerance,
            )
        except ValueError:
            # If brentq fails, fall back to ATM call
            optimal_call_strike = spot

        return self.build_collar(
            symbol=symbol,
            spot=spot,
            shares=shares,
            put_strike=put_strike,
            call_strike=optimal_call_strike,
            expiry=expiry,
            volatility=volatility,
            entry_date=entry_date,
        )

    def _get_volatility_model(self, volatility: float | None) -> VolatilityModel:
        """Get volatility model from explicit vol or instance model.

        Args:
            volatility: Explicit volatility value (optional).

        Returns:
            VolatilityModel to use for pricing.

        Raises:
            ValueError: If neither volatility nor volatility_model available.
        """
        if self.volatility_model is not None:
            return self.volatility_model
        elif volatility is not None:
            return FlatVolatility(volatility)
        else:
            raise ValueError(
                "Either volatility parameter or volatility_model in constructor must be provided"
            )
