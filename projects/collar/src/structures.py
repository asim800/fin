"""Core data structures for option collar positions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from lib.pricing import OptionGreeks


@dataclass
class CollarLeg:
    """Represents a single leg of a collar position.

    Attributes:
        option_type: Type of option ("call" or "put").
        strike: Strike price of the option.
        expiry: Expiration date of the option.
        position: Position direction ("long" or "short").
        contracts: Number of contracts (default 1).
    """

    option_type: Literal["call", "put"]
    strike: float
    expiry: date
    position: Literal["long", "short"]
    contracts: int = 1


@dataclass
class CollarPosition:
    """Represents a complete collar position.

    A collar consists of:
    - Long underlying shares
    - Long protective put
    - Short covered call

    Attributes:
        underlying_symbol: Ticker symbol of the underlying.
        underlying_shares: Number of shares held.
        underlying_entry_price: Entry price of the underlying.
        entry_date: Date the position was entered.
        long_put: The protective put leg.
        short_call: The covered call leg.
    """

    underlying_symbol: str
    underlying_shares: int
    underlying_entry_price: float
    entry_date: date
    long_put: CollarLeg
    short_call: CollarLeg

    @property
    def put_strike_pct(self) -> float:
        """Put strike as percentage of entry price."""
        return self.long_put.strike / self.underlying_entry_price

    @property
    def call_strike_pct(self) -> float:
        """Call strike as percentage of entry price."""
        return self.short_call.strike / self.underlying_entry_price

    @property
    def protection_level(self) -> float:
        """Maximum downside as percentage (negative value).

        Returns the percentage loss at which the put protection kicks in.
        """
        return self.put_strike_pct - 1.0

    @property
    def upside_cap(self) -> float:
        """Maximum upside as percentage.

        Returns the percentage gain at which gains are capped by the short call.
        """
        return self.call_strike_pct - 1.0


@dataclass
class CollarPricing:
    """Pricing information for a collar position.

    All values are per-share basis unless otherwise noted.

    Attributes:
        put_price: Put premium per share.
        call_price: Call premium per share.
        net_premium: Net cost per share (put_price - call_price).
        net_premium_pct: Net cost as percentage of spot price.
        put_greeks: Greeks for the put leg.
        call_greeks: Greeks for the call leg.
        net_delta: Net position delta (1 + put_delta - call_delta).
        net_gamma: Net position gamma (put_gamma - call_gamma).
        net_vega: Net position vega (put_vega - call_vega).
        net_theta: Net position theta (-put_theta + call_theta).
    """

    put_price: float
    call_price: float
    net_premium: float
    net_premium_pct: float
    put_greeks: OptionGreeks
    call_greeks: OptionGreeks
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float

    @classmethod
    def from_prices_and_greeks(
        cls,
        put_price: float,
        call_price: float,
        spot: float,
        put_greeks: OptionGreeks,
        call_greeks: OptionGreeks,
    ) -> CollarPricing:
        """Create CollarPricing from component prices and Greeks.

        Args:
            put_price: Put premium per share.
            call_price: Call premium per share.
            spot: Current spot price for percentage calculation.
            put_greeks: Greeks for the put leg.
            call_greeks: Greeks for the call leg.

        Returns:
            CollarPricing instance with computed net values.
        """
        net_premium = put_price - call_price

        return cls(
            put_price=put_price,
            call_price=call_price,
            net_premium=net_premium,
            net_premium_pct=net_premium / spot,
            put_greeks=put_greeks,
            call_greeks=call_greeks,
            # Position delta: long stock (1) + long put (negative) - short call
            net_delta=1.0 + put_greeks.delta - call_greeks.delta,
            # Long put gamma - short call gamma
            net_gamma=put_greeks.gamma - call_greeks.gamma,
            # Long put vega - short call vega
            net_vega=put_greeks.vega - call_greeks.vega,
            # Short put theta (receive decay) + short call theta (receive decay)
            # Long put loses theta, short call gains theta
            net_theta=-put_greeks.theta + call_greeks.theta,
        )
