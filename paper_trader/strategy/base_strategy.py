"""
BaseStrategy: the contract every strategy must implement.

CRITICAL DESIGN CONSTRAINT: a strategy interacts with the outside
world through exactly two objects — `self.exchange` (to place trades)
and `self.market` (to read prices). This module deliberately imports
nothing from `market.groww_feed` or any other concrete data-provider
implementation, only the abstract `MarketDataSource` interface. That
means it is structurally impossible for a subclass written against
this base to reach into Groww directly through inherited attributes —
it only ever sees the abstraction, whatever concrete feed is plugged
in behind it. This is what allows the data source to be swapped later
(WebSocket feed, historical replay engine, a different broker) without
touching a single line of strategy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.exchange import Exchange
from engine.position import Position
from market.market_data import MarketDataSource
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Parameters
    ----------
    exchange:
        The Exchange to place trades through. This is the ONLY way a
        strategy should ever open, close, or modify a position.
    market:
        The market data source to read prices from. This is the ONLY
        way a strategy should ever learn a current price.
    symbols:
        The symbols this strategy operates on.
    """

    def __init__(self, exchange: Exchange, market: MarketDataSource, symbols: list[str]) -> None:
        self.exchange = exchange
        self.market = market
        self.symbols = symbols

    def on_start(self) -> None:
        """Called once before the trading loop begins.

        Override for one-time setup (e.g. loading historical data to
        warm up an indicator, initializing an ML model). Default is a
        no-op — most simple strategies won't need this.
        """

    @abstractmethod
    def on_tick(self) -> None:
        """Called once per market data cycle by the main loop.

        This is where a strategy reads prices via `self.market.get_price()`
        (or `self.market.get_prices()`), evaluates its rules/signals, and
        calls `self.exchange.buy()`, `self.exchange.sell()`,
        `self.exchange.close_position()`, `self.exchange.modify_stop_loss()`,
        or `self.exchange.modify_target()` as appropriate. Every concrete
        strategy must implement this.
        """
        raise NotImplementedError

    def on_position_closed(self, position: Position) -> None:
        """Called when a position closes WITHOUT the strategy explicitly
        requesting it — i.e. an automatic stop-loss or target exit
        detected by `Exchange.process_tick()`.

        Override to react to automatic exits (e.g. clear internal
        tracking state, log the outcome, adjust future sizing).
        Default is a no-op.
        """

    def on_stop(self) -> None:
        """Called once when the trading loop is shutting down.

        Override for teardown (e.g. flushing logs, closing files).
        Default is a no-op.
        """
