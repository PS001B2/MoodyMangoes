"""
Market data abstraction.

This is the single most important seam in the whole framework: the
Exchange and every Strategy depend ONLY on MarketDataSource, never on
a concrete provider like Groww. To plug in a different data source
later (another broker, a WebSocket feed, a historical replay engine
for backtesting), you write a new class that implements
MarketDataSource and hand it to the Exchange at startup — nothing in
engine/ or strategy/ needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class PriceUnavailableError(Exception):
    """Raised when a price cannot be retrieved for a requested symbol.

    Callers (Exchange, PositionManager) should treat this as "try again
    later" rather than a fatal error — a single missed price update
    should not crash the trading loop.
    """

    def __init__(self, symbol: str, reason: str = "") -> None:
        self.symbol = symbol
        self.reason = reason
        message = f"Price unavailable for '{symbol}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class MarketDataSource(ABC):
    """Abstract interface for anything that can supply market prices.

    Concrete implementations (GrowwMarketFeed, a future WebSocket feed,
    a HistoricalReplayFeed for backtesting, etc.) must implement
    `get_price`. Everything else in the framework — Exchange,
    PositionManager, Strategy — depends only on this interface.
    """

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return the most recent known price for `symbol`.

        Parameters
        ----------
        symbol:
            Trading symbol, e.g. "RELIANCE".

        Returns
        -------
        float
            Last traded price.

        Raises
        ------
        PriceUnavailableError
            If the price cannot be retrieved right now.
        """
        raise NotImplementedError

    def get_prices(self, symbols: Iterable[str]) -> dict[str, float]:
        """Return prices for multiple symbols at once.

        Default implementation just calls get_price() per symbol, so
        subclasses only need to implement the single-symbol case to be
        usable. Subclasses backed by an API that supports true batch
        quotes (Groww's does, for example) can override this for
        efficiency without changing the public contract.

        A symbol that fails to price is simply omitted from the result
        rather than aborting the whole batch, since one bad/delisted
        symbol shouldn't block price updates for everything else.
        """
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                prices[symbol] = self.get_price(symbol)
            except PriceUnavailableError:
                continue
        return prices

    def is_available(self, symbol: str) -> bool:
        """Convenience check: True if a price can currently be fetched."""
        try:
            self.get_price(symbol)
            return True
        except PriceUnavailableError:
            return False
