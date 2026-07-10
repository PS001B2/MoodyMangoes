"""
ManualStrategy: a simple, illustrative rule-based strategy.

This is a TEMPLATE, not a production trading strategy. It demonstrates
the pattern for writing a BaseStrategy subclass: read prices via
self.market, make a decision, act via self.exchange. Replace the rule
in on_tick() with your own logic — the surrounding scaffolding
(tracking one open position per symbol, reacting to automatic exits)
is the reusable part.

The example rule implemented here is a simple SMA breakout: once a
symbol's rolling simple moving average has enough history, go long
when price closes above the SMA, with a fixed percentage stop-loss
and target. It intentionally does not short, pyramid, or manage
multiple concurrent positions per symbol — see Future Features in the
project's design doc for where those extensions would plug in.
"""

from __future__ import annotations

from collections import deque

from engine.order import OrderStatus
from engine.position import Position, PositionStatus
from engine.exchange import Exchange
from market.market_data import MarketDataSource, PriceUnavailableError
from strategy.base_strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


class ManualStrategy(BaseStrategy):
    """Example SMA-breakout strategy with fixed percentage SL/target.

    Parameters
    ----------
    exchange, market, symbols:
        See BaseStrategy.
    quantity:
        Fixed order quantity per entry.
    sma_period:
        Number of recent prices used to compute the simple moving average.
    stop_loss_pct:
        Stop-loss distance below entry price, as a percentage (e.g. 1.0 = 1%).
    target_pct:
        Target distance above entry price, as a percentage (e.g. 2.0 = 2%).
    """

    def __init__(
        self,
        exchange: Exchange,
        market: MarketDataSource,
        symbols: list[str],
        quantity: int = 1,
        sma_period: int = 5,
        stop_loss_pct: float = 1.0,
        target_pct: float = 2.0,
    ) -> None:
        super().__init__(exchange, market, symbols)
        self._quantity = quantity
        self._sma_period = sma_period
        self._stop_loss_pct = stop_loss_pct
        self._target_pct = target_pct

        self._price_history: dict[str, deque[float]] = {
            symbol: deque(maxlen=sma_period) for symbol in symbols
        }
        # Tracks at most one open position per symbol for this simple
        # example. A strategy supporting multiple concurrent positions
        # per symbol would track a list of position_ids here instead.
        self._open_position_id: dict[str, str] = {}

    def on_tick(self) -> None:
        for symbol in self.symbols:
            try:
                price = self.market.get_price(symbol)
            except PriceUnavailableError:
                continue

            history = self._price_history[symbol]
            history.append(price)

            self._sync_position_state(symbol)

            if symbol in self._open_position_id:
                continue  # already in a position for this symbol; example strategy doesn't add to it

            if len(history) < self._sma_period:
                continue  # not enough history yet to compute the SMA

            sma = sum(history) / len(history)
            if price > sma:
                self._enter_long(symbol, price)

    def _sync_position_state(self, symbol: str) -> None:
        """Drop tracking for a symbol if its position was already closed
        (e.g. by an automatic SL/target exit) since the last tick."""
        position_id = self._open_position_id.get(symbol)
        if position_id is None:
            return
        position = self.exchange.get_position(position_id)
        if position is None or position.status != PositionStatus.OPEN:
            del self._open_position_id[symbol]

    def _enter_long(self, symbol: str, price: float) -> None:
        stop_loss = price * (1 - self._stop_loss_pct / 100)
        target = price * (1 + self._target_pct / 100)

        order = self.exchange.buy(symbol, self._quantity, stop_loss=stop_loss, target=target)
        if order.status == OrderStatus.FILLED and order.position_id is not None:
            self._open_position_id[symbol] = order.position_id
            logger.info(
                "ManualStrategy entered long %s x%d @ %.2f (SL=%.2f, target=%.2f)",
                symbol, self._quantity, price, stop_loss, target,
            )
        else:
            logger.info(
                "ManualStrategy entry skipped for %s: order %s (%s)",
                symbol, order.status.value, order.rejection_reason,
            )

    def on_position_closed(self, position: Position) -> None:
        """Clear tracking when a position we opened is closed automatically."""
        for symbol, position_id in list(self._open_position_id.items()):
            if position_id == position.position_id:
                del self._open_position_id[symbol]
                logger.info(
                    "ManualStrategy position auto-closed: %s (%s) pnl=%.2f",
                    symbol, position.exit_reason.value if position.exit_reason else "?",
                    position.realized_pnl,
                )
