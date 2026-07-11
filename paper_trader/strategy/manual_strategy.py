"""
ManualStrategy: Buy-the-Dip strategy for testing.

This strategy is intended for stress-testing the paper trading engine,
not for profitability.

Logic:
- Track the previous price for each symbol.
- If the current price drops by `drop_pct` or more from the previous tick,
  open a long position.
- Let the exchange automatically exit using TP/SL.
- After the position closes, wait for the next qualifying drop.
"""

from __future__ import annotations

from paper_trader.engine.order import OrderStatus
from paper_trader.engine.position import Position, PositionStatus
from paper_trader.engine.exchange import Exchange
from paper_trader.market.market_data import MarketDataSource, PriceUnavailableError
from paper_trader.strategy.base_strategy import BaseStrategy
from paper_trader.utils.logger import get_logger

logger = get_logger(__name__)


class ManualStrategy(BaseStrategy):
    """Buy after a configurable percentage drop from the previous price."""

    def __init__(
        self,
        exchange: Exchange,
        market: MarketDataSource,
        symbols: list[str],
        quantity: int = 1,
        drop_pct: float = 0.20,
        stop_loss_pct: float = 0.10,
        target_pct: float = 0.10,
    ) -> None:
        super().__init__(exchange, market, symbols)

        self._quantity = quantity
        self._drop_pct = drop_pct
        self._stop_loss_pct = stop_loss_pct
        self._target_pct = target_pct

        # Stores the previous tick price for each symbol
        self._previous_price: dict[str, float] = {}

        # One open position per symbol
        self._open_position_id: dict[str, str] = {}

    def on_tick(self) -> None:
        for symbol in self.symbols:
            try:
                price = self.market.get_price(symbol)
            except PriceUnavailableError:
                continue

            self._sync_position_state(symbol)

            previous_price = self._previous_price.get(symbol)

            # First tick
            if previous_price is None:
                self._previous_price[symbol] = price
                continue

            # Skip if already in a position
            if symbol in self._open_position_id:
                self._previous_price[symbol] = price
                continue

            # Calculate percentage drop from previous tick
            drop_pct = ((previous_price - price) / previous_price) * 100

            if drop_pct >= self._drop_pct:
                self._enter_long(symbol, price)

            # Update previous price
            self._previous_price[symbol] = price

    def _sync_position_state(self, symbol: str) -> None:
        """Remove tracking if the position has already been closed."""
        position_id = self._open_position_id.get(symbol)
        if position_id is None:
            return

        position = self.exchange.get_position(position_id)

        if position is None or position.status != PositionStatus.OPEN:
            del self._open_position_id[symbol]

    def _enter_long(self, symbol: str, price: float) -> None:
        stop_loss = price * (1 - self._stop_loss_pct / 100)
        target = price * (1 + self._target_pct / 100)

        order = self.exchange.buy(
            symbol,
            self._quantity,
            stop_loss=stop_loss,
            target=target,
        )

        if order.status == OrderStatus.FILLED and order.position_id is not None:
            self._open_position_id[symbol] = order.position_id

            logger.info(
                "BUY %s x%d @ %.2f | Drop=%.3f%% | SL=%.2f | TP=%.2f",
                symbol,
                self._quantity,
                price,
                self._drop_pct,
                stop_loss,
                target,
            )

        else:
            logger.info(
                "Entry skipped for %s: %s (%s)",
                symbol,
                order.status.value,
                order.rejection_reason,
            )

    def on_position_closed(self, position: Position) -> None:
        """Clear tracking after automatic TP/SL exit."""
        for symbol, position_id in list(self._open_position_id.items()):
            if position_id == position.position_id:
                del self._open_position_id[symbol]

                logger.info(
                    "Position closed: %s (%s) PnL=%.2f",
                    symbol,
                    position.exit_reason.value if position.exit_reason else "?",
                    position.realized_pnl,
                )