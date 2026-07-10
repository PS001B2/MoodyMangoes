"""
PositionManager: keeps open positions marked-to-market and closes them
when stop-loss/target levels are hit.

Single responsibility: update prices/PnL on open positions, detect
stop-loss and target hits, and close positions (delegating capital
bookkeeping to Account). It does NOT store its own copy of the
position list — Account is the single source of truth for which
positions are open or closed, so there is never a sync question
between two registries. PositionManager is the behavior layer on top
of that data.
"""

from __future__ import annotations

from datetime import datetime

from engine.account import Account
from engine.order import OrderSide
from engine.position import ExitReason, Position
from market.market_data import MarketDataSource, PriceUnavailableError
from utils.logger import get_logger

logger = get_logger(__name__)


class PositionNotFoundError(Exception):
    """Raised when an operation references a position_id that isn't open."""

    def __init__(self, position_id: str) -> None:
        self.position_id = position_id
        super().__init__(f"No open position found with position_id '{position_id}'.")


class PositionManager:
    """Manages the lifecycle of open positions: pricing, exits, closing.

    Parameters
    ----------
    market:
        Source of live prices. Only MarketDataSource.get_price is used,
        so any implementation (Groww, a future WebSocket feed, a
        historical replay feed) works unchanged.
    account:
        The account whose open/closed position ledger this manager
        operates on.
    """

    def __init__(self, market: MarketDataSource, account: Account) -> None:
        self._market = market
        self._account = account

    # -- opening --------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        stop_loss: float | None = None,
        target: float | None = None,
        entry_time: datetime | None = None,
    ) -> Position:
        """Open a new position and reserve capital for it on the account.

        Raises
        ------
        InsufficientCapitalError
            Propagated from Account.reserve_capital if there isn't
            enough available capital.
        """
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            entry_time=entry_time or datetime.now(),
        )
        self._account.reserve_capital(position)
        logger.info(
            "Position opened: %s %s %s x%d @ %.2f (SL=%s, target=%s)",
            position.position_id, side.value, symbol, quantity, entry_price,
            stop_loss, target,
        )
        return position

    # -- price updates ---------------------------------------------------

    def update_prices(self) -> None:
        """Refresh current_price (and therefore unrealized PnL) for every open position.

        A symbol whose price can't be fetched right now is logged and
        skipped rather than raising — one flaky quote shouldn't halt
        mark-to-market for every other open position.
        """
        for position in self._account.open_positions:
            try:
                price = self._market.get_price(position.symbol)
            except PriceUnavailableError as exc:
                logger.warning(
                    "Skipping price update for %s (%s): %s",
                    position.position_id, position.symbol, exc,
                )
                continue
            position.update_price(price)

    # -- exit detection ---------------------------------------------------

    def check_exits(self) -> list[Position]:
        """Close any open position whose stop-loss or target has been hit.

        Should be called after update_prices() so current_price is
        fresh. Checks stop-loss before target when, hypothetically,
        both conditions are met on the same tick (conservative: assume
        the adverse move happened first).

        Returns
        -------
        list[Position]
            Positions closed during this call, in the order they were closed.
        """
        closed: list[Position] = []
        for position in list(self._account.open_positions):
            if position.is_stop_loss_hit():
                self.close_position(position.position_id, position.current_price, ExitReason.STOP_LOSS)
                closed.append(position)
            elif position.is_target_hit():
                self.close_position(position.position_id, position.current_price, ExitReason.TARGET)
                closed.append(position)
        return closed

    # -- closing ---------------------------------------------------------

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: ExitReason,
        exit_time: datetime | None = None,
    ) -> Position:
        """Close an open position and release its capital back to the account.

        Raises
        ------
        PositionNotFoundError
            If position_id doesn't correspond to a currently open position.
        """
        position = self._account.get_position(position_id)
        if position is None:
            raise PositionNotFoundError(position_id)

        position.close(exit_price, reason, exit_time)
        self._account.release_capital(position)
        logger.info(
            "Position closed: %s (%s) exit=%.2f reason=%s realized_pnl=%.2f",
            position_id, position.symbol, exit_price, reason.value, position.realized_pnl,
        )
        return position

    # -- modification ------------------------------------------------------

    def modify_stop_loss(self, position_id: str, new_stop_loss: float | None) -> Position:
        """Update the stop-loss level for an open position."""
        position = self._get_open_position_or_raise(position_id)
        position.stop_loss = new_stop_loss
        logger.info("Stop-loss modified for %s: %s", position_id, new_stop_loss)
        return position

    def modify_target(self, position_id: str, new_target: float | None) -> Position:
        """Update the target level for an open position."""
        position = self._get_open_position_or_raise(position_id)
        position.target = new_target
        logger.info("Target modified for %s: %s", position_id, new_target)
        return position

    # -- lookups ------------------------------------------------------------

    def get_position(self, position_id: str) -> Position | None:
        return self._account.get_position(position_id)

    def get_open_positions(self) -> list[Position]:
        return self._account.open_positions

    def get_positions_for_symbol(self, symbol: str) -> list[Position]:
        return self._account.open_positions_for_symbol(symbol)

    def get_closed_positions(self) -> list[Position]:
        return self._account.closed_positions

    def _get_open_position_or_raise(self, position_id: str) -> Position:
        position = self._account.get_position(position_id)
        if position is None:
            raise PositionNotFoundError(position_id)
        return position
