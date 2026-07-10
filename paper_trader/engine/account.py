"""
Account: capital and portfolio-level bookkeeping.

Single responsibility: track how much capital exists, how much is
free to trade with, and which positions (open/closed) belong to this
account. Account does NOT decide whether a trade should be allowed
(RiskManager's job) or orchestrate opening/closing trades (Exchange's
job) — it is the ledger those other components read from and write to.
"""

from __future__ import annotations

from datetime import datetime

from engine.position import Position, PositionStatus
from utils.logger import get_logger

logger = get_logger(__name__)


class InsufficientCapitalError(Exception):
    """Raised when an operation would require more capital than is available."""

    def __init__(self, required: float, available: float) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient capital: required {required:.2f}, available {available:.2f}"
        )


class Account:
    """Tracks capital and positions for a single paper trading account.

    Parameters
    ----------
    starting_capital:
        Initial capital the account is funded with. Fixed for the
        lifetime of the account (used as the baseline for total PnL).
    """

    def __init__(self, starting_capital: float) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be greater than 0.")

        self._starting_capital = starting_capital
        self._available_capital = starting_capital

        self._open_positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []

        # Capital reserved (cost basis) per open position, so the exact
        # amount can be released back on close regardless of price moves.
        self._reserved_capital: dict[str, float] = {}

        self._day_start_equity = starting_capital

        logger.info("Account created with starting capital %.2f", starting_capital)

    # -- read-only views -----------------------------------------------

    @property
    def starting_capital(self) -> float:
        return self._starting_capital

    @property
    def available_capital(self) -> float:
        """Free cash available to open new positions."""
        return self._available_capital

    @property
    def current_capital(self) -> float:
        """Total account equity: free cash + market value of open positions.

        Market value of each open position = its reserved (cost basis)
        capital plus its current unrealized PnL, so this always
        reflects live mark-to-market value, not just cash on hand.
        """
        market_value_of_positions = sum(
            self._reserved_capital[pid] + position.unrealized_pnl
            for pid, position in self._open_positions.items()
        )
        return self._available_capital + market_value_of_positions

    @property
    def open_positions(self) -> list[Position]:
        """All currently open positions (read-only snapshot)."""
        return list(self._open_positions.values())

    @property
    def closed_positions(self) -> list[Position]:
        """All closed positions, in the order they were closed (read-only snapshot)."""
        return list(self._closed_positions)

    @property
    def daily_pnl(self) -> float:
        """PnL since the last start_new_day() call (realized + unrealized)."""
        return self.current_capital - self._day_start_equity

    @property
    def total_pnl(self) -> float:
        """PnL since account inception (realized + unrealized)."""
        return self.current_capital - self._starting_capital

    def get_position(self, position_id: str) -> Position | None:
        """Look up a single open position by its ID, or None if not open."""
        return self._open_positions.get(position_id)

    def open_positions_for_symbol(self, symbol: str) -> list[Position]:
        """All currently open positions for a given symbol.

        Returns a list (not a single Position) because "multiple
        simultaneous positions" per symbol is a planned future feature;
        callers that only expect one position today can just take
        index 0 or len() == 0/1.
        """
        return [p for p in self._open_positions.values() if p.symbol == symbol]

    # -- mutating operations --------------------------------------------

    def has_sufficient_capital(self, amount: float) -> bool:
        """Check without raising whether `amount` of capital is currently free."""
        return amount <= self._available_capital

    def reserve_capital(self, position: Position, amount: float | None = None) -> None:
        """Reserve capital for a newly opened position and register it as open.

        Parameters
        ----------
        position:
            The Position being opened. Must currently be OPEN.
        amount:
            Capital to reserve. Defaults to entry_price * quantity (the
            full cost basis), which is correct for cash-segment equity.
            An explicit amount can be passed later for margin products
            (F&O) where reserved capital differs from notional value.

        Raises
        ------
        InsufficientCapitalError
            If the requested amount exceeds available_capital.
        """
        if position.status != PositionStatus.OPEN:
            raise ValueError("Cannot reserve capital for a position that is not OPEN.")

        reserved = amount if amount is not None else position.entry_price * position.quantity
        if reserved > self._available_capital:
            raise InsufficientCapitalError(required=reserved, available=self._available_capital)

        self._available_capital -= reserved
        self._reserved_capital[position.position_id] = reserved
        self._open_positions[position.position_id] = position

        logger.info(
            "Reserved %.2f for position %s (%s x%d @ %.2f). Available capital: %.2f",
            reserved, position.position_id, position.symbol, position.quantity,
            position.entry_price, self._available_capital,
        )

    def release_capital(self, position: Position) -> None:
        """Release reserved capital and record realized PnL for a closed position.

        Parameters
        ----------
        position:
            The Position being closed. Must already be CLOSED (i.e.
            Position.close() must have been called first, so
            realized_pnl is populated) and must currently be tracked
            as open on this account.
        """
        if position.status != PositionStatus.CLOSED:
            raise ValueError("Cannot release capital for a position that is not CLOSED.")
        if position.position_id not in self._open_positions:
            raise ValueError(f"Position {position.position_id} is not open on this account.")

        reserved = self._reserved_capital.pop(position.position_id)
        self._available_capital += reserved + position.realized_pnl

        del self._open_positions[position.position_id]
        self._closed_positions.append(position)

        logger.info(
            "Released position %s (%s). Realized PnL: %.2f. Available capital: %.2f",
            position.position_id, position.symbol, position.realized_pnl, self._available_capital,
        )

    def start_new_day(self) -> None:
        """Snapshot current equity as the baseline for daily_pnl.

        Should be called once at the start of each trading session
        (e.g. from main.py before the market opens). Does not affect
        available_capital, open_positions, or total_pnl.
        """
        self._day_start_equity = self.current_capital
        logger.info("New trading day started. Equity baseline: %.2f", self._day_start_equity)
