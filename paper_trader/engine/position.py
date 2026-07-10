"""
Position data model.

A Position represents a live (or previously live) holding in a single
symbol. It knows how to update itself given a new market price and how
to compute its own PnL, but it does NOT know when it should be closed
or scanned — that decision-making belongs to PositionManager, which
owns the collection of positions and consults the market feed and risk
rules. This keeps Position a small, single-responsibility data model
that's trivial to unit test in isolation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from engine.order import OrderSide


class PositionStatus(str, Enum):
    """Lifecycle states of a position."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    """Why a position was closed.

    Used by PositionManager when closing a position and recorded on
    the resulting Trade in analytics.trade_history, so performance
    reporting can later break down PnL by exit reason (e.g. "how much
    am I losing to stop-outs vs. manual exits?").
    """

    STOP_LOSS = "STOP_LOSS"
    TARGET = "TARGET"
    MANUAL = "MANUAL"
    SIGNAL = "SIGNAL"
    END_OF_DAY = "END_OF_DAY"


def _new_position_id() -> str:
    """Generate a short, unique position identifier."""
    return f"POS-{uuid.uuid4().hex[:12]}"


@dataclass
class Position:
    """A single open or closed holding in one symbol.

    Attributes
    ----------
    symbol:
        Trading symbol, e.g. "RELIANCE".
    side:
        BUY (long) or SELL (short). Determines the sign convention
        used when computing PnL.
    quantity:
        Number of shares/units held.
    entry_price:
        Price at which the position was opened.
    current_price:
        Most recent known market price; updated via update_price().
    stop_loss:
        Optional stop-loss price. None means no stop-loss is set.
    target:
        Optional target (take-profit) price. None means no target is set.
    entry_time:
        When the position was opened.
    exit_time:
        When the position was closed. None while still open.
    exit_price:
        Price at which the position was closed. None while still open.
    exit_reason:
        Why the position was closed. None while still open.
    status:
        OPEN or CLOSED.
    unrealized_pnl:
        Mark-to-market PnL while the position is open. Always 0 once closed.
    realized_pnl:
        Locked-in PnL once the position is closed. Always 0 while open.
    """

    symbol: str
    side: OrderSide
    quantity: int
    entry_price: float

    position_id: str = field(default_factory=_new_position_id)
    current_price: float = 0.0
    stop_loss: float | None = None
    target: float | None = None

    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None

    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Position quantity must be greater than 0.")
        if self.entry_price <= 0:
            raise ValueError("Position entry_price must be greater than 0.")
        if self.current_price == 0.0:
            self.current_price = self.entry_price
            self._recompute_unrealized_pnl()

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    @property
    def is_long(self) -> bool:
        return self.side == OrderSide.BUY

    def _signed_pnl(self, price: float) -> float:
        """PnL at a given price, respecting long/short direction."""
        direction = 1 if self.is_long else -1
        return (price - self.entry_price) * self.quantity * direction

    def _recompute_unrealized_pnl(self) -> None:
        self.unrealized_pnl = self._signed_pnl(self.current_price)

    def update_price(self, price: float) -> None:
        """Update the position's mark-to-market price and unrealized PnL.

        No-op if the position is already closed, since closed positions
        should not keep mark-to-marking against new prices.
        """
        if not self.is_open:
            return
        if price <= 0:
            raise ValueError("price must be greater than 0.")
        self.current_price = price
        self._recompute_unrealized_pnl()

    def is_stop_loss_hit(self) -> bool:
        """True if the current price has breached the stop-loss level."""
        if not self.is_open or self.stop_loss is None:
            return False
        return self.current_price <= self.stop_loss if self.is_long else self.current_price >= self.stop_loss

    def is_target_hit(self) -> bool:
        """True if the current price has reached the target level."""
        if not self.is_open or self.target is None:
            return False
        return self.current_price >= self.target if self.is_long else self.current_price <= self.target

    def close(self, exit_price: float, reason: ExitReason, exit_time: datetime | None = None) -> None:
        """Close the position, locking in realized PnL.

        Parameters
        ----------
        exit_price:
            Price at which the position is closed.
        reason:
            Why the position is being closed.
        exit_time:
            Timestamp of the close; defaults to now.
        """
        if not self.is_open:
            raise ValueError("Position is already closed.")
        if exit_price <= 0:
            raise ValueError("exit_price must be greater than 0.")

        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED
        self.realized_pnl = self._signed_pnl(exit_price)
        self.unrealized_pnl = 0.0
