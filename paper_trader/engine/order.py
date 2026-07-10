"""
Order data model.

An Order represents *intent* to trade plus its execution lifecycle. It
does not know how to get executed — that responsibility belongs to
OrderManager (creates/tracks orders) and Exchange (decides execution
and fills them). Keeping Order a plain data model means the same class
works unchanged whether the fill comes from the paper trading
simulator or, later, a real broker's order confirmation callback.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    """Direction of an order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types.

    Only MARKET orders are executed in the first version of the
    engine (see Exchange). LIMIT and STOP exist in the model now so
    the Order/OrderManager/Exchange interfaces don't need to change
    shape when limit/stop execution logic is added later.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    """Lifecycle states of an order.

    PENDING -> SUBMITTED -> FILLED | PARTIALLY_FILLED | REJECTED | CANCELLED

    PARTIALLY_FILLED is included even though the first version of the
    engine only supports full market fills, so downstream consumers
    (e.g. analytics) can already branch on it correctly once partial
    fills are implemented.
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


def _new_order_id() -> str:
    """Generate a short, unique order identifier."""
    return f"ORD-{uuid.uuid4().hex[:12]}"


@dataclass
class Order:
    """A single order submitted to the Exchange.

    Attributes
    ----------
    order_id:
        Unique identifier, auto-generated if not supplied.
    symbol:
        Trading symbol, e.g. "RELIANCE".
    side:
        BUY or SELL.
    quantity:
        Number of shares/units requested.
    order_type:
        MARKET, LIMIT, or STOP.
    limit_price:
        Required for LIMIT orders; the price at or better than which
        the order should fill. Ignored for MARKET orders.
    stop_price:
        Required for STOP orders; the trigger price. Ignored for
        MARKET orders.
    timestamp:
        When the order was created.
    status:
        Current lifecycle state.
    filled_quantity:
        Quantity filled so far (supports future partial fills).
    avg_fill_price:
        Volume-weighted average price of fills so far. None until at
        least one fill has occurred.
    rejection_reason:
        Populated when status is REJECTED, explaining why (e.g. failed
        a RiskManager check).
    """

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None

    order_id: str = field(default_factory=_new_order_id)
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float | None = None
    rejection_reason: str | None = None
    position_id: str | None = None
    """The Position this order resulted in (set once filled by Exchange).
    None for orders that were rejected/cancelled before ever opening or
    closing a position."""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be greater than 0.")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require a limit_price.")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP orders require a stop_price.")

    @property
    def is_open(self) -> bool:
        """True if the order can still be filled or cancelled."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)

    @property
    def remaining_quantity(self) -> int:
        """Quantity still unfilled."""
        return self.quantity - self.filled_quantity

    def mark_submitted(self) -> None:
        """Transition PENDING -> SUBMITTED."""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in status {self.status}.")
        self.status = OrderStatus.SUBMITTED

    def apply_fill(self, fill_quantity: int, fill_price: float) -> None:
        """Record a fill (full or partial) against this order.

        Updates filled_quantity, recomputes the volume-weighted average
        fill price, and transitions status to FILLED or
        PARTIALLY_FILLED as appropriate.
        """
        if not self.is_open:
            raise ValueError(f"Cannot fill an order in status {self.status}.")
        if fill_quantity <= 0:
            raise ValueError("fill_quantity must be greater than 0.")
        if fill_quantity > self.remaining_quantity:
            raise ValueError("fill_quantity exceeds remaining_quantity.")

        prior_value = (self.avg_fill_price or 0.0) * self.filled_quantity
        new_value = prior_value + (fill_price * fill_quantity)
        self.filled_quantity += fill_quantity
        self.avg_fill_price = new_value / self.filled_quantity

        self.status = (
            OrderStatus.FILLED
            if self.remaining_quantity == 0
            else OrderStatus.PARTIALLY_FILLED
        )

    def reject(self, reason: str) -> None:
        """Mark the order as rejected with a human-readable reason."""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason

    def cancel(self) -> None:
        """Cancel the order, if it is still open."""
        if not self.is_open:
            raise ValueError(f"Cannot cancel an order in status {self.status}.")
        self.status = OrderStatus.CANCELLED

    def link_position(self, position_id: str) -> None:
        """Record which Position this order resulted in (opening or closing)."""
        self.position_id = position_id
