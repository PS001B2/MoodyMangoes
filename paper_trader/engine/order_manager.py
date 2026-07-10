"""
OrderManager: owns the order registry and order lifecycle transitions.

Single responsibility: create orders, track every order ever created,
and apply lifecycle transitions (submit / fill / reject / cancel) by
delegating to the Order object's own transition methods. OrderManager
does NOT decide whether an order should be allowed to execute (that is
Exchange's job, consulting RiskManager) and does NOT know about prices,
positions, or capital.
"""

from __future__ import annotations

from engine.order import Order, OrderSide, OrderType
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderNotFoundError(Exception):
    """Raised when an operation references an order_id that doesn't exist."""

    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        super().__init__(f"No order found with order_id '{order_id}'.")


class OrderManager:
    """Creates and tracks the full lifecycle of every order.

    Orders are never deleted from the registry once created, so this
    class doubles as a complete historical order log — `get_open_orders`
    filters live ones, everything else is queryable by id or symbol.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}

    # -- creation ---------------------------------------------------

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Create a new order in PENDING status and register it.

        Parameters mirror Order's constructor; validation (e.g.
        quantity > 0, limit_price required for LIMIT orders) happens
        inside Order.__post_init__ and will raise ValueError if invalid.
        """
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        self._orders[order.order_id] = order
        logger.info(
            "Order created: %s %s %s x%d (%s)",
            order.order_id, order.side.value, order.symbol, order.quantity, order.order_type.value,
        )
        return order

    # -- lookups ------------------------------------------------------

    def get_order(self, order_id: str) -> Order:
        """Return the order with the given ID.

        Raises
        ------
        OrderNotFoundError
            If no order with that ID has ever been created.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise OrderNotFoundError(order_id)
        return order

    def get_open_orders(self) -> list[Order]:
        """All orders that are still PENDING, SUBMITTED, or PARTIALLY_FILLED."""
        return [o for o in self._orders.values() if o.is_open]

    def get_orders_for_symbol(self, symbol: str) -> list[Order]:
        """All orders (any status) ever created for a given symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_all_orders(self) -> list[Order]:
        """The complete order log, in creation order."""
        return list(self._orders.values())

    # -- lifecycle transitions -----------------------------------------

    def submit_order(self, order_id: str) -> Order:
        """Transition an order from PENDING to SUBMITTED."""
        order = self.get_order(order_id)
        order.mark_submitted()
        logger.info("Order submitted: %s", order_id)
        return order

    def fill_order(self, order_id: str, fill_quantity: int, fill_price: float) -> Order:
        """Apply a fill (full or partial) to an order.

        Delegates validation (order must be open, fill_quantity must
        not exceed remaining_quantity, etc.) to Order.apply_fill.
        """
        order = self.get_order(order_id)
        order.apply_fill(fill_quantity, fill_price)
        logger.info(
            "Order filled: %s qty=%d price=%.2f status=%s",
            order_id, fill_quantity, fill_price, order.status.value,
        )
        return order

    def reject_order(self, order_id: str, reason: str) -> Order:
        """Mark an order as REJECTED with a human-readable reason."""
        order = self.get_order(order_id)
        order.reject(reason)
        logger.info("Order rejected: %s reason=%s", order_id, reason)
        return order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel an order, if it is still open."""
        order = self.get_order(order_id)
        order.cancel()
        logger.info("Order cancelled: %s", order_id)
        return order
