"""
Exchange: the core orchestrator and the ONLY class strategies talk to.

Single responsibility: decide whether a proposed trade gets executed,
and if so, execute it — by coordinating OrderManager (order lifecycle),
RiskManager (policy checks), Account (capital), and PositionManager
(position lifecycle). Exchange contains no state of its own; it is a
thin coordination layer over those four collaborators.

This is the seam that will eventually be swapped for real broker
execution: a future LiveExchange (or a pluggable execution handler
inside this same class) would replace the "fill instantly at current
market price" logic below with an actual broker order placement +
async fill confirmation, while keeping this exact public interface
(buy/sell/close_position/modify_stop_loss/modify_target/cancel_order)
unchanged — so Strategy code never needs to change.
"""

from __future__ import annotations

from paper_trader.engine.account import Account, InsufficientCapitalError
from paper_trader.engine.order import Order, OrderSide, OrderType
from paper_trader.engine.order_manager import OrderManager
from paper_trader.engine.position import ExitReason, Position
from paper_trader.engine.position_manager import PositionManager, PositionNotFoundError
from paper_trader.engine.risk_manager import RiskManager, RiskViolationError
from paper_trader.market.market_data import MarketDataSource, PriceUnavailableError
from paper_trader.utils.logger import get_logger

logger = get_logger(__name__)


class Exchange:
    """The core paper trading exchange.

    Parameters
    ----------
    market:
        Source of live prices.
    account:
        The trading account whose capital and positions this exchange operates on.
    order_manager:
        Tracks the order registry and lifecycle transitions.
    position_manager:
        Manages position pricing, SL/target detection, and closing.
    risk_manager:
        Validates proposed trades against configured risk limits.

    Notes
    -----
    All four collaborators are constructor-injected (not built
    internally) so tests can wire in fakes/mocks for any of them
    independently, and so main.py remains the single place that knows
    how the concrete pieces are assembled.
    """

    def __init__(
        self,
        market: MarketDataSource,
        account: Account,
        order_manager: OrderManager,
        position_manager: PositionManager,
        risk_manager: RiskManager,
    ) -> None:
        self._market = market
        self._account = account
        self._order_manager = order_manager
        self._position_manager = position_manager
        self._risk_manager = risk_manager

    # -- entering positions -----------------------------------------------

    def buy(
        self,
        symbol: str,
        quantity: int,
        stop_loss: float | None = None,
        target: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Open a new long position in `symbol`.

        Always OPENS a new position — to exit an existing position, use
        close_position(). This keeps entry and exit explicit and
        unambiguous even when multiple positions in the same symbol
        are open simultaneously.

        Returns
        -------
        Order
            The resulting order. Check `order.status`: FILLED means a
            position was opened (order.position_id references it).
            REJECTED means it wasn't (order.rejection_reason explains why).
            A rejection is a normal, expected outcome (e.g. risk limit
            hit) — not an exception — so the strategy can inspect and
            react without wrapping every call in try/except.
        """
        return self._enter_position(symbol, OrderSide.BUY, quantity, stop_loss, target, order_type, limit_price, stop_price)

    def sell(
        self,
        symbol: str,
        quantity: int,
        stop_loss: float | None = None,
        target: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Open a new short position in `symbol`.

        Always OPENS a new (short) position — see buy() docstring for
        why entry and exit are kept explicit and separate.
        """
        return self._enter_position(symbol, OrderSide.SELL, quantity, stop_loss, target, order_type, limit_price, stop_price)

    def _enter_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        stop_loss: float | None,
        target: float | None,
        order_type: OrderType,
        limit_price: float | None,
        stop_price: float | None,
    ) -> Order:
        order = self._order_manager.create_order(
            symbol=symbol, side=side, quantity=quantity, order_type=order_type,
            limit_price=limit_price, stop_price=stop_price,
        )
        self._order_manager.submit_order(order.order_id)

        if order_type != OrderType.MARKET:
            self._order_manager.reject_order(
                order.order_id, "Only MARKET orders are executable in this version of the engine."
            )
            return order

        try:
            price = self._market.get_price(symbol)
        except PriceUnavailableError as exc:
            self._order_manager.reject_order(order.order_id, f"Price unavailable: {exc}")
            return order

        try:
            self._risk_manager.validate_new_order(symbol, quantity, price)
        except RiskViolationError as exc:
            self._order_manager.reject_order(order.order_id, str(exc))
            return order

        try:
            position = self._position_manager.open_position(
                symbol=symbol, side=side, quantity=quantity, entry_price=price,
                stop_loss=stop_loss, target=target,
            )
        except InsufficientCapitalError as exc:
            self._order_manager.reject_order(order.order_id, str(exc))
            return order

        self._order_manager.fill_order(order.order_id, quantity, price)
        order.link_position(position.position_id)
        logger.info(
            "%s executed: %s %s x%d @ %.2f -> position %s",
            side.value, order.order_id, symbol, quantity, price, position.position_id,
        )
        return order

    # -- exiting positions -----------------------------------------------

    def close_position(self, position_id: str, reason: ExitReason = ExitReason.MANUAL) -> Position:
        """Close an open position at the current market price.

        Unlike buy()/sell(), risk limits are NOT checked here — a
        position-reducing trade should never be blocked by risk policy.

        Raises
        ------
        PositionNotFoundError
            If position_id doesn't correspond to a currently open position.
        PriceUnavailableError
            If the current market price can't be fetched. The position
            remains open; callers should retry.
        """
        position = self._position_manager.get_position(position_id)
        if position is None:
            raise PositionNotFoundError(position_id)

        price = self._market.get_price(position.symbol)  # PriceUnavailableError propagates

        closing_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        order = self._order_manager.create_order(
            symbol=position.symbol, side=closing_side, quantity=position.quantity,
            order_type=OrderType.MARKET,
        )
        self._order_manager.submit_order(order.order_id)
        self._order_manager.fill_order(order.order_id, position.quantity, price)
        order.link_position(position_id)

        closed_position = self._position_manager.close_position(position_id, price, reason)
        logger.info(
            "Position closed via Exchange: %s (%s) @ %.2f reason=%s",
            position_id, position.symbol, price, reason.value,
        )
        return closed_position

    # -- modification -----------------------------------------------------

    def modify_stop_loss(self, position_id: str, new_stop_loss: float | None) -> Position:
        """Update the stop-loss level for an open position.

        Raises
        ------
        PositionNotFoundError
            If position_id doesn't correspond to a currently open position.
        """
        return self._position_manager.modify_stop_loss(position_id, new_stop_loss)

    def modify_target(self, position_id: str, new_target: float | None) -> Position:
        """Update the target level for an open position.

        Raises
        ------
        PositionNotFoundError
            If position_id doesn't correspond to a currently open position.
        """
        return self._position_manager.modify_target(position_id, new_target)

    def cancel_order(self, order_id: str) -> Order:
        """Cancel an order that is still open.

        In this version, MARKET orders fill synchronously inside
        buy()/sell(), so there is no window in which to cancel one.
        This method exists for interface completeness and becomes
        directly useful once LIMIT/STOP orders can rest on the book,
        and once real broker execution introduces asynchronous fills.

        Raises
        ------
        OrderNotFoundError
            If order_id doesn't correspond to any known order.
        """
        return self._order_manager.cancel_order(order_id)

    # -- market data tick processing ----------------------------------------

    def process_tick(self) -> list[Position]:
        """Refresh prices for all open positions and close any that hit SL/target.

        Intended to be called once per price update cycle from the main
        loop (e.g. main.py), independent of any buy()/sell() calls.
        This is how positions exit automatically when the market moves
        against/in favor of them, without the strategy having to poll.

        Returns
        -------
        list[Position]
            Positions that were automatically closed this tick.
        """
        self._position_manager.update_prices()
        return self._position_manager.check_exits()

    # -- read-only pass-throughs ---------------------------------------------

    def get_open_positions(self) -> list[Position]:
        return self._position_manager.get_open_positions()

    def get_position(self, position_id: str) -> Position | None:
        return self._position_manager.get_position(position_id)

    def get_order(self, order_id: str) -> Order:
        return self._order_manager.get_order(order_id)

    @property
    def account(self) -> Account:
        """Direct read access to the account (capital, PnL, closed positions)."""
        return self._account
