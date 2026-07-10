"""
RiskManager: the gatekeeper Exchange consults before approving any order.

Single responsibility: enforce risk POLICY (position sizing limits,
daily loss circuit-breaker, max concurrent open positions) against a
proposed trade. It does NOT check raw capital sufficiency — that
remains Account.reserve_capital's job. Keeping these separate means
risk rules can evolve (per-symbol limits, sector exposure, options
greeks limits, portfolio-level VaR, ...) without ever touching capital
bookkeeping, and vice versa.
"""

from __future__ import annotations

from engine.account import Account
from utils.config import RiskConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskViolationError(Exception):
    """Raised when a proposed trade violates a configured risk limit."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RiskManager:
    """Validates proposed trades against configured risk limits.

    Parameters
    ----------
    config:
        Risk limits (max position size %, max daily loss %, max open positions).
    account:
        The account whose capital and positions are checked against those limits.
    """

    def __init__(self, config: RiskConfig, account: Account) -> None:
        self._config = config
        self._account = account

    def validate_new_order(self, symbol: str, quantity: int, price: float) -> None:
        """Check whether a proposed new position is allowed under current risk limits.

        Parameters
        ----------
        symbol:
            Trading symbol the order is for (used only for error messages
            here; per-symbol limits are a natural future extension point).
        quantity:
            Proposed order quantity.
        price:
            Proposed execution price, used to compute notional exposure.

        Raises
        ------
        RiskViolationError
            If any configured risk limit would be breached. The
            exception's `reason` explains exactly which limit and why,
            so Exchange can reject the order with a clear message.
        """
        self._check_daily_loss_limit()
        self._check_max_open_positions()
        self._check_position_size(symbol, quantity, price)

    def is_daily_loss_limit_breached(self) -> bool:
        """True if today's PnL has breached the configured daily loss limit.

        Exposed separately (not just as part of validate_new_order) so
        callers like main.py can halt strategy execution entirely for
        the rest of the day, not just reject individual orders one at
        a time.
        """
        max_loss = self._account.starting_capital * (self._config.max_daily_loss_pct / 100)
        return self._account.daily_pnl <= -max_loss

    def max_allowed_quantity(self, price: float) -> int:
        """The largest quantity of a symbol at `price` that fits within
        the per-position size limit, given currently available capital.

        Useful for Exchange or Strategy to size orders correctly
        up-front, rather than guessing a quantity and having it rejected.
        """
        if price <= 0:
            raise ValueError("price must be greater than 0.")
        max_notional = self._account.available_capital * (self._config.max_position_size_pct / 100)
        return int(max_notional // price)

    # -- individual checks -----------------------------------------------

    def _check_daily_loss_limit(self) -> None:
        if self.is_daily_loss_limit_breached():
            raise RiskViolationError(
                f"Daily loss limit of {self._config.max_daily_loss_pct}% of starting "
                f"capital has been breached (daily PnL: {self._account.daily_pnl:.2f}). "
                f"New orders are blocked for the rest of the day."
            )

    def _check_max_open_positions(self) -> None:
        open_count = len(self._account.open_positions)
        if open_count >= self._config.max_open_positions:
            raise RiskViolationError(
                f"Max open positions limit reached "
                f"({open_count}/{self._config.max_open_positions})."
            )

    def _check_position_size(self, symbol: str, quantity: int, price: float) -> None:
        notional = quantity * price
        max_notional = self._account.available_capital * (self._config.max_position_size_pct / 100)
        if notional > max_notional:
            raise RiskViolationError(
                f"Position size for {symbol} ({notional:.2f}) exceeds max allowed "
                f"({self._config.max_position_size_pct}% of available capital = "
                f"{max_notional:.2f})."
            )
