"""
TradeHistory: an append-only log of completed trades.

Single responsibility: record and query completed trades, and export
them to CSV. This module has no dependency on Exchange, Account, or
any other engine component beyond reading a closed Position to build
a Trade record — it is a consumer of engine output, not a
collaborator the engine calls into. The composition root (main.py)
is responsible for calling record_trade() whenever Exchange reports a
position has closed (Exchange.close_position() and
Exchange.process_tick() both return the closed Position(s) for
exactly this purpose).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from engine.order import OrderSide
from engine.position import ExitReason, Position, PositionStatus
from utils.logger import get_logger

logger = get_logger(__name__)

_CSV_FIELDNAMES = [
    "position_id", "symbol", "side", "quantity",
    "entry_price", "exit_price", "entry_time", "exit_time",
    "pnl", "exit_reason",
]


@dataclass(frozen=True)
class Trade:
    """An immutable record of one completed round-trip trade.

    Unlike Position (which is mutable and represents a live holding),
    Trade is a frozen historical fact — once a trade is recorded it
    never changes, which is exactly the property you want for
    analytics and audit trails.
    """

    position_id: str
    symbol: str
    side: OrderSide
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: ExitReason

    @classmethod
    def from_position(cls, position: Position) -> "Trade":
        """Build a Trade record from a closed Position.

        Raises
        ------
        ValueError
            If the position is not CLOSED (an open position has no
            exit_price/exit_time/exit_reason to record yet).
        """
        if position.status != PositionStatus.CLOSED:
            raise ValueError(
                f"Cannot record a trade for position {position.position_id}: "
                f"it is not CLOSED (status={position.status})."
            )
        # These are guaranteed non-None once status is CLOSED (see Position.close).
        assert position.exit_price is not None
        assert position.exit_time is not None
        assert position.exit_reason is not None

        return cls(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            pnl=position.realized_pnl,
            exit_reason=position.exit_reason,
        )


class TradeHistory:
    """An append-only log of every completed trade in a session."""

    def __init__(self) -> None:
        self._trades: list[Trade] = []

    def record_trade(self, position: Position) -> Trade:
        """Convert a closed Position into a Trade and append it to the log.

        Raises
        ------
        ValueError
            If the position is not CLOSED.
        """
        trade = Trade.from_position(position)
        self._trades.append(trade)
        logger.info(
            "Trade recorded: %s %s qty=%d pnl=%.2f reason=%s",
            trade.symbol, trade.side.value, trade.quantity, trade.pnl, trade.exit_reason.value,
        )
        return trade

    def get_all_trades(self) -> list[Trade]:
        """All recorded trades, in the order they were recorded."""
        return list(self._trades)

    def get_trades_for_symbol(self, symbol: str) -> list[Trade]:
        """All recorded trades for a given symbol."""
        return [t for t in self._trades if t.symbol == symbol]

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    def export_to_csv(self, path: str | Path) -> Path:
        """Write the full trade log to a CSV file.

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created if
            they don't already exist.

        Returns
        -------
        Path
            The path written to (for convenience/chaining).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            for trade in self._trades:
                writer.writerow({
                    "position_id": trade.position_id,
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "pnl": trade.pnl,
                    "exit_reason": trade.exit_reason.value,
                })

        logger.info("Exported %d trades to %s", len(self._trades), path)
        return path
