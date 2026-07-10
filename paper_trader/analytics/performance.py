"""
Performance: computes trading performance metrics from completed trades.

Single responsibility: turn a list of Trade records into summary
statistics. Stateless and pure — no dependency on Account, Exchange,
or any live engine component — so it works identically whether fed
trades from a live paper trading session, a backtest, or a CSV
re-loaded from a previous session's export.
"""

from __future__ import annotations

from dataclasses import dataclass

from analytics.trade_history import Trade


@dataclass(frozen=True)
class PerformanceReport:
    """Summary performance statistics for a set of completed trades.

    Attributes
    ----------
    total_trades:
        Total number of closed trades.
    winning_trades / losing_trades:
        Counts of trades with positive / negative PnL. Trades with
        exactly zero PnL count toward total_trades but neither bucket.
    win_rate:
        Percentage (0-100) of trades that were winners.
    average_win:
        Mean PnL of winning trades (positive number). 0.0 if none.
    average_loss:
        Mean PnL magnitude of losing trades (positive number,
        i.e. already absolute-valued for readability). 0.0 if none.
    profit_factor:
        Gross profit / gross loss. float('inf') if there were wins and
        zero losses; 0.0 if there were no wins at all.
    max_drawdown:
        Peak-to-trough drawdown (positive number) on the cumulative
        realized-PnL curve, trades taken in exit-time order. NOTE:
        this reflects only realized trade PnL in sequence — it does
        NOT account for intraday unrealized mark-to-market swings on
        positions that were open but not yet closed. A true
        equity-curve drawdown would require periodic snapshots of
        Account.current_capital over time, which this module does not
        have access to.
    net_profit:
        Sum of PnL across all trades (can be negative).
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    net_profit: float


class PerformanceCalculator:
    """Computes a PerformanceReport from a list of Trade records."""

    @staticmethod
    def compute(trades: list[Trade]) -> PerformanceReport:
        """Compute performance metrics for the given trades.

        Parameters
        ----------
        trades:
            Completed trades, in any order (they are sorted internally
            by exit_time for the drawdown calculation).

        Returns
        -------
        PerformanceReport
            All-zero report if `trades` is empty.
        """
        if not trades:
            return PerformanceReport(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, average_win=0.0, average_loss=0.0,
                profit_factor=0.0, max_drawdown=0.0, net_profit=0.0,
            )

        ordered = sorted(trades, key=lambda t: t.exit_time)

        wins = [t.pnl for t in ordered if t.pnl > 0]
        losses = [t.pnl for t in ordered if t.pnl < 0]

        total_trades = len(ordered)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades) * 100.0

        average_win = (sum(wins) / winning_trades) if winning_trades else 0.0
        average_loss = (abs(sum(losses)) / losing_trades) if losing_trades else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        net_profit = sum(t.pnl for t in ordered)
        max_drawdown = PerformanceCalculator._max_drawdown(ordered)

        return PerformanceReport(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            net_profit=net_profit,
        )

    @staticmethod
    def _max_drawdown(ordered_trades: list[Trade]) -> float:
        """Peak-to-trough drawdown on the cumulative realized-PnL curve.

        See PerformanceReport.max_drawdown docstring for the important
        caveat that this excludes intraday unrealized swings.
        """
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for trade in ordered_trades:
            cumulative += trade.pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_dd = max(max_dd, drawdown)
        return max_dd
