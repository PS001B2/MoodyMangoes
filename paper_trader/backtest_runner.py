"""
backtest_runner.py: runs the paper trading engine against historical
CSV data instead of the live Groww feed.

This is a separate entry point from main.py because live trading and
backtesting have fundamentally different loop shapes: main.py waits
in real time between ticks (time.sleep), while a backtest should run
through months of history as fast as possible, driven by advancing
through data rows rather than a wall clock. Everything else — Account,
OrderManager, PositionManager, RiskManager, Exchange, Strategy — is
the exact same code used for live trading. Only the market data
source (HistoricalReplayFeed instead of GrowwMarketFeed) and the loop
driver differ.

Usage
-----
    python backtest_runner.py <folder_or_file_path> [symbol] [config.yaml]

Examples
--------
    python backtest_runner.py data/nsei/
    python backtest_runner.py data/nsei/ NSEI
    python backtest_runner.py data/nsei/ NSEI my_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from paper_trader.analytics.performance import PerformanceCalculator, PerformanceReport
from paper_trader.analytics.trade_history import TradeHistory
from paper_trader.engine.account import Account
from paper_trader.engine.exchange import Exchange
from paper_trader.engine.order_manager import OrderManager
from paper_trader.engine.position import ExitReason
from paper_trader.engine.position_manager import PositionManager
from paper_trader.engine.risk_manager import RiskManager
from paper_trader.market.historical_feed import HistoricalReplayFeed
from paper_trader.market.market_data import PriceUnavailableError
from paper_trader.strategy.manual_strategy import ManualStrategy
from paper_trader.utils.config import AppConfig, load_config
from paper_trader.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Log a progress line every time the replay crosses one of these
# percentage thresholds, so a long backtest doesn't sit silently for
# minutes with no feedback, but also doesn't spam the log every bar.
_PROGRESS_LOG_INTERVAL_PCT = 10.0


def run_backtest(data_path: str, symbol: str = "NSEI", config_path: str = "config.yaml") -> PerformanceReport:
    """Run a full backtest and return the resulting performance report.

    Parameters
    ----------
    data_path:
        Folder containing one or more CSV files (or a single CSV
        file), covering the historical period to replay.
    symbol:
        The symbol the data represents.
    config_path:
        Path to the YAML config (reused for starting capital, risk
        limits, and strategy parameters — the same config.yaml a live
        session would use).
    """
    config = load_config(config_path)
    setup_logging(config.logging)

    market = HistoricalReplayFeed(symbol=symbol, data_path=data_path)

    account = Account(config.capital.starting_capital)
    order_manager = OrderManager()
    position_manager = PositionManager(market, account)
    risk_manager = RiskManager(config.risk, account)
    exchange = Exchange(market, account, order_manager, position_manager, risk_manager)
    trade_history = TradeHistory()

    # Backtests replay exactly one symbol's data at a time, regardless
    # of how many symbols config.yaml lists for live trading.
    strategy = ManualStrategy(
        exchange=exchange,
        market=market,
        symbols=[symbol],
        quantity=config.trading.quantity,
        sma_period=config.trading.sma_period,
        stop_loss_pct=config.trading.stop_loss_pct,
        target_pct=config.trading.target_pct,
    )

    logger.info(
        "Starting backtest: symbol=%s bars=%d starting_capital=%.2f",
        symbol, market.bar_count, config.capital.starting_capital,
    )

    account.start_new_day()
    strategy.on_start()

    _replay_loop(market, exchange, strategy, account, trade_history)
    _close_remaining_positions(exchange, trade_history)

    strategy.on_stop()
    report = _log_summary(trade_history, account, market)
    return report


def _replay_loop(
    market: HistoricalReplayFeed,
    exchange: Exchange,
    strategy: ManualStrategy,
    account: Account,
    trade_history: TradeHistory,
) -> None:
    """Step through every bar, running the strategy and processing exits.

    Mirrors PaperTradingApp._tick() from main.py exactly, except there
    is no time.sleep() between iterations, and the "clock" advances by
    calling market.advance() instead of waiting for wall-clock time to
    pass. A new trading day (detected via a change in the bar's date)
    resets the daily PnL baseline, matching what a live session does
    once per calendar day.
    """
    current_day = market.current_timestamp.date()
    next_progress_threshold = _PROGRESS_LOG_INTERVAL_PCT
    bar_index = 0

    while True:
        bar_date = market.current_timestamp.date()
        if bar_date != current_day:
            account.start_new_day()
            current_day = bar_date

        try:
            strategy.on_tick()
        except Exception:
            logger.exception("Strategy.on_tick() raised an unhandled exception; continuing.")

        auto_closed = exchange.process_tick()
        for position in auto_closed:
            trade_history.record_trade(position)
            try:
                strategy.on_position_closed(position)
            except Exception:
                logger.exception("Strategy.on_position_closed() raised an unhandled exception; continuing.")

        bar_index += 1
        if market.progress_pct >= next_progress_threshold:
            logger.info(
                "Backtest progress: %.0f%% (%d/%d bars, at %s)",
                market.progress_pct, bar_index, market.bar_count, market.current_timestamp,
            )
            next_progress_threshold += _PROGRESS_LOG_INTERVAL_PCT

        if not market.advance():
            break


def _close_remaining_positions(exchange: Exchange, trade_history: TradeHistory) -> None:
    """Close any positions still open when the historical data runs out.

    Without this, a position opened near the end of the dataset but
    never hit by its stop-loss/target would stay OPEN forever and
    silently never appear in trade_history or the performance report —
    even though it has a real (unrealized) PnL. Marking it closed with
    ExitReason.END_OF_DAY makes the backtest's numbers complete and
    mirrors what a live session does at actual market close (a real
    broker or your own end-of-day logic would need to decide whether
    to square off open positions too).
    """
    open_positions = exchange.get_open_positions()
    if not open_positions:
        return

    logger.info("%d position(s) still open at end of data; closing at last known price.", len(open_positions))
    for position in open_positions:
        try:
            closed = exchange.close_position(position.position_id, reason=ExitReason.END_OF_DAY)
            trade_history.record_trade(closed)
        except PriceUnavailableError as exc:
            logger.warning(
                "Could not close position %s at end of data (price unavailable): %s",
                position.position_id, exc,
            )


def _log_summary(trade_history: TradeHistory, account: Account, market: HistoricalReplayFeed) -> PerformanceReport:
    report = PerformanceCalculator.compute(trade_history.get_all_trades())
    logger.info(
        "Backtest complete: bars=%d trades=%d win_rate=%.1f%% net_profit=%.2f "
        "max_drawdown=%.2f profit_factor=%s | final_equity=%.2f total_pnl=%.2f",
        market.bar_count, report.total_trades, report.win_rate, report.net_profit,
        report.max_drawdown, report.profit_factor, account.current_capital, account.total_pnl,
    )
    if report.total_trades > 0:
        csv_path = trade_history.export_to_csv("logs/backtest_trades.csv")
        logger.info("Trade log exported to %s", csv_path)
    else:
        logger.info("No trades were taken during this backtest.")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python backtest_runner.py <folder_or_file_path> [symbol] [config.yaml]",
            file=sys.stderr,
        )
        sys.exit(1)

    arg_data_path = sys.argv[1]
    arg_symbol = sys.argv[2] if len(sys.argv) > 2 else "NSEI"
    arg_config_path = sys.argv[3] if len(sys.argv) > 3 else "config.yaml"

    run_backtest(arg_data_path, arg_symbol, arg_config_path)
