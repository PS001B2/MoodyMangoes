"""
main.py: composition root and entry point for the paper trading engine.

This is the ONLY file in the project that wires concrete
implementations together — it is the one place that knows
GrowwMarketFeed exists, which strategy is running, and how the tick
loop is driven. Every other module (engine/, analytics/, strategy/)
depends only on abstractions. That is what makes this file safe to
change — swapping the data source, the strategy, or the run mode —
without touching anything else in the codebase.

Usage
-----
    python main.py [path/to/config.yaml]

Defaults to "config.yaml" in the current directory if no path is given.
"""

from __future__ import annotations

import signal
import sys
import time
from types import FrameType

from analytics.performance import PerformanceCalculator
from analytics.trade_history import TradeHistory
from engine.account import Account
from engine.exchange import Exchange
from engine.order_manager import OrderManager
from engine.position_manager import PositionManager
from engine.risk_manager import RiskManager
from market.groww_feed import GrowwMarketFeed
from strategy.base_strategy import BaseStrategy
from strategy.manual_strategy import ManualStrategy
from utils.config import AppConfig, ConfigError, load_config
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class PaperTradingApp:
    """Wires together and runs one paper trading session.

    Construction assembles the full dependency graph:
    GrowwMarketFeed -> Account -> OrderManager/PositionManager/RiskManager
    -> Exchange -> Strategy, plus TradeHistory for recording completed
    trades. Nothing here is a singleton or module-level global — a
    fresh PaperTradingApp is a fresh, isolated session.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        self.market = GrowwMarketFeed(config.groww)
        self.account = Account(config.capital.starting_capital)
        self.order_manager = OrderManager()
        self.position_manager = PositionManager(self.market, self.account)
        self.risk_manager = RiskManager(config.risk, self.account)
        self.exchange = Exchange(
            self.market, self.account, self.order_manager, self.position_manager, self.risk_manager,
        )
        self.trade_history = TradeHistory()

        # This is the one line that would change to run a different
        # strategy — everything above it and below it stays the same.
        self.strategy: BaseStrategy = ManualStrategy(
            exchange=self.exchange,
            market=self.market,
            symbols=config.trading.symbols,
            quantity=config.trading.quantity,
            sma_period=config.trading.sma_period,
            stop_loss_pct=config.trading.stop_loss_pct,
            target_pct=config.trading.target_pct,
        )

        self._running = False

    def run(self) -> None:
        """Run the tick loop until stopped (Ctrl+C, SIGTERM, or self.stop())."""
        logger.info(
            "Starting paper trading session. Symbols=%s starting_capital=%.2f",
            self._config.trading.symbols, self._config.capital.starting_capital,
        )
        self.account.start_new_day()
        self.strategy.on_start()
        self._running = True

        try:
            while self._running:
                self._tick()
                time.sleep(self._config.trading.tick_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self.stop()

    def _tick(self) -> None:
        """One full cycle: let the strategy act, then process automatic exits.

        Both the strategy call and the exit-callback are wrapped so a
        bug in strategy code (a bad indicator calculation, a division
        by zero, whatever) logs and the loop keeps running rather than
        crashing the whole session.
        """
        try:
            self.strategy.on_tick()
        except Exception:
            logger.exception("Strategy.on_tick() raised an unhandled exception; continuing.")

        auto_closed = self.exchange.process_tick()
        for position in auto_closed:
            self.trade_history.record_trade(position)
            try:
                self.strategy.on_position_closed(position)
            except Exception:
                logger.exception("Strategy.on_position_closed() raised an unhandled exception; continuing.")

    def stop(self) -> None:
        """Stop the loop, run strategy teardown, and log a session summary."""
        if not self._running:
            return
        self._running = False
        try:
            self.strategy.on_stop()
        except Exception:
            logger.exception("Strategy.on_stop() raised an unhandled exception.")
        self._log_summary()

    def _log_summary(self) -> None:
        report = PerformanceCalculator.compute(self.trade_history.get_all_trades())
        logger.info(
            "Session summary: trades=%d win_rate=%.1f%% net_profit=%.2f max_drawdown=%.2f "
            "profit_factor=%s | account current_capital=%.2f total_pnl=%.2f",
            report.total_trades, report.win_rate, report.net_profit, report.max_drawdown,
            report.profit_factor, self.account.current_capital, self.account.total_pnl,
        )
        if report.total_trades > 0:
            csv_path = self.trade_history.export_to_csv("logs/trades.csv")
            logger.info("Trade log exported to %s", csv_path)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        # Logging isn't configured yet at this point (it depends on the
        # config we just failed to load), so this one message goes to
        # stderr directly rather than through the logger.
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config.logging)
    app = PaperTradingApp(config)

    def _handle_sigterm(signum: int, frame: FrameType | None) -> None:
        logger.info("Received termination signal (%s).", signum)
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    app.run()


if __name__ == "__main__":
    main()
