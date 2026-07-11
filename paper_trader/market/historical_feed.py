"""
HistoricalReplayFeed: a MarketDataSource that replays historical bar
data instead of fetching live prices.

Loads one or more CSV files (columns: Datetime, Close, High, Low,
Open, Volume — as exported by common historical data providers),
merges them into a single chronologically-sorted timeline, and steps
through it bar-by-bar via advance(). Everything downstream — Exchange,
PositionManager, RiskManager, Strategy — is completely unaware this
isn't a live feed; it just calls get_price(symbol) like always.

This is what makes backtesting "free": the exact same engine and
strategy code that runs live against Groww runs unchanged against
history, just driven by a different clock (advance() instead of
time.sleep()).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from paper_trader.market.market_data import MarketDataSource, PriceUnavailableError
from paper_trader.utils.logger import get_logger

logger = get_logger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%z"


class HistoricalReplayFeed(MarketDataSource):
    """Replays historical bar data (e.g. 1-minute candles) as a market feed.

    Parameters
    ----------
    symbol:
        The symbol this data represents. The CSV files have no symbol
        column, so it's supplied explicitly here. get_price() only
        ever answers for this symbol.
    data_path:
        Either a single CSV file, or a directory containing multiple
        CSV files (all matching "*.csv") that together cover the full
        date range to replay. Files are merged and sorted
        chronologically automatically, with overlapping timestamps
        across files de-duplicated — you don't need to pre-sort or
        concatenate anything yourself; just point this at the folder.

    Usage
    -----
        feed = HistoricalReplayFeed(symbol="NSEI", data_path="data/nsei/")
        while True:
            price = feed.get_price("NSEI")   # current bar's close price
            ...  # strategy / exchange logic runs against this price
            if not feed.advance():
                break  # ran out of data
    """

    def __init__(self, symbol: str, data_path: str | Path) -> None:
        self._symbol = symbol
        self._bars: list[tuple[datetime, float]] = self._load_bars(Path(data_path))
        if not self._bars:
            raise ValueError(f"No usable bars were loaded from {data_path}.")
        self._cursor = 0
        logger.info(
            "Loaded %d bars for %s, spanning %s -> %s",
            len(self._bars), symbol, self._bars[0][0], self._bars[-1][0],
        )

    # -- MarketDataSource interface --------------------------------------------

    def get_price(self, symbol: str) -> float:
        """Return the close price of the CURRENT bar (does not auto-advance)."""
        if symbol != self._symbol:
            raise PriceUnavailableError(
                symbol, f"HistoricalReplayFeed only has data loaded for '{self._symbol}'."
            )
        return self._bars[self._cursor][1]

    # -- replay control -----------------------------------------------------------

    def advance(self) -> bool:
        """Move the replay forward to the next bar.

        Returns
        -------
        bool
            True if there was a next bar to move to and the cursor
            advanced. False if the data is exhausted (cursor stays on
            the final bar) — the signal for the backtest loop to stop.
        """
        if self._cursor + 1 >= len(self._bars):
            return False
        self._cursor += 1
        return True

    def reset(self) -> None:
        """Rewind to the first bar, e.g. to re-run a backtest from scratch."""
        self._cursor = 0

    @property
    def current_timestamp(self) -> datetime:
        """Timestamp of the bar currently being served by get_price()."""
        return self._bars[self._cursor][0]

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    @property
    def is_finished(self) -> bool:
        """True once the cursor is on the final bar (advance() would return False)."""
        return self._cursor >= len(self._bars) - 1

    @property
    def progress_pct(self) -> float:
        """How far through the dataset the replay currently is (0-100)."""
        if len(self._bars) <= 1:
            return 100.0
        return (self._cursor / (len(self._bars) - 1)) * 100.0

    # -- loading ------------------------------------------------------------------

    def _load_bars(self, data_path: Path) -> list[tuple[datetime, float]]:
        """Load, merge, de-duplicate, and sort bars from one file or a folder of files."""
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if data_path.is_dir():
            csv_files = sorted(data_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No .csv files found in directory: {data_path}")
        else:
            csv_files = [data_path]

        bars_by_timestamp: dict[datetime, float] = {}
        duplicate_count = 0
        malformed_count = 0

        for csv_file in csv_files:
            rows_loaded = self._load_one_file(csv_file, bars_by_timestamp)
            logger.info("Loaded %d rows from %s", rows_loaded, csv_file.name)

        # Recount duplicates/malformed globally isn't tracked per-file above
        # to keep _load_one_file simple; instead we log a single summary here.
        return sorted(bars_by_timestamp.items(), key=lambda pair: pair[0])

    def _load_one_file(self, csv_file: Path, bars_by_timestamp: dict[datetime, float]) -> int:
        """Parse a single CSV file, merging its rows into bars_by_timestamp in place.

        Returns the number of valid rows loaded from this file. Rows
        with missing/unparseable data are skipped (logged, not fatal) —
        one bad row in a huge historical file shouldn't abort the load.
        """
        rows_loaded = 0
        malformed_count = 0
        duplicate_count = 0

        with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.warning("Skipping empty file: %s", csv_file)
                return 0

            # Normalize header lookups (case/whitespace) so "Datetime" vs
            # "datetime" vs " Datetime " don't cause a spurious failure.
            fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
            datetime_col = fieldnames.get("datetime")
            close_col = fieldnames.get("close")
            if datetime_col is None or close_col is None:
                raise ValueError(
                    f"{csv_file} is missing a required 'Datetime' or 'Close' column. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                raw_ts = (row.get(datetime_col) or "").strip()
                raw_close = (row.get(close_col) or "").strip()
                if not raw_ts or not raw_close:
                    malformed_count += 1
                    continue
                try:
                    timestamp = datetime.strptime(raw_ts, _TIMESTAMP_FORMAT)
                    close_price = float(raw_close)
                except (ValueError, TypeError):
                    malformed_count += 1
                    continue

                if timestamp in bars_by_timestamp:
                    duplicate_count += 1
                bars_by_timestamp[timestamp] = close_price
                rows_loaded += 1

        if malformed_count:
            logger.warning("%s: skipped %d malformed/incomplete rows.", csv_file.name, malformed_count)
        if duplicate_count:
            logger.warning(
                "%s: %d rows had timestamps already seen in another file; this file's value was kept.",
                csv_file.name, duplicate_count,
            )
        return rows_loaded
