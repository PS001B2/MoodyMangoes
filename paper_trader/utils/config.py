"""
Configuration loading for the paper trading framework.

Design notes
------------
- Structural settings (starting capital, risk limits, logging level, etc.)
  live in a YAML file and are version-controllable.
- Secrets (API keys/tokens) are NEVER read from the YAML file directly.
  They are read from environment variables, so credentials never end up
  committed to source control. The YAML file may reference the *name*
  of an env var, but not its value.
- All configuration is exposed as immutable, typed dataclasses rather
  than raw dicts, so the rest of the codebase gets IDE autocomplete
  and fails fast (at startup) if a required value is missing, instead
  of failing deep inside a strategy at 2pm on a trading day.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or invalid."""


@dataclass(frozen=True)
class CapitalConfig:
    """Starting capital and currency for the paper trading account."""

    starting_capital: float
    currency: str = "INR"

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ConfigError("starting_capital must be greater than 0.")


@dataclass(frozen=True)
class RiskConfig:
    """Global risk limits enforced by the RiskManager.

    These are intentionally simple for the paper-trading first version.
    More sophisticated fields (per-symbol limits, max sector exposure,
    options greeks limits, etc.) can be added later without breaking
    callers, since consumers always access fields by name.
    """

    max_position_size_pct: float = 10.0
    """Max percentage of available capital allowed in a single position."""

    max_daily_loss_pct: float = 5.0
    """Max percentage of starting capital allowed to be lost in a day
    before the RiskManager should refuse new orders."""

    max_open_positions: int = 5

    def __post_init__(self) -> None:
        if not (0 < self.max_position_size_pct <= 100):
            raise ConfigError("max_position_size_pct must be between 0 and 100.")
        if not (0 < self.max_daily_loss_pct <= 100):
            raise ConfigError("max_daily_loss_pct must be between 0 and 100.")
        if self.max_open_positions <= 0:
            raise ConfigError("max_open_positions must be greater than 0.")


@dataclass(frozen=True)
class GrowwAPIConfig:
    """Connection details for the Groww market data feed.

    api_key / api_secret / access_token are read from environment
    variables at load time (see load_config). The YAML file only
    stores the *env var names* to use, keeping secrets out of version
    control.

    Two auth modes are supported (mirroring Groww's official SDK):
    - Provide api_key + api_secret: GrowwMarketFeed will exchange these
      for an access_token itself, once, on first use.
    - Provide access_token directly: skips the exchange step. Useful
      if a token was already generated elsewhere (e.g. TOTP flow via
      Groww's web console) and you'd rather manage refresh externally.
    """

    api_key: str = field(repr=False, default="")
    api_secret: str = field(repr=False, default="")
    access_token: str = field(repr=False, default="")
    default_exchange: str = "NSE"
    default_segment: str = "CASH"
    request_timeout_seconds: float = 5.0

    def __repr__(self) -> str:  # avoid ever accidentally logging secrets
        return "GrowwAPIConfig(api_key=***, api_secret=***, access_token=***)"


@dataclass(frozen=True)
class LoggingConfig:
    """Controls how utils.logger configures the standard logging module."""

    level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/paper_trader.log"

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ConfigError(
                f"Invalid logging level '{self.level}'. Must be one of {valid_levels}."
            )


@dataclass(frozen=True)
class TradingConfig:
    """Controls which symbols are traded and basic strategy/loop parameters.

    Kept deliberately simple for the first version (one strategy, fixed
    parameters for all symbols). As multiple strategies / per-symbol
    configuration become real requirements, this is the natural place
    to grow — e.g. a list of per-strategy configs instead of one flat set.
    """

    symbols: list[str] = field(default_factory=lambda: ["RELIANCE"])
    tick_interval_seconds: float = 5.0
    quantity: int = 1
    sma_period: int = 5
    stop_loss_pct: float = 1.0
    target_pct: float = 2.0

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ConfigError("trading.symbols must contain at least one symbol.")
        if self.tick_interval_seconds <= 0:
            raise ConfigError("trading.tick_interval_seconds must be greater than 0.")
        if self.quantity <= 0:
            raise ConfigError("trading.quantity must be greater than 0.")
        if self.sma_period <= 0:
            raise ConfigError("trading.sma_period must be greater than 0.")


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object for the whole application."""

    capital: CapitalConfig
    risk: RiskConfig
    groww: GrowwAPIConfig
    logging: LoggingConfig
    trading: TradingConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level.")
    return data


def _resolve_env(value: str | None, env_var_name: str, required: bool) -> str:
    """Resolve a secret from the environment.

    `value` is whatever was found under the key in YAML: we expect it to
    be the *name* of an environment variable (e.g. "GROWW_API_KEY"), not
    the secret itself. If nothing is configured in YAML, we fall back to
    a sensible default env var name.
    """
    env_name = value or env_var_name
    resolved = os.environ.get(env_name, "")
    if required and not resolved:
        raise ConfigError(
            f"Environment variable '{env_name}' is not set. "
            f"Set it before starting the application."
        )
    return resolved


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate application configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    AppConfig
        Fully validated, immutable configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, malformed, or fails validation.
    """
    raw = _read_yaml(Path(path))

    capital_raw = raw.get("capital", {})
    risk_raw = raw.get("risk", {})
    groww_raw = raw.get("groww", {})
    logging_raw = raw.get("logging", {})

    capital = CapitalConfig(
        starting_capital=float(capital_raw.get("starting_capital", 100000)),
        currency=capital_raw.get("currency", "INR"),
    )

    risk = RiskConfig(
        max_position_size_pct=float(risk_raw.get("max_position_size_pct", 10.0)),
        max_daily_loss_pct=float(risk_raw.get("max_daily_loss_pct", 5.0)),
        max_open_positions=int(risk_raw.get("max_open_positions", 5)),
    )

    groww = GrowwAPIConfig(
        api_key=_resolve_env(
            groww_raw.get("api_key_env"), "GROWW_API_KEY", required=False
        ),
        api_secret=_resolve_env(
            groww_raw.get("api_secret_env"), "GROWW_API_SECRET", required=False
        ),
        access_token=_resolve_env(
            groww_raw.get("access_token_env"), "GROWW_ACCESS_TOKEN", required=False
        ),
        default_exchange=groww_raw.get("default_exchange", "NSE"),
        default_segment=groww_raw.get("default_segment", "CASH"),
        request_timeout_seconds=float(groww_raw.get("request_timeout_seconds", 5.0)),
    )

    logging_cfg = LoggingConfig(
        level=logging_raw.get("level", "INFO"),
        log_to_file=bool(logging_raw.get("log_to_file", True)),
        log_file_path=logging_raw.get("log_file_path", "logs/paper_trader.log"),
    )

    trading_raw = raw.get("trading", {})
    trading = TradingConfig(
        symbols=list(trading_raw.get("symbols", ["RELIANCE"])),
        tick_interval_seconds=float(trading_raw.get("tick_interval_seconds", 5.0)),
        quantity=int(trading_raw.get("quantity", 1)),
        sma_period=int(trading_raw.get("sma_period", 5)),
        stop_loss_pct=float(trading_raw.get("stop_loss_pct", 1.0)),
        target_pct=float(trading_raw.get("target_pct", 2.0)),
    )

    return AppConfig(capital=capital, risk=risk, groww=groww, logging=logging_cfg, trading=trading)
