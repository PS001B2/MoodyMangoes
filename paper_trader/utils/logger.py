"""
Centralized logging configuration for the paper trading framework.

Usage
-----
In main.py, once, at startup::

    from utils.config import load_config
    from utils.logger import setup_logging

    config = load_config("config.yaml")
    setup_logging(config.logging)

In every other module::

    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Position opened: %s", symbol)

No module should ever call `print()` for operational output — this
keeps behavior consistent (levels, timestamps, file output) across the
entire codebase and makes it trivial to redirect logs later (e.g. to a
monitoring service) by changing this one file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from paper_trader.utils.config import LoggingConfig

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(config: LoggingConfig) -> None:
    """Configure the root logger for the application.

    This should be called exactly once, at application startup (in
    main.py). Calling it again is a no-op, so it's safe even if a
    module accidentally imports something that triggers a second call.

    Parameters
    ----------
    config:
        Logging configuration (level, whether to log to file, file path).
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(config.level.upper())

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if config.log_to_file:
        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True
    root_logger.debug("Logging configured (level=%s).", config.level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Parameters
    ----------
    name:
        Conventionally `__name__` of the calling module, so log lines
        show exactly which module emitted them.

    Returns
    -------
    logging.Logger
        A logger that inherits handlers/level from the root logger
        configured via `setup_logging`. If `setup_logging` has not been
        called yet, Python's logging defaults apply (WARNING level,
        stderr output) so nothing crashes during early imports/tests.
    """
    return logging.getLogger(name)
