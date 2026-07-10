"""
Groww implementation of MarketDataSource.

Wraps Groww's official `growwapi` Python SDK. This is the ONLY file in
the framework that should ever import `growwapi` or know anything
about Groww's authentication flow, request/response shapes, or SDK
quirks. Everything upstream (Exchange, PositionManager, Strategy)
talks to the MarketDataSource interface only.

Authentication
---------------
Groww's SDK supports two ways to obtain a client:

1. api_key + api_secret -> GrowwAPI.get_access_token(...) -> access_token
2. A pre-generated access_token used directly.

GrowwMarketFeed accepts a GrowwAPIConfig and picks whichever mode has
credentials available, authenticating lazily on first use rather than
in __init__, so constructing a GrowwMarketFeed never fails just
because credentials happen to be missing at startup (e.g. in a test
environment that never actually calls get_price).
"""

from __future__ import annotations

from typing import Any

from market.market_data import MarketDataSource, PriceUnavailableError
from utils.config import GrowwAPIConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class GrowwMarketFeed(MarketDataSource):
    """Fetches live prices from the Groww API.

    Parameters
    ----------
    config:
        Groww connection details (credentials, default exchange/segment).
    """

    def __init__(self, config: GrowwAPIConfig) -> None:
        self._config = config
        self._client: Any = None
        self._price_cache: dict[str, float] = {}

    def get_price(self, symbol: str) -> float:
        """Fetch the last traded price for `symbol` from Groww.

        Raises
        ------
        PriceUnavailableError
            If authentication fails, the SDK call fails, or the
            response does not contain a usable price.
        """
        client = self._ensure_client()

        try:
            quote = client.get_quote(
                exchange=self._exchange_constant(client),
                segment=self._segment_constant(client),
                trading_symbol=symbol,
            )
        except PriceUnavailableError:
            raise
        except Exception as exc:  # SDK can raise various exception types
            logger.warning("Groww get_quote failed for %s: %s", symbol, exc)
            raise PriceUnavailableError(symbol, str(exc)) from exc

        price = quote.get("last_price") if isinstance(quote, dict) else None
        if price is None:
            raise PriceUnavailableError(symbol, "response did not include 'last_price'")

        price = float(price)
        self._price_cache[symbol] = price
        return price

    def get_last_known_price(self, symbol: str) -> float | None:
        """Return the last successfully fetched price for `symbol`, if any.

        Useful as a fallback display value while a fresh fetch is
        retried, without pretending it's a live price.
        """
        return self._price_cache.get(symbol)

    # -- internals ---------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily authenticate and cache the Groww SDK client."""
        if self._client is not None:
            return self._client

        try:
            from growwapi import GrowwAPI
        except ImportError as exc:
            raise PriceUnavailableError(
                "*",
                "The 'growwapi' package is not installed. "
                "Run: pip install growwapi",
            ) from exc

        access_token = self._config.access_token
        if not access_token:
            if not (self._config.api_key and self._config.api_secret):
                raise PriceUnavailableError(
                    "*",
                    "Groww credentials not configured. Set GROWW_API_KEY and "
                    "GROWW_API_SECRET (or GROWW_ACCESS_TOKEN) environment variables.",
                )
            try:
                access_token = GrowwAPI.get_access_token(
                    api_key=self._config.api_key,
                    secret=self._config.api_secret,
                )
            except Exception as exc:
                raise PriceUnavailableError("*", f"Groww authentication failed: {exc}") from exc

        try:
            self._client = GrowwAPI(access_token)
        except Exception as exc:
            raise PriceUnavailableError("*", f"Failed to initialize Groww client: {exc}") from exc

        logger.info("Authenticated with Groww API.")
        return self._client

    def _exchange_constant(self, client: Any) -> str:
        """Resolve the configured exchange (e.g. "NSE") to the SDK's constant.

        Falls back to the raw string if the SDK doesn't expose a
        matching constant, so config typos fail at the API call rather
        than silently here.
        """
        attr_name = f"EXCHANGE_{self._config.default_exchange.upper()}"
        return getattr(client, attr_name, self._config.default_exchange)

    def _segment_constant(self, client: Any) -> str:
        """Resolve the configured segment (e.g. "CASH") to the SDK's constant."""
        attr_name = f"SEGMENT_{self._config.default_segment.upper()}"
        return getattr(client, attr_name, self._config.default_segment)
