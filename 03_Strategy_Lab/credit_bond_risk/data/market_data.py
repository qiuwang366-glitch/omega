"""
Credit Bond Risk - Market Data Service

Provides market data from various sources:
- Bloomberg Terminal (via xbbg)
- Wind Terminal
- Local database cache

Design Pattern: Strategy Pattern for multiple data sources
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MarketDataSource(str, Enum):
    """Available market data sources"""
    BLOOMBERG = "bloomberg"
    WIND = "wind"
    DATABASE = "database"
    MOCK = "mock"


@dataclass
class PriceQuote:
    """Real-time price quote"""
    isin: str
    source: MarketDataSource
    timestamp: datetime

    # Pricing
    bid_price: float | None = None
    ask_price: float | None = None
    mid_price: float | None = None
    last_price: float | None = None

    # Yields & Spreads
    bid_yield: float | None = None
    ask_yield: float | None = None
    ytm: float | None = None
    oas: float | None = None
    z_spread: float | None = None
    g_spread: float | None = None

    # Risk metrics
    duration: float | None = None
    convexity: float | None = None

    # Metadata
    currency: str = "USD"
    settlement_date: date | None = None
    accrued_interest: float | None = None

    @property
    def spread(self) -> float | None:
        """Return best available spread (OAS preferred)"""
        return self.oas or self.z_spread or self.g_spread


@dataclass
class SpreadHistory:
    """Historical spread data for a security"""
    isin: str
    source: MarketDataSource
    data: dict[str, float] = field(default_factory=dict)  # date -> spread
    spread_type: str = "OAS"

    @property
    def dates(self) -> list[str]:
        return sorted(self.data.keys())

    @property
    def values(self) -> list[float]:
        return [self.data[d] for d in self.dates]

    def get_latest(self) -> float | None:
        if not self.data:
            return None
        return self.data[max(self.data.keys())]

    def get_percentile(self, value: float) -> float:
        """Calculate percentile rank of value in history"""
        if not self.data:
            return 0.5
        values = sorted(self.values)
        count_below = sum(1 for v in values if v <= value)
        return count_below / len(values)

    def get_zscore(self, value: float) -> float | None:
        """Calculate z-score of value vs history"""
        if len(self.data) < 30:
            return None
        import statistics
        values = self.values
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        if std == 0:
            return 0.0
        return (value - mean) / std


class MarketDataConfig(BaseModel):
    """Configuration for market data service"""
    # Bloomberg
    bbg_host: str = "localhost"
    bbg_port: int = 8194
    bbg_timeout_ms: int = 5000

    # Wind
    wind_username: str | None = None
    wind_password: str | None = None

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_dir: str = "data/cache/market_data"

    # Defaults
    default_source: MarketDataSource = MarketDataSource.MOCK
    lookback_days: int = 252


# =============================================================================
# Abstract Market Data Fetcher
# =============================================================================


class MarketDataFetcher(ABC):
    """Abstract base class for market data fetchers"""

    SOURCE: MarketDataSource

    def __init__(self, config: MarketDataConfig | None = None):
        self.config = config or MarketDataConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.SOURCE.value}")

    @abstractmethod
    def get_quote(self, isin: str) -> PriceQuote | None:
        """Get real-time quote for a single security"""
        pass

    @abstractmethod
    def get_quotes(self, isins: list[str]) -> dict[str, PriceQuote]:
        """Get real-time quotes for multiple securities"""
        pass

    @abstractmethod
    def get_spread_history(
        self,
        isin: str,
        days: int = 252,
        spread_type: str = "OAS",
    ) -> SpreadHistory | None:
        """Get historical spread data"""
        pass

    def is_available(self) -> bool:
        """Check if data source is available"""
        return True


# =============================================================================
# Bloomberg Implementation
# =============================================================================


class BloombergFetcher(MarketDataFetcher):
    """
    Bloomberg Terminal data fetcher via xbbg library.

    Requires:
    - Bloomberg Terminal or B-PIPE connection
    - xbbg library: pip install xbbg

    Usage:
        fetcher = BloombergFetcher()
        quote = fetcher.get_quote("US912828ZT05")
    """

    SOURCE = MarketDataSource.BLOOMBERG

    FIELD_MAPPING = {
        "PX_BID": "bid_price",
        "PX_ASK": "ask_price",
        "PX_MID": "mid_price",
        "PX_LAST": "last_price",
        "YLD_YTM_BID": "bid_yield",
        "YLD_YTM_ASK": "ask_yield",
        "YLD_YTM_MID": "ytm",
        "OAS_SPREAD_BID": "oas",
        "Z_SPREAD_BID": "z_spread",
        "G_SPREAD_BID": "g_spread",
        "DUR_ADJ_MID": "duration",
        "CONVEXITY": "convexity",
        "CRNCY": "currency",
        "SETTLE_DT": "settlement_date",
        "INT_ACC": "accrued_interest",
    }

    def __init__(self, config: MarketDataConfig | None = None):
        super().__init__(config)
        self._blp = None

    def _get_blp(self):
        """Lazy initialization of Bloomberg connection"""
        if self._blp is None:
            try:
                from xbbg import blp
                self._blp = blp
                self.logger.info("Bloomberg connection initialized")
            except ImportError:
                self.logger.warning("xbbg not installed. Run: pip install xbbg")
                raise
        return self._blp

    def is_available(self) -> bool:
        """Check if Bloomberg is available"""
        try:
            self._get_blp()
            return True
        except Exception:
            return False

    def get_quote(self, isin: str) -> PriceQuote | None:
        """Get real-time quote from Bloomberg"""
        try:
            blp = self._get_blp()

            # Convert ISIN to Bloomberg ticker
            ticker = f"/isin/{isin}"

            # Request fields
            fields = list(self.FIELD_MAPPING.keys())
            data = blp.bdp(ticker, fields)

            if data.empty:
                self.logger.warning(f"No data returned for {isin}")
                return None

            # Build PriceQuote
            row = data.iloc[0]
            quote_data = {"isin": isin, "source": self.SOURCE, "timestamp": datetime.now()}

            for bbg_field, attr in self.FIELD_MAPPING.items():
                if bbg_field in row.index:
                    quote_data[attr] = row[bbg_field]

            return PriceQuote(**quote_data)

        except Exception as e:
            self.logger.error(f"Bloomberg quote failed for {isin}: {e}")
            return None

    def get_quotes(self, isins: list[str]) -> dict[str, PriceQuote]:
        """Get real-time quotes for multiple securities"""
        try:
            blp = self._get_blp()

            # Convert ISINs to Bloomberg tickers
            tickers = [f"/isin/{isin}" for isin in isins]

            # Batch request
            fields = list(self.FIELD_MAPPING.keys())
            data = blp.bdp(tickers, fields)

            quotes = {}
            for isin, ticker in zip(isins, tickers):
                if ticker in data.index:
                    row = data.loc[ticker]
                    quote_data = {"isin": isin, "source": self.SOURCE, "timestamp": datetime.now()}

                    for bbg_field, attr in self.FIELD_MAPPING.items():
                        if bbg_field in row.index:
                            quote_data[attr] = row[bbg_field]

                    quotes[isin] = PriceQuote(**quote_data)

            return quotes

        except Exception as e:
            self.logger.error(f"Bloomberg batch quote failed: {e}")
            return {}

    def get_spread_history(
        self,
        isin: str,
        days: int = 252,
        spread_type: str = "OAS",
    ) -> SpreadHistory | None:
        """Get historical spread data from Bloomberg"""
        try:
            blp = self._get_blp()

            ticker = f"/isin/{isin}"
            end_date = date.today()
            start_date = end_date - timedelta(days=int(days * 1.5))  # Extra buffer for weekends

            # Map spread type to Bloomberg field
            field_map = {
                "OAS": "OAS_SPREAD_BID",
                "Z_SPREAD": "Z_SPREAD_BID",
                "G_SPREAD": "G_SPREAD_BID",
            }
            field = field_map.get(spread_type, "OAS_SPREAD_BID")

            data = blp.bdh(
                ticker,
                field,
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d"),
            )

            if data.empty:
                return None

            # Convert to dict
            spread_data = {}
            for idx, row in data.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                spread_data[date_str] = row.iloc[0]

            return SpreadHistory(
                isin=isin,
                source=self.SOURCE,
                data=spread_data,
                spread_type=spread_type,
            )

        except Exception as e:
            self.logger.error(f"Bloomberg history failed for {isin}: {e}")
            return None


# =============================================================================
# Wind Implementation
# =============================================================================


class WindFetcher(MarketDataFetcher):
    """
    Wind Terminal data fetcher.

    Requires:
    - Wind Terminal subscription
    - WindPy library

    Usage:
        fetcher = WindFetcher()
        quote = fetcher.get_quote("220210.IB")
    """

    SOURCE = MarketDataSource.WIND

    def __init__(self, config: MarketDataConfig | None = None):
        super().__init__(config)
        self._w = None

    def _get_wind(self):
        """Lazy initialization of Wind connection"""
        if self._w is None:
            try:
                from WindPy import w
                w.start()
                self._w = w
                self.logger.info("Wind connection initialized")
            except ImportError:
                self.logger.warning("WindPy not installed")
                raise
        return self._w

    def is_available(self) -> bool:
        try:
            self._get_wind()
            return True
        except Exception:
            return False

    def get_quote(self, isin: str) -> PriceQuote | None:
        """Get real-time quote from Wind"""
        try:
            w = self._get_wind()

            # Wind uses different identifiers, may need mapping
            fields = ["close", "ytm_b", "duration", "convexity"]
            data = w.wsq(isin, fields)

            if data.ErrorCode != 0:
                self.logger.warning(f"Wind error for {isin}: {data.ErrorCode}")
                return None

            return PriceQuote(
                isin=isin,
                source=self.SOURCE,
                timestamp=datetime.now(),
                last_price=data.Data[0][0] if data.Data[0] else None,
                ytm=data.Data[1][0] if data.Data[1] else None,
                duration=data.Data[2][0] if data.Data[2] else None,
                convexity=data.Data[3][0] if data.Data[3] else None,
            )

        except Exception as e:
            self.logger.error(f"Wind quote failed for {isin}: {e}")
            return None

    def get_quotes(self, isins: list[str]) -> dict[str, PriceQuote]:
        """Get real-time quotes from Wind"""
        quotes = {}
        for isin in isins:
            quote = self.get_quote(isin)
            if quote:
                quotes[isin] = quote
        return quotes

    def get_spread_history(
        self,
        isin: str,
        days: int = 252,
        spread_type: str = "OAS",
    ) -> SpreadHistory | None:
        """Get historical spread data from Wind"""
        try:
            w = self._get_wind()

            end_date = date.today()
            start_date = end_date - timedelta(days=int(days * 1.5))

            # Wind field for spread
            field = "creditspread" if spread_type != "OAS" else "oas"

            data = w.wsd(
                isin,
                field,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            if data.ErrorCode != 0:
                return None

            spread_data = {}
            for i, dt in enumerate(data.Times):
                date_str = dt.strftime("%Y-%m-%d")
                spread_data[date_str] = data.Data[0][i]

            return SpreadHistory(
                isin=isin,
                source=self.SOURCE,
                data=spread_data,
                spread_type=spread_type,
            )

        except Exception as e:
            self.logger.error(f"Wind history failed for {isin}: {e}")
            return None


# =============================================================================
# Mock Implementation
# =============================================================================


class MockMarketFetcher(MarketDataFetcher):
    """Mock market data for testing"""

    SOURCE = MarketDataSource.MOCK

    def get_quote(self, isin: str) -> PriceQuote | None:
        import random
        return PriceQuote(
            isin=isin,
            source=self.SOURCE,
            timestamp=datetime.now(),
            mid_price=100 + random.uniform(-5, 5),
            ytm=4.5 + random.uniform(-1, 1),
            oas=150 + random.uniform(-30, 30),
            duration=5 + random.uniform(-2, 2),
        )

    def get_quotes(self, isins: list[str]) -> dict[str, PriceQuote]:
        return {isin: self.get_quote(isin) for isin in isins}

    def get_spread_history(
        self,
        isin: str,
        days: int = 252,
        spread_type: str = "OAS",
    ) -> SpreadHistory | None:
        from .mock_data import generate_mock_spread_history
        data = generate_mock_spread_history(isin, days)
        return SpreadHistory(
            isin=isin,
            source=self.SOURCE,
            data=data,
            spread_type=spread_type,
        )


# =============================================================================
# Market Data Service (Unified Interface)
# =============================================================================


class MarketDataService:
    """
    Unified market data service with multi-source fallback.

    Features:
    - Automatic source selection based on availability
    - Caching layer
    - Batch optimization

    Usage:
        service = MarketDataService()
        quote = service.get_quote("US912828ZT05")
        history = service.get_spread_history("US912828ZT05")
    """

    def __init__(self, config: MarketDataConfig | None = None):
        self.config = config or MarketDataConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize fetchers by priority
        self._fetchers: dict[MarketDataSource, MarketDataFetcher] = {}
        self._init_fetchers()

        # Cache
        self._quote_cache: dict[str, tuple[datetime, PriceQuote]] = {}
        self._history_cache: dict[str, tuple[datetime, SpreadHistory]] = {}

    def _init_fetchers(self) -> None:
        """Initialize available fetchers"""
        # Always have mock available
        self._fetchers[MarketDataSource.MOCK] = MockMarketFetcher(self.config)

        # Try Bloomberg
        try:
            bbg = BloombergFetcher(self.config)
            if bbg.is_available():
                self._fetchers[MarketDataSource.BLOOMBERG] = bbg
        except Exception:
            pass

        # Try Wind
        try:
            wind = WindFetcher(self.config)
            if wind.is_available():
                self._fetchers[MarketDataSource.WIND] = wind
        except Exception:
            pass

        self.logger.info(f"Available market data sources: {list(self._fetchers.keys())}")

    def _get_fetcher(self, source: MarketDataSource | None = None) -> MarketDataFetcher:
        """Get fetcher for specified or default source"""
        if source and source in self._fetchers:
            return self._fetchers[source]

        # Priority: Bloomberg > Wind > Mock
        for src in [MarketDataSource.BLOOMBERG, MarketDataSource.WIND, MarketDataSource.MOCK]:
            if src in self._fetchers:
                return self._fetchers[src]

        return self._fetchers[MarketDataSource.MOCK]

    def get_quote(
        self,
        isin: str,
        source: MarketDataSource | None = None,
        use_cache: bool = True,
    ) -> PriceQuote | None:
        """Get real-time quote with caching"""
        cache_key = f"{isin}:{source or 'default'}"

        # Check cache
        if use_cache and cache_key in self._quote_cache:
            timestamp, quote = self._quote_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.config.cache_ttl_seconds:
                return quote

        # Fetch
        fetcher = self._get_fetcher(source)
        quote = fetcher.get_quote(isin)

        # Cache result
        if quote and use_cache:
            self._quote_cache[cache_key] = (datetime.now(), quote)

        return quote

    def get_quotes(
        self,
        isins: list[str],
        source: MarketDataSource | None = None,
    ) -> dict[str, PriceQuote]:
        """Get quotes for multiple securities"""
        fetcher = self._get_fetcher(source)
        return fetcher.get_quotes(isins)

    def get_spread_history(
        self,
        isin: str,
        days: int = 252,
        spread_type: str = "OAS",
        source: MarketDataSource | None = None,
        use_cache: bool = True,
    ) -> SpreadHistory | None:
        """Get historical spread data with caching"""
        cache_key = f"{isin}:{days}:{spread_type}:{source or 'default'}"

        # Check cache
        if use_cache and cache_key in self._history_cache:
            timestamp, history = self._history_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.config.cache_ttl_seconds * 10:  # Longer TTL for history
                return history

        # Fetch
        fetcher = self._get_fetcher(source)
        history = fetcher.get_spread_history(isin, days, spread_type)

        # Cache result
        if history and use_cache:
            self._history_cache[cache_key] = (datetime.now(), history)

        return history

    def clear_cache(self) -> None:
        """Clear all caches"""
        self._quote_cache.clear()
        self._history_cache.clear()

    @property
    def available_sources(self) -> list[MarketDataSource]:
        """List available data sources"""
        return list(self._fetchers.keys())
