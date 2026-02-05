"""
Credit Bond Risk - Data Layer

Centralized data access layer providing:
- Abstract data provider interfaces
- Market data fetching (BBG/Wind)
- News aggregation (RSS/API)
- Filing/announcement parsing
- Mock data generation for testing

Architecture Pattern:
    Provider (ABC) → Concrete Implementations → Unified Data Interface

Usage:
    from data import DataProvider, MockDataProvider, get_data_provider
    from data.market_data import MarketDataService
    from data.news_fetcher import NewsAggregator
"""

from .provider import (
    DataProvider,
    DataProviderType,
    get_data_provider,
)
from .mock_data import (
    MockDataProvider,
    generate_mock_obligors,
    generate_mock_exposures,
    generate_mock_alerts,
    generate_mock_news,
)
from .market_data import (
    MarketDataService,
    MarketDataSource,
    PriceQuote,
    SpreadHistory,
)
from .news_fetcher import (
    NewsFetcher,
    NewsAggregator,
    RawNewsItem,
    AnalyzedNewsItem,
)
from .filing_parser import (
    FilingParser,
    FilingType,
    ParsedFiling,
)

__all__ = [
    # Provider
    "DataProvider",
    "DataProviderType",
    "get_data_provider",
    # Mock
    "MockDataProvider",
    "generate_mock_obligors",
    "generate_mock_exposures",
    "generate_mock_alerts",
    "generate_mock_news",
    # Market Data
    "MarketDataService",
    "MarketDataSource",
    "PriceQuote",
    "SpreadHistory",
    # News
    "NewsFetcher",
    "NewsAggregator",
    "RawNewsItem",
    "AnalyzedNewsItem",
    # Filing
    "FilingParser",
    "FilingType",
    "ParsedFiling",
]
