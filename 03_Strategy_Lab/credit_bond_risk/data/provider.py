"""
Credit Bond Risk - Abstract Data Provider Interface

Defines the contract for all data providers, enabling:
- Pluggable data sources (Mock, Database, Bloomberg, etc.)
- Consistent API across different implementations
- Easy testing with mock providers

Design Pattern: Strategy Pattern for data access
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

# Import from core models - these are the standard data types
import sys
from pathlib import Path

# Ensure core module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    Obligor,
    BondPosition,
    CreditExposure,
    RiskAlert,
    NewsItem,
)
from core.enums import Sector, Region, CreditRating


class DataProviderType(str, Enum):
    """Available data provider types"""
    MOCK = "mock"           # Mock data for testing/demo
    DATABASE = "database"   # DuckDB/SQLite
    BLOOMBERG = "bloomberg" # Bloomberg Terminal
    WIND = "wind"           # Wind Terminal (China)
    HYBRID = "hybrid"       # Mixed sources


class DataProviderConfig(BaseModel):
    """Configuration for data providers"""
    provider_type: DataProviderType = DataProviderType.MOCK
    db_path: str | None = None
    cache_ttl_seconds: int = 300  # 5 minutes
    total_aum_usd: float = 50_000_000_000  # $50B default

    # Bloomberg config
    bbg_host: str | None = None
    bbg_port: int = 8194

    # Wind config
    wind_username: str | None = None
    wind_password: str | None = None


class DataProvider(ABC):
    """
    Abstract base class for all data providers.

    Implements the Strategy pattern for data access, allowing
    seamless switching between mock data, database, and live feeds.

    Usage:
        provider = get_data_provider(DataProviderType.MOCK)
        obligors = provider.get_obligors()
        exposures = provider.get_exposures()
    """

    def __init__(self, config: DataProviderConfig | None = None):
        self.config = config or DataProviderConfig()
        self._cache: dict[str, tuple[datetime, Any]] = {}

    @abstractmethod
    def get_obligors(self) -> dict[str, Obligor]:
        """
        Fetch all obligor master data.

        Returns:
            Dict mapping obligor_id to Obligor model
        """
        pass

    @abstractmethod
    def get_obligor(self, obligor_id: str) -> Obligor | None:
        """
        Fetch single obligor by ID.

        Args:
            obligor_id: Unique obligor identifier

        Returns:
            Obligor model or None if not found
        """
        pass

    @abstractmethod
    def get_positions(self, obligor_id: str | None = None) -> list[BondPosition]:
        """
        Fetch bond positions.

        Args:
            obligor_id: Optional filter by obligor

        Returns:
            List of BondPosition models
        """
        pass

    @abstractmethod
    def get_exposures(self) -> list[CreditExposure]:
        """
        Get aggregated credit exposures by obligor.

        Returns:
            List of CreditExposure models (pre-aggregated)
        """
        pass

    @abstractmethod
    def get_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[RiskAlert]:
        """
        Fetch risk alerts with optional filters.

        Args:
            status: Filter by alert status (PENDING, RESOLVED, etc.)
            severity: Filter by severity (CRITICAL, WARNING, INFO)
            limit: Maximum number of alerts to return

        Returns:
            List of RiskAlert models, sorted by timestamp desc
        """
        pass

    @abstractmethod
    def get_news(
        self,
        obligor_id: str | None = None,
        days: int = 7,
        limit: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch news items with optional filters.

        Args:
            obligor_id: Filter by related obligor
            days: Lookback period in days
            limit: Maximum number of items

        Returns:
            List of NewsItem models, sorted by timestamp desc
        """
        pass

    @abstractmethod
    def get_spread_history(
        self,
        obligor_id: str,
        days: int = 252,
    ) -> dict[str, float]:
        """
        Get historical OAS/spread data for an obligor.

        Args:
            obligor_id: Obligor identifier
            days: Lookback period (default 252 trading days = 1 year)

        Returns:
            Dict mapping date string (YYYY-MM-DD) to OAS value
        """
        pass

    # Convenience methods with default implementations

    def get_total_aum(self) -> float:
        """Get total AUM from config"""
        return self.config.total_aum_usd

    def get_exposures_by_sector(self) -> dict[Sector, list[CreditExposure]]:
        """Group exposures by sector"""
        exposures = self.get_exposures()
        result: dict[Sector, list[CreditExposure]] = {}
        for exp in exposures:
            sector = exp.obligor.sector
            if sector not in result:
                result[sector] = []
            result[sector].append(exp)
        return result

    def get_exposures_by_region(self) -> dict[Region, list[CreditExposure]]:
        """Group exposures by region"""
        exposures = self.get_exposures()
        result: dict[Region, list[CreditExposure]] = {}
        for exp in exposures:
            region = exp.obligor.region
            if region not in result:
                result[region] = []
            result[region].append(exp)
        return result

    def get_exposures_by_rating(self) -> dict[CreditRating, list[CreditExposure]]:
        """Group exposures by internal rating"""
        exposures = self.get_exposures()
        result: dict[CreditRating, list[CreditExposure]] = {}
        for exp in exposures:
            rating = exp.obligor.rating_internal
            if rating not in result:
                result[rating] = []
            result[rating].append(exp)
        return result

    def get_top_exposures(self, n: int = 10) -> list[CreditExposure]:
        """Get top N exposures by market value"""
        exposures = self.get_exposures()
        return sorted(exposures, key=lambda x: x.total_market_usd, reverse=True)[:n]

    def get_watchlist_obligors(self) -> list[Obligor]:
        """Get obligors with negative outlook or on watch"""
        from core.enums import RatingOutlook
        obligors = self.get_obligors()
        return [
            o for o in obligors.values()
            if o.rating_outlook in (RatingOutlook.NEGATIVE, RatingOutlook.WATCH_NEG)
        ]

    # Cache management

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if still valid"""
        if key in self._cache:
            timestamp, value = self._cache[key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.config.cache_ttl_seconds:
                return value
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value"""
        self._cache[key] = (datetime.now(), value)

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()


def get_data_provider(
    provider_type: DataProviderType = DataProviderType.MOCK,
    config: DataProviderConfig | None = None,
) -> DataProvider:
    """
    Factory function to create data provider instances.

    Args:
        provider_type: Type of provider to create
        config: Optional configuration

    Returns:
        Concrete DataProvider implementation

    Example:
        # For testing/demo
        provider = get_data_provider(DataProviderType.MOCK)

        # For production
        provider = get_data_provider(
            DataProviderType.DATABASE,
            DataProviderConfig(db_path="data/credit_risk.duckdb")
        )
    """
    if config is None:
        config = DataProviderConfig(provider_type=provider_type)

    if provider_type == DataProviderType.MOCK:
        from .mock_data import MockDataProvider
        return MockDataProvider(config)

    elif provider_type == DataProviderType.DATABASE:
        from .database import DatabaseProvider
        return DatabaseProvider(config)

    elif provider_type == DataProviderType.BLOOMBERG:
        from .market_data import BloombergProvider
        return BloombergProvider(config)

    else:
        # Default to mock for unimplemented providers
        from .mock_data import MockDataProvider
        return MockDataProvider(config)
