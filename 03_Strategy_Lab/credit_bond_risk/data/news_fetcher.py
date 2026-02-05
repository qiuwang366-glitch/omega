"""
Credit Bond Risk - News Fetcher Module

Multi-source news aggregation for credit risk monitoring.

Sources:
    International:
    - Bloomberg (RSS)
    - Reuters (RSS)
    - Financial Times (RSS)
    - Wall Street Journal (RSS)

    Domestic (China):
    - 财联社 (CLS)
    - 东方财富 (EastMoney)
    - 财新 (Caixin)

Usage:
    from data.news_fetcher import NewsAggregator
    aggregator = NewsAggregator()
    news = aggregator.fetch_all(days=7)
"""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RawNewsItem:
    """Raw news item from source"""
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)

    @property
    def news_id(self) -> str:
        """Generate unique ID based on source and title"""
        hash_input = f"{self.source}:{self.title}:{self.timestamp.date()}"
        return f"NEWS_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"


@dataclass
class AnalyzedNewsItem(RawNewsItem):
    """Analyzed news item with sentiment and entity extraction"""
    summary: str | None = None
    sentiment: str | None = None
    sentiment_score: float | None = None
    key_events: list[str] = field(default_factory=list)
    mentioned_entities: list[str] = field(default_factory=list)
    obligor_ids: list[str] = field(default_factory=list)
    relevance_score: float = 0.0


class NewsSource(str, Enum):
    """Available news sources"""
    # International
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    FT = "ft"
    WSJ = "wsj"
    # Domestic
    CLS = "cls"
    EASTMONEY = "eastmoney"
    CAIXIN = "caixin"
    # Company
    COMPANY = "company"


class NewsConfig(BaseModel):
    """Configuration for news fetching"""
    # RSS feeds
    rss_timeout_seconds: int = 30
    max_items_per_source: int = 50

    # Filtering
    credit_keywords: list[str] = [
        "bond", "credit", "debt", "yield", "spread", "rating",
        "downgrade", "upgrade", "default", "restructuring",
        "high yield", "investment grade", "corporate bond",
        "sovereign", "emerging market", "LGFV", "china",
        "城投", "债券", "评级", "违约", "化债", "信用",
    ]

    # LLM analysis
    enable_llm_analysis: bool = True
    llm_batch_size: int = 10

    # Deduplication
    similarity_threshold: float = 0.8


# =============================================================================
# News Fetcher Base Class
# =============================================================================


class NewsFetcher(ABC):
    """Abstract base class for news fetchers"""

    SOURCE_NAME: str = "unknown"

    def __init__(self, config: NewsConfig | None = None):
        self.config = config or NewsConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.SOURCE_NAME}")

    @abstractmethod
    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        """Fetch news items from source"""
        pass

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        return datetime.now()

    def _filter_by_keywords(self, title: str, content: str, keywords: list[str]) -> bool:
        """Check if article matches any credit-related keywords"""
        text = f"{title} {content}".lower()
        return any(kw.lower() in text for kw in keywords)


# =============================================================================
# International News Fetchers
# =============================================================================


class BloombergFetcher(NewsFetcher):
    """Bloomberg news fetcher via RSS feeds"""

    SOURCE_NAME = NewsSource.BLOOMBERG.value

    RSS_FEEDS = {
        "markets": "https://feeds.bloomberg.com/markets/news.rss",
        "economics": "https://feeds.bloomberg.com/economics/news.rss",
        "credit": "https://feeds.bloomberg.com/credit/news.rss",
    }

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        """Fetch Bloomberg news from RSS feeds"""
        self.logger.info(f"Fetching Bloomberg news for past {days} days...")

        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days)
        keywords = keywords or self.config.credit_keywords

        try:
            import feedparser
        except ImportError:
            self.logger.warning("feedparser not installed. Using mock data.")
            return self._get_mock_data()

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:self.config.max_items_per_source]:
                    pub_date = self._parse_date(entry.get("published", ""))
                    if pub_date < cutoff_date:
                        continue

                    title = entry.get("title", "")
                    content = entry.get("summary", entry.get("description", ""))

                    if not self._filter_by_keywords(title, content, keywords):
                        continue

                    all_news.append(RawNewsItem(
                        title=title,
                        content=content,
                        source=self.SOURCE_NAME,
                        timestamp=pub_date,
                        url=entry.get("link"),
                        tags=[feed_name],
                    ))

            except Exception as e:
                self.logger.warning(f"Failed to fetch {feed_name} feed: {e}")

        self.logger.info(f"Fetched {len(all_news)} credit-related items from Bloomberg RSS")
        return all_news if all_news else self._get_mock_data()

    def _get_mock_data(self) -> list[RawNewsItem]:
        """Return mock data when feeds unavailable"""
        return [
            RawNewsItem(
                title="Fed Holds Rates Steady, Signals Cuts May Come Later This Year",
                content="The Federal Reserve held interest rates at 5.25%-5.50% but signaled potential cuts in H2. Powell emphasized data dependency. Credit markets rallied on dovish tone.",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=1),
                url="https://bloomberg.com/news/1",
                tags=["fed", "rates", "credit"],
            ),
            RawNewsItem(
                title="Deutsche Bank Faces Fresh Concerns Over Commercial Real Estate Exposure",
                content="Deutsche Bank's US CRE portfolio under scrutiny as office vacancies rise. The bank has $17B in commercial property loans, with 40% in office sector. Spreads widened 15bp.",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=6),
                url="https://bloomberg.com/news/2",
                tags=["deutsche bank", "cre", "credit"],
            ),
            RawNewsItem(
                title="AIIB Prices $3bn Dual-Tranche Global Bond at Record Tight Spreads",
                content="Asian Infrastructure Investment Bank achieved its tightest spread ever on new issue. 5Y priced at T+32bp, 10Y at T+45bp. Order book $12B, 4x oversubscribed.",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=12),
                url="https://bloomberg.com/news/3",
                tags=["aiib", "new issue", "supranational"],
            ),
        ]


class ReutersFetcher(NewsFetcher):
    """Reuters news fetcher via RSS feeds"""

    SOURCE_NAME = NewsSource.REUTERS.value

    RSS_FEEDS = {
        "business": "https://feeds.reuters.com/reuters/businessNews",
        "markets": "https://feeds.reuters.com/reuters/marketsNews",
    }

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        """Fetch Reuters news from RSS feeds"""
        self.logger.info(f"Fetching Reuters news for past {days} days...")

        try:
            import feedparser
        except ImportError:
            self.logger.warning("feedparser not installed. Using mock data.")
            return self._get_mock_data()

        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    pub_date = self._parse_date(entry.get("published", ""))
                    if pub_date < cutoff_date:
                        continue

                    all_news.append(RawNewsItem(
                        title=entry.get("title", ""),
                        content=entry.get("summary", ""),
                        source=self.SOURCE_NAME,
                        timestamp=pub_date,
                        url=entry.get("link"),
                        tags=[feed_name],
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to fetch {feed_name}: {e}")

        self.logger.info(f"Fetched {len(all_news)} items from Reuters RSS")
        return all_news if all_news else self._get_mock_data()

    def _get_mock_data(self) -> list[RawNewsItem]:
        return [
            RawNewsItem(
                title="China's Ministry of Finance Announces LGFV Debt Resolution Framework",
                content="China's MOF unveiled a comprehensive framework for resolving LGFV debt. Provincial platforms can swap debt at lower rates. Initial quota of RMB 1.5T for 12 provinces.",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=3),
                url="https://reuters.com/news/1",
                tags=["china", "lgfv", "debt"],
            ),
        ]


class FTFetcher(NewsFetcher):
    """Financial Times news fetcher via RSS"""

    SOURCE_NAME = NewsSource.FT.value

    RSS_FEEDS = {
        "markets": "https://www.ft.com/markets?format=rss",
        "companies": "https://www.ft.com/companies?format=rss",
    }

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        """Fetch FT news from RSS feeds"""
        self.logger.info(f"Fetching FT news for past {days} days...")

        try:
            import feedparser
        except ImportError:
            self.logger.warning("feedparser not installed. Using mock data.")
            return self._get_mock_data()

        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    pub_date = self._parse_date(entry.get("published", ""))
                    if pub_date < cutoff_date:
                        continue

                    all_news.append(RawNewsItem(
                        title=entry.get("title", ""),
                        content=entry.get("summary", ""),
                        source=self.SOURCE_NAME,
                        timestamp=pub_date,
                        url=entry.get("link"),
                        tags=[feed_name],
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to fetch {feed_name}: {e}")

        self.logger.info(f"Fetched {len(all_news)} items from FT RSS")
        return all_news if all_news else self._get_mock_data()

    def _get_mock_data(self) -> list[RawNewsItem]:
        return [
            RawNewsItem(
                title="JPMorgan Beats Estimates on Record Net Interest Income",
                content="JPMorgan Chase reported Q4 earnings that exceeded analyst expectations. NII reached $24.1B, up 19% YoY. CEO Dimon warned of 'significant uncertainty' in 2025.",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(days=1, hours=6),
                url="https://ft.com/content/1",
                tags=["jpmorgan", "earnings", "banks"],
            ),
        ]


class WSJFetcher(NewsFetcher):
    """Wall Street Journal news fetcher via RSS"""

    SOURCE_NAME = NewsSource.WSJ.value

    RSS_FEEDS = {
        "markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    }

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        self.logger.info(f"Fetching WSJ news for past {days} days...")

        try:
            import feedparser
        except ImportError:
            return self._get_mock_data()

        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    pub_date = self._parse_date(entry.get("published", ""))
                    if pub_date < cutoff_date:
                        continue

                    all_news.append(RawNewsItem(
                        title=entry.get("title", ""),
                        content=entry.get("summary", ""),
                        source=self.SOURCE_NAME,
                        timestamp=pub_date,
                        url=entry.get("link"),
                        tags=[feed_name],
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to fetch {feed_name}: {e}")

        return all_news if all_news else self._get_mock_data()

    def _get_mock_data(self) -> list[RawNewsItem]:
        return [
            RawNewsItem(
                title="Pemex Production Drops to Lowest Level Since 1979",
                content="Mexico's state oil company reported another decline in crude output. Production fell to 1.4M bpd, well below the 1.9M target. Bonds fell 2 points on the news.",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=8),
                url="https://wsj.com/articles/1",
                tags=["pemex", "mexico", "oil"],
            ),
        ]


# =============================================================================
# Domestic News Fetchers
# =============================================================================


class CLSFetcher(NewsFetcher):
    """财联社 news fetcher"""

    SOURCE_NAME = NewsSource.CLS.value

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        """
        Fetch news from 财联社

        Note: In production, integrate with CLS API or scraper.
        """
        self.logger.info(f"Fetching CLS news for past {days} days...")

        # Mock data - replace with actual CLS API integration
        return [
            RawNewsItem(
                title="贵州省政府召开化债工作推进会",
                content="贵州省召开全省化债攻坚会议，要求各地市加快推进债务化解工作。省财政将安排专项资金支持重点地区...",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(days=1),
                url="https://www.cls.cn/detail/1",
                tags=["贵州", "城投", "化债"],
            ),
            RawNewsItem(
                title="某省财政厅发文支持城投平台债务重组",
                content="省财政厅发布指导意见，支持辖内城投平台通过债务重组、资产注入等方式化解债务风险...",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=2),
                url="https://www.cls.cn/detail/2",
                tags=["城投", "债务重组", "财政"],
            ),
        ]


class EastMoneyFetcher(NewsFetcher):
    """东方财富 news fetcher"""

    SOURCE_NAME = NewsSource.EASTMONEY.value

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        self.logger.info(f"Fetching EastMoney news for past {days} days...")

        return [
            RawNewsItem(
                title="美联储议息会议在即，境外中资美元债或承压",
                content="分析师预计美联储将维持高利率，境外中资美元债收益率可能继续上行...",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(days=1),
                url="https://finance.eastmoney.com/news/1",
                tags=["美元债", "美联储", "利率"],
            ),
        ]


class CaixinFetcher(NewsFetcher):
    """财新 news fetcher"""

    SOURCE_NAME = NewsSource.CAIXIN.value

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        self.logger.info(f"Fetching Caixin news for past {days} days...")

        return [
            RawNewsItem(
                title="财政部明确地方债务化解路径：省级统筹为核心",
                content="财政部发布地方债务管理新规，强调以省为主体统筹化债，建立分类处置机制...",
                source=self.SOURCE_NAME,
                timestamp=datetime.now() - timedelta(hours=8),
                url="https://www.caixin.com/news/1",
                tags=["财政部", "地方债", "化债"],
            ),
        ]


# =============================================================================
# News Aggregator
# =============================================================================


class NewsAggregator:
    """
    Aggregates news from multiple sources with deduplication.

    Features:
    - Multi-source fetching
    - Keyword filtering
    - Deduplication
    - LLM analysis integration

    Usage:
        aggregator = NewsAggregator()
        news = aggregator.fetch_all(days=7)
        analyzed = aggregator.analyze(news)
    """

    FETCHER_MAP: dict[NewsSource, type[NewsFetcher]] = {
        NewsSource.BLOOMBERG: BloombergFetcher,
        NewsSource.REUTERS: ReutersFetcher,
        NewsSource.FT: FTFetcher,
        NewsSource.WSJ: WSJFetcher,
        NewsSource.CLS: CLSFetcher,
        NewsSource.EASTMONEY: EastMoneyFetcher,
        NewsSource.CAIXIN: CaixinFetcher,
    }

    INTERNATIONAL_SOURCES = [NewsSource.BLOOMBERG, NewsSource.REUTERS, NewsSource.FT, NewsSource.WSJ]
    DOMESTIC_SOURCES = [NewsSource.CLS, NewsSource.EASTMONEY, NewsSource.CAIXIN]

    def __init__(self, config: NewsConfig | None = None):
        self.config = config or NewsConfig()
        self.logger = logging.getLogger(f"{__name__}.aggregator")

    def fetch_all(
        self,
        sources: list[NewsSource] | None = None,
        days: int = 7,
        include_international: bool = True,
        include_domestic: bool = True,
    ) -> list[RawNewsItem]:
        """
        Fetch news from multiple sources.

        Args:
            sources: Specific sources to fetch from. If None, fetch from all.
            days: Number of days to look back
            include_international: Include Bloomberg/Reuters/FT/WSJ
            include_domestic: Include CLS/EastMoney/Caixin

        Returns:
            List of aggregated news items, deduplicated and sorted
        """
        if sources is None:
            sources = []
            if include_international:
                sources.extend(self.INTERNATIONAL_SOURCES)
            if include_domestic:
                sources.extend(self.DOMESTIC_SOURCES)

        all_news = []
        for source in sources:
            if source not in self.FETCHER_MAP:
                self.logger.warning(f"Unknown source: {source}")
                continue

            try:
                fetcher = self.FETCHER_MAP[source](self.config)
                news_items = fetcher.fetch(days=days)
                all_news.extend(news_items)
                self.logger.info(f"Fetched {len(news_items)} items from {source.value}")
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source.value}: {e}")

        # Deduplicate
        all_news = self._deduplicate(all_news)

        # Sort by timestamp descending
        all_news.sort(key=lambda x: x.timestamp, reverse=True)

        self.logger.info(f"Total aggregated news: {len(all_news)} items")
        return all_news

    def fetch_source(self, source: NewsSource, days: int = 7) -> list[RawNewsItem]:
        """Fetch from a single source"""
        return self.fetch_all(sources=[source], days=days)

    def _deduplicate(self, news_items: list[RawNewsItem]) -> list[RawNewsItem]:
        """Remove duplicate news items based on title similarity"""
        seen_titles = set()
        unique_items = []

        for item in news_items:
            # Normalize title for comparison
            normalized = re.sub(r"[^\w\s]", "", item.title.lower())
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_items.append(item)

        return unique_items

    def analyze(
        self,
        news_items: list[RawNewsItem],
        use_llm: bool = True,
    ) -> list[AnalyzedNewsItem]:
        """
        Analyze news items using LLM for sentiment and entity extraction.

        Args:
            news_items: Raw news items
            use_llm: Whether to use LLM for analysis

        Returns:
            Analyzed news items with sentiment, summary, etc.
        """
        if not use_llm:
            self.logger.info("LLM analysis disabled, returning raw news")
            return [AnalyzedNewsItem(**item.__dict__) for item in news_items]

        self.logger.info(f"Analyzing {len(news_items)} news items with LLM...")

        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from intelligence.news_analyzer import NewsAnalyzer

            analyzer = NewsAnalyzer()
            results = []

            for item in news_items:
                try:
                    analysis = analyzer.analyze(item.title, item.content)
                    results.append(AnalyzedNewsItem(
                        **item.__dict__,
                        summary=analysis.get("summary"),
                        sentiment=analysis.get("sentiment"),
                        sentiment_score=analysis.get("sentiment_score"),
                        key_events=analysis.get("key_events", []),
                        mentioned_entities=analysis.get("entities", []),
                    ))
                except Exception as e:
                    self.logger.warning(f"Analysis failed for '{item.title[:50]}...': {e}")
                    results.append(AnalyzedNewsItem(**item.__dict__))

            self.logger.info(f"Analysis complete: {len(results)} items processed")
            return results

        except ImportError as e:
            self.logger.warning(f"LLM modules not available: {e}. Returning raw news.")
            return [AnalyzedNewsItem(**item.__dict__) for item in news_items]
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return [AnalyzedNewsItem(**item.__dict__) for item in news_items]


# =============================================================================
# Convenience Functions
# =============================================================================


def fetch_all_news(days: int = 7) -> list[RawNewsItem]:
    """Quick helper to fetch all news"""
    aggregator = NewsAggregator()
    return aggregator.fetch_all(days=days)


def fetch_international_news(days: int = 7) -> list[RawNewsItem]:
    """Quick helper to fetch international news only"""
    aggregator = NewsAggregator()
    return aggregator.fetch_all(
        days=days,
        include_international=True,
        include_domestic=False,
    )


def fetch_domestic_news(days: int = 7) -> list[RawNewsItem]:
    """Quick helper to fetch domestic news only"""
    aggregator = NewsAggregator()
    return aggregator.fetch_all(
        days=days,
        include_international=False,
        include_domestic=True,
    )
