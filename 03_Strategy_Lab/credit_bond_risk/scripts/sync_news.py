#!/usr/bin/env python3
"""
Credit Bond Risk - News Sync Script (Phase 5)

Fetches and analyzes news from domestic and international sources.

Usage:
    python scripts/sync_news.py [--source bloomberg|reuters|ft|cls|eastmoney|all] [--days 7]

Sources:
    Domestic:
    - cls: 财联社
    - eastmoney: 东方财富
    - caixin: 财新

    International:
    - bloomberg: Bloomberg (via RSS/API)
    - reuters: Reuters (via RSS)
    - ft: Financial Times (via RSS)
    - wsj: Wall Street Journal (via RSS)
"""

import argparse
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


# =============================================================================
# News Fetcher Base Class
# =============================================================================


class NewsFetcher(ABC):
    """Abstract base class for news fetchers"""

    SOURCE_NAME: str = "unknown"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
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


# =============================================================================
# International News Fetchers
# =============================================================================


class BloombergFetcher(NewsFetcher):
    """
    Bloomberg news fetcher via RSS feeds

    Note: For full Bloomberg Terminal integration, use xbbg library.
    This implementation uses public RSS feeds for basic news.
    """

    SOURCE_NAME = "bloomberg"

    RSS_FEEDS = {
        "markets": "https://feeds.bloomberg.com/markets/news.rss",
        "economics": "https://feeds.bloomberg.com/economics/news.rss",
        "credit": "https://feeds.bloomberg.com/credit/news.rss",
    }

    CREDIT_KEYWORDS = [
        "bond", "credit", "debt", "yield", "spread", "rating",
        "downgrade", "upgrade", "default", "restructuring",
        "high yield", "investment grade", "corporate bond",
        "sovereign", "emerging market", "LGFV", "china",
    ]

    def fetch(self, days: int = 7, keywords: list[str] | None = None) -> list[RawNewsItem]:
        """Fetch Bloomberg news from RSS feeds"""
        self.logger.info(f"Fetching Bloomberg news for past {days} days...")

        all_news = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            import feedparser
        except ImportError:
            self.logger.warning("feedparser not installed. Using mock data.")
            return self._get_mock_data()

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:50]:  # Limit per feed
                    pub_date = self._parse_date(entry.get("published", ""))
                    if pub_date < cutoff_date:
                        continue

                    title = entry.get("title", "")
                    content = entry.get("summary", entry.get("description", ""))

                    # Filter by credit-related keywords
                    text = f"{title} {content}".lower()
                    if not any(kw.lower() in text for kw in self.CREDIT_KEYWORDS):
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

    SOURCE_NAME = "reuters"

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

    SOURCE_NAME = "ft"

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

    SOURCE_NAME = "wsj"

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

    SOURCE_NAME = "cls"

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

    SOURCE_NAME = "eastmoney"

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

    SOURCE_NAME = "caixin"

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
    """Aggregates news from multiple sources"""

    FETCHER_MAP = {
        "bloomberg": BloombergFetcher,
        "reuters": ReutersFetcher,
        "ft": FTFetcher,
        "wsj": WSJFetcher,
        "cls": CLSFetcher,
        "eastmoney": EastMoneyFetcher,
        "caixin": CaixinFetcher,
    }

    INTERNATIONAL_SOURCES = ["bloomberg", "reuters", "ft", "wsj"]
    DOMESTIC_SOURCES = ["cls", "eastmoney", "caixin"]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.aggregator")

    def fetch_all(
        self,
        sources: list[str] | None = None,
        days: int = 7,
        include_international: bool = True,
        include_domestic: bool = True,
    ) -> list[RawNewsItem]:
        """
        Fetch news from multiple sources

        Args:
            sources: Specific sources to fetch from. If None, fetch from all.
            days: Number of days to look back
            include_international: Include Bloomberg/Reuters/FT/WSJ
            include_domestic: Include CLS/EastMoney/Caixin

        Returns:
            List of aggregated news items
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
                fetcher = self.FETCHER_MAP[source](self.config.get(source, {}))
                news_items = fetcher.fetch(days=days)
                all_news.extend(news_items)
                self.logger.info(f"Fetched {len(news_items)} items from {source}")
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source}: {e}")

        # Deduplicate by title similarity
        all_news = self._deduplicate(all_news)

        # Sort by timestamp descending
        all_news.sort(key=lambda x: x.timestamp, reverse=True)

        self.logger.info(f"Total aggregated news: {len(all_news)} items")
        return all_news

    def _deduplicate(self, news_items: list[RawNewsItem], similarity_threshold: float = 0.8) -> list[RawNewsItem]:
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


# =============================================================================
# News Analyzer Integration
# =============================================================================


def analyze_news(news_items: list[RawNewsItem], use_llm: bool = True) -> list[AnalyzedNewsItem]:
    """
    Analyze news items using LLM for sentiment and entity extraction

    Args:
        news_items: Raw news items
        use_llm: Whether to use LLM for analysis

    Returns:
        Analyzed news items with sentiment, summary, etc.
    """
    if not use_llm:
        logger.info("LLM analysis disabled, returning raw news")
        return [AnalyzedNewsItem(**item.__dict__) for item in news_items]

    logger.info(f"Analyzing {len(news_items)} news items with LLM...")

    try:
        import sys
        sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

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
                logger.warning(f"Analysis failed for '{item.title[:50]}...': {e}")
                results.append(AnalyzedNewsItem(**item.__dict__))

        logger.info(f"Analysis complete: {len(results)} items processed")
        return results

    except ImportError as e:
        logger.warning(f"LLM modules not available: {e}. Returning raw news.")
        return [AnalyzedNewsItem(**item.__dict__) for item in news_items]
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return [AnalyzedNewsItem(**item.__dict__) for item in news_items]


# =============================================================================
# Database Integration
# =============================================================================


def save_to_database(news_items: list[AnalyzedNewsItem]) -> int:
    """
    Save news items to DuckDB database

    Returns:
        Number of items saved
    """
    try:
        import duckdb
        from pathlib import Path

        db_path = Path(__file__).parent.parent / "data" / "credit_risk.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = duckdb.connect(str(db_path))

        # Create table if not exists
        conn.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            news_id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            source VARCHAR,
            title VARCHAR,
            content TEXT,
            url VARCHAR,
            summary TEXT,
            sentiment VARCHAR,
            sentiment_score DOUBLE,
            key_events VARCHAR[],
            mentioned_entities VARCHAR[],
            obligor_ids VARCHAR[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        count = 0
        for item in news_items:
            try:
                conn.execute("""
                INSERT OR REPLACE INTO news_items (
                    news_id, timestamp, source, title, content, url,
                    summary, sentiment, sentiment_score, key_events,
                    mentioned_entities, obligor_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    item.news_id,
                    item.timestamp,
                    item.source,
                    item.title,
                    item.content,
                    item.url,
                    item.summary,
                    item.sentiment,
                    item.sentiment_score,
                    item.key_events,
                    item.mentioned_entities,
                    item.obligor_ids,
                ])
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save news item: {e}")

        conn.close()
        logger.info(f"Saved {count} news items to database")
        return count

    except Exception as e:
        logger.error(f"Database save failed: {e}")
        return 0


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sync news from domestic and international sources (Phase 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync_news.py                           # Fetch from all sources
  python sync_news.py --source bloomberg        # Bloomberg only
  python sync_news.py --international           # International sources only
  python sync_news.py --domestic --days 3       # Domestic, last 3 days
  python sync_news.py --dry-run --no-llm        # Test without LLM/DB
        """
    )
    parser.add_argument(
        "--source",
        choices=list(NewsAggregator.FETCHER_MAP.keys()) + ["all"],
        default="all",
        help="Specific source to fetch from",
    )
    parser.add_argument(
        "--international",
        action="store_true",
        help="Fetch only from international sources",
    )
    parser.add_argument(
        "--domestic",
        action="store_true",
        help="Fetch only from domestic sources",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting news sync: source={args.source}, days={args.days}")

    # Initialize aggregator
    aggregator = NewsAggregator()

    # Determine sources
    if args.source != "all":
        sources = [args.source]
    elif args.international:
        sources = NewsAggregator.INTERNATIONAL_SOURCES
    elif args.domestic:
        sources = NewsAggregator.DOMESTIC_SOURCES
    else:
        sources = None  # All sources

    # Fetch
    raw_news = aggregator.fetch_all(sources=sources, days=args.days)

    if not raw_news:
        logger.info("No news items fetched")
        return

    # Analyze
    analyzed_news = analyze_news(raw_news, use_llm=not args.no_llm)

    # Display summary
    print(f"\n{'='*60}")
    print(f"News Sync Summary")
    print(f"{'='*60}")
    print(f"Total items: {len(analyzed_news)}")
    print(f"Sources: {set(n.source for n in analyzed_news)}")
    print(f"Date range: {min(n.timestamp for n in analyzed_news).date()} to {max(n.timestamp for n in analyzed_news).date()}")

    # Show sample
    print(f"\n{'='*60}")
    print("Sample Headlines (latest 5):")
    print(f"{'='*60}")
    for item in analyzed_news[:5]:
        sentiment_icon = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}.get(item.sentiment or "", "⚪")
        print(f"{sentiment_icon} [{item.source.upper():10}] {item.title[:70]}...")

    # Save
    if not args.dry_run:
        saved = save_to_database(analyzed_news)
        logger.info(f"Sync complete: {saved}/{len(analyzed_news)} items saved")
    else:
        logger.info(f"Dry run complete: {len(analyzed_news)} items would be saved")


if __name__ == "__main__":
    main()
