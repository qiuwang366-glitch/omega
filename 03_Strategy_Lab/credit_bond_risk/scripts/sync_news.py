#!/usr/bin/env python3
"""
Credit Bond Risk - News Sync Script (Phase 5)

Fetches and analyzes news from domestic and international sources.
This script is a CLI wrapper around the data.news_fetcher module.

Usage:
    python scripts/sync_news.py [--source bloomberg|reuters|ft|cls|eastmoney|all] [--days 7]

Examples:
    python sync_news.py                           # Fetch from all sources
    python sync_news.py --source bloomberg        # Bloomberg only
    python sync_news.py --international           # International sources only
    python sync_news.py --domestic --days 3       # Domestic, last 3 days
    python sync_news.py --dry-run --no-llm        # Test without LLM/DB
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.news_fetcher import (
    NewsAggregator,
    NewsSource,
    NewsConfig,
    RawNewsItem,
    AnalyzedNewsItem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Database Integration
# =============================================================================


def save_to_database(news_items: list[AnalyzedNewsItem]) -> int:
    """
    Save news items to DuckDB database.

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

    except ImportError:
        logger.warning("duckdb not installed. Run: pip install duckdb")
        return 0
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
        choices=[s.value for s in NewsSource] + ["all"],
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

    # Initialize aggregator with config
    config = NewsConfig(enable_llm_analysis=not args.no_llm)
    aggregator = NewsAggregator(config)

    # Determine sources
    if args.source != "all":
        sources = [NewsSource(args.source)]
    elif args.international:
        sources = NewsAggregator.INTERNATIONAL_SOURCES
    elif args.domestic:
        sources = NewsAggregator.DOMESTIC_SOURCES
    else:
        sources = None  # All sources

    # Fetch
    raw_news = aggregator.fetch_all(
        sources=sources,
        days=args.days,
        include_international=not args.domestic if args.domestic else True,
        include_domestic=not args.international if args.international else True,
    )

    if not raw_news:
        logger.info("No news items fetched")
        return

    # Analyze
    analyzed_news = aggregator.analyze(raw_news, use_llm=not args.no_llm)

    # Display summary
    print(f"\n{'='*60}")
    print(f"News Sync Summary")
    print(f"{'='*60}")
    print(f"Total items: {len(analyzed_news)}")
    print(f"Sources: {set(n.source for n in analyzed_news)}")
    if analyzed_news:
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
