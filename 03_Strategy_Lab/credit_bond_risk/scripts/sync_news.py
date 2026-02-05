#!/usr/bin/env python3
"""
Credit Bond Risk - News Sync Script

Fetches and analyzes news from configured sources.

Usage:
    python scripts/sync_news.py [--source cls|eastmoney|all] [--days 7]

Sources:
    - cls: 财联社
    - eastmoney: 东方财富
    - bloomberg: Bloomberg (requires terminal)
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_cls_news(days: int = 7) -> list[dict]:
    """
    Fetch news from 财联社

    Note: This is a placeholder. In production, implement actual API integration.
    """
    logger.info(f"Fetching CLS news for past {days} days...")

    # Placeholder - return mock data
    mock_news = [
        {
            "title": "某省财政厅发文支持城投平台债务重组",
            "content": "省财政厅发布指导意见，支持辖内城投平台通过债务重组、资产注入等方式化解债务风险...",
            "source": "cls",
            "timestamp": datetime.now() - timedelta(hours=2),
            "url": "https://example.com/news/1",
        },
        {
            "title": "某市城建投资被曝现金流紧张",
            "content": "据知情人士透露，该公司近期应收账款回款困难，部分项目支出延迟...",
            "source": "cls",
            "timestamp": datetime.now() - timedelta(hours=5),
            "url": "https://example.com/news/2",
        },
    ]

    logger.info(f"Fetched {len(mock_news)} news items from CLS")
    return mock_news


def fetch_eastmoney_news(days: int = 7) -> list[dict]:
    """
    Fetch news from 东方财富

    Note: This is a placeholder. In production, implement actual API/scraping.
    """
    logger.info(f"Fetching EastMoney news for past {days} days...")

    mock_news = [
        {
            "title": "美联储议息会议在即，境外中资美元债或承压",
            "content": "分析师预计美联储将维持高利率，境外中资美元债收益率可能继续上行...",
            "source": "eastmoney",
            "timestamp": datetime.now() - timedelta(days=1),
            "url": "https://example.com/news/3",
        },
    ]

    logger.info(f"Fetched {len(mock_news)} news items from EastMoney")
    return mock_news


def analyze_news(news_items: list[dict], use_llm: bool = True) -> list[dict]:
    """
    Analyze news items using LLM

    Args:
        news_items: Raw news items
        use_llm: Whether to use LLM for analysis

    Returns:
        Analyzed news items with sentiment, summary, etc.
    """
    if not use_llm:
        logger.info("LLM analysis disabled, returning raw news")
        return news_items

    logger.info(f"Analyzing {len(news_items)} news items with LLM...")

    try:
        from ..intelligence.news_analyzer import NewsAnalyzer, BatchNewsProcessor
        from ..core.models import NewsItem

        analyzer = NewsAnalyzer()
        processor = BatchNewsProcessor(analyzer)

        # Convert to NewsItem objects
        news_objs = []
        for i, item in enumerate(news_items):
            news_objs.append(NewsItem(
                news_id=f"NEWS_{datetime.now().strftime('%Y%m%d')}_{i:04d}",
                title=item["title"],
                content=item["content"],
                source=item["source"],
                timestamp=item["timestamp"],
                url=item.get("url"),
            ))

        # Process batch
        analyzed = processor.process_batch(news_objs)

        # Convert back to dicts
        results = []
        for news in analyzed:
            results.append({
                "news_id": news.news_id,
                "title": news.title,
                "content": news.content,
                "source": news.source,
                "timestamp": news.timestamp,
                "url": news.url,
                "summary": news.summary,
                "sentiment": news.sentiment.value if news.sentiment else None,
                "sentiment_score": news.sentiment_score,
                "key_events": news.key_events,
            })

        logger.info(f"Analysis complete: {len(results)} items processed")
        return results

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return news_items


def save_to_database(news_items: list[dict]) -> int:
    """
    Save news items to database

    Returns:
        Number of items saved
    """
    try:
        import duckdb
        from .init_db import DB_PATH

        conn = duckdb.connect(str(DB_PATH))

        count = 0
        for item in news_items:
            try:
                conn.execute("""
                INSERT OR REPLACE INTO news_items (
                    news_id, timestamp, source, title, content, url,
                    summary, sentiment, sentiment_score, key_events
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    item.get("news_id", f"NEWS_{datetime.now().timestamp()}"),
                    item["timestamp"],
                    item["source"],
                    item["title"],
                    item["content"],
                    item.get("url"),
                    item.get("summary"),
                    item.get("sentiment"),
                    item.get("sentiment_score"),
                    item.get("key_events"),
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Sync news from various sources")
    parser.add_argument(
        "--source",
        choices=["cls", "eastmoney", "all"],
        default="all",
        help="News source to fetch from",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to fetch",
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
    args = parser.parse_args()

    logger.info(f"Starting news sync: source={args.source}, days={args.days}")

    # Fetch news
    all_news = []

    if args.source in ["cls", "all"]:
        all_news.extend(fetch_cls_news(args.days))

    if args.source in ["eastmoney", "all"]:
        all_news.extend(fetch_eastmoney_news(args.days))

    logger.info(f"Total news items fetched: {len(all_news)}")

    if not all_news:
        logger.info("No news items to process")
        return

    # Analyze
    analyzed_news = analyze_news(all_news, use_llm=not args.no_llm)

    # Save
    if not args.dry_run:
        saved = save_to_database(analyzed_news)
        logger.info(f"Sync complete: {saved}/{len(analyzed_news)} items saved")
    else:
        logger.info(f"Dry run complete: {len(analyzed_news)} items would be saved")

        # Print sample
        for item in analyzed_news[:3]:
            print(f"\n--- {item['title']} ---")
            print(f"Source: {item['source']}")
            print(f"Sentiment: {item.get('sentiment', 'N/A')}")
            print(f"Summary: {item.get('summary', 'N/A')}")


if __name__ == "__main__":
    main()
