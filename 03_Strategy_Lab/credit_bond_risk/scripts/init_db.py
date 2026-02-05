#!/usr/bin/env python3
"""
Credit Bond Risk - Database Initialization

Creates and initializes the credit risk database tables.

Usage:
    python scripts/init_db.py [--reset]

Tables:
    - obligors: 发行人主数据
    - credit_exposures: 信用曝光快照
    - risk_alerts: 风险预警记录
    - news_items: 新闻存储
    - signal_results: 信号计算结果
    - document_embeddings: 向量存储
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed. Run: pip install duckdb")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "credit_risk.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get database connection"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def create_tables(conn: duckdb.DuckDBPyConnection, reset: bool = False) -> None:
    """Create all database tables"""

    if reset:
        logger.warning("Resetting database - all data will be lost!")
        tables = [
            "document_embeddings",
            "signal_results",
            "news_items",
            "risk_alerts",
            "credit_exposures",
            "obligors",
        ]
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    # Obligors table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS obligors (
        obligor_id VARCHAR PRIMARY KEY,
        name_cn VARCHAR NOT NULL,
        name_en VARCHAR,

        -- Classification
        sector VARCHAR NOT NULL,        -- LGFV, SOE, FINANCIAL, CORP
        sub_sector VARCHAR NOT NULL,
        province VARCHAR,
        city VARCHAR,

        -- Ratings
        rating_internal VARCHAR NOT NULL,
        rating_outlook VARCHAR DEFAULT 'STABLE',
        rating_moody VARCHAR,
        rating_sp VARCHAR,
        rating_fitch VARCHAR,

        -- Financials (snapshot)
        total_debt_cny DOUBLE,
        revenue_cny DOUBLE,
        debt_to_ebitda DOUBLE,
        interest_coverage DOUBLE,

        -- AI enhanced
        embedding_vector DOUBLE[],
        risk_narrative TEXT,
        similar_obligors VARCHAR[],

        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Credit exposures table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS credit_exposures (
        snapshot_date DATE NOT NULL,
        obligor_id VARCHAR NOT NULL REFERENCES obligors(obligor_id),

        -- Aggregated metrics
        total_nominal_usd DOUBLE NOT NULL,
        total_market_usd DOUBLE NOT NULL,
        pct_of_aum DOUBLE NOT NULL,

        -- Weighted metrics
        weighted_avg_duration DOUBLE,
        weighted_avg_oas DOUBLE,
        credit_dv01_usd DOUBLE,

        -- Maturity profile (JSON)
        maturity_profile JSON,

        -- Bond count
        num_bonds INTEGER,

        PRIMARY KEY (snapshot_date, obligor_id)
    )
    """)

    # Risk alerts table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS risk_alerts (
        alert_id VARCHAR PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,

        -- Classification
        severity VARCHAR NOT NULL,      -- INFO, WARNING, CRITICAL
        category VARCHAR NOT NULL,      -- CONCENTRATION, RATING, SPREAD, NEWS
        signal_name VARCHAR NOT NULL,

        -- Obligor
        obligor_id VARCHAR NOT NULL REFERENCES obligors(obligor_id),
        obligor_name VARCHAR NOT NULL,

        -- Alert details
        message TEXT NOT NULL,
        metric_value DOUBLE NOT NULL,
        threshold DOUBLE NOT NULL,

        -- Status
        status VARCHAR DEFAULT 'PENDING',  -- PENDING, INVESTIGATING, RESOLVED, DISMISSED
        assigned_to VARCHAR,
        resolution_note TEXT,
        resolved_at TIMESTAMP,

        -- AI enhanced
        ai_summary TEXT,
        related_news VARCHAR[]
    )
    """)

    # News items table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS news_items (
        news_id VARCHAR PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        source VARCHAR NOT NULL,
        title VARCHAR NOT NULL,
        content TEXT NOT NULL,
        url VARCHAR,

        -- AI analysis
        obligor_ids VARCHAR[],
        summary TEXT,
        sentiment VARCHAR,              -- POSITIVE, NEUTRAL, NEGATIVE
        sentiment_score DOUBLE,
        key_events VARCHAR[],

        -- Embedding
        embedding DOUBLE[],

        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Signal results table (for historical tracking)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS signal_results (
        id INTEGER PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        signal_name VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        obligor_id VARCHAR NOT NULL,

        -- Values
        value DOUBLE NOT NULL,
        z_score DOUBLE,
        percentile DOUBLE,

        -- Thresholds
        threshold_warning DOUBLE NOT NULL,
        threshold_critical DOUBLE NOT NULL,

        -- Status
        is_triggered BOOLEAN NOT NULL,
        severity VARCHAR NOT NULL,

        -- Metadata (JSON)
        metadata JSON
    )
    """)

    # Document embeddings table (for RAG)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS document_embeddings (
        doc_id VARCHAR PRIMARY KEY,
        content TEXT NOT NULL,
        embedding DOUBLE[] NOT NULL,

        -- Source info
        source_type VARCHAR NOT NULL,   -- news, filing, research, obligor_profile
        source_id VARCHAR,
        obligor_id VARCHAR,
        timestamp TIMESTAMP,

        -- Metadata (JSON)
        metadata JSON,

        -- Search index
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exposures_date ON credit_exposures(snapshot_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON risk_alerts(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON risk_alerts(severity)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_items(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signal_results(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_obligor ON document_embeddings(obligor_id)")

    logger.info("All tables created successfully")


def insert_sample_data(conn: duckdb.DuckDBPyConnection) -> None:
    """Insert sample data for testing"""

    # Check if data exists
    result = conn.execute("SELECT COUNT(*) FROM obligors").fetchone()
    if result and result[0] > 0:
        logger.info("Sample data already exists, skipping")
        return

    logger.info("Inserting sample data...")

    # Sample obligors
    sample_obligors = [
        ("OBL001", "某省城投集团", "LGFV", "省级城投", "云南", "AA", "STABLE"),
        ("OBL002", "某市城建投资", "LGFV", "地级市城投", "重庆", "AA-", "NEGATIVE"),
        ("OBL003", "某央企集团", "SOE", "央企", None, "AAA", "STABLE"),
        ("OBL004", "某股份制银行", "FINANCIAL", "股份制银行", None, "AA+", "STABLE"),
        ("OBL005", "某地方国企", "SOE", "地方国企", "四川", "AA", "WATCH_NEG"),
    ]

    for oid, name, sector, sub, province, rating, outlook in sample_obligors:
        conn.execute("""
        INSERT INTO obligors (obligor_id, name_cn, sector, sub_sector, province, rating_internal, rating_outlook)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [oid, name, sector, sub, province, rating, outlook])

    logger.info(f"Inserted {len(sample_obligors)} sample obligors")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Initialize credit risk database")
    parser.add_argument("--reset", action="store_true", help="Reset database (delete all data)")
    parser.add_argument("--sample", action="store_true", help="Insert sample data")
    args = parser.parse_args()

    logger.info(f"Database path: {DB_PATH}")

    conn = get_connection()

    try:
        create_tables(conn, reset=args.reset)

        if args.sample:
            insert_sample_data(conn)

        # Print summary
        tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """).fetchall()

        logger.info(f"Database contains {len(tables)} tables:")
        for (table,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  - {table}: {count} rows")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
