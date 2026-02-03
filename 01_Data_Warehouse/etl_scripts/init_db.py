"""
init_db.py - Portfolio Position ETL Pipeline
Loads position data from CSV into DuckDB with carry calculations.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb
import numpy as np
import pandas as pd


# ==========================================
# 0. Config & Constants
# ==========================================
BASE_DIR = Path(__file__).parent.parent
DEFAULT_DB_PATH = BASE_DIR / "db" / "portfolio.duckdb"
DEFAULT_RAW_DIR = BASE_DIR / "raw_landing"

COL_MAP: dict[str, str] = {
    "ISIN": "isin", "TICKER": "ticker", "债券名称": "security_name",
    "分类1": "sector_l1", "分类2": "sector_l2", "CCY": "currency",
    "AccSection": "accounting_raw", "Portfolio": "portfolio_id",
    "Nominal（USD）": "nominal_usd", "Nominal（原币）": "nominal_local",
    "摊余成本（USD）": "book_value_usd", "Maket Value（USD）": "market_value_usd",
    "EffectiveYield": "yield_effective", "FTP Rate": "ftp_rate",
    "Duration": "duration_modified", "Maturity": "maturity_date",
}

ACC_MAP: dict[str, str] = {"HTM": "AC", "AFS": "FVOCI", "HFT": "FVTPL", "Trading": "FVTPL"}

NUM_COLS: list[str] = [
    "nominal_usd", "nominal_local", "book_value_usd", "market_value_usd",
    "yield_effective", "ftp_rate", "duration_modified",
]


# ==========================================
# 1. Vectorized Helpers
# ==========================================
def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Vectorized Excel string -> float conversion."""
    s = s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    s = s.replace(["-", "", "nan", "None"], "0")
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def compute_fx_rate(df: pd.DataFrame) -> np.ndarray:
    """Vectorized FX rate: USD=1, else Nominal(USD)/Nominal(Local)."""
    return np.where(
        df["currency"] == "USD",
        1.0,
        np.where(df["nominal_local"] != 0, df["nominal_usd"] / df["nominal_local"], 0.0),
    )


def compute_carry_base(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized carry base calculation:
    - AC/FVOCI: Book Value
    - FVTPL/OTHER: Market Value (fallback to Book if MV=0)
    """
    is_accrual = df["accounting_type"].isin(["AC", "FVOCI"])
    mv_or_bv = np.where(df["market_value_usd"] != 0, df["market_value_usd"], df["book_value_usd"])
    return np.where(is_accrual, df["book_value_usd"], mv_or_bv)


# ==========================================
# 2. Database Context Manager
# ==========================================
@contextmanager
def duckdb_connection(db_path: Path) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager for DuckDB connection with auto-close."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


# ==========================================
# 3. Schema Definitions
# ==========================================
DDL_SECURITY_MASTER = """
CREATE TABLE IF NOT EXISTS security_master (
    isin VARCHAR PRIMARY KEY,
    ticker VARCHAR,
    name VARCHAR,
    currency VARCHAR(3),
    sector_l1 VARCHAR,
    sector_l2 VARCHAR,
    maturity_date DATE,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_POSITIONS_DAILY = """
CREATE TABLE IF NOT EXISTS positions_daily (
    snapshot_date DATE,
    isin VARCHAR,
    portfolio_id VARCHAR,
    accounting_type VARCHAR,
    currency VARCHAR(3),
    -- Exposure
    nominal_usd DOUBLE,
    nominal_local DOUBLE,
    book_value_usd DOUBLE,
    market_value_usd DOUBLE,
    -- Pricing
    fx_rate DOUBLE,
    yield_effective DOUBLE,
    ftp_rate DOUBLE,
    duration DOUBLE,
    -- PnL Metrics
    carry_base_usd DOUBLE,
    carry_annual_usd DOUBLE,
    PRIMARY KEY (snapshot_date, isin, portfolio_id)
);
"""


# ==========================================
# 4. ETL Pipeline
# ==========================================
def extract(csv_path: Path) -> pd.DataFrame:
    """Extract: Load CSV and rename columns."""
    df = pd.read_csv(csv_path)
    df.rename(columns=COL_MAP, inplace=True)
    return df


def transform(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Transform: Clean data and compute derived fields."""
    # Vectorized numeric cleaning
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])

    # Accounting classification
    df["accounting_type"] = df["accounting_raw"].map(ACC_MAP).fillna("OTHER")

    # Vectorized calculations
    df["fx_rate"] = compute_fx_rate(df)
    df["carry_base_usd"] = compute_carry_base(df)
    df["carry_annual_usd"] = df["carry_base_usd"] * (df["yield_effective"] / 100.0)

    # Date handling
    df["snapshot_date"] = pd.to_datetime(snapshot_date).date()
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], dayfirst=True, errors="coerce").dt.date

    return df


def load(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, snapshot_date: str) -> None:
    """Load: Upsert data into DuckDB tables."""
    conn.execute(DDL_SECURITY_MASTER)
    conn.execute(DDL_POSITIONS_DAILY)

    # Upsert security master
    conn.execute("""
        INSERT OR IGNORE INTO security_master
        SELECT DISTINCT isin, ticker, security_name, currency, sector_l1, sector_l2, maturity_date, CURRENT_TIMESTAMP
        FROM df WHERE isin IS NOT NULL
    """)

    # Delete-insert pattern for positions (idempotent reload)
    conn.execute("DELETE FROM positions_daily WHERE snapshot_date = ?", [snapshot_date])
    conn.execute("""
        INSERT INTO positions_daily
        SELECT snapshot_date, isin, portfolio_id, accounting_type, currency,
               nominal_usd, nominal_local, book_value_usd, market_value_usd,
               fx_rate, yield_effective, ftp_rate, duration_modified,
               carry_base_usd, carry_annual_usd
        FROM df WHERE isin IS NOT NULL
    """)


def validate(conn: duckdb.DuckDBPyConnection) -> None:
    """Print validation summary."""
    print("\n--- Validation Report ---")
    print("\nFX Rates (Non-USD):")
    print(conn.execute("""
        SELECT currency, ROUND(fx_rate, 4) as fx_rate, COUNT(*) as cnt
        FROM positions_daily WHERE currency != 'USD'
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 5
    """).df().to_string(index=False))

    print("\nCarry by Accounting Type:")
    print(conn.execute("""
        SELECT accounting_type,
               ROUND(SUM(book_value_usd)/1e9, 2) as book_bn,
               ROUND(SUM(carry_annual_usd)/1e6, 1) as carry_mm
        FROM positions_daily GROUP BY 1 ORDER BY 2 DESC
    """).df().to_string(index=False))

    print("\nPortfolio Summary:")
    print(conn.execute("""
        SELECT COUNT(*) as positions,
               ROUND(SUM(book_value_usd)/1e9, 2) as total_book_bn,
               ROUND(SUM(carry_annual_usd)/1e6, 1) as total_carry_mm,
               ROUND(SUM(carry_annual_usd)/SUM(book_value_usd)*100, 2) as avg_yield_pct
        FROM positions_daily
    """).df().to_string(index=False))


# ==========================================
# 5. CLI Entry Point
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize portfolio database from CSV")
    parser.add_argument("--csv", type=Path, help="Path to position CSV file")
    parser.add_argument("--date", type=str, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="DuckDB file path")
    args = parser.parse_args()

    # Auto-detect CSV if not specified
    if args.csv:
        csv_path = args.csv
        snapshot_date = args.date or csv_path.stem.replace("position", "")
    else:
        csvs = sorted(DEFAULT_RAW_DIR.glob("position*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No position*.csv found in {DEFAULT_RAW_DIR}")
        csv_path = csvs[-1]  # Latest file
        snapshot_date = args.date or csv_path.stem.replace("position", "")

    # Format date
    if len(snapshot_date) == 8:  # 20251231 -> 2025-12-31
        snapshot_date = f"{snapshot_date[:4]}-{snapshot_date[4:6]}-{snapshot_date[6:]}"

    print(f"[ETL] CSV: {csv_path}")
    print(f"[ETL] Date: {snapshot_date}")
    print(f"[ETL] DB: {args.db}")

    # Run pipeline
    df = extract(csv_path)
    df = transform(df, snapshot_date)

    with duckdb_connection(args.db) as conn:
        load(conn, df, snapshot_date)
        validate(conn)

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
