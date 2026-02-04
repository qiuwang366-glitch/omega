"""
position_loader.py - Position Data Loader & Maturity Analysis
================================================================
Loads bond-level position data and provides maturity analysis functions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, datetime
from typing import Any

import pandas as pd
import numpy as np

# Add data warehouse to path
_project_root = Path(__file__).parent.parent.parent
_data_warehouse = _project_root / "01_Data_Warehouse"


# ============================================================================
# 1. Data Loading
# ============================================================================
def load_position_data(as_of_date: str = "20251231") -> pd.DataFrame:
    """
    Load position data from CSV.

    Args:
        as_of_date: Date string for position file (default: 20251231)

    Returns:
        DataFrame with cleaned position data
    """
    file_path = _data_warehouse / "raw_landing" / f"position{as_of_date}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Position file not found: {file_path}")

    # Load with proper encoding
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse maturity date
    df["MaturityDate"] = pd.to_datetime(df["Maturity"], format="%d/%m/%Y", errors="coerce")
    df["MaturityYear"] = df["MaturityDate"].dt.year
    df["MaturityMonth"] = df["MaturityDate"].dt.month

    # Clean numeric columns
    numeric_cols = [
        "Nominal（USD）", "Nominal(亿美元）", "摊余成本（USD）",
        "Duration", "EffectiveYield", "WAL"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # Standardize column names for easier access
    df = df.rename(columns={
        "Nominal（USD）": "NominalUSD",
        "Nominal(亿美元）": "NominalYi",
        "摊余成本（USD）": "AmortizedCostUSD",
        "摊余成本（亿USD）": "AmortizedCostYi",
        "EffectiveYield": "Yield",
        "Duration": "Duration",
        "债券名称": "BondName",
        "分类1": "Category1",
        "分类2": "Category2",
    })

    return df


def get_position_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Get high-level position summary statistics."""
    return {
        "total_positions": len(df),
        "total_nominal_usd": df["NominalUSD"].sum(),
        "total_nominal_yi": df["NominalYi"].sum() if "NominalYi" in df.columns else df["NominalUSD"].sum() / 100_000_000,
        "unique_currencies": df["CCY"].nunique(),
        "currencies": df["CCY"].unique().tolist(),
        "avg_duration": df["Duration"].mean(),
        "avg_yield": df["Yield"].mean() if "Yield" in df.columns else None,
    }


# ============================================================================
# 2. Maturity Analysis
# ============================================================================
def get_maturity_by_year(
    df: pd.DataFrame,
    start_year: int = 2025,
    end_year: int = 2035,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate maturities by year.

    Args:
        df: Position DataFrame
        start_year: Start year for analysis
        end_year: End year (inclusive)
        group_by: Optional column to group by (e.g., "CCY", "Category1")

    Returns:
        DataFrame with maturity amounts by year
    """
    # Filter to valid maturity dates
    valid_df = df[df["MaturityYear"].notna() & (df["MaturityYear"] >= start_year) & (df["MaturityYear"] <= end_year)].copy()

    if group_by and group_by in valid_df.columns:
        result = valid_df.groupby(["MaturityYear", group_by]).agg({
            "NominalUSD": "sum",
            "ISIN": "count",
            "Duration": "mean",
        }).reset_index()
        result = result.rename(columns={"ISIN": "PositionCount"})

        # Pivot for easier visualization
        pivot = result.pivot_table(
            index="MaturityYear",
            columns=group_by,
            values="NominalUSD",
            aggfunc="sum",
            fill_value=0
        )
        return pivot
    else:
        result = valid_df.groupby("MaturityYear").agg({
            "NominalUSD": "sum",
            "ISIN": "count",
            "Duration": "mean",
            "Yield": "mean" if "Yield" in valid_df.columns else "count",
        }).reset_index()
        result = result.rename(columns={"ISIN": "PositionCount"})
        return result


def get_maturity_schedule_monthly(
    df: pd.DataFrame,
    year: int = 2026,
) -> pd.DataFrame:
    """
    Get monthly maturity schedule for a specific year.

    Args:
        df: Position DataFrame
        year: Target year

    Returns:
        DataFrame with monthly maturity breakdown
    """
    year_df = df[df["MaturityYear"] == year].copy()

    if year_df.empty:
        return pd.DataFrame()

    monthly = year_df.groupby("MaturityMonth").agg({
        "NominalUSD": "sum",
        "ISIN": "count",
        "Duration": "mean",
        "Yield": "mean" if "Yield" in year_df.columns else "count",
        "BondName": lambda x: list(x),
    }).reset_index()

    monthly = monthly.rename(columns={
        "ISIN": "PositionCount",
        "BondName": "Bonds"
    })

    # Add month names
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    monthly["MonthName"] = monthly["MaturityMonth"].map(month_names)

    return monthly


def get_bonds_maturing_in_year(
    df: pd.DataFrame,
    year: int = 2026,
    currency: str | None = None,
) -> pd.DataFrame:
    """
    Get detailed list of bonds maturing in a specific year.

    Args:
        df: Position DataFrame
        year: Target maturity year
        currency: Optional currency filter

    Returns:
        DataFrame with bond details
    """
    mask = df["MaturityYear"] == year
    if currency:
        mask = mask & (df["CCY"] == currency)

    result = df[mask].copy()

    if result.empty:
        return pd.DataFrame()

    # Select and order columns
    cols_to_show = [
        "ISIN", "BondName", "CCY", "NominalUSD", "NominalYi",
        "Duration", "Yield", "MaturityDate", "Category1", "Category2"
    ]
    cols_available = [c for c in cols_to_show if c in result.columns]

    result = result[cols_available].sort_values("MaturityDate")

    # Format maturity date
    result["MaturityDate"] = result["MaturityDate"].dt.strftime("%Y-%m-%d")

    # Convert to 亿美元
    if "NominalUSD" in result.columns:
        result["NominalYi"] = result["NominalUSD"] / 100_000_000

    return result


def get_maturity_concentration(
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identify maturity concentration risk - largest single maturity dates.

    Args:
        df: Position DataFrame
        top_n: Number of top concentration points

    Returns:
        DataFrame with concentration analysis
    """
    valid_df = df[df["MaturityDate"].notna()].copy()

    # Group by exact maturity date
    concentration = valid_df.groupby("MaturityDate").agg({
        "NominalUSD": "sum",
        "ISIN": "count",
        "BondName": lambda x: ", ".join(x.head(3)) + ("..." if len(x) > 3 else ""),
    }).reset_index()

    concentration = concentration.rename(columns={
        "ISIN": "PositionCount",
        "BondName": "SampleBonds"
    })

    concentration = concentration.sort_values("NominalUSD", ascending=False).head(top_n)
    concentration["MaturityDate"] = concentration["MaturityDate"].dt.strftime("%Y-%m-%d")
    concentration["NominalYi"] = concentration["NominalUSD"] / 100_000_000

    return concentration


# ============================================================================
# 3. Duration & Risk Metrics
# ============================================================================
def get_duration_profile_by_maturity(
    df: pd.DataFrame,
    buckets: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Analyze duration profile by maturity bucket.

    Args:
        df: Position DataFrame
        buckets: List of (start_year, end_year) tuples for bucketing

    Returns:
        DataFrame with duration profile by bucket
    """
    if buckets is None:
        current_year = datetime.now().year
        buckets = [
            (current_year, current_year),
            (current_year + 1, current_year + 1),
            (current_year + 2, current_year + 3),
            (current_year + 4, current_year + 6),
            (current_year + 7, current_year + 10),
            (current_year + 11, 2099),
        ]

    valid_df = df[df["MaturityYear"].notna()].copy()

    records = []
    for start, end in buckets:
        mask = (valid_df["MaturityYear"] >= start) & (valid_df["MaturityYear"] <= end)
        bucket_df = valid_df[mask]

        if bucket_df.empty:
            continue

        total_nominal = bucket_df["NominalUSD"].sum()
        weighted_dur = (bucket_df["NominalUSD"] * bucket_df["Duration"]).sum() / total_nominal if total_nominal > 0 else 0
        weighted_yield = (bucket_df["NominalUSD"] * bucket_df["Yield"]).sum() / total_nominal if total_nominal > 0 and "Yield" in bucket_df.columns else 0

        label = f"{start}" if start == end else f"{start}-{min(end, 2050)}"
        if end > 2050:
            label = f"{start}+"

        records.append({
            "Bucket": label,
            "StartYear": start,
            "EndYear": end,
            "NominalUSD": total_nominal,
            "NominalYi": total_nominal / 100_000_000,
            "PositionCount": len(bucket_df),
            "AvgDuration": weighted_dur,
            "AvgYield": weighted_yield,
            "WeightPct": 0,  # Will be filled after
        })

    result = pd.DataFrame(records)

    if not result.empty:
        total = result["NominalUSD"].sum()
        result["WeightPct"] = result["NominalUSD"] / total * 100 if total > 0 else 0

    return result


def calculate_key_rate_durations(
    df: pd.DataFrame,
    key_rates: list[float] | None = None,
) -> dict[str, float]:
    """
    Calculate approximate key rate durations.

    Args:
        df: Position DataFrame
        key_rates: Key rate tenors in years

    Returns:
        Dict mapping tenor -> KRD contribution
    """
    if key_rates is None:
        key_rates = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

    valid_df = df[df["Duration"].notna() & df["NominalUSD"].notna()].copy()

    if valid_df.empty:
        return {f"{kr}Y": 0.0 for kr in key_rates}

    total_nominal = valid_df["NominalUSD"].sum()

    # Approximate KRD by bucketing positions near key rates
    krds = {}
    for kr in key_rates:
        # Simple bucketing: positions within ±1 year of key rate
        lower = kr - 1 if kr > 1 else 0
        upper = kr + 1

        mask = (valid_df["Duration"] >= lower) & (valid_df["Duration"] < upper)
        bucket = valid_df[mask]

        if not bucket.empty:
            contribution = (bucket["NominalUSD"] * bucket["Duration"]).sum() / total_nominal
        else:
            contribution = 0.0

        krds[f"{kr}Y"] = contribution

    return krds


# ============================================================================
# 4. Currency Analysis
# ============================================================================
def get_currency_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Get position breakdown by currency."""
    valid_df = df[df["NominalUSD"].notna()].copy()

    result = valid_df.groupby("CCY").agg({
        "NominalUSD": "sum",
        "ISIN": "count",
        "Duration": "mean",
        "Yield": "mean" if "Yield" in valid_df.columns else "count",
    }).reset_index()

    result = result.rename(columns={"ISIN": "PositionCount"})
    result["NominalYi"] = result["NominalUSD"] / 100_000_000

    total = result["NominalUSD"].sum()
    result["WeightPct"] = result["NominalUSD"] / total * 100 if total > 0 else 0

    return result.sort_values("NominalUSD", ascending=False)


def get_maturity_by_currency_year(
    df: pd.DataFrame,
    start_year: int = 2025,
    end_year: int = 2035,
) -> pd.DataFrame:
    """
    Get maturity schedule by currency and year.

    Returns pivot table: rows=Year, columns=Currency, values=NominalYi
    """
    valid_df = df[
        df["MaturityYear"].notna() &
        (df["MaturityYear"] >= start_year) &
        (df["MaturityYear"] <= end_year)
    ].copy()

    pivot = valid_df.pivot_table(
        index="MaturityYear",
        columns="CCY",
        values="NominalUSD",
        aggfunc="sum",
        fill_value=0
    )

    # Convert to 亿美元
    pivot = pivot / 100_000_000

    return pivot


# ============================================================================
# 5. Sector/Category Analysis
# ============================================================================
def get_category_breakdown(
    df: pd.DataFrame,
    category_col: str = "Category1",
) -> pd.DataFrame:
    """Get position breakdown by category."""
    if category_col not in df.columns:
        return pd.DataFrame()

    valid_df = df[df["NominalUSD"].notna()].copy()

    result = valid_df.groupby(category_col).agg({
        "NominalUSD": "sum",
        "ISIN": "count",
        "Duration": "mean",
    }).reset_index()

    result = result.rename(columns={"ISIN": "PositionCount"})
    result["NominalYi"] = result["NominalUSD"] / 100_000_000

    total = result["NominalUSD"].sum()
    result["WeightPct"] = result["NominalUSD"] / total * 100 if total > 0 else 0

    return result.sort_values("NominalUSD", ascending=False)


# ============================================================================
# 6. Utility Functions
# ============================================================================
def format_number_yi(value: float, decimals: int = 2) -> str:
    """Format number in 亿美元."""
    return f"{value:.{decimals}f}"


def format_number_mm(value: float, decimals: int = 1) -> str:
    """Format number in MM USD."""
    return f"{value/1e6:.{decimals}f}MM"


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    try:
        df = load_position_data()
        print(f"Loaded {len(df)} positions")

        summary = get_position_summary(df)
        print(f"\nSummary:")
        print(f"  Total Nominal: {summary['total_nominal_yi']:.2f}亿美元")
        print(f"  Currencies: {summary['currencies']}")
        print(f"  Avg Duration: {summary['avg_duration']:.2f}Y")

        print("\n2026 Maturing Bonds:")
        bonds_2026 = get_bonds_maturing_in_year(df, 2026)
        print(bonds_2026.head(10))

        print("\nMaturity by Year:")
        yearly = get_maturity_by_year(df, 2025, 2035)
        print(yearly)

    except Exception as e:
        print(f"Error: {e}")
