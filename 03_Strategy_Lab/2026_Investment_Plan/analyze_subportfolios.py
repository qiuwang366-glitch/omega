"""
analyze_subportfolios.py - 2026 Investment Plan Sub-Portfolio Analysis
Analyzes USD SSA and AUD Rates portfolios for reinvestment planning.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 0. Config
# ==========================================
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "../../01_Data_Warehouse/db/portfolio.duckdb"
OUTPUT_MD = BASE_DIR / "2026_Subportfolio_Profile.md"
OUTPUT_PNG = BASE_DIR / "maturity_schedule_2026.png"

SNAPSHOT_DATE = "2025-12-31"
MATURITY_START = "2026-01-01"
MATURITY_END = "2026-12-31"

PortfolioType = Literal["USD_SSA", "AUD_Rates"]


# ==========================================
# 1. Data Classes
# ==========================================
@dataclass
class PortfolioProfile:
    """Stock profile metrics for a sub-portfolio."""
    name: str
    aum_usd: float
    weighted_yield: float
    weighted_duration: float
    annual_carry_mm: float
    position_count: int


@dataclass
class MaturitySchedule:
    """Monthly maturity schedule."""
    portfolio: str
    df: pd.DataFrame  # month, nominal_usd, weighted_yield


# ==========================================
# 2. SQL Queries
# ==========================================
def get_portfolio_filter(portfolio: PortfolioType) -> str:
    """Return SQL WHERE clause for portfolio definition."""
    if portfolio == "USD_SSA":
        # USD Supranational/Agency: currency=USD AND sector_l2 in SSA-like categories
        return """
            p.currency = 'USD'
            AND s.sector_l2 IN ('Supra', 'DM Agency', 'CN Agency')
        """
    else:  # AUD_Rates
        # AUD Non-Credit: currency=AUD AND sector_l1 NOT IN (Fins, Corps)
        return """
            p.currency = 'AUD'
            AND s.sector_l1 NOT IN ('Fins', 'Corps')
        """


SQL_PROFILE = """
SELECT
    COUNT(*) as position_count,
    SUM(p.book_value_usd) as aum_usd,
    SUM(p.book_value_usd * COALESCE(p.yield_effective, 0)) / NULLIF(SUM(p.book_value_usd), 0) as weighted_yield,
    SUM(p.book_value_usd * COALESCE(p.duration, 0)) / NULLIF(SUM(p.book_value_usd), 0) as weighted_duration,
    SUM(p.carry_annual_usd) as annual_carry
FROM positions_daily p
JOIN security_master s ON p.isin = s.isin
WHERE p.snapshot_date = ?
  AND {filter}
"""

SQL_MATURITY_SCHEDULE = """
SELECT
    EXTRACT(MONTH FROM s.maturity_date) as month,
    SUM(p.nominal_usd) as nominal_usd,
    SUM(p.nominal_usd * COALESCE(p.yield_effective, 0)) / NULLIF(SUM(p.nominal_usd), 0) as weighted_yield
FROM positions_daily p
JOIN security_master s ON p.isin = s.isin
WHERE p.snapshot_date = ?
  AND s.maturity_date >= ?
  AND s.maturity_date <= ?
  AND {filter}
GROUP BY 1
ORDER BY 1
"""


# ==========================================
# 3. Analysis Functions
# ==========================================
def analyze_profile(conn: duckdb.DuckDBPyConnection, portfolio: PortfolioType) -> PortfolioProfile:
    """Calculate stock profile metrics for a sub-portfolio."""
    sql = SQL_PROFILE.format(filter=get_portfolio_filter(portfolio))
    result = conn.execute(sql, [SNAPSHOT_DATE]).fetchone()

    return PortfolioProfile(
        name="USD SSA" if portfolio == "USD_SSA" else "AUD Rates",
        position_count=result[0] or 0,
        aum_usd=result[1] or 0.0,
        weighted_yield=result[2] or 0.0,
        weighted_duration=result[3] or 0.0,
        annual_carry_mm=(result[4] or 0.0) / 1e6,
    )


def analyze_maturity_schedule(
    conn: duckdb.DuckDBPyConnection, portfolio: PortfolioType
) -> MaturitySchedule:
    """Get 2026 maturity schedule by month."""
    sql = SQL_MATURITY_SCHEDULE.format(filter=get_portfolio_filter(portfolio))
    df = conn.execute(sql, [SNAPSHOT_DATE, MATURITY_START, MATURITY_END]).df()

    # Ensure all 12 months are present
    all_months = pd.DataFrame({"month": range(1, 13)})
    df = all_months.merge(df, on="month", how="left").fillna(0)
    df["nominal_usd"] = df["nominal_usd"].astype(float)
    df["weighted_yield"] = df["weighted_yield"].astype(float)

    return MaturitySchedule(
        portfolio="USD SSA" if portfolio == "USD_SSA" else "AUD Rates",
        df=df,
    )


# ==========================================
# 4. Report Generation
# ==========================================
def generate_markdown_report(
    profiles: list[PortfolioProfile],
    schedules: list[MaturitySchedule],
) -> str:
    """Generate markdown report."""
    lines = [
        "# 2026 Sub-Portfolio Investment Profile",
        f"> Generated: {date.today()} | Snapshot: {SNAPSHOT_DATE}",
        "",
        "## 1. Stock Profile (End of 2025)",
        "",
        "| Metric | USD SSA | AUD Rates |",
        "|--------|--------:|----------:|",
    ]

    p_usd, p_aud = profiles[0], profiles[1]
    lines.extend([
        f"| **AUM (USD Bn)** | {p_usd.aum_usd/1e9:.2f} | {p_aud.aum_usd/1e9:.2f} |",
        f"| **Positions** | {p_usd.position_count} | {p_aud.position_count} |",
        f"| **Wtd Avg Yield (%)** | {p_usd.weighted_yield:.2f} | {p_aud.weighted_yield:.2f} |",
        f"| **Wtd Avg Duration** | {p_usd.weighted_duration:.2f} | {p_aud.weighted_duration:.2f} |",
        f"| **Annual Carry (USD MM)** | {p_usd.annual_carry_mm:.1f} | {p_aud.annual_carry_mm:.1f} |",
        "",
    ])

    # Maturity Wall Section
    lines.extend([
        "## 2. 2026 Maturity Wall (Reinvestment Risk)",
        "",
        "| Month | USD SSA Maturing (MM) | USD SSA Exit Yield | AUD Rates Maturing (MM) | AUD Rates Exit Yield |",
        "|------:|----------------------:|-------------------:|------------------------:|---------------------:|",
    ])

    s_usd, s_aud = schedules[0], schedules[1]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    total_usd, total_aud = 0.0, 0.0
    for i in range(12):
        usd_nom = s_usd.df.iloc[i]["nominal_usd"] / 1e6
        usd_yld = s_usd.df.iloc[i]["weighted_yield"]
        aud_nom = s_aud.df.iloc[i]["nominal_usd"] / 1e6
        aud_yld = s_aud.df.iloc[i]["weighted_yield"]
        total_usd += usd_nom
        total_aud += aud_nom

        usd_yld_str = f"{usd_yld:.2f}%" if usd_nom > 0 else "-"
        aud_yld_str = f"{aud_yld:.2f}%" if aud_nom > 0 else "-"

        lines.append(
            f"| {month_names[i]} | {usd_nom:.1f} | {usd_yld_str} | {aud_nom:.1f} | {aud_yld_str} |"
        )

    lines.extend([
        f"| **Total** | **{total_usd:.1f}** | - | **{total_aud:.1f}** | - |",
        "",
        "## 3. Key Observations",
        "",
        f"- **USD SSA**: {total_usd:.0f}MM maturing in 2026 ({total_usd/p_usd.aum_usd*1e6*100:.1f}% of AUM)",
        f"- **AUD Rates**: {total_aud:.0f}MM maturing in 2026 ({total_aud/p_aud.aum_usd*1e6*100:.1f}% of AUM)" if p_aud.aum_usd > 0 else "- **AUD Rates**: No maturities in 2026",
        "",
        "---",
        f"*Chart: [maturity_schedule_2026.png](./maturity_schedule_2026.png)*",
    ])

    return "\n".join(lines)


def generate_chart(schedules: list[MaturitySchedule]) -> None:
    """Generate dual-axis maturity schedule chart."""
    s_usd, s_aud = schedules[0], schedules[1]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(len(months))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart: Maturity Volume
    usd_bars = s_usd.df["nominal_usd"].values / 1e6
    aud_bars = s_aud.df["nominal_usd"].values / 1e6

    bars1 = ax1.bar(x - width/2, usd_bars, width, label="USD SSA", color="#2E86AB", alpha=0.8)
    bars2 = ax1.bar(x + width/2, aud_bars, width, label="AUD Rates", color="#A23B72", alpha=0.8)

    ax1.set_xlabel("Month", fontsize=11)
    ax1.set_ylabel("Maturing Nominal (USD MM)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.legend(loc="upper left")
    ax1.set_ylim(bottom=0)

    # Line chart: Exit Yield (secondary axis)
    ax2 = ax1.twinx()

    # Only plot yield where there's volume
    usd_yields = s_usd.df["weighted_yield"].values
    aud_yields = s_aud.df["weighted_yield"].values

    # Mask zero-volume months for cleaner lines
    usd_yields_masked = np.where(usd_bars > 0, usd_yields, np.nan)
    aud_yields_masked = np.where(aud_bars > 0, aud_yields, np.nan)

    ax2.plot(x, usd_yields_masked, "o--", color="#1B4F72", label="USD SSA Yield", linewidth=2, markersize=6)
    ax2.plot(x, aud_yields_masked, "s--", color="#6C3483", label="AUD Rates Yield", linewidth=2, markersize=6)

    ax2.set_ylabel("Exit Yield (%)", fontsize=11)
    ax2.legend(loc="upper right")

    # Title and styling
    plt.title("2026 Maturity Schedule: Reinvestment Risk Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()

    # Save
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] Saved: {OUTPUT_PNG}")


# ==========================================
# 5. Main
# ==========================================
def main() -> None:
    print(f"[Analysis] Connecting to {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        # Analyze profiles
        print("[Analysis] Computing portfolio profiles...")
        profiles = [
            analyze_profile(conn, "USD_SSA"),
            analyze_profile(conn, "AUD_Rates"),
        ]

        for p in profiles:
            print(f"  - {p.name}: {p.position_count} positions, ${p.aum_usd/1e9:.2f}Bn AUM")

        # Analyze maturity schedules
        print("[Analysis] Computing 2026 maturity schedules...")
        schedules = [
            analyze_maturity_schedule(conn, "USD_SSA"),
            analyze_maturity_schedule(conn, "AUD_Rates"),
        ]

        for s in schedules:
            total = s.df["nominal_usd"].sum() / 1e6
            print(f"  - {s.portfolio}: ${total:.1f}MM maturing in 2026")

        # Generate outputs
        print("[Analysis] Generating markdown report...")
        report = generate_markdown_report(profiles, schedules)
        OUTPUT_MD.write_text(report, encoding="utf-8")
        print(f"[Report] Saved: {OUTPUT_MD}")

        print("[Analysis] Generating chart...")
        generate_chart(schedules)

    finally:
        conn.close()

    print("\n[OK] Analysis complete.")


if __name__ == "__main__":
    main()
