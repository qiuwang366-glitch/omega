#!/usr/bin/env python3
"""
Credit Bond Risk Intelligence Platform - Standalone Dashboard

This is a self-contained Streamlit application that can be run directly.

Usage:
    cd 03_Strategy_Lab/credit_bond_risk
    streamlit run app.py
"""

import streamlit as st
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Any
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pydantic import BaseModel, Field, computed_field

# Ensure data module is importable
sys.path.insert(0, str(Path(__file__).parent))

# Import from data layer
from data.mock_data import (
    MockDataProvider,
    generate_mock_obligors,
    generate_mock_exposures,
    generate_mock_alerts,
    generate_mock_news,
)
from data.provider import DataProviderConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Config (must be first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title="Credit Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Enums
# =============================================================================


class Sector(str, Enum):
    # China Onshore/Offshore
    LGFV = "LGFV"
    SOE = "SOE"
    FINANCIAL = "FINANCIAL"
    CORP = "CORP"
    SOVEREIGN = "SOVEREIGN"
    # International
    DM_SOVEREIGN = "DM_SOVEREIGN"
    EM_SOVEREIGN = "EM_SOVEREIGN"
    SUPRA = "SUPRA"
    US_CORP = "US_CORP"
    EU_CORP = "EU_CORP"
    G_SIB = "G-SIB"
    EM_FIN = "EM_FIN"
    HY = "HY"


class Region(str, Enum):
    CHINA_ONSHORE = "CHINA_ONSHORE"
    CHINA_OFFSHORE = "CHINA_OFFSHORE"
    US = "US"
    EU = "EU"
    UK = "UK"
    JAPAN = "JAPAN"
    LATAM = "LATAM"
    CEEMEA = "CEEMEA"
    ASIA_EX_CHINA = "ASIA_EX_CHINA"
    SUPRANATIONAL = "SUPRANATIONAL"


class CreditRating(str, Enum):
    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB = "BB"
    B = "B"
    NR = "NR"


class RatingOutlook(str, Enum):
    POSITIVE = "POSITIVE"
    STABLE = "STABLE"
    NEGATIVE = "NEGATIVE"
    WATCH_NEG = "WATCH_NEG"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    CONCENTRATION = "CONCENTRATION"
    RATING = "RATING"
    SPREAD = "SPREAD"
    NEWS = "NEWS"


class AlertStatus(str, Enum):
    PENDING = "PENDING"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


RATING_SCORE = {
    CreditRating.AAA: 100,
    CreditRating.AA_PLUS: 95,
    CreditRating.AA: 90,
    CreditRating.AA_MINUS: 85,
    CreditRating.A_PLUS: 80,
    CreditRating.A: 75,
    CreditRating.A_MINUS: 70,
    CreditRating.BBB_PLUS: 65,
    CreditRating.BBB: 60,
    CreditRating.BBB_MINUS: 55,
    CreditRating.BB: 45,
    CreditRating.B: 30,
    CreditRating.NR: 50,
}


# =============================================================================
# Color Scheme
# =============================================================================


@dataclass
class ColorScheme:
    """Premium dark theme color scheme"""
    bg_primary: str = "#0d1117"
    bg_secondary: str = "#161b22"
    bg_tertiary: str = "#21262d"
    text_primary: str = "#f0f6fc"
    text_secondary: str = "#8b949e"
    text_muted: str = "#6e7681"
    accent_blue: str = "#58a6ff"
    accent_green: str = "#3fb950"
    accent_yellow: str = "#d29922"
    accent_orange: str = "#db6d28"
    accent_red: str = "#f85149"
    accent_purple: str = "#a371f7"
    severity_critical: str = "#f85149"
    severity_warning: str = "#d29922"
    severity_info: str = "#58a6ff"
    severity_success: str = "#3fb950"

    @classmethod
    def get_severity_color(cls, severity: str) -> str:
        scheme = cls()
        mapping = {
            "CRITICAL": scheme.severity_critical,
            "WARNING": scheme.severity_warning,
            "INFO": scheme.severity_info,
        }
        return mapping.get(severity.upper(), scheme.text_secondary)

    @classmethod
    def get_rating_color(cls, rating: str) -> str:
        scheme = cls()
        if "AAA" in rating.upper():
            return "#238636"
        elif "AA" in rating.upper():
            return "#3fb950"
        elif rating.upper().startswith("A"):
            return "#7ee787"
        elif "BBB" in rating.upper():
            return "#d29922"
        elif "BB" in rating.upper():
            return "#db6d28"
        else:
            return "#f85149"

    @classmethod
    def get_sector_color(cls, sector: str) -> str:
        scheme = cls()
        mapping = {
            # China
            "LGFV": scheme.accent_blue,
            "SOE": scheme.accent_purple,
            "FINANCIAL": scheme.accent_green,
            "CORP": scheme.accent_yellow,
            "SOVEREIGN": "#238636",
            # International
            "DM_SOVEREIGN": "#238636",
            "EM_SOVEREIGN": "#7ee787",
            "SUPRA": "#a371f7",
            "US_CORP": "#db6d28",
            "EU_CORP": "#d29922",
            "G-SIB": "#58a6ff",
            "EM_FIN": "#3fb950",
            "HY": "#f85149",
        }
        return mapping.get(sector.upper(), scheme.text_secondary)

    @classmethod
    def get_region_color(cls, region: str) -> str:
        scheme = cls()
        mapping = {
            "CHINA_ONSHORE": "#f85149",
            "CHINA_OFFSHORE": "#db6d28",
            "US": "#58a6ff",
            "EU": "#a371f7",
            "UK": "#3fb950",
            "JAPAN": "#d29922",
            "LATAM": "#7ee787",
            "CEEMEA": "#f0883e",
            "ASIA_EX_CHINA": "#8957e5",
            "SUPRANATIONAL": "#238636",
        }
        return mapping.get(region.upper(), scheme.text_secondary)


def get_premium_layout(title: str = "", height: int = 400) -> dict:
    scheme = ColorScheme()
    return {
        "title": {"text": title, "font": {"size": 16, "color": scheme.text_primary}, "x": 0.02},
        "paper_bgcolor": scheme.bg_primary,
        "plot_bgcolor": scheme.bg_secondary,
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "font": {"family": "Inter, sans-serif", "color": scheme.text_secondary},
        "xaxis": {"gridcolor": scheme.bg_tertiary, "linecolor": scheme.bg_tertiary},
        "yaxis": {"gridcolor": scheme.bg_tertiary, "linecolor": scheme.bg_tertiary},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": scheme.text_secondary}},
    }


# =============================================================================
# Data Models
# =============================================================================


class Obligor(BaseModel):
    obligor_id: str
    name_cn: str
    name_en: str | None = None
    sector: Sector
    sub_sector: str
    region: Region = Region.CHINA_OFFSHORE
    country: str | None = None
    province: str | None = None
    rating_internal: CreditRating
    rating_outlook: RatingOutlook = RatingOutlook.STABLE
    # Fundamentals (for international issuers)
    ticker: str | None = None
    lei: str | None = None  # Legal Entity Identifier
    parent_entity: str | None = None

    @computed_field
    @property
    def rating_score(self) -> int:
        return RATING_SCORE.get(self.rating_internal, 50)

    @computed_field
    @property
    def display_name(self) -> str:
        """Display name preferring English for international issuers"""
        if self.region not in (Region.CHINA_ONSHORE, Region.CHINA_OFFSHORE) and self.name_en:
            return self.name_en
        return self.name_cn


class BondPosition(BaseModel):
    isin: str
    obligor_id: str
    bond_name: str | None = None
    currency: str = "USD"
    maturity_date: date
    coupon: float
    nominal: float
    nominal_usd: float
    book_value_usd: float
    market_value_usd: float
    duration: float
    oas: float | None = None

    @computed_field
    @property
    def years_to_maturity(self) -> float:
        days = (self.maturity_date - date.today()).days
        return max(0, days / 365.25)

    @computed_field
    @property
    def credit_dv01(self) -> float:
        return self.market_value_usd * self.duration * 0.0001


class CreditExposure(BaseModel):
    obligor: Obligor
    bonds: list[BondPosition] = Field(default_factory=list)
    total_nominal_usd: float = 0
    total_market_usd: float = 0
    pct_of_aum: float = 0
    weighted_avg_duration: float = 0
    weighted_avg_oas: float = 0
    credit_dv01_usd: float = 0
    maturity_profile: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_positions(cls, obligor: Obligor, positions: list[BondPosition], total_aum: float) -> "CreditExposure":
        if not positions:
            return cls(obligor=obligor)

        total_nominal = sum(p.nominal_usd for p in positions)
        total_market = sum(p.market_value_usd for p in positions)
        total_dv01 = sum(p.credit_dv01 for p in positions)

        if total_market > 0:
            weighted_duration = sum(p.market_value_usd * p.duration for p in positions) / total_market
            oas_positions = [p for p in positions if p.oas is not None]
            weighted_oas = sum(p.market_value_usd * p.oas for p in oas_positions) / sum(p.market_value_usd for p in oas_positions) if oas_positions else 0
        else:
            weighted_duration = 0
            weighted_oas = 0

        maturity_buckets = {"0-1Y": 0, "1-3Y": 0, "3-5Y": 0, "5-10Y": 0, "10Y+": 0}
        for p in positions:
            ytm = p.years_to_maturity
            if ytm <= 1:
                maturity_buckets["0-1Y"] += p.nominal_usd
            elif ytm <= 3:
                maturity_buckets["1-3Y"] += p.nominal_usd
            elif ytm <= 5:
                maturity_buckets["3-5Y"] += p.nominal_usd
            elif ytm <= 10:
                maturity_buckets["5-10Y"] += p.nominal_usd
            else:
                maturity_buckets["10Y+"] += p.nominal_usd

        return cls(
            obligor=obligor,
            bonds=positions,
            total_nominal_usd=total_nominal,
            total_market_usd=total_market,
            pct_of_aum=total_market / total_aum if total_aum > 0 else 0,
            weighted_avg_duration=weighted_duration,
            weighted_avg_oas=weighted_oas,
            credit_dv01_usd=total_dv01,
            maturity_profile=maturity_buckets,
        )


class RiskAlert(BaseModel):
    alert_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: Severity
    category: AlertCategory
    obligor_id: str
    obligor_name: str
    signal_name: str
    message: str
    metric_value: float
    threshold: float
    status: AlertStatus = AlertStatus.PENDING
    ai_summary: str | None = None


class NewsItem(BaseModel):
    news_id: str
    timestamp: datetime
    source: str
    title: str
    content: str
    obligor_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    sentiment: Sentiment | None = None
    sentiment_score: float | None = None


# =============================================================================
# Chart Components
# =============================================================================


def create_concentration_chart(exposures: list[CreditExposure], top_n: int = 15) -> go.Figure:
    sorted_exposures = sorted(exposures, key=lambda x: x.total_market_usd, reverse=True)[:top_n]
    names = [e.obligor.display_name for e in sorted_exposures]
    values = [e.total_market_usd / 1e6 for e in sorted_exposures]
    pcts = [e.pct_of_aum for e in sorted_exposures]
    sectors = [e.obligor.sector.value for e in sorted_exposures]
    colors = [ColorScheme.get_sector_color(s) for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"${v:.0f}M ({p:.1%})" for v, p in zip(values, pcts)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>MV: $%{x:.0f}M<extra></extra>",
    ))
    fig.update_layout(**get_premium_layout("Top Issuers by Market Value", height=max(400, top_n * 30)))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def create_region_distribution_chart(exposures: list[CreditExposure]) -> go.Figure:
    """Create region distribution pie chart"""
    region_totals: dict[str, float] = {}
    for exp in exposures:
        region = exp.obligor.region.value
        region_totals[region] = region_totals.get(region, 0) + exp.total_market_usd

    labels = list(region_totals.keys())
    values = [v / 1e6 for v in region_totals.values()]
    colors = [ColorScheme.get_region_color(r) for r in labels]

    # Create display labels
    label_map = {
        "CHINA_OFFSHORE": "China Offshore",
        "CHINA_ONSHORE": "China Onshore",
        "US": "United States",
        "EU": "Europe",
        "UK": "United Kingdom",
        "JAPAN": "Japan",
        "LATAM": "Latin America",
        "CEEMEA": "CEEMEA",
        "ASIA_EX_CHINA": "Asia ex-China",
        "SUPRANATIONAL": "Supranational",
    }
    display_labels = [label_map.get(l, l) for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=display_labels, values=values, hole=0.5, marker_colors=colors,
        textinfo="label+percent", textposition="outside",
    )])
    fig.update_layout(**get_premium_layout("Regional Distribution", height=400))
    fig.update_layout(showlegend=False)
    return fig


def create_currency_breakdown_chart(exposures: list[CreditExposure]) -> go.Figure:
    """Create currency breakdown chart"""
    scheme = ColorScheme()
    currency_totals: dict[str, float] = {}
    for exp in exposures:
        for bond in exp.bonds:
            currency_totals[bond.currency] = currency_totals.get(bond.currency, 0) + bond.market_value_usd

    labels = list(currency_totals.keys())
    values = [v / 1e6 for v in currency_totals.values()]

    ccy_colors = {
        "USD": scheme.accent_blue,
        "EUR": scheme.accent_purple,
        "GBP": scheme.accent_green,
        "JPY": scheme.accent_yellow,
        "CNH": scheme.accent_orange,
        "CNY": scheme.accent_red,
    }
    colors = [ccy_colors.get(l, scheme.text_secondary) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"${v:.0f}M" for v in values], textposition="outside",
    )])
    fig.update_layout(**get_premium_layout("Currency Breakdown", height=350))
    return fig


def create_dv01_decomposition_chart(exposures: list[CreditExposure]) -> go.Figure:
    """Create DV01 decomposition by sector"""
    scheme = ColorScheme()
    sector_dv01: dict[str, float] = {}
    for exp in exposures:
        sector = exp.obligor.sector.value
        sector_dv01[sector] = sector_dv01.get(sector, 0) + exp.credit_dv01_usd

    # Sort by DV01 descending
    sorted_items = sorted(sector_dv01.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] / 1e6 for item in sorted_items]  # Convert to $M
    colors = [ColorScheme.get_sector_color(l) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"${v:.2f}M" for v in values], textposition="outside",
    )])
    fig.update_layout(**get_premium_layout("Credit DV01 by Sector ($M/bp)", height=400))
    fig.update_yaxes(autorange="reversed")
    return fig


def create_rating_distribution_chart(exposures: list[CreditExposure]) -> go.Figure:
    rating_totals: dict[str, float] = {}
    for exp in exposures:
        rating = exp.obligor.rating_internal.value
        rating_totals[rating] = rating_totals.get(rating, 0) + exp.total_market_usd

    labels = list(rating_totals.keys())
    values = [v / 1e6 for v in rating_totals.values()]
    colors = [ColorScheme.get_rating_color(r) for r in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.5, marker_colors=colors,
        textinfo="label+percent", textposition="outside",
    )])
    fig.update_layout(**get_premium_layout("评级分布", height=400))
    fig.update_layout(showlegend=False)
    return fig


def create_maturity_profile_chart(exposures: list[CreditExposure]) -> go.Figure:
    scheme = ColorScheme()
    buckets = ["0-1Y", "1-3Y", "3-5Y", "5-10Y", "10Y+"]
    bucket_totals = {b: 0 for b in buckets}
    for exp in exposures:
        for bucket, value in exp.maturity_profile.items():
            if bucket in buckets:
                bucket_totals[bucket] += value
    values = [bucket_totals[b] / 1e6 for b in buckets]

    fig = go.Figure(data=[go.Bar(
        x=buckets, y=values,
        marker_color=[scheme.accent_green, scheme.accent_blue, scheme.accent_purple, scheme.accent_orange, scheme.accent_red],
        text=[f"${v:.0f}M" for v in values], textposition="outside",
    )])
    fig.update_layout(**get_premium_layout("到期分布", height=350))
    return fig


def create_sector_concentration_chart(exposures: list[CreditExposure]) -> go.Figure:
    sector_totals: dict[str, float] = {}
    for exp in exposures:
        sector = exp.obligor.sector.value
        sector_totals[sector] = sector_totals.get(sector, 0) + exp.total_market_usd

    labels = list(sector_totals.keys())
    values = [v / 1e6 for v in sector_totals.values()]
    colors = [ColorScheme.get_sector_color(s) for s in labels]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6, marker_colors=colors)])
    fig.update_layout(**get_premium_layout("行业分布", height=350))
    fig.update_layout(showlegend=True)
    return fig


def create_risk_heatmap(exposures: list[CreditExposure]) -> go.Figure:
    scheme = ColorScheme()
    rating_buckets = ["AAA/AA", "A", "BBB", "BB/B"]
    duration_buckets = ["0-2Y", "2-5Y", "5-10Y", "10Y+"]
    matrix = np.zeros((len(rating_buckets), len(duration_buckets)))

    for exp in exposures:
        rating = exp.obligor.rating_internal.value
        if "AAA" in rating or "AA" in rating:
            r_idx = 0
        elif rating.startswith("A"):
            r_idx = 1
        elif "BBB" in rating:
            r_idx = 2
        else:
            r_idx = 3

        dur = exp.weighted_avg_duration
        if dur <= 2:
            d_idx = 0
        elif dur <= 5:
            d_idx = 1
        elif dur <= 10:
            d_idx = 2
        else:
            d_idx = 3
        matrix[r_idx, d_idx] += exp.total_market_usd / 1e6

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=duration_buckets, y=rating_buckets,
        colorscale=[[0, scheme.bg_secondary], [0.5, scheme.accent_blue], [1, scheme.accent_orange]],
        text=[[f"${v:.0f}M" for v in row] for row in matrix],
        texttemplate="%{text}", textfont={"size": 12},
    ))
    fig.update_layout(**get_premium_layout("风险矩阵 (评级 × 久期)", height=350))
    fig.update_yaxes(autorange="reversed")
    return fig


# =============================================================================
# Mock Data Generation (Using Data Layer)
# =============================================================================


def generate_mock_data() -> tuple[dict[str, Obligor], list[CreditExposure], list[RiskAlert], list[NewsItem]]:
    """
    Generate comprehensive mock data using the data layer.

    This function now delegates to the data layer modules for
    better code organization and reusability.
    """
    # Use the centralized mock data generators from data layer
    obligors = generate_mock_obligors()
    exposures = generate_mock_exposures(obligors)
    alerts = generate_mock_alerts()
    news_items = generate_mock_news()

    return obligors, exposures, alerts, news_items


# =============================================================================
# Alert Table Component
# =============================================================================


def render_alert_table(alerts: list[RiskAlert], show_filters: bool = True) -> list[RiskAlert]:
    if not alerts:
        st.info("暂无预警")
        return []

    filtered_alerts = alerts

    if show_filters:
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.multiselect("严重程度", options=["CRITICAL", "WARNING", "INFO"], default=["CRITICAL", "WARNING"])
            if severity_filter:
                filtered_alerts = [a for a in filtered_alerts if a.severity.value in severity_filter]
        with col2:
            category_filter = st.multiselect("类别", options=list(set(a.category.value for a in alerts)), default=None)
            if category_filter:
                filtered_alerts = [a for a in filtered_alerts if a.category.value in category_filter]
        with col3:
            status_filter = st.multiselect("状态", options=["PENDING", "INVESTIGATING", "RESOLVED", "DISMISSED"], default=["PENDING", "INVESTIGATING"])
            if status_filter:
                filtered_alerts = [a for a in filtered_alerts if a.status.value in status_filter]

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    filtered_alerts.sort(key=lambda a: (severity_order.get(a.severity.value, 3), -a.timestamp.timestamp()))

    table_data = []
    for alert in filtered_alerts[:20]:
        severity_icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(alert.severity.value, "⚪")
        table_data.append({
            "": severity_icon,
            "时间": alert.timestamp.strftime("%m-%d %H:%M"),
            "发行人": alert.obligor_name,
            "类别": alert.category.value,
            "消息": alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
            "指标": f"{alert.metric_value:.2f}",
            "阈值": f"{alert.threshold:.2f}",
            "状态": alert.status.value,
        })

    df = pd.DataFrame(table_data)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        f"显示 {min(20, len(filtered_alerts))}/{len(filtered_alerts)} 条预警 | "
        f"🔴 {sum(1 for a in filtered_alerts if a.severity == Severity.CRITICAL)} 严重 | "
        f"🟡 {sum(1 for a in filtered_alerts if a.severity == Severity.WARNING)} 警告"
    )
    return filtered_alerts


# =============================================================================
# Session State
# =============================================================================


def init_session_state():
    if "mock_data" not in st.session_state:
        obligors, exposures, alerts, news = generate_mock_data()
        st.session_state.obligors = obligors
        st.session_state.exposures = exposures
        st.session_state.alerts = alerts
        st.session_state.news = news
        st.session_state.mock_data = True
    if "active_page" not in st.session_state:
        st.session_state.active_page = "overview"


# =============================================================================
# Page Renderers
# =============================================================================


def render_overview_page():
    exposures = st.session_state.exposures
    alerts = st.session_state.alerts

    total_market = sum(e.total_market_usd for e in exposures)
    total_obligors = len(exposures)
    total_dv01 = sum(e.credit_dv01_usd for e in exposures)
    active_alerts = len([a for a in alerts if a.status == AlertStatus.PENDING])
    critical_alerts = len([a for a in alerts if a.severity == Severity.CRITICAL])

    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total AUM", f"${total_market/1e9:.2f}B")
    with col2:
        st.metric("Issuers", f"{total_obligors}")
    with col3:
        avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in exposures) / total_market if total_market > 0 else 0
        st.metric("Wtd OAS", f"{avg_oas:.0f}bp")
    with col4:
        avg_dur = sum(e.weighted_avg_duration * e.total_market_usd for e in exposures) / total_market if total_market > 0 else 0
        st.metric("Wtd Duration", f"{avg_dur:.2f}Y")
    with col5:
        st.metric("Active Alerts", f"{active_alerts}", delta=f"{critical_alerts} critical" if critical_alerts else None, delta_color="inverse")

    st.divider()

    # Row 1: Concentration + Rating
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Issuer Concentration")
        st.plotly_chart(create_concentration_chart(exposures, top_n=12), use_container_width=True)
    with col2:
        st.subheader("Rating Distribution")
        st.plotly_chart(create_rating_distribution_chart(exposures), use_container_width=True)

    # Row 2: Region + Sector
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regional Allocation")
        st.plotly_chart(create_region_distribution_chart(exposures), use_container_width=True)
    with col2:
        st.subheader("Sector Allocation")
        st.plotly_chart(create_sector_concentration_chart(exposures), use_container_width=True)

    # Row 3: Currency + Maturity
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Currency Breakdown")
        st.plotly_chart(create_currency_breakdown_chart(exposures), use_container_width=True)
    with col2:
        st.subheader("Maturity Profile")
        st.plotly_chart(create_maturity_profile_chart(exposures), use_container_width=True)

    # Risk Heatmap
    st.subheader("Risk Matrix (Rating × Duration)")
    st.plotly_chart(create_risk_heatmap(exposures), use_container_width=True)


def render_panorama_page():
    """Portfolio Panorama - Comprehensive risk analytics view"""
    exposures = st.session_state.exposures
    alerts = st.session_state.alerts
    scheme = ColorScheme()

    total_market = sum(e.total_market_usd for e in exposures)
    total_dv01 = sum(e.credit_dv01_usd for e in exposures)

    st.subheader("Portfolio Panorama | 组合全景图")

    # Executive Summary Cards
    st.markdown("### Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    # Calculate key metrics
    ig_exposure = sum(e.total_market_usd for e in exposures if e.obligor.rating_internal.value in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"])
    hy_exposure = total_market - ig_exposure
    china_exposure = sum(e.total_market_usd for e in exposures if e.obligor.region in (Region.CHINA_OFFSHORE, Region.CHINA_ONSHORE))
    dm_exposure = sum(e.total_market_usd for e in exposures if e.obligor.region in (Region.US, Region.EU, Region.UK, Region.JAPAN))
    em_exposure = total_market - china_exposure - dm_exposure

    with col1:
        st.markdown(f"""
        <div style="background:{scheme.bg_secondary};padding:16px;border-radius:8px;border-left:4px solid {scheme.accent_green};">
            <div style="color:{scheme.text_muted};font-size:12px;">Investment Grade</div>
            <div style="color:{scheme.text_primary};font-size:24px;font-weight:600;">${ig_exposure/1e9:.2f}B</div>
            <div style="color:{scheme.accent_green};font-size:14px;">{ig_exposure/total_market:.1%} of AUM</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{scheme.bg_secondary};padding:16px;border-radius:8px;border-left:4px solid {scheme.accent_red};">
            <div style="color:{scheme.text_muted};font-size:12px;">High Yield / Sub-IG</div>
            <div style="color:{scheme.text_primary};font-size:24px;font-weight:600;">${hy_exposure/1e9:.2f}B</div>
            <div style="color:{scheme.accent_orange};font-size:14px;">{hy_exposure/total_market:.1%} of AUM</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background:{scheme.bg_secondary};padding:16px;border-radius:8px;border-left:4px solid {scheme.accent_blue};">
            <div style="color:{scheme.text_muted};font-size:12px;">Credit DV01 (Total)</div>
            <div style="color:{scheme.text_primary};font-size:24px;font-weight:600;">${total_dv01/1e6:.2f}M</div>
            <div style="color:{scheme.accent_blue};font-size:14px;">per 1bp spread move</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        neg_outlook = len([e for e in exposures if e.obligor.rating_outlook in (RatingOutlook.NEGATIVE, RatingOutlook.WATCH_NEG)])
        st.markdown(f"""
        <div style="background:{scheme.bg_secondary};padding:16px;border-radius:8px;border-left:4px solid {scheme.accent_yellow};">
            <div style="color:{scheme.text_muted};font-size:12px;">Negative Outlook Issuers</div>
            <div style="color:{scheme.text_primary};font-size:24px;font-weight:600;">{neg_outlook}</div>
            <div style="color:{scheme.accent_yellow};font-size:14px;">{neg_outlook/len(exposures):.1%} of issuers</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Geographic Breakdown Table
    st.markdown("### Geographic Risk Decomposition")

    region_data = []
    for region in Region:
        region_exps = [e for e in exposures if e.obligor.region == region]
        if region_exps:
            mv = sum(e.total_market_usd for e in region_exps)
            dv01 = sum(e.credit_dv01_usd for e in region_exps)
            avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in region_exps) / mv if mv > 0 else 0
            avg_dur = sum(e.weighted_avg_duration * e.total_market_usd for e in region_exps) / mv if mv > 0 else 0
            region_data.append({
                "Region": region.value.replace("_", " ").title(),
                "MV ($M)": f"{mv/1e6:,.0f}",
                "% AUM": f"{mv/total_market:.1%}",
                "DV01 ($K)": f"{dv01/1e3:,.0f}",
                "Wtd OAS (bp)": f"{avg_oas:.0f}",
                "Wtd Duration": f"{avg_dur:.2f}",
                "# Issuers": len(region_exps),
            })

    if region_data:
        df_region = pd.DataFrame(region_data)
        st.dataframe(df_region, use_container_width=True, hide_index=True)

    st.divider()

    # Sector Risk Table
    st.markdown("### Sector Risk Decomposition")

    col1, col2 = st.columns([1, 1])

    with col1:
        sector_data = []
        for sector in Sector:
            sector_exps = [e for e in exposures if e.obligor.sector == sector]
            if sector_exps:
                mv = sum(e.total_market_usd for e in sector_exps)
                dv01 = sum(e.credit_dv01_usd for e in sector_exps)
                avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in sector_exps) / mv if mv > 0 else 0
                sector_data.append({
                    "Sector": sector.value,
                    "MV ($M)": f"{mv/1e6:,.0f}",
                    "% AUM": f"{mv/total_market:.1%}",
                    "DV01 ($K)": f"{dv01/1e3:,.0f}",
                    "Wtd OAS": f"{avg_oas:.0f}bp",
                })

        if sector_data:
            df_sector = pd.DataFrame(sector_data)
            st.dataframe(df_sector, use_container_width=True, hide_index=True)

    with col2:
        st.plotly_chart(create_dv01_decomposition_chart(exposures), use_container_width=True)

    st.divider()

    # Top Risk Exposures
    st.markdown("### Top 10 Risk Exposures (by Credit DV01)")

    sorted_by_dv01 = sorted(exposures, key=lambda x: x.credit_dv01_usd, reverse=True)[:10]
    top_risk_data = []
    for exp in sorted_by_dv01:
        top_risk_data.append({
            "Issuer": exp.obligor.display_name,
            "Sector": exp.obligor.sector.value,
            "Region": exp.obligor.region.value.replace("_", " ").title(),
            "Rating": exp.obligor.rating_internal.value,
            "Outlook": exp.obligor.rating_outlook.value,
            "MV ($M)": f"{exp.total_market_usd/1e6:,.0f}",
            "Duration": f"{exp.weighted_avg_duration:.2f}",
            "OAS (bp)": f"{exp.weighted_avg_oas:.0f}",
            "DV01 ($K)": f"{exp.credit_dv01_usd/1e3:,.0f}",
        })

    df_top_risk = pd.DataFrame(top_risk_data)
    st.dataframe(df_top_risk, use_container_width=True, hide_index=True)

    st.divider()

    # Watchlist - Negative Outlook Issuers
    st.markdown("### Watchlist: Negative Outlook / Under Review")

    watchlist = [e for e in exposures if e.obligor.rating_outlook in (RatingOutlook.NEGATIVE, RatingOutlook.WATCH_NEG)]
    if watchlist:
        watchlist_data = []
        for exp in sorted(watchlist, key=lambda x: x.total_market_usd, reverse=True):
            watchlist_data.append({
                "Issuer": exp.obligor.display_name,
                "Sector": exp.obligor.sector.value,
                "Region": exp.obligor.region.value.replace("_", " ").title(),
                "Rating": exp.obligor.rating_internal.value,
                "Outlook": "⚠️ " + exp.obligor.rating_outlook.value,
                "MV ($M)": f"{exp.total_market_usd/1e6:,.0f}",
                "% AUM": f"{exp.pct_of_aum:.2%}",
                "OAS (bp)": f"{exp.weighted_avg_oas:.0f}",
            })
        df_watchlist = pd.DataFrame(watchlist_data)
        st.dataframe(df_watchlist, use_container_width=True, hide_index=True)
    else:
        st.success("No issuers on negative outlook watchlist")

    st.divider()

    # Maturity Wall Analysis
    st.markdown("### Maturity Wall Analysis (Next 12 Months)")

    # Aggregate bonds maturing in next 12 months by issuer
    maturity_wall = []
    for exp in exposures:
        maturing_bonds = [b for b in exp.bonds if b.years_to_maturity <= 1.0]
        if maturing_bonds:
            maturing_mv = sum(b.market_value_usd for b in maturing_bonds)
            maturity_wall.append({
                "Issuer": exp.obligor.display_name,
                "Rating": exp.obligor.rating_internal.value,
                "Region": exp.obligor.region.value.replace("_", " ").title(),
                "Maturing MV ($M)": f"{maturing_mv/1e6:,.0f}",
                "% of Issuer Total": f"{maturing_mv/exp.total_market_usd:.1%}" if exp.total_market_usd > 0 else "N/A",
                "Bonds Maturing": len(maturing_bonds),
            })

    if maturity_wall:
        maturity_wall = sorted(maturity_wall, key=lambda x: float(x["Maturing MV ($M)"].replace(",", "")), reverse=True)
        df_maturity = pd.DataFrame(maturity_wall[:15])
        st.dataframe(df_maturity, use_container_width=True, hide_index=True)

        total_maturing = sum(float(m["Maturing MV ($M)"].replace(",", "")) for m in maturity_wall)
        st.caption(f"**Total maturing in 12M:** ${total_maturing:,.0f}M ({total_maturing*1e6/total_market:.1%} of AUM)")
    else:
        st.info("No significant maturities in the next 12 months")


def render_issuer_page():
    """Issuer Detail Page - Deep dive into individual obligors"""
    exposures = st.session_state.exposures
    obligors = st.session_state.obligors
    alerts = st.session_state.alerts
    news_items = st.session_state.news
    scheme = ColorScheme()

    st.subheader("Issuer Analysis | 发行人分析")

    # Issuer selector
    sorted_exposures = sorted(exposures, key=lambda x: x.total_market_usd, reverse=True)
    issuer_options = {e.obligor.obligor_id: f"{e.obligor.display_name} (${e.total_market_usd/1e6:.0f}M)" for e in sorted_exposures}

    selected_id = st.selectbox(
        "Select Issuer",
        options=list(issuer_options.keys()),
        format_func=lambda x: issuer_options[x],
    )

    if not selected_id:
        st.info("Select an issuer to view details")
        return

    # Get selected exposure
    exp = next((e for e in exposures if e.obligor.obligor_id == selected_id), None)
    if not exp:
        return

    obligor = exp.obligor

    st.divider()

    # Issuer Header Card
    col1, col2 = st.columns([2, 1])

    with col1:
        # Rating badge color
        rating_color = ColorScheme.get_rating_color(obligor.rating_internal.value)
        outlook_icon = {"POSITIVE": "📈", "STABLE": "➡️", "NEGATIVE": "📉", "WATCH_NEG": "⚠️"}.get(obligor.rating_outlook.value, "")

        st.markdown(f"""
        <div style="background:{scheme.bg_secondary};padding:20px;border-radius:12px;border-left:5px solid {ColorScheme.get_sector_color(obligor.sector.value)};">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <h2 style="color:{scheme.text_primary};margin:0;">{obligor.display_name}</h2>
                    <p style="color:{scheme.text_muted};margin:4px 0;">{obligor.name_cn if obligor.name_en else ''}</p>
                </div>
                <div style="text-align:right;">
                    <span style="background:{rating_color};color:white;padding:6px 12px;border-radius:6px;font-weight:600;font-size:18px;">
                        {obligor.rating_internal.value}
                    </span>
                    <p style="color:{scheme.text_muted};margin:8px 0 0 0;">{outlook_icon} {obligor.rating_outlook.value}</p>
                </div>
            </div>
            <div style="margin-top:16px;display:flex;gap:24px;flex-wrap:wrap;">
                <div><span style="color:{scheme.text_muted};">Sector:</span> <span style="color:{scheme.text_primary};">{obligor.sector.value}</span></div>
                <div><span style="color:{scheme.text_muted};">Region:</span> <span style="color:{scheme.text_primary};">{obligor.region.value.replace('_', ' ').title()}</span></div>
                <div><span style="color:{scheme.text_muted};">Country:</span> <span style="color:{scheme.text_primary};">{obligor.country or 'N/A'}</span></div>
                {f'<div><span style="color:{scheme.text_muted};">Ticker:</span> <span style="color:{scheme.accent_blue};">{obligor.ticker}</span></div>' if obligor.ticker else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Key metrics
        st.markdown(f"""
        <div style="background:{scheme.bg_secondary};padding:20px;border-radius:12px;">
            <h4 style="color:{scheme.text_primary};margin:0 0 12px 0;">Exposure Summary</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div>
                    <div style="color:{scheme.text_muted};font-size:12px;">Market Value</div>
                    <div style="color:{scheme.text_primary};font-size:20px;font-weight:600;">${exp.total_market_usd/1e6:,.0f}M</div>
                </div>
                <div>
                    <div style="color:{scheme.text_muted};font-size:12px;">% of AUM</div>
                    <div style="color:{scheme.text_primary};font-size:20px;font-weight:600;">{exp.pct_of_aum:.2%}</div>
                </div>
                <div>
                    <div style="color:{scheme.text_muted};font-size:12px;">Wtd Duration</div>
                    <div style="color:{scheme.text_primary};font-size:20px;font-weight:600;">{exp.weighted_avg_duration:.2f}Y</div>
                </div>
                <div>
                    <div style="color:{scheme.text_muted};font-size:12px;">Wtd OAS</div>
                    <div style="color:{scheme.text_primary};font-size:20px;font-weight:600;">{exp.weighted_avg_oas:.0f}bp</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Bond Holdings Table
    st.markdown("### Bond Holdings")

    bond_data = []
    for bond in sorted(exp.bonds, key=lambda b: b.maturity_date):
        bond_data.append({
            "ISIN": bond.isin,
            "Currency": bond.currency,
            "Coupon": f"{bond.coupon:.2f}%",
            "Maturity": bond.maturity_date.strftime("%Y-%m-%d"),
            "YTM (yrs)": f"{bond.years_to_maturity:.1f}",
            "Nominal ($M)": f"{bond.nominal_usd/1e6:,.1f}",
            "MV ($M)": f"{bond.market_value_usd/1e6:,.1f}",
            "Duration": f"{bond.duration:.2f}",
            "OAS (bp)": f"{bond.oas:.0f}" if bond.oas else "N/A",
        })

    if bond_data:
        df_bonds = pd.DataFrame(bond_data)
        st.dataframe(df_bonds, use_container_width=True, hide_index=True)

    st.divider()

    # Peer Comparison
    st.markdown("### Peer Comparison")

    # Find peers (same sector, similar rating)
    peer_exposures = [e for e in exposures
                      if e.obligor.sector == obligor.sector
                      and e.obligor.obligor_id != obligor.obligor_id][:5]

    if peer_exposures:
        peer_data = [{"Issuer": obligor.display_name, "Rating": obligor.rating_internal.value,
                      "MV ($M)": f"{exp.total_market_usd/1e6:,.0f}",
                      "Duration": f"{exp.weighted_avg_duration:.2f}",
                      "OAS (bp)": f"{exp.weighted_avg_oas:.0f}", "Type": "Selected"}]

        for peer in peer_exposures:
            peer_data.append({
                "Issuer": peer.obligor.display_name,
                "Rating": peer.obligor.rating_internal.value,
                "MV ($M)": f"{peer.total_market_usd/1e6:,.0f}",
                "Duration": f"{peer.weighted_avg_duration:.2f}",
                "OAS (bp)": f"{peer.weighted_avg_oas:.0f}",
                "Type": "Peer",
            })

        df_peers = pd.DataFrame(peer_data)
        st.dataframe(df_peers, use_container_width=True, hide_index=True)

        # OAS comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[obligor.display_name] + [p.obligor.display_name for p in peer_exposures],
            y=[exp.weighted_avg_oas] + [p.weighted_avg_oas for p in peer_exposures],
            marker_color=[scheme.accent_blue] + [scheme.text_muted] * len(peer_exposures),
        ))
        fig.update_layout(**get_premium_layout("OAS Comparison vs Peers", height=300))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No peers found in the same sector")

    st.divider()

    # Related Alerts
    st.markdown("### Related Alerts")
    issuer_alerts = [a for a in alerts if a.obligor_id == selected_id]
    if issuer_alerts:
        for alert in issuer_alerts:
            severity_color = ColorScheme.get_severity_color(alert.severity.value)
            st.markdown(f"""
            <div style="background:{scheme.bg_secondary};padding:12px 16px;margin:8px 0;border-radius:8px;border-left:4px solid {severity_color};">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{scheme.text_primary};font-weight:600;">{alert.message}</span>
                    <span style="color:{scheme.text_muted};font-size:12px;">{alert.timestamp.strftime('%m-%d %H:%M')}</span>
                </div>
                {f'<div style="color:{scheme.text_secondary};margin-top:8px;font-size:14px;">{alert.ai_summary}</div>' if alert.ai_summary else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No active alerts for this issuer")

    # Related News
    st.markdown("### Related News")
    issuer_news = [n for n in news_items if selected_id in n.obligor_ids]
    if issuer_news:
        for news in issuer_news[:5]:
            sentiment_color = {
                Sentiment.POSITIVE: scheme.severity_success,
                Sentiment.NEUTRAL: scheme.text_secondary,
                Sentiment.NEGATIVE: scheme.severity_critical,
            }.get(news.sentiment, scheme.text_secondary)

            st.markdown(f"""
            <div style="background:{scheme.bg_secondary};padding:12px 16px;margin:8px 0;border-radius:8px;border-left:3px solid {sentiment_color};">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{scheme.text_primary};font-weight:500;">{news.title}</span>
                    <span style="color:{scheme.text_muted};font-size:12px;">{news.timestamp.strftime('%m-%d %H:%M')} · {news.source}</span>
                </div>
                <div style="color:{scheme.text_secondary};margin-top:6px;font-size:14px;">{news.summary or ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent news for this issuer")


def render_alerts_page():
    st.subheader("🚨 Alerts Center")
    alerts = st.session_state.alerts

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 严重", len([a for a in alerts if a.severity == Severity.CRITICAL]))
    with col2:
        st.metric("🟡 警告", len([a for a in alerts if a.severity == Severity.WARNING]))
    with col3:
        st.metric("待处理", len([a for a in alerts if a.status == AlertStatus.PENDING]))
    with col4:
        st.metric("已解决", len([a for a in alerts if a.status == AlertStatus.RESOLVED]))

    st.divider()
    render_alert_table(alerts, show_filters=True)


def render_news_page():
    st.subheader("📰 新闻流")
    news_items = st.session_state.news
    scheme = ColorScheme()

    for news in sorted(news_items, key=lambda x: x.timestamp, reverse=True):
        sentiment_color = {
            Sentiment.POSITIVE: scheme.severity_success,
            Sentiment.NEUTRAL: scheme.text_secondary,
            Sentiment.NEGATIVE: scheme.severity_critical,
        }.get(news.sentiment, scheme.text_secondary)

        sentiment_icon = {Sentiment.POSITIVE: "🟢", Sentiment.NEUTRAL: "⚪", Sentiment.NEGATIVE: "🔴"}.get(news.sentiment, "⚪")

        st.markdown(f"""
        <div style="background-color:{scheme.bg_secondary};border-left:3px solid {sentiment_color};padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:600;color:{scheme.text_primary};">{sentiment_icon} {news.title}</span>
                <span style="color:{scheme.text_muted};font-size:12px;">{news.timestamp.strftime('%m-%d %H:%M')} · {news.source}</span>
            </div>
            <div style="color:{scheme.text_secondary};margin-top:8px;font-size:14px;">{news.summary or news.content[:100] + '...'}</div>
        </div>
        """, unsafe_allow_html=True)

        if news.obligor_ids:
            names = [st.session_state.obligors[oid].display_name for oid in news.obligor_ids if oid in st.session_state.obligors]
            if names:
                st.caption(f"Related Issuers: {', '.join(names)}")


def render_chat_page():
    st.subheader("💬 AI问答")
    st.info("基于RAG的信用知识库问答（Demo模式）")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("输入问题，例如：云南城投最近有什么风险？"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                if "云南" in prompt:
                    response = """根据近期资料分析：

**云南城投整体情况**：
1. 近期省财政厅出台支持政策，整体信用环境有所改善
2. 部分地市级平台仍存在现金流压力
3. 建议关注省级平台，谨慎对待区县级平台

**相关新闻**：
- 省财政厅发文支持城投平台债务重组（正面）

**建议**：维持持有，关注政策执行效果"""
                else:
                    response = f"已收到您的问题：{prompt}\n\n正在检索相关资料...（Demo模式下功能有限）"
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


# =============================================================================
# Main Application
# =============================================================================


def main():
    init_session_state()
    scheme = ColorScheme()

    # Custom CSS
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {scheme.bg_primary}; }}
    [data-testid="stSidebar"] {{ background-color: {scheme.bg_secondary}; }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📊 Credit Intelligence")
        st.caption("Global Credit Risk Platform | 信用债风险预警平台")
        st.divider()

        page = st.radio(
            "Navigation", options=["overview", "panorama", "issuer", "alerts", "news", "chat"],
            format_func=lambda x: {
                "overview": "📈 Overview",
                "panorama": "🌍 Panorama",
                "issuer": "🏢 Issuer",
                "alerts": "🚨 Alerts",
                "news": "📰 News",
                "chat": "💬 AI Q&A"
            }[x],
            label_visibility="collapsed",
        )
        st.session_state.active_page = page

        st.divider()

        # Quick Stats
        exposures = st.session_state.exposures
        total_mv = sum(e.total_market_usd for e in exposures)
        st.caption(f"**AUM:** ${total_mv/1e9:.1f}B | **Issuers:** {len(exposures)}")

        st.divider()

        alerts = st.session_state.alerts
        pending = len([a for a in alerts if a.status == AlertStatus.PENDING])
        critical = len([a for a in alerts if a.severity == Severity.CRITICAL])

        if critical > 0:
            st.error(f"🔴 {critical} critical alerts pending")
        elif pending > 0:
            st.warning(f"🟡 {pending} alerts pending")
        else:
            st.success("✅ No pending alerts")

        st.divider()

        if st.button("🔄 Refresh Data", use_container_width=True):
            obligors, exposures, alerts, news = generate_mock_data()
            st.session_state.obligors = obligors
            st.session_state.exposures = exposures
            st.session_state.alerts = alerts
            st.session_state.news = news
            st.rerun()

        st.divider()
        st.caption("v2.0 | Phase 4 Complete")

    # Main content
    st.title("Credit Intelligence Platform")

    if st.session_state.active_page == "overview":
        render_overview_page()
    elif st.session_state.active_page == "panorama":
        render_panorama_page()
    elif st.session_state.active_page == "issuer":
        render_issuer_page()
    elif st.session_state.active_page == "alerts":
        render_alerts_page()
    elif st.session_state.active_page == "news":
        render_news_page()
    elif st.session_state.active_page == "chat":
        render_chat_page()


if __name__ == "__main__":
    main()
