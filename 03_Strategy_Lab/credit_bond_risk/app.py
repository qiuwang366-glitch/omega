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
import random

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pydantic import BaseModel, Field, computed_field

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
    LGFV = "LGFV"
    SOE = "SOE"
    FINANCIAL = "FINANCIAL"
    CORP = "CORP"


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
            "LGFV": scheme.accent_blue,
            "SOE": scheme.accent_purple,
            "FINANCIAL": scheme.accent_green,
            "CORP": scheme.accent_yellow,
        }
        return mapping.get(sector.upper(), scheme.text_secondary)


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
    province: str | None = None
    rating_internal: CreditRating
    rating_outlook: RatingOutlook = RatingOutlook.STABLE

    @computed_field
    @property
    def rating_score(self) -> int:
        return RATING_SCORE.get(self.rating_internal, 50)


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
    names = [e.obligor.name_cn for e in sorted_exposures]
    values = [e.total_market_usd / 1e6 for e in sorted_exposures]
    pcts = [e.pct_of_aum for e in sorted_exposures]
    sectors = [e.obligor.sector.value for e in sorted_exposures]
    colors = [ColorScheme.get_sector_color(s) for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"${v:.0f}M ({p:.1%})" for v, p in zip(values, pcts)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>市值: $%{x:.0f}M<extra></extra>",
    ))
    fig.update_layout(**get_premium_layout("Top发行人持仓", height=max(400, top_n * 30)))
    fig.update_layout(showlegend=False)
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
# Mock Data Generation
# =============================================================================


def generate_mock_data() -> tuple[dict[str, Obligor], list[CreditExposure], list[RiskAlert], list[NewsItem]]:
    obligor_templates = [
        ("OBL001", "某省城投集团", Sector.LGFV, "省级城投", "云南", CreditRating.AA, RatingOutlook.STABLE),
        ("OBL002", "某市城建投资", Sector.LGFV, "地级市城投", "重庆", CreditRating.AA_MINUS, RatingOutlook.NEGATIVE),
        ("OBL003", "某央企集团", Sector.SOE, "央企", None, CreditRating.AAA, RatingOutlook.STABLE),
        ("OBL004", "某股份制银行", Sector.FINANCIAL, "股份制银行", None, CreditRating.AA_PLUS, RatingOutlook.STABLE),
        ("OBL005", "某地方国企", Sector.SOE, "地方国企", "四川", CreditRating.AA, RatingOutlook.WATCH_NEG),
        ("OBL006", "某区县城投", Sector.LGFV, "区县城投", "贵州", CreditRating.AA_MINUS, RatingOutlook.NEGATIVE),
        ("OBL007", "某科技企业", Sector.CORP, "科技", "北京", CreditRating.A, RatingOutlook.POSITIVE),
        ("OBL008", "某城商行", Sector.FINANCIAL, "城商行", "江苏", CreditRating.AA, RatingOutlook.STABLE),
    ]

    obligors = {}
    exposures = []

    for oid, name, sector, sub, province, rating, outlook in obligor_templates:
        obligor = Obligor(
            obligor_id=oid, name_cn=name, sector=sector, sub_sector=sub,
            province=province, rating_internal=rating, rating_outlook=outlook,
        )
        obligors[oid] = obligor

        bonds = []
        for i in range(random.randint(2, 5)):
            maturity_years = random.uniform(0.5, 8)
            nominal = random.uniform(50, 300) * 1e6
            bonds.append(BondPosition(
                isin=f"{oid}-BOND-{i+1}", obligor_id=oid, bond_name=f"{name}债券{i+1}",
                currency="USD", maturity_date=date.today() + timedelta(days=int(maturity_years * 365)),
                coupon=random.uniform(3, 6), nominal=nominal, nominal_usd=nominal,
                book_value_usd=nominal * random.uniform(0.95, 1.02),
                market_value_usd=nominal * random.uniform(0.90, 1.05),
                duration=maturity_years * 0.9, oas=random.uniform(80, 400),
            ))
        exposures.append(CreditExposure.from_positions(obligor, bonds, 50e9))

    alerts = [
        RiskAlert(alert_id="ALT001", severity=Severity.CRITICAL, category=AlertCategory.RATING,
                  obligor_id="OBL002", obligor_name="某市城建投资", signal_name="rating_change",
                  message="评级下调至AA-，展望负面", metric_value=2.0, threshold=1.0, status=AlertStatus.PENDING),
        RiskAlert(alert_id="ALT002", severity=Severity.WARNING, category=AlertCategory.SPREAD,
                  obligor_id="OBL006", obligor_name="某区县城投", signal_name="spread_percentile",
                  message="OAS突破历史92%分位", metric_value=0.92, threshold=0.85, status=AlertStatus.INVESTIGATING),
        RiskAlert(alert_id="ALT003", severity=Severity.WARNING, category=AlertCategory.NEWS,
                  obligor_id="OBL005", obligor_name="某地方国企", signal_name="news_sentiment",
                  message="近7天舆情负面 (sentiment: -0.45)", metric_value=-0.45, threshold=-0.30,
                  status=AlertStatus.PENDING, ai_summary="近期有关于该企业现金流紧张的报道，建议关注其短期偿债能力。"),
        RiskAlert(alert_id="ALT004", severity=Severity.CRITICAL, category=AlertCategory.CONCENTRATION,
                  obligor_id="OBL001", obligor_name="某省城投集团", signal_name="concentration_single",
                  message="单一发行人占比超过5%", metric_value=0.052, threshold=0.05, status=AlertStatus.PENDING),
    ]

    news_items = [
        NewsItem(news_id="NEWS001", timestamp=datetime.now() - timedelta(hours=2), source="cls",
                 title="某省财政厅发文支持城投平台债务重组",
                 content="省财政厅发布指导意见，支持辖内城投平台通过债务重组、资产注入等方式化解债务风险...",
                 obligor_ids=["OBL001"], summary="省级支持政策出台，利好区域城投",
                 sentiment=Sentiment.POSITIVE, sentiment_score=0.6),
        NewsItem(news_id="NEWS002", timestamp=datetime.now() - timedelta(hours=5), source="bloomberg",
                 title="某市城建投资被曝现金流紧张",
                 content="据知情人士透露，该公司近期应收账款回款困难，部分项目支出延迟...",
                 obligor_ids=["OBL002"], summary="现金流压力显现，关注再融资能力",
                 sentiment=Sentiment.NEGATIVE, sentiment_score=-0.7),
        NewsItem(news_id="NEWS003", timestamp=datetime.now() - timedelta(days=1), source="eastmoney",
                 title="美联储议息会议在即，境外中资美元债或承压",
                 content="分析师预计美联储将维持高利率，境外中资美元债收益率可能继续上行...",
                 obligor_ids=[], summary="宏观利率风险提示", sentiment=Sentiment.NEUTRAL, sentiment_score=-0.1),
    ]

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
    active_alerts = len([a for a in alerts if a.status == AlertStatus.PENDING])
    critical_alerts = len([a for a in alerts if a.severity == Severity.CRITICAL])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总市值", f"${total_market/1e9:.2f}B")
    with col2:
        st.metric("发行人数", f"{total_obligors}")
    with col3:
        st.metric("活跃预警", f"{active_alerts}", delta=f"-{critical_alerts} 严重" if critical_alerts else None, delta_color="inverse")
    with col4:
        avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in exposures) / total_market if total_market > 0 else 0
        st.metric("加权OAS", f"{avg_oas:.0f}bp")

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("持仓集中度")
        st.plotly_chart(create_concentration_chart(exposures, top_n=10), use_container_width=True)
    with col2:
        st.subheader("评级分布")
        st.plotly_chart(create_rating_distribution_chart(exposures), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("行业分布")
        st.plotly_chart(create_sector_concentration_chart(exposures), use_container_width=True)
    with col2:
        st.subheader("到期分布")
        st.plotly_chart(create_maturity_profile_chart(exposures), use_container_width=True)

    st.subheader("风险矩阵")
    st.plotly_chart(create_risk_heatmap(exposures), use_container_width=True)


def render_alerts_page():
    st.subheader("🚨 预警中心")
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
            names = [st.session_state.obligors[oid].name_cn for oid in news.obligor_ids if oid in st.session_state.obligors]
            if names:
                st.caption(f"关联发行人: {', '.join(names)}")


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
        st.caption("信用债风险预警平台")
        st.divider()

        page = st.radio(
            "导航", options=["overview", "alerts", "news", "chat"],
            format_func=lambda x: {"overview": "📈 组合概览", "alerts": "🚨 预警中心", "news": "📰 新闻流", "chat": "💬 AI问答"}[x],
            label_visibility="collapsed",
        )
        st.session_state.active_page = page

        st.divider()

        alerts = st.session_state.alerts
        pending = len([a for a in alerts if a.status == AlertStatus.PENDING])
        critical = len([a for a in alerts if a.severity == Severity.CRITICAL])

        if critical > 0:
            st.error(f"🔴 {critical} 条严重预警待处理")
        elif pending > 0:
            st.warning(f"🟡 {pending} 条预警待处理")
        else:
            st.success("✅ 无待处理预警")

        st.divider()

        if st.button("🔄 刷新数据", use_container_width=True):
            obligors, exposures, alerts, news = generate_mock_data()
            st.session_state.obligors = obligors
            st.session_state.exposures = exposures
            st.session_state.alerts = alerts
            st.session_state.news = news
            st.rerun()

    # Main content
    st.title("Credit Intelligence Platform")

    if st.session_state.active_page == "overview":
        render_overview_page()
    elif st.session_state.active_page == "alerts":
        render_alerts_page()
    elif st.session_state.active_page == "news":
        render_news_page()
    elif st.session_state.active_page == "chat":
        render_chat_page()


if __name__ == "__main__":
    main()
