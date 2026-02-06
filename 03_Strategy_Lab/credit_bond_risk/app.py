#!/usr/bin/env python3
"""
信用债风险智能平台 | Credit Bond Risk Intelligence Platform
Scandinavian Minimal Design | 北欧极简设计风格

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
    get_obligor_relationships,
    OBLIGOR_RELATIONSHIPS,
)
from data.provider import DataProviderConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Config (must be first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title="信用债风险智能平台",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Enums (Simplified for app.py)
# =============================================================================


class Sector(str, Enum):
    LGFV = "LGFV"
    SOE = "SOE"
    FINANCIAL = "FINANCIAL"
    CORP = "CORP"
    SOVEREIGN = "SOVEREIGN"
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
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC = "CCC"
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
    CreditRating.AAA: 100, CreditRating.AA_PLUS: 95, CreditRating.AA: 90,
    CreditRating.AA_MINUS: 85, CreditRating.A_PLUS: 80, CreditRating.A: 75,
    CreditRating.A_MINUS: 70, CreditRating.BBB_PLUS: 65, CreditRating.BBB: 60,
    CreditRating.BBB_MINUS: 55, CreditRating.BB_PLUS: 50, CreditRating.BB: 45,
    CreditRating.BB_MINUS: 40, CreditRating.B_PLUS: 35, CreditRating.B: 30,
    CreditRating.B_MINUS: 25, CreditRating.CCC: 15, CreditRating.NR: 50,
}


# =============================================================================
# Scandinavian Color Scheme | 北欧极简配色
# =============================================================================


@dataclass
class NordicTheme:
    """Scandinavian minimal design color palette | 斯堪的纳维亚极简设计配色"""

    # Background - Warm whites and soft grays
    bg_primary: str = "#FAFAFA"       # 米白色背景
    bg_secondary: str = "#FFFFFF"     # 纯白卡片
    bg_tertiary: str = "#F5F5F5"      # 浅灰分隔
    bg_sidebar: str = "#F8F9FA"       # 侧边栏

    # Text - Soft blacks and grays
    text_primary: str = "#1A1A2E"     # 深蓝黑 - 主文字
    text_secondary: str = "#4A4A68"   # 灰蓝 - 次要文字
    text_muted: str = "#8E8EA0"       # 柔和灰 - 辅助文字
    text_light: str = "#B0B0C0"       # 浅灰 - 提示文字

    # Accent - Nordic nature inspired
    accent_blue: str = "#5B8DEF"      # 北欧蓝 - 主强调色
    accent_green: str = "#4CAF7C"     # 森林绿 - 正面/成功
    accent_amber: str = "#E5A94D"     # 琥珀黄 - 警告
    accent_coral: str = "#E57373"     # 珊瑚红 - 严重/错误
    accent_purple: str = "#9575CD"    # 薰衣草紫 - 次要强调
    accent_teal: str = "#4DB6AC"      # 青绿色 - 信息

    # Severity colors
    severity_critical: str = "#D84315"  # 深珊瑚红
    severity_warning: str = "#F9A825"   # 琥珀黄
    severity_info: str = "#5B8DEF"      # 北欧蓝

    # Rating gradient (AAA to CCC)
    rating_aaa: str = "#2E7D5A"       # 深森林绿
    rating_aa: str = "#4CAF7C"        # 森林绿
    rating_a: str = "#81C784"         # 浅绿
    rating_bbb: str = "#E5A94D"       # 琥珀黄
    rating_bb: str = "#FF8A65"        # 珊瑚橙
    rating_b: str = "#E57373"         # 珊瑚红
    rating_ccc: str = "#D84315"       # 深红

    # Sector colors
    sector_lgfv: str = "#5B8DEF"      # 蓝 - 城投
    sector_soe: str = "#9575CD"       # 紫 - 央企
    sector_financial: str = "#4CAF7C" # 绿 - 金融
    sector_corp: str = "#E5A94D"      # 黄 - 企业
    sector_supra: str = "#4DB6AC"     # 青 - 超主权

    # Border and shadow
    border_light: str = "#E8E8EC"     # 浅边框
    shadow: str = "0 1px 3px rgba(0,0,0,0.08)"

    @classmethod
    def get_severity_color(cls, severity: str) -> str:
        scheme = cls()
        mapping = {
            "CRITICAL": scheme.severity_critical,
            "WARNING": scheme.severity_warning,
            "INFO": scheme.severity_info,
        }
        return mapping.get(severity.upper(), scheme.text_muted)

    @classmethod
    def get_rating_color(cls, rating: str) -> str:
        scheme = cls()
        r = rating.upper()
        if "AAA" in r:
            return scheme.rating_aaa
        elif "AA" in r:
            return scheme.rating_aa
        elif r.startswith("A"):
            return scheme.rating_a
        elif "BBB" in r:
            return scheme.rating_bbb
        elif "BB" in r:
            return scheme.rating_bb
        elif r.startswith("B"):
            return scheme.rating_b
        else:
            return scheme.rating_ccc

    @classmethod
    def get_sector_color(cls, sector: str) -> str:
        scheme = cls()
        mapping = {
            "LGFV": scheme.sector_lgfv,
            "SOE": scheme.sector_soe,
            "FINANCIAL": scheme.sector_financial,
            "CORP": scheme.sector_corp,
            "US_CORP": scheme.accent_amber,
            "EU_CORP": scheme.accent_purple,
            "G-SIB": scheme.accent_blue,
            "EM_SOVEREIGN": scheme.accent_teal,
            "SUPRA": scheme.sector_supra,
            "HY": scheme.severity_critical,
        }
        return mapping.get(sector.upper(), scheme.text_muted)

    @classmethod
    def get_region_color(cls, region: str) -> str:
        scheme = cls()
        mapping = {
            "CHINA_OFFSHORE": "#E57373",
            "CHINA_ONSHORE": "#EF9A9A",
            "US": "#5B8DEF",
            "EU": "#9575CD",
            "UK": "#4CAF7C",
            "JAPAN": "#E5A94D",
            "LATAM": "#81C784",
            "CEEMEA": "#FF8A65",
            "ASIA_EX_CHINA": "#4DB6AC",
            "SUPRANATIONAL": "#2E7D5A",
        }
        return mapping.get(region.upper(), scheme.text_muted)


def get_nordic_layout(title: str = "", height: int = 380) -> dict:
    """Get Plotly layout with Nordic minimal style"""
    theme = NordicTheme()
    return {
        "title": {
            "text": title,
            "font": {"size": 14, "color": theme.text_primary, "family": "Inter, -apple-system, sans-serif"},
            "x": 0.02, "xanchor": "left",
        },
        "paper_bgcolor": theme.bg_secondary,
        "plot_bgcolor": theme.bg_secondary,
        "height": height,
        "margin": {"l": 50, "r": 20, "t": 45, "b": 40},
        "font": {"family": "Inter, -apple-system, sans-serif", "color": theme.text_secondary, "size": 11},
        "xaxis": {
            "gridcolor": theme.border_light,
            "linecolor": theme.border_light,
            "tickfont": {"color": theme.text_muted, "size": 10},
            "showgrid": True,
            "gridwidth": 1,
        },
        "yaxis": {
            "gridcolor": theme.border_light,
            "linecolor": theme.border_light,
            "tickfont": {"color": theme.text_muted, "size": 10},
            "showgrid": True,
            "gridwidth": 1,
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "font": {"color": theme.text_secondary, "size": 10},
            "bordercolor": theme.border_light,
            "borderwidth": 1,
        },
        "hoverlabel": {
            "bgcolor": theme.bg_secondary,
            "font": {"color": theme.text_primary, "size": 11},
            "bordercolor": theme.border_light,
        },
    }


# =============================================================================
# Data Models (Simplified)
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
    ticker: str | None = None
    lei: str | None = None
    parent_entity: str | None = None

    @computed_field
    @property
    def rating_score(self) -> int:
        return RATING_SCORE.get(self.rating_internal, 50)

    @computed_field
    @property
    def display_name(self) -> str:
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
            obligor=obligor, bonds=positions, total_nominal_usd=total_nominal,
            total_market_usd=total_market, pct_of_aum=total_market / total_aum if total_aum > 0 else 0,
            weighted_avg_duration=weighted_duration, weighted_avg_oas=weighted_oas,
            credit_dv01_usd=total_dv01, maturity_profile=maturity_buckets,
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
# Chart Components - Nordic Style
# =============================================================================


def create_concentration_chart(exposures: list[CreditExposure], top_n: int = 12) -> go.Figure:
    """发行人集中度图表"""
    sorted_exposures = sorted(exposures, key=lambda x: x.total_market_usd, reverse=True)[:top_n]
    names = [e.obligor.name_cn[:8] for e in sorted_exposures]
    values = [e.total_market_usd / 1e6 for e in sorted_exposures]
    pcts = [e.pct_of_aum for e in sorted_exposures]
    sectors = [e.obligor.sector.value for e in sorted_exposures]
    colors = [NordicTheme.get_sector_color(s) for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"${v:.0f}M ({p:.1%})" for v, p in zip(values, pcts)],
        textposition="outside", textfont={"size": 10, "color": NordicTheme().text_secondary},
        hovertemplate="<b>%{y}</b><br>市值: $%{x:.0f}M<extra></extra>",
    ))
    fig.update_layout(**get_nordic_layout("发行人集中度 | Top Issuers", height=max(350, top_n * 28)))
    fig.update_layout(showlegend=False, bargap=0.3)
    fig.update_yaxes(autorange="reversed")
    return fig


def create_rating_distribution_chart(exposures: list[CreditExposure]) -> go.Figure:
    """评级分布图表"""
    rating_totals: dict[str, float] = {}
    for exp in exposures:
        rating = exp.obligor.rating_internal.value
        rating_totals[rating] = rating_totals.get(rating, 0) + exp.total_market_usd
    labels = list(rating_totals.keys())
    values = [v / 1e6 for v in rating_totals.values()]
    colors = [NordicTheme.get_rating_color(r) for r in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55, marker_colors=colors,
        textinfo="label+percent", textposition="outside",
        textfont={"size": 10, "color": NordicTheme().text_secondary},
        pull=[0.02] * len(labels),
    )])
    fig.update_layout(**get_nordic_layout("评级分布 | Rating", height=320))
    fig.update_layout(showlegend=False)
    return fig


def create_region_distribution_chart(exposures: list[CreditExposure]) -> go.Figure:
    """区域分布图表"""
    region_totals: dict[str, float] = {}
    for exp in exposures:
        region = exp.obligor.region.value
        region_totals[region] = region_totals.get(region, 0) + exp.total_market_usd
    label_map = {
        "CHINA_OFFSHORE": "中国离岸", "CHINA_ONSHORE": "中国在岸", "US": "美国",
        "EU": "欧洲", "UK": "英国", "JAPAN": "日本", "LATAM": "拉美",
        "CEEMEA": "中东欧/中东", "ASIA_EX_CHINA": "亚太", "SUPRANATIONAL": "超主权",
    }
    labels = list(region_totals.keys())
    display_labels = [label_map.get(l, l) for l in labels]
    values = [v / 1e6 for v in region_totals.values()]
    colors = [NordicTheme.get_region_color(r) for r in labels]

    fig = go.Figure(data=[go.Pie(
        labels=display_labels, values=values, hole=0.55, marker_colors=colors,
        textinfo="label+percent", textposition="outside",
        textfont={"size": 10},
    )])
    fig.update_layout(**get_nordic_layout("区域分布 | Region", height=320))
    fig.update_layout(showlegend=False)
    return fig


def create_sector_distribution_chart(exposures: list[CreditExposure]) -> go.Figure:
    """行业分布图表"""
    sector_totals: dict[str, float] = {}
    for exp in exposures:
        sector = exp.obligor.sector.value
        sector_totals[sector] = sector_totals.get(sector, 0) + exp.total_market_usd
    sector_map = {
        "LGFV": "城投", "SOE": "央企", "FINANCIAL": "金融", "G-SIB": "G-SIB银行",
        "US_CORP": "美企", "EU_CORP": "欧企", "EM_SOVEREIGN": "新兴主权",
        "SUPRA": "超主权", "HY": "高收益",
    }
    labels = list(sector_totals.keys())
    display_labels = [sector_map.get(l, l) for l in labels]
    values = [v / 1e6 for v in sector_totals.values()]
    colors = [NordicTheme.get_sector_color(s) for s in labels]

    fig = go.Figure(data=[go.Pie(
        labels=display_labels, values=values, hole=0.55, marker_colors=colors,
        textinfo="label+percent", textposition="outside",
        textfont={"size": 10},
    )])
    fig.update_layout(**get_nordic_layout("行业分布 | Sector", height=320))
    fig.update_layout(showlegend=False)
    return fig


def create_maturity_profile_chart(exposures: list[CreditExposure]) -> go.Figure:
    """到期分布图表"""
    theme = NordicTheme()
    buckets = ["0-1Y", "1-3Y", "3-5Y", "5-10Y", "10Y+"]
    bucket_labels = ["1年内", "1-3年", "3-5年", "5-10年", "10年+"]
    bucket_totals = {b: 0 for b in buckets}
    for exp in exposures:
        for bucket, value in exp.maturity_profile.items():
            if bucket in buckets:
                bucket_totals[bucket] += value
    values = [bucket_totals[b] / 1e6 for b in buckets]
    colors = [theme.accent_green, theme.accent_blue, theme.accent_purple, theme.accent_amber, theme.accent_coral]

    fig = go.Figure(data=[go.Bar(
        x=bucket_labels, y=values, marker_color=colors,
        text=[f"${v:.0f}M" for v in values], textposition="outside",
        textfont={"size": 10, "color": theme.text_secondary},
    )])
    fig.update_layout(**get_nordic_layout("到期分布 | Maturity Profile", height=300))
    fig.update_layout(bargap=0.4)
    return fig


def create_currency_breakdown_chart(exposures: list[CreditExposure]) -> go.Figure:
    """货币分布图表"""
    theme = NordicTheme()
    currency_totals: dict[str, float] = {}
    for exp in exposures:
        for bond in exp.bonds:
            currency_totals[bond.currency] = currency_totals.get(bond.currency, 0) + bond.market_value_usd
    labels = list(currency_totals.keys())
    values = [v / 1e6 for v in currency_totals.values()]
    ccy_colors = {
        "USD": theme.accent_blue, "EUR": theme.accent_purple, "GBP": theme.accent_green,
        "JPY": theme.accent_amber, "CNH": theme.accent_coral, "CNY": "#EF9A9A",
    }
    colors = [ccy_colors.get(l, theme.text_muted) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"${v:.0f}M" for v in values], textposition="outside",
        textfont={"size": 10, "color": theme.text_secondary},
    )])
    fig.update_layout(**get_nordic_layout("货币分布 | Currency", height=300))
    fig.update_layout(bargap=0.4)
    return fig


def create_dv01_decomposition_chart(exposures: list[CreditExposure]) -> go.Figure:
    """信用DV01分解图表"""
    sector_dv01: dict[str, float] = {}
    for exp in exposures:
        sector = exp.obligor.sector.value
        sector_dv01[sector] = sector_dv01.get(sector, 0) + exp.credit_dv01_usd
    sorted_items = sorted(sector_dv01.items(), key=lambda x: x[1], reverse=True)
    sector_map = {
        "LGFV": "城投", "SOE": "央企", "FINANCIAL": "金融", "G-SIB": "G-SIB",
        "US_CORP": "美企", "EU_CORP": "欧企", "EM_SOVEREIGN": "新兴主权",
        "SUPRA": "超主权", "HY": "高收益",
    }
    labels = [sector_map.get(item[0], item[0]) for item in sorted_items]
    values = [item[1] / 1e6 for item in sorted_items]
    colors = [NordicTheme.get_sector_color(item[0]) for item in sorted_items]

    fig = go.Figure(data=[go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"${v:.2f}M" for v in values], textposition="outside",
        textfont={"size": 10},
    )])
    fig.update_layout(**get_nordic_layout("信用DV01分解 ($/bp)", height=350))
    fig.update_yaxes(autorange="reversed")
    return fig


def create_risk_heatmap(exposures: list[CreditExposure]) -> go.Figure:
    """风险矩阵热力图"""
    theme = NordicTheme()
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
        colorscale=[[0, "#F5F5F5"], [0.5, theme.accent_blue], [1, theme.accent_coral]],
        text=[[f"${v:.0f}M" for v in row] for row in matrix],
        texttemplate="%{text}", textfont={"size": 11, "color": theme.text_primary},
        hovertemplate="评级: %{y}<br>久期: %{x}<br>市值: %{z:.0f}M<extra></extra>",
    ))
    fig.update_layout(**get_nordic_layout("风险矩阵 | 评级 × 久期", height=300))
    fig.update_yaxes(autorange="reversed")
    return fig


def create_oas_trend_chart(exposures: list[CreditExposure]) -> go.Figure:
    """OAS趋势模拟图表"""
    theme = NordicTheme()
    # 模拟过去30天的OAS趋势
    dates = [date.today() - timedelta(days=i) for i in range(30, 0, -1)]
    date_strs = [d.strftime("%m-%d") for d in dates]

    # 获取不同评级的平均OAS
    ig_oas = []
    hy_oas = []
    base_ig = 85
    base_hy = 450

    for i in range(30):
        ig_oas.append(base_ig + np.random.normal(0, 5) + (i - 15) * 0.3)
        hy_oas.append(base_hy + np.random.normal(0, 20) + (i - 15) * 1.5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_strs, y=ig_oas, mode='lines', name='投资级 IG',
        line=dict(color=theme.accent_blue, width=2),
        hovertemplate="日期: %{x}<br>OAS: %{y:.0f}bp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=date_strs, y=hy_oas, mode='lines', name='高收益 HY',
        line=dict(color=theme.accent_coral, width=2),
        yaxis='y2',
        hovertemplate="日期: %{x}<br>OAS: %{y:.0f}bp<extra></extra>",
    ))
    layout = get_nordic_layout("利差走势 | OAS Trend (30D)", height=280)
    layout['yaxis2'] = dict(
        overlaying='y', side='right', showgrid=False,
        tickfont={"color": theme.accent_coral, "size": 10},
    )
    layout['legend'] = dict(
        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
        font={"size": 10},
    )
    fig.update_layout(**layout)
    return fig


# =============================================================================
# Mock Data Generation
# =============================================================================


def generate_mock_data() -> tuple[dict[str, Obligor], list[CreditExposure], list[RiskAlert], list[NewsItem]]:
    """Generate mock data using data layer"""
    obligors = generate_mock_obligors()
    exposures = generate_mock_exposures(obligors)
    alerts = generate_mock_alerts()
    news_items = generate_mock_news()
    return obligors, exposures, alerts, news_items


# =============================================================================
# UI Components - Nordic Style
# =============================================================================


def render_kpi_card(title: str, value: str, subtitle: str = "", delta: str = "", delta_color: str = "normal"):
    """渲染KPI卡片 - 北欧极简风格"""
    theme = NordicTheme()
    delta_html = ""
    if delta:
        color = theme.accent_green if delta_color == "positive" else (theme.accent_coral if delta_color == "negative" else theme.text_muted)
        delta_html = f'<div style="font-size:12px;color:{color};margin-top:4px;">{delta}</div>'

    st.markdown(f"""
    <div style="
        background:{theme.bg_secondary};
        border:1px solid {theme.border_light};
        border-radius:12px;
        padding:20px 24px;
        text-align:left;
    ">
        <div style="font-size:12px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">{title}</div>
        <div style="font-size:28px;font-weight:600;color:{theme.text_primary};line-height:1.2;">{value}</div>
        {f'<div style="font-size:12px;color:{theme.text_secondary};margin-top:6px;">{subtitle}</div>' if subtitle else ''}
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_alert_badge(severity: str, count: int):
    """渲染预警徽章"""
    theme = NordicTheme()
    colors = {
        "CRITICAL": (theme.severity_critical, "严重"),
        "WARNING": (theme.severity_warning, "警告"),
        "INFO": (theme.severity_info, "提示"),
    }
    color, label = colors.get(severity.upper(), (theme.text_muted, severity))
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:500;">{label} {count}</span>'


def render_alert_table(alerts: list[RiskAlert], show_filters: bool = True) -> list[RiskAlert]:
    """渲染预警表格"""
    theme = NordicTheme()
    if not alerts:
        st.info("暂无预警信息")
        return []

    filtered_alerts = alerts

    if show_filters:
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.multiselect("严重程度", options=["CRITICAL", "WARNING", "INFO"],
                                              default=["CRITICAL", "WARNING"],
                                              format_func=lambda x: {"CRITICAL": "严重", "WARNING": "警告", "INFO": "提示"}[x])
            if severity_filter:
                filtered_alerts = [a for a in filtered_alerts if a.severity.value in severity_filter]
        with col2:
            category_filter = st.multiselect("类别", options=list(set(a.category.value for a in alerts)),
                                              format_func=lambda x: {"CONCENTRATION": "集中度", "RATING": "评级", "SPREAD": "利差", "NEWS": "新闻"}[x])
            if category_filter:
                filtered_alerts = [a for a in filtered_alerts if a.category.value in category_filter]
        with col3:
            status_filter = st.multiselect("状态", options=["PENDING", "INVESTIGATING", "RESOLVED"],
                                            default=["PENDING", "INVESTIGATING"],
                                            format_func=lambda x: {"PENDING": "待处理", "INVESTIGATING": "调查中", "RESOLVED": "已解决", "DISMISSED": "已忽略"}[x])
            if status_filter:
                filtered_alerts = [a for a in filtered_alerts if a.status.value in status_filter]

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    filtered_alerts.sort(key=lambda a: (severity_order.get(a.severity.value, 3), -a.timestamp.timestamp()))

    table_data = []
    for alert in filtered_alerts[:15]:
        severity_icon = {"CRITICAL": "●", "WARNING": "●", "INFO": "●"}.get(alert.severity.value, "○")
        severity_color = NordicTheme.get_severity_color(alert.severity.value)
        category_map = {"CONCENTRATION": "集中度", "RATING": "评级", "SPREAD": "利差", "NEWS": "新闻"}
        status_map = {"PENDING": "待处理", "INVESTIGATING": "调查中", "RESOLVED": "已解决", "DISMISSED": "已忽略"}
        table_data.append({
            "级别": f"{severity_icon}",
            "时间": alert.timestamp.strftime("%m-%d %H:%M"),
            "发行人": alert.obligor_name[:10],
            "类别": category_map.get(alert.category.value, alert.category.value),
            "消息": alert.message[:40] + "..." if len(alert.message) > 40 else alert.message,
            "状态": status_map.get(alert.status.value, alert.status.value),
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    # Summary
    critical_count = sum(1 for a in filtered_alerts if a.severity == Severity.CRITICAL)
    warning_count = sum(1 for a in filtered_alerts if a.severity == Severity.WARNING)
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-top:12px;">
        {render_alert_badge("CRITICAL", critical_count)}
        {render_alert_badge("WARNING", warning_count)}
        <span style="color:{theme.text_muted};font-size:12px;line-height:24px;">共 {len(filtered_alerts)} 条预警</span>
    </div>
    """, unsafe_allow_html=True)

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
    """总览页面"""
    theme = NordicTheme()
    exposures = st.session_state.exposures
    alerts = st.session_state.alerts

    total_market = sum(e.total_market_usd for e in exposures)
    total_obligors = len([e for e in exposures if e.total_market_usd > 0])
    total_dv01 = sum(e.credit_dv01_usd for e in exposures)
    active_alerts = len([a for a in alerts if a.status == AlertStatus.PENDING])
    critical_alerts = len([a for a in alerts if a.severity == Severity.CRITICAL and a.status == AlertStatus.PENDING])
    avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in exposures) / total_market if total_market > 0 else 0
    avg_dur = sum(e.weighted_avg_duration * e.total_market_usd for e in exposures) / total_market if total_market > 0 else 0

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("资产规模", f"${total_market/1e9:.1f}B", "总市值")
    with col2:
        render_kpi_card("发行人数量", f"{total_obligors}", "活跃持仓")
    with col3:
        render_kpi_card("加权OAS", f"{avg_oas:.0f}bp", "信用利差")
    with col4:
        render_kpi_card("加权久期", f"{avg_dur:.2f}Y", "修正久期")
    with col5:
        delta_text = f"{critical_alerts} 严重" if critical_alerts > 0 else "无严重预警"
        delta_color = "negative" if critical_alerts > 0 else "positive"
        render_kpi_card("待处理预警", f"{active_alerts}", "", delta_text, delta_color)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Charts Row 1
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(create_concentration_chart(exposures, top_n=10), use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(create_rating_distribution_chart(exposures), use_container_width=True, config={'displayModeBar': False})

    # Charts Row 2
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_region_distribution_chart(exposures), use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(create_sector_distribution_chart(exposures), use_container_width=True, config={'displayModeBar': False})
    with col3:
        st.plotly_chart(create_maturity_profile_chart(exposures), use_container_width=True, config={'displayModeBar': False})

    # Charts Row 3
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_risk_heatmap(exposures), use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(create_oas_trend_chart(exposures), use_container_width=True, config={'displayModeBar': False})


def render_panorama_page():
    """全景分析页面"""
    theme = NordicTheme()
    exposures = st.session_state.exposures
    total_market = sum(e.total_market_usd for e in exposures)
    total_dv01 = sum(e.credit_dv01_usd for e in exposures)

    # Executive Summary
    st.markdown(f"""
    <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:24px;margin-bottom:24px;">
        <h3 style="color:{theme.text_primary};margin:0 0 16px 0;font-weight:500;">执行摘要 | Executive Summary</h3>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;">
    """, unsafe_allow_html=True)

    ig_exposure = sum(e.total_market_usd for e in exposures if e.obligor.rating_internal.value in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"])
    hy_exposure = total_market - ig_exposure
    china_exposure = sum(e.total_market_usd for e in exposures if e.obligor.region in (Region.CHINA_OFFSHORE, Region.CHINA_ONSHORE))
    neg_outlook = len([e for e in exposures if e.obligor.rating_outlook in (RatingOutlook.NEGATIVE, RatingOutlook.WATCH_NEG)])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:{theme.bg_tertiary};border-radius:8px;">
            <div style="font-size:11px;color:{theme.text_muted};margin-bottom:8px;">投资级敞口</div>
            <div style="font-size:24px;font-weight:600;color:{theme.accent_green};">${ig_exposure/1e9:.2f}B</div>
            <div style="font-size:12px;color:{theme.text_secondary};">{ig_exposure/total_market:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:{theme.bg_tertiary};border-radius:8px;">
            <div style="font-size:11px;color:{theme.text_muted};margin-bottom:8px;">高收益敞口</div>
            <div style="font-size:24px;font-weight:600;color:{theme.accent_coral};">${hy_exposure/1e9:.2f}B</div>
            <div style="font-size:12px;color:{theme.text_secondary};">{hy_exposure/total_market:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:{theme.bg_tertiary};border-radius:8px;">
            <div style="font-size:11px;color:{theme.text_muted};margin-bottom:8px;">信用DV01</div>
            <div style="font-size:24px;font-weight:600;color:{theme.accent_blue};">${total_dv01/1e6:.2f}M</div>
            <div style="font-size:12px;color:{theme.text_secondary};">每bp变动</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:{theme.bg_tertiary};border-radius:8px;">
            <div style="font-size:11px;color:{theme.text_muted};margin-bottom:8px;">负面展望</div>
            <div style="font-size:24px;font-weight:600;color:{theme.accent_amber};">{neg_outlook}</div>
            <div style="font-size:12px;color:{theme.text_secondary};">发行人</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Regional Breakdown
    st.markdown(f"### 区域风险分解")
    region_data = []
    for region in Region:
        region_exps = [e for e in exposures if e.obligor.region == region]
        if region_exps:
            mv = sum(e.total_market_usd for e in region_exps)
            dv01 = sum(e.credit_dv01_usd for e in region_exps)
            avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in region_exps) / mv if mv > 0 else 0
            avg_dur = sum(e.weighted_avg_duration * e.total_market_usd for e in region_exps) / mv if mv > 0 else 0
            region_map = {"CHINA_OFFSHORE": "中国离岸", "CHINA_ONSHORE": "中国在岸", "US": "美国", "EU": "欧洲", "UK": "英国", "JAPAN": "日本", "LATAM": "拉美", "CEEMEA": "中东欧/中东", "ASIA_EX_CHINA": "亚太", "SUPRANATIONAL": "超主权"}
            region_data.append({
                "区域": region_map.get(region.value, region.value),
                "市值 ($M)": f"{mv/1e6:,.0f}",
                "占比": f"{mv/total_market:.1%}",
                "DV01 ($K)": f"{dv01/1e3:,.0f}",
                "加权OAS": f"{avg_oas:.0f}bp",
                "加权久期": f"{avg_dur:.2f}",
                "发行人数": len(region_exps),
            })
    if region_data:
        df_region = pd.DataFrame(region_data)
        st.dataframe(df_region, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Top Risk Exposures
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 风险敞口 Top 10 (按DV01)")
        sorted_by_dv01 = sorted(exposures, key=lambda x: x.credit_dv01_usd, reverse=True)[:10]
        top_risk_data = []
        for exp in sorted_by_dv01:
            top_risk_data.append({
                "发行人": exp.obligor.name_cn[:10],
                "行业": exp.obligor.sector.value,
                "评级": exp.obligor.rating_internal.value,
                "市值": f"${exp.total_market_usd/1e6:,.0f}M",
                "DV01": f"${exp.credit_dv01_usd/1e3:,.0f}K",
            })
        st.dataframe(pd.DataFrame(top_risk_data), use_container_width=True, hide_index=True, height=380)

    with col2:
        st.markdown(f"### 关注名单 | Watchlist")
        watchlist = [e for e in exposures if e.obligor.rating_outlook in (RatingOutlook.NEGATIVE, RatingOutlook.WATCH_NEG)]
        if watchlist:
            watchlist_data = []
            for exp in sorted(watchlist, key=lambda x: x.total_market_usd, reverse=True)[:10]:
                outlook_icon = "⚠" if exp.obligor.rating_outlook == RatingOutlook.WATCH_NEG else "↓"
                watchlist_data.append({
                    "发行人": exp.obligor.name_cn[:10],
                    "评级": exp.obligor.rating_internal.value,
                    "展望": outlook_icon,
                    "市值": f"${exp.total_market_usd/1e6:,.0f}M",
                    "OAS": f"{exp.weighted_avg_oas:.0f}bp",
                })
            st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True, hide_index=True, height=380)
        else:
            st.success("暂无负面展望发行人")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # DV01 Chart
    st.plotly_chart(create_dv01_decomposition_chart(exposures), use_container_width=True, config={'displayModeBar': False})


def render_issuer_page():
    """发行人分析页面"""
    theme = NordicTheme()
    exposures = st.session_state.exposures
    obligors = st.session_state.obligors
    alerts = st.session_state.alerts
    news_items = st.session_state.news

    # Issuer selector
    sorted_exposures = sorted(exposures, key=lambda x: x.total_market_usd, reverse=True)
    issuer_options = {e.obligor.obligor_id: f"{e.obligor.name_cn} (${e.total_market_usd/1e6:.0f}M)" for e in sorted_exposures if e.total_market_usd > 0}

    selected_id = st.selectbox("选择发行人", options=list(issuer_options.keys()), format_func=lambda x: issuer_options[x])

    if not selected_id:
        return

    exp = next((e for e in exposures if e.obligor.obligor_id == selected_id), None)
    if not exp:
        return

    obligor = exp.obligor
    rating_color = NordicTheme.get_rating_color(obligor.rating_internal.value)
    outlook_icon = {"POSITIVE": "↑", "STABLE": "→", "NEGATIVE": "↓", "WATCH_NEG": "⚠"}.get(obligor.rating_outlook.value, "")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Header Card
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:24px;border-left:4px solid {NordicTheme.get_sector_color(obligor.sector.value)};">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <h2 style="color:{theme.text_primary};margin:0;font-weight:500;">{obligor.name_cn}</h2>
                    <p style="color:{theme.text_muted};margin:4px 0 0 0;font-size:14px;">{obligor.name_en or ''}</p>
                </div>
                <div style="text-align:right;">
                    <span style="background:{rating_color};color:white;padding:8px 16px;border-radius:8px;font-weight:600;font-size:18px;">{obligor.rating_internal.value}</span>
                    <p style="color:{theme.text_muted};margin:8px 0 0 0;font-size:13px;">{outlook_icon} {obligor.rating_outlook.value}</p>
                </div>
            </div>
            <div style="margin-top:20px;display:flex;gap:32px;flex-wrap:wrap;">
                <div><span style="color:{theme.text_muted};font-size:12px;">行业</span><br><span style="color:{theme.text_primary};">{obligor.sector.value}</span></div>
                <div><span style="color:{theme.text_muted};font-size:12px;">区域</span><br><span style="color:{theme.text_primary};">{obligor.region.value.replace('_', ' ')}</span></div>
                <div><span style="color:{theme.text_muted};font-size:12px;">国家</span><br><span style="color:{theme.text_primary};">{obligor.country or 'N/A'}</span></div>
                {f'<div><span style="color:{theme.text_muted};font-size:12px;">代码</span><br><span style="color:{theme.accent_blue};">{obligor.ticker}</span></div>' if obligor.ticker else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:24px;">
            <h4 style="color:{theme.text_primary};margin:0 0 16px 0;font-weight:500;">敞口概览</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div>
                    <div style="color:{theme.text_muted};font-size:11px;">市值</div>
                    <div style="color:{theme.text_primary};font-size:22px;font-weight:600;">${exp.total_market_usd/1e6:,.0f}M</div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:11px;">占比</div>
                    <div style="color:{theme.text_primary};font-size:22px;font-weight:600;">{exp.pct_of_aum:.2%}</div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:11px;">加权久期</div>
                    <div style="color:{theme.text_primary};font-size:22px;font-weight:600;">{exp.weighted_avg_duration:.2f}Y</div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:11px;">加权OAS</div>
                    <div style="color:{theme.text_primary};font-size:22px;font-weight:600;">{exp.weighted_avg_oas:.0f}bp</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Bond Holdings
    st.markdown("### 债券持仓")
    bond_data = []
    for bond in sorted(exp.bonds, key=lambda b: b.maturity_date):
        bond_data.append({
            "ISIN": bond.isin,
            "货币": bond.currency,
            "票息": f"{bond.coupon:.2f}%",
            "到期日": bond.maturity_date.strftime("%Y-%m-%d"),
            "剩余年限": f"{bond.years_to_maturity:.1f}",
            "面值 ($M)": f"{bond.nominal_usd/1e6:,.1f}",
            "市值 ($M)": f"{bond.market_value_usd/1e6:,.1f}",
            "久期": f"{bond.duration:.2f}",
            "OAS": f"{bond.oas:.0f}" if bond.oas else "-",
        })
    if bond_data:
        st.dataframe(pd.DataFrame(bond_data), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Related Alerts & News
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 相关预警")
        issuer_alerts = [a for a in alerts if a.obligor_id == selected_id]
        if issuer_alerts:
            for alert in issuer_alerts[:5]:
                severity_color = NordicTheme.get_severity_color(alert.severity.value)
                st.markdown(f"""
                <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-left:3px solid {severity_color};padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{theme.text_primary};font-weight:500;font-size:13px;">{alert.message[:50]}</span>
                        <span style="color:{theme.text_muted};font-size:11px;">{alert.timestamp.strftime('%m-%d %H:%M')}</span>
                    </div>
                    {f'<div style="color:{theme.text_secondary};margin-top:8px;font-size:12px;">{alert.ai_summary}</div>' if alert.ai_summary else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("暂无相关预警")

    with col2:
        st.markdown("### 相关新闻")
        issuer_news = [n for n in news_items if selected_id in n.obligor_ids]
        if issuer_news:
            for news in issuer_news[:5]:
                sentiment_color = {
                    Sentiment.POSITIVE: theme.accent_green,
                    Sentiment.NEUTRAL: theme.text_muted,
                    Sentiment.NEGATIVE: theme.accent_coral,
                }.get(news.sentiment, theme.text_muted)
                st.markdown(f"""
                <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-left:3px solid {sentiment_color};padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{theme.text_primary};font-weight:500;font-size:13px;">{news.title[:40]}...</span>
                        <span style="color:{theme.text_muted};font-size:11px;">{news.source}</span>
                    </div>
                    <div style="color:{theme.text_secondary};margin-top:6px;font-size:12px;">{news.summary or ''}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无相关新闻")


def render_alerts_page():
    """预警中心页面"""
    theme = NordicTheme()
    alerts = st.session_state.alerts

    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        critical = len([a for a in alerts if a.severity == Severity.CRITICAL])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:20px;text-align:center;border-top:3px solid {theme.severity_critical};">
            <div style="font-size:32px;font-weight:600;color:{theme.severity_critical};">{critical}</div>
            <div style="font-size:12px;color:{theme.text_muted};margin-top:4px;">严重预警</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        warning = len([a for a in alerts if a.severity == Severity.WARNING])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:20px;text-align:center;border-top:3px solid {theme.severity_warning};">
            <div style="font-size:32px;font-weight:600;color:{theme.severity_warning};">{warning}</div>
            <div style="font-size:12px;color:{theme.text_muted};margin-top:4px;">警告</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        pending = len([a for a in alerts if a.status == AlertStatus.PENDING])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:20px;text-align:center;border-top:3px solid {theme.accent_blue};">
            <div style="font-size:32px;font-weight:600;color:{theme.accent_blue};">{pending}</div>
            <div style="font-size:12px;color:{theme.text_muted};margin-top:4px;">待处理</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        resolved = len([a for a in alerts if a.status == AlertStatus.RESOLVED])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:20px;text-align:center;border-top:3px solid {theme.accent_green};">
            <div style="font-size:32px;font-weight:600;color:{theme.accent_green};">{resolved}</div>
            <div style="font-size:12px;color:{theme.text_muted};margin-top:4px;">已解决</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    render_alert_table(alerts, show_filters=True)


def render_news_page():
    """新闻流页面"""
    theme = NordicTheme()
    news_items = st.session_state.news

    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <span style="font-size:13px;color:{theme.text_muted};">共 {len(news_items)} 条新闻 · 最近更新 {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)

    for news in sorted(news_items, key=lambda x: x.timestamp, reverse=True):
        sentiment_color = {
            Sentiment.POSITIVE: theme.accent_green,
            Sentiment.NEUTRAL: theme.text_muted,
            Sentiment.NEGATIVE: theme.accent_coral,
        }.get(news.sentiment, theme.text_muted)
        sentiment_label = {"POSITIVE": "正面", "NEUTRAL": "中性", "NEGATIVE": "负面"}.get(news.sentiment.value if news.sentiment else "", "")

        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-left:3px solid {sentiment_color};padding:16px 20px;margin:12px 0;border-radius:0 12px 12px 0;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div style="flex:1;">
                    <span style="font-weight:500;color:{theme.text_primary};font-size:14px;">{news.title}</span>
                </div>
                <div style="text-align:right;margin-left:16px;">
                    <span style="color:{theme.text_muted};font-size:11px;">{news.timestamp.strftime('%m-%d %H:%M')}</span>
                    <br><span style="color:{theme.text_muted};font-size:11px;">{news.source}</span>
                </div>
            </div>
            <div style="color:{theme.text_secondary};margin-top:10px;font-size:13px;line-height:1.5;">{news.summary or news.content[:150] + '...'}</div>
            <div style="margin-top:10px;display:flex;gap:8px;align-items:center;">
                <span style="background:{sentiment_color}20;color:{sentiment_color};padding:2px 8px;border-radius:4px;font-size:11px;">{sentiment_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if news.obligor_ids:
            names = [st.session_state.obligors[oid].name_cn for oid in news.obligor_ids if oid in st.session_state.obligors][:3]
            if names:
                st.caption(f"相关发行人: {', '.join(names)}")


def render_chat_page():
    """AI问答页面"""
    theme = NordicTheme()

    st.markdown(f"""
    <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:16px 20px;margin-bottom:20px;">
        <div style="color:{theme.text_primary};font-weight:500;">信用知识库问答 (演示模式)</div>
        <div style="color:{theme.text_muted};font-size:12px;margin-top:4px;">基于RAG技术，支持自然语言查询信用风险信息</div>
    </div>
    """, unsafe_allow_html=True)

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
            with st.spinner("分析中..."):
                if "云南" in prompt:
                    response = """**云南城投整体分析**

根据系统数据，云南省城投敞口情况如下：

**1. 敞口概况**
- 省级平台：云南省城投集团 (AA/稳定)
- 地市平台：昆明城投 (AA-/稳定)、曲靖城投 (A+/负面)
- 总敞口占比约 6.5%，略超区域限额

**2. 近期风险信号**
- 曲靖城投利差周变动+45bp，触发预警
- 省财政收入超预期，对省级平台形成支撑

**3. 政策动态**
- 财政部发布城投化债新政，省级平台可开展债务置换
- 预计对云南省级平台形成直接利好

**建议**：维持省级平台持仓，地市级平台需谨慎增持"""
                elif "贵州" in prompt or "六盘水" in prompt:
                    response = """**贵州城投风险分析**

**1. 重点预警**
- 六盘水城投被曝财务造假，省财政厅已介入调查
- 贵州交投利差突破历史92%分位

**2. 担保链风险**
- 六盘水城投为贵州交投子公司
- 存在担保链传染风险

**3. 化债进展**
- 省政府召开化债攻坚会，承诺兜底支持
- 执行力度仍需持续观察

**建议**：密切关注调查进展，考虑对冲或减持地市级平台"""
                else:
                    response = f"已收到您的问题：{prompt}\n\n正在检索相关资料...（演示模式下功能有限）"
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


# =============================================================================
# Main Application
# =============================================================================


def main():
    init_session_state()
    theme = NordicTheme()

    # Custom CSS - Nordic Minimal Style
    st.markdown(f"""
    <style>
    /* Base styles */
    .stApp {{
        background-color: {theme.bg_primary};
    }}
    [data-testid="stSidebar"] {{
        background-color: {theme.bg_sidebar};
        border-right: 1px solid {theme.border_light};
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {theme.text_secondary};
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {theme.text_primary} !important;
        font-weight: 500 !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {theme.text_primary};
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme.text_muted};
    }}

    /* Cards and containers */
    .stDataFrame {{
        border: 1px solid {theme.border_light};
        border-radius: 8px;
    }}

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > label {{
        color: {theme.text_secondary};
    }}

    /* Selectbox */
    .stSelectbox > div > div {{
        background-color: {theme.bg_secondary};
        border-color: {theme.border_light};
    }}

    /* Remove Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Smooth transitions */
    * {{
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:8px 0 16px 0;">
            <h2 style="color:{theme.text_primary};margin:0;font-weight:500;">◈ 信用债风险平台</h2>
            <p style="color:{theme.text_muted};font-size:12px;margin:8px 0 0 0;">Credit Risk Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        page = st.radio(
            "导航", options=["overview", "panorama", "issuer", "alerts", "news", "chat"],
            format_func=lambda x: {
                "overview": "◈  总览",
                "panorama": "◎  全景分析",
                "issuer": "◇  发行人",
                "alerts": "◆  预警中心",
                "news": "◇  新闻流",
                "chat": "◈  AI问答"
            }[x],
            label_visibility="collapsed",
        )
        st.session_state.active_page = page

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Quick Stats
        exposures = st.session_state.exposures
        total_mv = sum(e.total_market_usd for e in exposures)
        total_issuers = len([e for e in exposures if e.total_market_usd > 0])

        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:8px;padding:16px;">
            <div style="font-size:11px;color:{theme.text_muted};margin-bottom:12px;">快速统计</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{theme.text_muted};font-size:12px;">资产规模</span>
                <span style="color:{theme.text_primary};font-weight:500;">${total_mv/1e9:.1f}B</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{theme.text_muted};font-size:12px;">发行人</span>
                <span style="color:{theme.text_primary};font-weight:500;">{total_issuers}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Alert Status
        alerts = st.session_state.alerts
        pending = len([a for a in alerts if a.status == AlertStatus.PENDING])
        critical = len([a for a in alerts if a.severity == Severity.CRITICAL and a.status == AlertStatus.PENDING])

        if critical > 0:
            st.markdown(f"""
            <div style="background:{theme.severity_critical}15;border:1px solid {theme.severity_critical}40;border-radius:8px;padding:12px;">
                <div style="color:{theme.severity_critical};font-size:12px;font-weight:500;">● {critical} 严重预警待处理</div>
            </div>
            """, unsafe_allow_html=True)
        elif pending > 0:
            st.markdown(f"""
            <div style="background:{theme.severity_warning}15;border:1px solid {theme.severity_warning}40;border-radius:8px;padding:12px;">
                <div style="color:{theme.severity_warning};font-size:12px;font-weight:500;">● {pending} 预警待处理</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{theme.accent_green}15;border:1px solid {theme.accent_green}40;border-radius:8px;padding:12px;">
                <div style="color:{theme.accent_green};font-size:12px;font-weight:500;">✓ 无待处理预警</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if st.button("刷新数据", use_container_width=True):
            obligors, exposures, alerts, news = generate_mock_data()
            st.session_state.obligors = obligors
            st.session_state.exposures = exposures
            st.session_state.alerts = alerts
            st.session_state.news = news
            st.rerun()

        st.markdown(f"""
        <div style="position:absolute;bottom:20px;left:20px;right:20px;">
            <div style="color:{theme.text_light};font-size:11px;text-align:center;">v3.0 · 2026 Edition</div>
        </div>
        """, unsafe_allow_html=True)

    # Main content
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
