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


def render_styled_table(df: pd.DataFrame, max_height: str = "auto"):
    """渲染北欧风格HTML表格 - 替代st.dataframe以确保主题一致"""
    theme = NordicTheme()
    if df.empty:
        st.info("暂无数据")
        return

    # Header
    header_cells = "".join(
        f'<th style="padding:12px 14px;color:{theme.text_muted};font-size:11px;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.04em;border-bottom:2px solid {theme.border_light};'
        f'text-align:left;white-space:nowrap;">{col}</th>'
        for col in df.columns
    )

    # Rows
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = theme.bg_secondary if i % 2 == 0 else theme.bg_primary
        cells = "".join(
            f'<td style="padding:10px 14px;color:{theme.text_primary};font-size:13px;'
            f'border-bottom:1px solid {theme.bg_tertiary};white-space:nowrap;">{val}</td>'
            for val in row
        )
        rows_html += f'<tr style="background:{bg};">{cells}</tr>'

    overflow_style = f"max-height:{max_height};overflow-y:auto;" if max_height != "auto" else ""

    st.markdown(f"""
    <div style="border:1px solid {theme.border_light};border-radius:12px;overflow:hidden;background:{theme.bg_secondary};{overflow_style}">
        <table style="width:100%;border-collapse:collapse;font-family:'Inter',-apple-system,sans-serif;">
            <thead style="position:sticky;top:0;z-index:1;">
                <tr style="background:{theme.bg_tertiary};">{header_cells}</tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


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
        border-radius:14px;
        padding:22px 24px;
        text-align:left;
        box-shadow:0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    ">
        <div style="font-size:11px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;font-weight:500;">{title}</div>
        <div style="font-size:30px;font-weight:600;color:{theme.text_primary};line-height:1.15;letter-spacing:-0.02em;">{value}</div>
        {f'<div style="font-size:12px;color:{theme.text_secondary};margin-top:8px;">{subtitle}</div>' if subtitle else ''}
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
        # Render styled HTML table for consistent light theme
        rows_html = ""
        for i, row in enumerate(table_data):
            severity_val = row["级别"]
            severity_color = NordicTheme.get_severity_color(
                "CRITICAL" if severity_val == "●" and i < len(filtered_alerts) and filtered_alerts[i].severity == Severity.CRITICAL
                else "WARNING" if severity_val == "●" and i < len(filtered_alerts) and filtered_alerts[i].severity == Severity.WARNING
                else "INFO"
            )
            bg = theme.bg_secondary if i % 2 == 0 else theme.bg_primary
            rows_html += f"""
            <tr style="background:{bg};">
                <td style="padding:10px 12px;color:{severity_color};font-size:16px;text-align:center;border-bottom:1px solid {theme.border_light};">{row['级别']}</td>
                <td style="padding:10px 12px;color:{theme.text_muted};font-size:12px;border-bottom:1px solid {theme.border_light};white-space:nowrap;">{row['时间']}</td>
                <td style="padding:10px 12px;color:{theme.text_primary};font-weight:500;font-size:13px;border-bottom:1px solid {theme.border_light};">{row['发行人']}</td>
                <td style="padding:10px 12px;color:{theme.text_secondary};font-size:12px;border-bottom:1px solid {theme.border_light};">{row['类别']}</td>
                <td style="padding:10px 12px;color:{theme.text_primary};font-size:13px;border-bottom:1px solid {theme.border_light};max-width:300px;">{row['消息']}</td>
                <td style="padding:10px 12px;border-bottom:1px solid {theme.border_light};">
                    <span style="background:{theme.bg_tertiary};color:{theme.text_secondary};padding:3px 10px;border-radius:12px;font-size:11px;font-weight:500;">{row['状态']}</span>
                </td>
            </tr>"""

        st.markdown(f"""
        <div style="border:1px solid {theme.border_light};border-radius:12px;overflow:hidden;background:{theme.bg_secondary};">
            <table style="width:100%;border-collapse:collapse;font-family:'Inter',-apple-system,sans-serif;">
                <thead>
                    <tr style="background:{theme.bg_tertiary};">
                        <th style="padding:12px;color:{theme.text_muted};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid {theme.border_light};width:40px;"></th>
                        <th style="padding:12px;color:{theme.text_muted};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid {theme.border_light};text-align:left;">时间</th>
                        <th style="padding:12px;color:{theme.text_muted};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid {theme.border_light};text-align:left;">发行人</th>
                        <th style="padding:12px;color:{theme.text_muted};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid {theme.border_light};text-align:left;">类别</th>
                        <th style="padding:12px;color:{theme.text_muted};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid {theme.border_light};text-align:left;">消息</th>
                        <th style="padding:12px;color:{theme.text_muted};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid {theme.border_light};text-align:left;">状态</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Summary
    critical_count = sum(1 for a in filtered_alerts if a.severity == Severity.CRITICAL)
    warning_count = sum(1 for a in filtered_alerts if a.severity == Severity.WARNING)
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-top:16px;align-items:center;">
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
    <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:28px;margin-bottom:28px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <div style="width:4px;height:20px;background:linear-gradient(180deg,{theme.accent_blue},{theme.accent_purple});border-radius:2px;"></div>
            <h3 style="color:{theme.text_primary};margin:0;font-weight:600;font-size:16px;">执行摘要 | Executive Summary</h3>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;">
    """, unsafe_allow_html=True)

    ig_exposure = sum(e.total_market_usd for e in exposures if e.obligor.rating_internal.value in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"])
    hy_exposure = total_market - ig_exposure
    china_exposure = sum(e.total_market_usd for e in exposures if e.obligor.region in (Region.CHINA_OFFSHORE, Region.CHINA_ONSHORE))
    neg_outlook = len([e for e in exposures if e.obligor.rating_outlook in (RatingOutlook.NEGATIVE, RatingOutlook.WATCH_NEG)])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 16px;background:{theme.bg_tertiary};border-radius:10px;border:1px solid {theme.border_light};">
            <div style="font-size:10px;color:{theme.text_muted};margin-bottom:10px;text-transform:uppercase;letter-spacing:0.08em;font-weight:500;">投资级敞口</div>
            <div style="font-size:26px;font-weight:700;color:{theme.accent_green};letter-spacing:-0.02em;">${ig_exposure/1e9:.2f}B</div>
            <div style="font-size:12px;color:{theme.text_secondary};margin-top:4px;">{ig_exposure/total_market:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 16px;background:{theme.bg_tertiary};border-radius:10px;border:1px solid {theme.border_light};">
            <div style="font-size:10px;color:{theme.text_muted};margin-bottom:10px;text-transform:uppercase;letter-spacing:0.08em;font-weight:500;">高收益敞口</div>
            <div style="font-size:26px;font-weight:700;color:{theme.accent_coral};letter-spacing:-0.02em;">${hy_exposure/1e9:.2f}B</div>
            <div style="font-size:12px;color:{theme.text_secondary};margin-top:4px;">{hy_exposure/total_market:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 16px;background:{theme.bg_tertiary};border-radius:10px;border:1px solid {theme.border_light};">
            <div style="font-size:10px;color:{theme.text_muted};margin-bottom:10px;text-transform:uppercase;letter-spacing:0.08em;font-weight:500;">信用DV01</div>
            <div style="font-size:26px;font-weight:700;color:{theme.accent_blue};letter-spacing:-0.02em;">${total_dv01/1e6:.2f}M</div>
            <div style="font-size:12px;color:{theme.text_secondary};margin-top:4px;">每bp变动</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 16px;background:{theme.bg_tertiary};border-radius:10px;border:1px solid {theme.border_light};">
            <div style="font-size:10px;color:{theme.text_muted};margin-bottom:10px;text-transform:uppercase;letter-spacing:0.08em;font-weight:500;">负面展望</div>
            <div style="font-size:26px;font-weight:700;color:{theme.accent_amber};letter-spacing:-0.02em;">{neg_outlook}</div>
            <div style="font-size:12px;color:{theme.text_secondary};margin-top:4px;">发行人</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Regional Breakdown
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <div style="width:3px;height:16px;background:{theme.accent_blue};border-radius:2px;"></div>
        <span style="color:{theme.text_primary};font-weight:600;font-size:15px;">区域风险分解</span>
    </div>""", unsafe_allow_html=True)
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
        render_styled_table(df_region)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Top Risk Exposures
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:3px;height:16px;background:{theme.accent_coral};border-radius:2px;"></div>
            <span style="color:{theme.text_primary};font-weight:600;font-size:15px;">风险敞口 Top 10 (按DV01)</span>
        </div>""", unsafe_allow_html=True)
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
        render_styled_table(pd.DataFrame(top_risk_data), max_height="380px")

    with col2:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:3px;height:16px;background:{theme.accent_amber};border-radius:2px;"></div>
            <span style="color:{theme.text_primary};font-weight:600;font-size:15px;">关注名单 | Watchlist</span>
        </div>""", unsafe_allow_html=True)
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
            render_styled_table(pd.DataFrame(watchlist_data), max_height="380px")
        else:
            st.success("暂无负面展望发行人")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # DV01 Chart
    st.plotly_chart(create_dv01_decomposition_chart(exposures), use_container_width=True, config={'displayModeBar': False})


def _compute_issuer_risk_scores(obligor: Obligor, exp: "CreditExposure", alerts: list[RiskAlert], news_items: list[NewsItem], exposures: list["CreditExposure"]) -> dict:
    """Compute composite risk scores for an issuer (Aladdin-style)."""
    selected_id = obligor.obligor_id

    # 1. Credit Quality Score (based on rating + outlook)
    rating_base = RATING_SCORE.get(obligor.rating_internal, 50)
    outlook_adj = {"POSITIVE": 3, "STABLE": 0, "NEGATIVE": -8, "WATCH_NEG": -15}.get(obligor.rating_outlook.value, 0)
    credit_score = max(0, min(100, rating_base + outlook_adj))

    # 2. Concentration Score (inverse - lower concentration = higher score)
    pct = exp.pct_of_aum
    if pct < 0.01:
        conc_score = 95
    elif pct < 0.02:
        conc_score = 85
    elif pct < 0.03:
        conc_score = 70
    elif pct < 0.05:
        conc_score = 50
    else:
        conc_score = max(10, 100 - pct * 1000)

    # 3. Spread Score (tighter spread = higher score)
    oas = exp.weighted_avg_oas
    if oas < 50:
        spread_score = 95
    elif oas < 100:
        spread_score = 85
    elif oas < 200:
        spread_score = 70
    elif oas < 400:
        spread_score = 50
    else:
        spread_score = max(10, 100 - oas / 10)

    # 4. News Sentiment Score
    issuer_news = [n for n in news_items if selected_id in n.obligor_ids]
    if issuer_news:
        avg_sentiment = np.mean([n.sentiment_score for n in issuer_news if n.sentiment_score is not None] or [0])
        news_score = max(0, min(100, 50 + avg_sentiment * 50))
    else:
        news_score = 60  # neutral default

    # 5. Alert Score (fewer/less severe alerts = higher score)
    issuer_alerts = [a for a in alerts if a.obligor_id == selected_id]
    critical_count = sum(1 for a in issuer_alerts if a.severity == Severity.CRITICAL)
    warning_count = sum(1 for a in issuer_alerts if a.severity == Severity.WARNING)
    alert_score = max(0, 100 - critical_count * 25 - warning_count * 10)

    # 6. Liquidity / Market Score (based on number of bonds and total size)
    n_bonds = len(exp.bonds)
    total_mv = exp.total_market_usd
    if total_mv > 500e6 and n_bonds >= 4:
        liquidity_score = 90
    elif total_mv > 200e6 and n_bonds >= 3:
        liquidity_score = 75
    elif total_mv > 50e6:
        liquidity_score = 60
    else:
        liquidity_score = 40

    # Composite (weighted)
    weights = {"credit": 0.30, "spread": 0.20, "concentration": 0.10, "news": 0.15, "alerts": 0.15, "liquidity": 0.10}
    composite = (
        credit_score * weights["credit"]
        + spread_score * weights["spread"]
        + conc_score * weights["concentration"]
        + news_score * weights["news"]
        + alert_score * weights["alerts"]
        + liquidity_score * weights["liquidity"]
    )

    return {
        "composite": round(composite, 1),
        "credit": round(credit_score, 1),
        "spread": round(spread_score, 1),
        "concentration": round(conc_score, 1),
        "news": round(news_score, 1),
        "alerts": round(alert_score, 1),
        "liquidity": round(liquidity_score, 1),
    }


def _render_risk_gauge_svg(score: float, size: int = 140) -> str:
    """Render a circular risk gauge as inline SVG."""
    theme = NordicTheme()
    # Determine color based on score
    if score >= 80:
        color = theme.accent_green
        label = "LOW RISK"
    elif score >= 60:
        color = theme.accent_blue
        label = "MODERATE"
    elif score >= 40:
        color = theme.accent_amber
        label = "ELEVATED"
    else:
        color = theme.severity_critical
        label = "HIGH RISK"

    cx, cy = size // 2, size // 2
    r = size // 2 - 12
    circumference = 2 * 3.14159 * r
    pct = score / 100
    dash = circumference * pct
    gap = circumference - dash

    return f"""
    <div style="text-align:center;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{theme.bg_tertiary}" stroke-width="10"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{dash:.1f} {gap:.1f}"
                    stroke-linecap="round" transform="rotate(-90 {cx} {cy})"
                    style="transition: stroke-dasharray 0.8s ease;"/>
            <text x="{cx}" y="{cy - 6}" text-anchor="middle" font-size="28" font-weight="700" fill="{theme.text_primary}" font-family="Inter, sans-serif">{score:.0f}</text>
            <text x="{cx}" y="{cy + 16}" text-anchor="middle" font-size="10" font-weight="600" fill="{color}" font-family="Inter, sans-serif" letter-spacing="0.08em">{label}</text>
        </svg>
    </div>"""


def _create_issuer_yield_curve(bonds: list[BondPosition]) -> go.Figure:
    """Create issuer's bond yield curve (tenor vs OAS/coupon)."""
    theme = NordicTheme()
    if not bonds:
        return go.Figure()

    sorted_bonds = sorted(bonds, key=lambda b: b.years_to_maturity)
    tenors = [b.years_to_maturity for b in sorted_bonds]
    oas_vals = [b.oas for b in sorted_bonds if b.oas]
    coupon_vals = [b.coupon for b in sorted_bonds]
    sizes = [max(8, min(30, b.nominal_usd / 1e6 / 10)) for b in sorted_bonds]
    hover_texts = [
        f"<b>{b.isin}</b><br>到期: {b.maturity_date.strftime('%Y-%m-%d')}<br>"
        f"票息: {b.coupon:.2f}%<br>OAS: {b.oas:.0f}bp<br>"
        f"面值: ${b.nominal_usd/1e6:.0f}M<br>久期: {b.duration:.2f}"
        for b in sorted_bonds
    ]

    fig = go.Figure()

    # OAS scatter (bubble size = nominal)
    if oas_vals and len(oas_vals) == len(sorted_bonds):
        fig.add_trace(go.Scatter(
            x=tenors, y=oas_vals, mode='markers+lines',
            name='OAS (bp)',
            marker=dict(size=sizes, color=theme.accent_blue, opacity=0.8,
                        line=dict(width=1.5, color='white')),
            line=dict(color=theme.accent_blue, width=1.5, dash='dot'),
            text=hover_texts, hovertemplate="%{text}<extra></extra>",
        ))

    # Coupon scatter on secondary y-axis
    fig.add_trace(go.Scatter(
        x=tenors, y=coupon_vals, mode='markers',
        name='Coupon (%)',
        marker=dict(size=8, color=theme.accent_amber, symbol='diamond',
                    line=dict(width=1, color='white')),
        yaxis='y2',
        hovertemplate="期限: %{x:.1f}Y<br>票息: %{y:.2f}%<extra></extra>",
    ))

    # Fitted curve (simple interpolation for visual)
    if len(tenors) >= 3 and oas_vals and len(oas_vals) == len(sorted_bonds):
        t_smooth = np.linspace(min(tenors), max(tenors), 50)
        try:
            from scipy.interpolate import UnivariateSpline
            spline = UnivariateSpline(tenors, oas_vals, s=len(tenors) * 100, k=min(3, len(tenors) - 1))
            oas_smooth = spline(t_smooth)
            fig.add_trace(go.Scatter(
                x=t_smooth, y=oas_smooth, mode='lines',
                name='拟合曲线', line=dict(color=theme.accent_blue, width=2),
                hoverinfo='skip', opacity=0.4,
            ))
        except Exception:
            pass

    layout = get_nordic_layout("发行人收益率曲线 | Issuer Yield Curve", height=340)
    layout['xaxis']['title'] = dict(text="期限 (Years)", font=dict(size=11, color=theme.text_muted))
    layout['yaxis']['title'] = dict(text="OAS (bp)", font=dict(size=11, color=theme.accent_blue))
    layout['yaxis2'] = dict(
        title=dict(text="Coupon (%)", font=dict(size=11, color=theme.accent_amber)),
        overlaying='y', side='right', showgrid=False,
        tickfont=dict(color=theme.accent_amber, size=10),
    )
    layout['legend'] = dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10))
    fig.update_layout(**layout)
    return fig


def _create_issuer_oas_history(obligor_id: str, base_oas: float) -> go.Figure:
    """Create OAS history trend chart for an issuer."""
    theme = NordicTheme()
    np.random.seed(hash(obligor_id) % (2**31))

    # Generate 252 trading days of history
    days = 252
    dates = []
    oas_series = []
    oas = base_oas
    current = date.today()

    for i in range(days, 0, -1):
        d = current - timedelta(days=i)
        if d.weekday() < 5:
            change = np.random.normal(0, base_oas * 0.015)
            reversion = (base_oas - oas) * 0.03
            oas = max(10, oas + change + reversion)
            dates.append(d)
            oas_series.append(round(oas, 1))

    oas_arr = np.array(oas_series)
    mean_oas = float(np.mean(oas_arr))
    std_oas = float(np.std(oas_arr))
    current_oas = oas_series[-1] if oas_series else base_oas
    pctile = float(np.sum(oas_arr <= current_oas) / len(oas_arr) * 100) if oas_arr.size > 0 else 50

    # Moving averages
    ma_20 = pd.Series(oas_series).rolling(20).mean().tolist()
    ma_60 = pd.Series(oas_series).rolling(60).mean().tolist()

    fig = go.Figure()

    # Bollinger bands (mean +/- 2 std rolling)
    rolling_mean = pd.Series(oas_series).rolling(60).mean()
    rolling_std = pd.Series(oas_series).rolling(60).std()
    upper = (rolling_mean + 2 * rolling_std).tolist()
    lower = (rolling_mean - 2 * rolling_std).tolist()

    fig.add_trace(go.Scatter(
        x=dates, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=lower, mode='lines', line=dict(width=0), showlegend=False,
        fill='tonexty', fillcolor=f'{theme.accent_blue}10', hoverinfo='skip',
    ))

    # OAS line
    fig.add_trace(go.Scatter(
        x=dates, y=oas_series, mode='lines', name='OAS',
        line=dict(color=theme.accent_blue, width=2),
        hovertemplate="日期: %{x}<br>OAS: %{y:.0f}bp<extra></extra>",
    ))

    # Moving averages
    fig.add_trace(go.Scatter(
        x=dates, y=ma_20, mode='lines', name='MA20',
        line=dict(color=theme.accent_amber, width=1, dash='dot'),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=ma_60, mode='lines', name='MA60',
        line=dict(color=theme.accent_coral, width=1, dash='dash'),
    ))

    # Annotations
    fig.add_annotation(
        x=dates[-1], y=current_oas, text=f"  {current_oas:.0f}bp",
        showarrow=False, font=dict(size=12, color=theme.accent_blue, weight="bold"),
        xanchor='left',
    )

    layout = get_nordic_layout(f"利差走势 | OAS History  (当前 {current_oas:.0f}bp · {pctile:.0f}th %ile · μ={mean_oas:.0f} σ={std_oas:.0f})", height=320)
    layout['legend'] = dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10))
    fig.update_layout(**layout)
    return fig


def _create_issuer_maturity_wall(bonds: list[BondPosition]) -> go.Figure:
    """Create maturity wall chart for an issuer's bonds."""
    theme = NordicTheme()
    if not bonds:
        return go.Figure()

    # Group by year
    year_buckets: dict[int, float] = {}
    for bond in bonds:
        yr = bond.maturity_date.year
        year_buckets[yr] = year_buckets.get(yr, 0) + bond.nominal_usd

    years = sorted(year_buckets.keys())
    values = [year_buckets[y] / 1e6 for y in years]

    # Color: near-term = warm, far-term = cool
    colors = []
    for y in years:
        diff = y - date.today().year
        if diff <= 1:
            colors.append(theme.accent_coral)
        elif diff <= 2:
            colors.append(theme.accent_amber)
        elif diff <= 5:
            colors.append(theme.accent_blue)
        else:
            colors.append(theme.accent_green)

    fig = go.Figure(data=[go.Bar(
        x=[str(y) for y in years], y=values,
        marker_color=colors,
        text=[f"${v:.0f}M" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=theme.text_secondary),
        hovertemplate="年份: %{x}<br>到期面值: $%{y:.0f}M<extra></extra>",
    )])
    layout = get_nordic_layout("到期墙 | Maturity Wall", height=280)
    layout['bargap'] = 0.35
    fig.update_layout(**layout)
    return fig


def _create_peer_comparison_chart(obligor: Obligor, exp: "CreditExposure", exposures: list["CreditExposure"]) -> go.Figure:
    """Create peer comparison chart (OAS vs peers in same sector/rating bucket)."""
    theme = NordicTheme()

    # Find peers: same sector or adjacent rating
    peers = []
    for e in exposures:
        if e.obligor.obligor_id == obligor.obligor_id:
            continue
        if e.total_market_usd <= 0:
            continue
        same_sector = e.obligor.sector == obligor.sector
        rating_diff = abs(RATING_SCORE.get(e.obligor.rating_internal, 50) - RATING_SCORE.get(obligor.rating_internal, 50))
        same_rating_bucket = rating_diff <= 15
        if same_sector or same_rating_bucket:
            peers.append(e)

    # Sort by OAS, take top 10
    peers = sorted(peers, key=lambda e: e.weighted_avg_oas, reverse=True)[:10]

    # Add the target obligor
    all_issuers = peers + [exp]
    all_issuers = sorted(all_issuers, key=lambda e: e.weighted_avg_oas, reverse=True)

    names = []
    oas_vals = []
    colors = []
    for e in all_issuers:
        label = e.obligor.name_cn[:8]
        names.append(label)
        oas_vals.append(e.weighted_avg_oas)
        if e.obligor.obligor_id == obligor.obligor_id:
            colors.append(theme.accent_blue)
        else:
            colors.append(theme.bg_tertiary)

    fig = go.Figure(data=[go.Bar(
        y=names, x=oas_vals, orientation='h',
        marker_color=colors,
        marker_line=dict(width=[2 if c == theme.accent_blue else 0 for c in colors], color=theme.accent_blue),
        text=[f"{v:.0f}bp" for v in oas_vals],
        textposition='outside',
        textfont=dict(size=10, color=theme.text_secondary),
        hovertemplate="<b>%{y}</b><br>OAS: %{x:.0f}bp<extra></extra>",
    )])
    layout = get_nordic_layout("同业比较 | Peer Comparison (OAS)", height=max(250, len(all_issuers) * 28))
    layout['bargap'] = 0.3
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")
    return fig


def _create_sentiment_timeline(news_items: list[NewsItem]) -> go.Figure:
    """Create sentiment timeline for issuer-related news."""
    theme = NordicTheme()
    if not news_items:
        return go.Figure()

    sorted_news = sorted(news_items, key=lambda n: n.timestamp)
    times = [n.timestamp for n in sorted_news]
    scores = [n.sentiment_score if n.sentiment_score is not None else 0 for n in sorted_news]
    colors = []
    for s in scores:
        if s > 0.2:
            colors.append(theme.accent_green)
        elif s < -0.2:
            colors.append(theme.accent_coral)
        else:
            colors.append(theme.text_muted)

    hover_texts = [f"<b>{n.title[:30]}...</b><br>来源: {n.source}<br>情感: {n.sentiment_score or 0:.2f}" for n in sorted_news]

    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=theme.border_light, line_width=1)

    # Area fill
    fig.add_trace(go.Scatter(
        x=times, y=scores, mode='lines',
        line=dict(color=theme.accent_blue, width=1.5),
        fill='tozeroy', fillcolor=f'{theme.accent_blue}15',
        hoverinfo='skip', showlegend=False,
    ))

    # Dots
    fig.add_trace(go.Scatter(
        x=times, y=scores, mode='markers',
        marker=dict(size=10, color=colors, line=dict(width=1.5, color='white')),
        text=hover_texts, hovertemplate="%{text}<extra></extra>",
        name='新闻情感',
    ))

    layout = get_nordic_layout("新闻情感走势 | Sentiment Timeline", height=240)
    layout['yaxis']['range'] = [-1.1, 1.1]
    layout['yaxis']['title'] = dict(text="Sentiment", font=dict(size=10, color=theme.text_muted))
    layout['showlegend'] = False
    fig.update_layout(**layout)
    return fig


def render_issuer_page():
    """发行人深度分析页面 | Issuer Deep-Dive (Aladdin-inspired)"""
    theme = NordicTheme()
    exposures = st.session_state.exposures
    obligors = st.session_state.obligors
    alerts = st.session_state.alerts
    news_items = st.session_state.news

    # ── Issuer Selector ──────────────────────────────────────────────
    sorted_exposures = sorted(exposures, key=lambda x: x.total_market_usd, reverse=True)
    issuer_options = {e.obligor.obligor_id: f"{e.obligor.name_cn} ({e.obligor.rating_internal.value}) · ${e.total_market_usd/1e6:.0f}M" for e in sorted_exposures if e.total_market_usd > 0}

    selected_id = st.selectbox("选择发行人 | Select Issuer", options=list(issuer_options.keys()), format_func=lambda x: issuer_options[x])

    if not selected_id:
        return

    exp = next((e for e in exposures if e.obligor.obligor_id == selected_id), None)
    if not exp:
        return

    obligor = exp.obligor
    rating_color = NordicTheme.get_rating_color(obligor.rating_internal.value)
    sector_color = NordicTheme.get_sector_color(obligor.sector.value)
    outlook_map = {"POSITIVE": ("↑", theme.accent_green), "STABLE": ("→", theme.text_muted), "NEGATIVE": ("↓", theme.accent_coral), "WATCH_NEG": ("⚠", theme.severity_critical)}
    outlook_icon, outlook_color = outlook_map.get(obligor.rating_outlook.value, ("", theme.text_muted))
    issuer_alerts = [a for a in alerts if a.obligor_id == selected_id]
    issuer_news = [n for n in news_items if selected_id in n.obligor_ids]
    risk_scores = _compute_issuer_risk_scores(obligor, exp, alerts, news_items, exposures)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: Header Card + Risk Gauge
    # ══════════════════════════════════════════════════════════════════
    col_header, col_gauge, col_exposure = st.columns([5, 2, 3])

    with col_header:
        # Sector tag
        sector_map = {"LGFV": "城投", "SOE": "央企", "FINANCIAL": "金融", "G-SIB": "G-SIB", "US_CORP": "美企", "EU_CORP": "欧企", "EM_SOVEREIGN": "新兴主权", "SUPRA": "超主权", "HY": "高收益", "CORP": "企业"}
        sector_label = sector_map.get(obligor.sector.value, obligor.sector.value)
        region_map = {"CHINA_OFFSHORE": "中国离岸", "CHINA_ONSHORE": "中国在岸", "US": "美国", "EU": "欧洲", "UK": "英国", "JAPAN": "日本", "LATAM": "拉美", "CEEMEA": "中东欧", "ASIA_EX_CHINA": "亚太", "SUPRANATIONAL": "超主权"}
        region_label = region_map.get(obligor.region.value, obligor.region.value)

        # External ratings (simulated for demo)
        ext_ratings_html = ""
        rating_val = obligor.rating_internal.value
        ext_map = {
            "AAA": ("Aaa", "AAA", "AAA"), "AA+": ("Aa1", "AA+", "AA+"), "AA": ("Aa2", "AA", "AA"),
            "AA-": ("Aa3", "AA-", "AA-"), "A+": ("A1", "A+", "A+"), "A": ("A2", "A", "A"),
            "A-": ("A3", "A-", "A-"), "BBB+": ("Baa1", "BBB+", "BBB+"), "BBB": ("Baa2", "BBB", "BBB"),
            "BBB-": ("Baa3", "BBB-", "BBB-"), "BB+": ("Ba1", "BB+", "BB+"), "BB": ("Ba2", "BB", "BB"),
            "BB-": ("Ba3", "BB-", "BB-"), "B+": ("B1", "B+", "B+"), "B": ("B2", "B", "B"),
            "B-": ("B3", "B-", "B-"), "CCC": ("Caa1", "CCC+", "CCC"),
        }
        ext = ext_map.get(rating_val, (rating_val, rating_val, rating_val))
        ext_ratings_html = f"""
        <div style="display:flex;gap:8px;margin-top:12px;">
            <span style="background:{theme.bg_tertiary};padding:3px 8px;border-radius:4px;font-size:10px;color:{theme.text_secondary};font-weight:500;">Moody's: {ext[0]}</span>
            <span style="background:{theme.bg_tertiary};padding:3px 8px;border-radius:4px;font-size:10px;color:{theme.text_secondary};font-weight:500;">S&P: {ext[1]}</span>
            <span style="background:{theme.bg_tertiary};padding:3px 8px;border-radius:4px;font-size:10px;color:{theme.text_secondary};font-weight:500;">Fitch: {ext[2]}</span>
        </div>"""

        alert_badges = ""
        crit_count = sum(1 for a in issuer_alerts if a.severity == Severity.CRITICAL)
        warn_count = sum(1 for a in issuer_alerts if a.severity == Severity.WARNING)
        if crit_count:
            alert_badges += f'<span style="background:{theme.severity_critical};color:white;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;margin-left:8px;">{crit_count} CRITICAL</span>'
        if warn_count:
            alert_badges += f'<span style="background:{theme.severity_warning};color:white;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;margin-left:4px;">{warn_count} WARNING</span>'

        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:24px 28px;border-left:4px solid {sector_color};box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div style="flex:1;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <h2 style="color:{theme.text_primary};margin:0;font-weight:600;font-size:22px;letter-spacing:-0.01em;">{obligor.name_cn}</h2>
                        {alert_badges}
                    </div>
                    <p style="color:{theme.text_muted};margin:4px 0 0 0;font-size:13px;">{obligor.name_en or ''}{(' · ' + obligor.ticker) if obligor.ticker else ''}</p>
                    {ext_ratings_html}
                </div>
                <div style="text-align:center;margin-left:20px;">
                    <span style="background:linear-gradient(135deg, {rating_color}, {rating_color}CC);color:white;padding:12px 20px;border-radius:12px;font-weight:700;font-size:22px;letter-spacing:0.02em;box-shadow:0 3px 8px {rating_color}40;display:inline-block;">{obligor.rating_internal.value}</span>
                    <p style="color:{outlook_color};margin:8px 0 0 0;font-size:12px;font-weight:600;">{outlook_icon} {obligor.rating_outlook.value}</p>
                </div>
            </div>
            <div style="margin-top:18px;display:flex;gap:28px;flex-wrap:wrap;padding-top:14px;border-top:1px solid {theme.bg_tertiary};">
                <div><span style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">行业</span><br><span style="color:{sector_color};font-size:13px;font-weight:600;">{sector_label}</span></div>
                <div><span style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">子行业</span><br><span style="color:{theme.text_primary};font-size:13px;font-weight:500;">{obligor.sub_sector}</span></div>
                <div><span style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">区域</span><br><span style="color:{theme.text_primary};font-size:13px;font-weight:500;">{region_label}</span></div>
                <div><span style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">国家</span><br><span style="color:{theme.text_primary};font-size:13px;font-weight:500;">{obligor.country or 'N/A'}</span></div>
                <div><span style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">债券数</span><br><span style="color:{theme.text_primary};font-size:13px;font-weight:500;">{len(exp.bonds)}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        gauge_svg = _render_risk_gauge_svg(risk_scores["composite"], size=150)
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:16px 12px;box-shadow:0 1px 3px rgba(0,0,0,0.04);text-align:center;height:100%;">
            <div style="font-size:10px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">COMPOSITE RISK</div>
            {gauge_svg}
        </div>
        """, unsafe_allow_html=True)

    with col_exposure:
        # KPI metrics
        total_dv01 = exp.credit_dv01_usd
        avg_coupon = np.mean([b.coupon for b in exp.bonds]) if exp.bonds else 0
        avg_ytm = np.mean([b.years_to_maturity for b in exp.bonds]) if exp.bonds else 0
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:10px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:16px;">EXPOSURE SUMMARY</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
                <div>
                    <div style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;">市值</div>
                    <div style="color:{theme.text_primary};font-size:22px;font-weight:700;letter-spacing:-0.02em;">${exp.total_market_usd/1e6:,.0f}<span style="font-size:12px;font-weight:400;color:{theme.text_muted};">M</span></div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;">占AUM</div>
                    <div style="color:{theme.text_primary};font-size:22px;font-weight:700;letter-spacing:-0.02em;">{exp.pct_of_aum:.2%}</div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;">加权久期</div>
                    <div style="color:{theme.accent_blue};font-size:20px;font-weight:700;">{exp.weighted_avg_duration:.2f}<span style="font-size:11px;font-weight:400;">Y</span></div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;">加权OAS</div>
                    <div style="color:{theme.accent_blue};font-size:20px;font-weight:700;">{exp.weighted_avg_oas:.0f}<span style="font-size:11px;font-weight:400;">bp</span></div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;">Credit DV01</div>
                    <div style="color:{theme.accent_purple};font-size:20px;font-weight:700;">${total_dv01/1e3:,.0f}<span style="font-size:11px;font-weight:400;">K</span></div>
                </div>
                <div>
                    <div style="color:{theme.text_muted};font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;">平均票息</div>
                    <div style="color:{theme.accent_amber};font-size:20px;font-weight:700;">{avg_coupon:.2f}<span style="font-size:11px;font-weight:400;">%</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: Risk Score Breakdown Bar
    # ══════════════════════════════════════════════════════════════════
    score_cats = [
        ("信用质量", "credit", theme.accent_green),
        ("利差水平", "spread", theme.accent_blue),
        ("集中度", "concentration", theme.accent_purple),
        ("新闻情感", "news", theme.accent_amber),
        ("预警状态", "alerts", theme.accent_coral),
        ("流动性", "liquidity", theme.accent_teal),
    ]
    bars_html = ""
    for label, key, color in score_cats:
        val = risk_scores[key]
        bar_width = max(2, val)
        bars_html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:68px;font-size:11px;color:{theme.text_secondary};font-weight:500;text-align:right;">{label}</div>
            <div style="flex:1;background:{theme.bg_tertiary};border-radius:4px;height:18px;overflow:hidden;">
                <div style="width:{bar_width}%;height:100%;background:linear-gradient(90deg, {color}CC, {color});border-radius:4px;transition:width 0.5s ease;"></div>
            </div>
            <div style="width:36px;font-size:12px;font-weight:600;color:{theme.text_primary};text-align:right;">{val:.0f}</div>
        </div>"""

    st.markdown(f"""
    <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:18px 24px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-size:12px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">RISK SCORE BREAKDOWN</span>
            <span style="font-size:11px;color:{theme.text_muted};">Composite: <b style="color:{theme.text_primary};font-size:14px;">{risk_scores['composite']:.0f}</b>/100</span>
        </div>
        {bars_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: Tabbed Deep-Dive
    # ══════════════════════════════════════════════════════════════════
    tab_portfolio, tab_market, tab_news, tab_signals = st.tabs([
        "债券组合 | Portfolio",
        "市场 & 利差 | Market",
        "新闻情报 | News Intel",
        "风险信号 | Signals",
    ])

    # ── Tab 1: Bond Portfolio ─────────────────────────────────────────
    with tab_portfolio:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        col_curve, col_wall = st.columns([3, 2])
        with col_curve:
            fig_curve = _create_issuer_yield_curve(exp.bonds)
            st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': False})
        with col_wall:
            fig_wall = _create_issuer_maturity_wall(exp.bonds)
            st.plotly_chart(fig_wall, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Bond Holdings Table
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:14px;font-weight:600;color:{theme.text_primary};">债券持仓明细</span>
            <span style="font-size:11px;color:{theme.text_muted};">{len(exp.bonds)} bonds · ${exp.total_nominal_usd/1e6:,.0f}M nominal</span>
        </div>
        """, unsafe_allow_html=True)

        bond_data = []
        for bond in sorted(exp.bonds, key=lambda b: b.maturity_date):
            ytm = bond.years_to_maturity
            ytm_color = theme.accent_coral if ytm < 1 else (theme.accent_amber if ytm < 2 else theme.text_primary)
            bond_data.append({
                "ISIN": bond.isin,
                "债券名称": (bond.bond_name or "")[:25],
                "货币": bond.currency,
                "票息": f"{bond.coupon:.2f}%",
                "到期日": bond.maturity_date.strftime("%Y-%m-%d"),
                "剩余 (Y)": f"{ytm:.1f}",
                "面值 ($M)": f"{bond.nominal_usd/1e6:,.1f}",
                "市值 ($M)": f"{bond.market_value_usd/1e6:,.1f}",
                "久期": f"{bond.duration:.2f}",
                "OAS (bp)": f"{bond.oas:.0f}" if bond.oas else "-",
                "DV01 ($K)": f"{bond.credit_dv01/1e3:,.1f}",
            })
        if bond_data:
            render_styled_table(pd.DataFrame(bond_data), max_height="360px")

    # ── Tab 2: Market & Spread ────────────────────────────────────────
    with tab_market:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # OAS History
        fig_oas = _create_issuer_oas_history(selected_id, exp.weighted_avg_oas)
        st.plotly_chart(fig_oas, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        col_peer, col_radar = st.columns([3, 2])
        with col_peer:
            fig_peer = _create_peer_comparison_chart(obligor, exp, exposures)
            st.plotly_chart(fig_peer, use_container_width=True, config={'displayModeBar': False})

        with col_radar:
            # Risk radar chart
            categories = ['信用', '利差', '集中度', '情感', '预警', '流动性']
            values = [risk_scores['credit'], risk_scores['spread'], risk_scores['concentration'],
                      risk_scores['news'], risk_scores['alerts'], risk_scores['liquidity']]
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed, theta=categories_closed,
                fill='toself',
                fillcolor=f'{theme.accent_blue}20',
                line=dict(color=theme.accent_blue, width=2),
                marker=dict(size=6, color=theme.accent_blue),
                name='Risk Profile',
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                                    tickfont=dict(size=9, color=theme.text_muted),
                                    gridcolor=theme.border_light),
                    angularaxis=dict(tickfont=dict(size=11, color=theme.text_secondary),
                                     gridcolor=theme.border_light),
                    bgcolor=theme.bg_secondary,
                ),
                paper_bgcolor=theme.bg_secondary,
                height=320,
                margin=dict(l=60, r=60, t=40, b=40),
                title=dict(text="风险画像 | Risk Radar", font=dict(size=14, color=theme.text_primary), x=0.02),
                showlegend=False,
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # ── Tab 3: News Intelligence ──────────────────────────────────────
    with tab_news:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if issuer_news:
            # Sentiment summary bar
            pos_count = sum(1 for n in issuer_news if n.sentiment == Sentiment.POSITIVE)
            neg_count = sum(1 for n in issuer_news if n.sentiment == Sentiment.NEGATIVE)
            neu_count = len(issuer_news) - pos_count - neg_count
            total_n = len(issuer_news)
            avg_sent = np.mean([n.sentiment_score for n in issuer_news if n.sentiment_score is not None] or [0])

            st.markdown(f"""
            <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:12px;padding:16px 24px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <span style="font-size:12px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">NEWS SENTIMENT OVERVIEW</span>
                    <span style="font-size:11px;color:{theme.text_muted};">{total_n} articles · Avg: <b style="color:{'#4CAF7C' if avg_sent > 0.1 else ('#E57373' if avg_sent < -0.1 else theme.text_primary)};">{avg_sent:.2f}</b></span>
                </div>
                <div style="display:flex;gap:16px;align-items:center;">
                    <div style="display:flex;gap:4px;flex:1;height:8px;border-radius:4px;overflow:hidden;">
                        <div style="width:{pos_count/total_n*100:.0f}%;background:{theme.accent_green};border-radius:4px 0 0 4px;"></div>
                        <div style="width:{neu_count/total_n*100:.0f}%;background:{theme.border_light};"></div>
                        <div style="width:{neg_count/total_n*100:.0f}%;background:{theme.accent_coral};border-radius:0 4px 4px 0;"></div>
                    </div>
                    <div style="display:flex;gap:12px;font-size:11px;">
                        <span style="color:{theme.accent_green};font-weight:500;">+{pos_count}</span>
                        <span style="color:{theme.text_muted};">~{neu_count}</span>
                        <span style="color:{theme.accent_coral};font-weight:500;">-{neg_count}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sentiment Timeline
            fig_sent = _create_sentiment_timeline(issuer_news)
            st.plotly_chart(fig_sent, use_container_width=True, config={'displayModeBar': False})

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # News Cards (enriched)
            for news in sorted(issuer_news, key=lambda n: n.timestamp, reverse=True):
                sentiment_color = {
                    Sentiment.POSITIVE: theme.accent_green,
                    Sentiment.NEUTRAL: theme.text_muted,
                    Sentiment.NEGATIVE: theme.accent_coral,
                }.get(news.sentiment, theme.text_muted)
                sentiment_label = {
                    Sentiment.POSITIVE: "POSITIVE",
                    Sentiment.NEUTRAL: "NEUTRAL",
                    Sentiment.NEGATIVE: "NEGATIVE",
                }.get(news.sentiment, "N/A")
                score_val = news.sentiment_score if news.sentiment_score is not None else 0
                time_str = news.timestamp.strftime("%m-%d %H:%M")

                # Determine impact indicator
                impact_bar = ""
                if abs(score_val) > 0.6:
                    impact_bar = f'<span style="background:{theme.severity_critical}20;color:{theme.severity_critical};padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;margin-left:6px;">HIGH IMPACT</span>'
                elif abs(score_val) > 0.3:
                    impact_bar = f'<span style="background:{theme.severity_warning}20;color:{theme.severity_warning};padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;margin-left:6px;">MEDIUM</span>'

                st.markdown(f"""
                <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-left:4px solid {sentiment_color};padding:16px 20px;margin:8px 0;border-radius:0 10px 10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div style="flex:1;">
                            <div style="display:flex;align-items:center;gap:6px;">
                                <span style="color:{theme.text_primary};font-weight:600;font-size:14px;line-height:1.4;">{news.title}</span>
                            </div>
                            <div style="display:flex;gap:10px;margin-top:6px;align-items:center;">
                                <span style="background:{theme.bg_tertiary};padding:2px 8px;border-radius:4px;font-size:10px;color:{theme.text_secondary};font-weight:500;">{news.source}</span>
                                <span style="font-size:10px;color:{theme.text_muted};">{time_str}</span>
                                <span style="background:{sentiment_color}18;color:{sentiment_color};padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">{sentiment_label} {score_val:+.2f}</span>
                                {impact_bar}
                            </div>
                        </div>
                    </div>
                    {f'<div style="color:{theme.text_secondary};margin-top:10px;font-size:12px;line-height:1.6;padding:10px 12px;background:{theme.bg_primary};border-radius:6px;border:1px solid {theme.bg_tertiary};">💡 <b>AI Summary:</b> {news.summary}</div>' if news.summary else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:{theme.text_muted};">
                <div style="font-size:36px;margin-bottom:12px;">📰</div>
                <div style="font-size:14px;">暂无该发行人相关新闻</div>
                <div style="font-size:12px;margin-top:4px;">No news available for this issuer</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 4: Risk Signals & Relationships ───────────────────────────
    with tab_signals:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        col_alerts, col_rel = st.columns([3, 2])

        with col_alerts:
            st.markdown(f"""
            <div style="font-size:14px;font-weight:600;color:{theme.text_primary};margin-bottom:12px;">活跃预警 | Active Alerts</div>
            """, unsafe_allow_html=True)

            if issuer_alerts:
                for alert in sorted(issuer_alerts, key=lambda a: (0 if a.severity == Severity.CRITICAL else 1, -a.timestamp.timestamp())):
                    severity_color = NordicTheme.get_severity_color(alert.severity.value)
                    severity_icon = {"CRITICAL": "●", "WARNING": "▲", "INFO": "○"}.get(alert.severity.value, "○")
                    category_map = {"CONCENTRATION": "集中度", "RATING": "评级", "SPREAD": "利差", "NEWS": "新闻"}
                    status_map = {"PENDING": "待处理", "INVESTIGATING": "调查中", "RESOLVED": "已解决", "DISMISSED": "已忽略", "ESCALATED": "已升级"}
                    cat_label = category_map.get(alert.category.value, alert.category.value)
                    status_label = status_map.get(alert.status.value, alert.status.value)

                    st.markdown(f"""
                    <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-left:4px solid {severity_color};padding:14px 18px;margin:8px 0;border-radius:0 10px 10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="display:flex;align-items:center;gap:8px;">
                                <span style="color:{severity_color};font-size:14px;">{severity_icon}</span>
                                <span style="color:{theme.text_primary};font-weight:500;font-size:13px;">{alert.message}</span>
                            </div>
                            <div style="display:flex;gap:8px;align-items:center;">
                                <span style="background:{theme.bg_tertiary};padding:2px 8px;border-radius:4px;font-size:10px;color:{theme.text_secondary};font-weight:500;">{cat_label}</span>
                                <span style="background:{theme.bg_tertiary};padding:2px 8px;border-radius:4px;font-size:10px;color:{theme.text_secondary};font-weight:500;">{status_label}</span>
                            </div>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-top:8px;">
                            <span style="color:{theme.text_muted};font-size:11px;">{alert.timestamp.strftime('%Y-%m-%d %H:%M')}</span>
                            <span style="color:{theme.text_muted};font-size:11px;">指标: {alert.metric_value:.2f} / 阈值: {alert.threshold:.2f}</span>
                        </div>
                        {f'<div style="color:{theme.text_secondary};margin-top:10px;font-size:12px;line-height:1.5;padding:8px 12px;background:{theme.bg_primary};border-radius:6px;border:1px solid {theme.bg_tertiary};">🤖 {alert.ai_summary}</div>' if alert.ai_summary else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align:center;padding:40px 20px;color:{theme.accent_green};background:{theme.accent_green}08;border-radius:10px;border:1px solid {theme.accent_green}20;">
                    <div style="font-size:24px;margin-bottom:8px;">✓</div>
                    <div style="font-size:13px;font-weight:500;">无活跃预警</div>
                    <div style="font-size:11px;color:{theme.text_muted};margin-top:4px;">All signals within normal range</div>
                </div>
                """, unsafe_allow_html=True)

        with col_rel:
            # Relationship Map
            st.markdown(f"""
            <div style="font-size:14px;font-weight:600;color:{theme.text_primary};margin-bottom:12px;">关联图谱 | Relationship Map</div>
            """, unsafe_allow_html=True)

            relationships = OBLIGOR_RELATIONSHIPS
            related_entities = []

            # Find relationships where this obligor is involved
            if selected_id in relationships:
                rel = relationships[selected_id]
                if "parent" in rel:
                    parent_id = rel["parent"]
                    parent_ob = obligors.get(parent_id)
                    if parent_ob:
                        related_entities.append(("parent", parent_ob, rel.get("relationship", "母公司"), rel.get("support_level", "")))
                if "guarantor" in rel:
                    g_id = rel["guarantor"]
                    g_ob = obligors.get(g_id)
                    if g_ob:
                        related_entities.append(("guarantor", g_ob, rel.get("relationship", "担保"), ""))

            # Find children
            for oid, rel in relationships.items():
                if rel.get("parent") == selected_id:
                    child_ob = obligors.get(oid)
                    if child_ob:
                        related_entities.append(("child", child_ob, rel.get("relationship", "子公司"), rel.get("support_level", "")))
                if rel.get("guarantor") == selected_id:
                    g_ob = obligors.get(oid)
                    if g_ob:
                        related_entities.append(("guaranteed", g_ob, rel.get("relationship", "被担保"), ""))

            if related_entities:
                for rel_type, rel_ob, rel_label, support in related_entities:
                    icon = {"parent": "⬆", "child": "⬇", "guarantor": "🛡", "guaranteed": "🔗"}.get(rel_type, "·")
                    type_label = {"parent": "母公司", "child": "子公司", "guarantor": "担保方", "guaranteed": "被担保方"}.get(rel_type, rel_label)
                    rel_rating_color = NordicTheme.get_rating_color(rel_ob.rating_internal.value)
                    support_html = f'<span style="background:{theme.bg_tertiary};padding:1px 6px;border-radius:3px;font-size:9px;color:{theme.text_muted};margin-left:4px;">支持: {support}</span>' if support else ""

                    st.markdown(f"""
                    <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};padding:12px 16px;margin:6px 0;border-radius:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="margin-right:6px;">{icon}</span>
                                <span style="color:{theme.text_primary};font-weight:500;font-size:13px;">{rel_ob.name_cn}</span>
                                {support_html}
                            </div>
                            <span style="background:{rel_rating_color};color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">{rel_ob.rating_internal.value}</span>
                        </div>
                        <div style="margin-top:6px;display:flex;gap:8px;">
                            <span style="font-size:10px;color:{theme.text_muted};background:{theme.bg_tertiary};padding:2px 6px;border-radius:3px;">{type_label}</span>
                            <span style="font-size:10px;color:{theme.text_muted};background:{theme.bg_tertiary};padding:2px 6px;border-radius:3px;">{rel_ob.sector.value}</span>
                            <span style="font-size:10px;color:{theme.text_muted};">{rel_ob.rating_outlook.value}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Related group total exposure
                group_ids = [selected_id] + [e[1].obligor_id for e in related_entities]
                group_exposure = sum(e.total_market_usd for e in exposures if e.obligor.obligor_id in group_ids)
                group_pct = group_exposure / 50e9

                st.markdown(f"""
                <div style="background:{theme.accent_blue}08;border:1px solid {theme.accent_blue}20;border-radius:8px;padding:12px 16px;margin-top:12px;">
                    <div style="font-size:10px;color:{theme.text_muted};text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">GROUP EXPOSURE</div>
                    <div style="display:flex;justify-content:space-between;margin-top:6px;">
                        <span style="color:{theme.text_primary};font-size:18px;font-weight:700;">${group_exposure/1e6:,.0f}M</span>
                        <span style="color:{theme.accent_blue};font-size:14px;font-weight:600;">{group_pct:.2%} AUM</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align:center;padding:40px 20px;color:{theme.text_muted};background:{theme.bg_tertiary};border-radius:10px;">
                    <div style="font-size:24px;margin-bottom:8px;">🔗</div>
                    <div style="font-size:12px;">暂无已知关联实体</div>
                    <div style="font-size:11px;margin-top:4px;">No known relationships</div>
                </div>
                """, unsafe_allow_html=True)


def render_alerts_page():
    """预警中心页面"""
    theme = NordicTheme()
    alerts = st.session_state.alerts

    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        critical = len([a for a in alerts if a.severity == Severity.CRITICAL])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:24px 20px;text-align:center;border-top:3px solid {theme.severity_critical};box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:36px;font-weight:700;color:{theme.severity_critical};letter-spacing:-0.02em;">{critical}</div>
            <div style="font-size:11px;color:{theme.text_muted};margin-top:6px;text-transform:uppercase;letter-spacing:0.06em;font-weight:500;">严重预警</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        warning = len([a for a in alerts if a.severity == Severity.WARNING])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:24px 20px;text-align:center;border-top:3px solid {theme.severity_warning};box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:36px;font-weight:700;color:{theme.severity_warning};letter-spacing:-0.02em;">{warning}</div>
            <div style="font-size:11px;color:{theme.text_muted};margin-top:6px;text-transform:uppercase;letter-spacing:0.06em;font-weight:500;">警告</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        pending = len([a for a in alerts if a.status == AlertStatus.PENDING])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:24px 20px;text-align:center;border-top:3px solid {theme.accent_blue};box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:36px;font-weight:700;color:{theme.accent_blue};letter-spacing:-0.02em;">{pending}</div>
            <div style="font-size:11px;color:{theme.text_muted};margin-top:6px;text-transform:uppercase;letter-spacing:0.06em;font-weight:500;">待处理</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        resolved = len([a for a in alerts if a.status == AlertStatus.RESOLVED])
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:14px;padding:24px 20px;text-align:center;border-top:3px solid {theme.accent_green};box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:36px;font-weight:700;color:{theme.accent_green};letter-spacing:-0.02em;">{resolved}</div>
            <div style="font-size:11px;color:{theme.text_muted};margin-top:6px;text-transform:uppercase;letter-spacing:0.06em;font-weight:500;">已解决</div>
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

    # Custom CSS - Nordic Minimal Style (Premium)
    st.markdown(f"""
    <style>
    /* ===== Force Light Mode ===== */
    :root {{
        color-scheme: light !important;
    }}

    /* ===== Base Layout ===== */
    .stApp {{
        background-color: {theme.bg_primary};
        font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {{
        background-color: {theme.bg_sidebar};
        border-right: 1px solid {theme.border_light};
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {theme.text_secondary};
    }}
    [data-testid="stSidebar"] .stRadio > label {{
        color: {theme.text_secondary};
    }}
    [data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {{
        display: none;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        padding: 8px 12px;
        border-radius: 8px;
        margin: 2px 0;
        transition: background-color 0.15s ease;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background-color: {theme.bg_tertiary};
    }}

    /* ===== Typography ===== */
    h1 {{
        color: {theme.text_primary} !important;
        font-weight: 600 !important;
        font-size: 1.6rem !important;
        letter-spacing: -0.02em;
    }}
    h2, h3 {{
        color: {theme.text_primary} !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em;
    }}
    p, span, div {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    /* ===== Metrics ===== */
    [data-testid="stMetricValue"] {{
        color: {theme.text_primary};
        font-weight: 600;
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme.text_muted};
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }}

    /* ===== DataFrames / Tables - CRITICAL FIX ===== */
    .stDataFrame {{
        border: 1px solid {theme.border_light};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Force light background on all dataframe internals */
    .stDataFrame [data-testid="stDataFrameResizable"],
    .stDataFrame iframe {{
        background-color: {theme.bg_secondary} !important;
        border-radius: 10px;
    }}

    /* Glide Data Editor (Streamlit's internal table renderer) */
    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background: {theme.bg_secondary} !important;
    }}

    /* Override dark theme inside dataframe canvas */
    .stDataFrame div[class*="glideDataEditor"],
    .stDataFrame div[data-testid="glideDataEditor"] {{
        background: {theme.bg_secondary} !important;
    }}

    /* Table cell backgrounds */
    .stDataFrame .dvn-scroller {{
        background: {theme.bg_secondary} !important;
    }}

    /* Header row styling */
    .stDataFrame [data-testid="StyledDataFrameHeaderCell"],
    .stDataFrame th {{
        background-color: {theme.bg_tertiary} !important;
        color: {theme.text_primary} !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 2px solid {theme.border_light} !important;
    }}

    /* Data cells */
    .stDataFrame [data-testid="StyledDataFrameCell"],
    .stDataFrame td {{
        background-color: {theme.bg_secondary} !important;
        color: {theme.text_primary} !important;
        font-size: 13px !important;
        border-bottom: 1px solid {theme.bg_tertiary} !important;
    }}

    /* Alternating row colors */
    .stDataFrame [data-testid="StyledDataFrameCell"]:nth-child(even) {{
        background-color: {theme.bg_primary} !important;
    }}

    /* DataEditor toolbar */
    .stDataFrame [data-testid="stElementToolbar"] {{
        background-color: {theme.bg_secondary} !important;
        border-color: {theme.border_light} !important;
    }}

    /* Column config dropdown */
    .stDataFrame [data-testid="column-header-cell"] {{
        background-color: {theme.bg_tertiary} !important;
        color: {theme.text_primary} !important;
    }}

    /* ===== Selectbox / Inputs ===== */
    .stSelectbox > div > div {{
        background-color: {theme.bg_secondary};
        border-color: {theme.border_light};
        border-radius: 8px;
    }}
    .stSelectbox [data-baseweb="select"] {{
        background-color: {theme.bg_secondary} !important;
    }}
    .stMultiSelect [data-baseweb="select"] {{
        background-color: {theme.bg_secondary} !important;
        border-color: {theme.border_light} !important;
    }}

    /* ===== Buttons ===== */
    .stButton > button {{
        background-color: {theme.bg_secondary};
        color: {theme.text_primary};
        border: 1px solid {theme.border_light};
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        padding: 8px 16px;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: {theme.bg_tertiary};
        border-color: {theme.accent_blue};
        color: {theme.accent_blue};
    }}

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: transparent;
        border-bottom: 1px solid {theme.border_light};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {theme.text_muted};
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-size: 13px;
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {theme.text_primary};
        background-color: {theme.bg_tertiary};
    }}
    .stTabs [aria-selected="true"] {{
        color: {theme.accent_blue} !important;
        border-bottom: 2px solid {theme.accent_blue} !important;
        background-color: transparent !important;
    }}

    /* ===== Chat ===== */
    [data-testid="stChatMessage"] {{
        background-color: {theme.bg_secondary} !important;
        border: 1px solid {theme.border_light};
        border-radius: 12px;
        padding: 16px;
    }}
    .stChatInput > div {{
        border-color: {theme.border_light} !important;
        border-radius: 12px !important;
    }}
    .stChatInput textarea {{
        background-color: {theme.bg_secondary} !important;
        color: {theme.text_primary} !important;
    }}

    /* ===== Expander ===== */
    .streamlit-expanderHeader {{
        background-color: {theme.bg_secondary} !important;
        border: 1px solid {theme.border_light};
        border-radius: 8px;
        color: {theme.text_primary} !important;
        font-weight: 500;
    }}
    .streamlit-expanderContent {{
        background-color: {theme.bg_secondary} !important;
        border: 1px solid {theme.border_light};
        border-top: none;
        border-radius: 0 0 8px 8px;
    }}

    /* ===== Alerts ===== */
    .stAlert {{
        background-color: {theme.bg_secondary} !important;
        border: 1px solid {theme.border_light} !important;
        border-radius: 8px;
        color: {theme.text_primary} !important;
    }}

    /* ===== Spinner ===== */
    .stSpinner > div {{
        color: {theme.accent_blue} !important;
    }}

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {theme.bg_primary};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {theme.border_light};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme.text_light};
    }}

    /* ===== Remove Streamlit Branding ===== */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ===== Smooth Transitions ===== */
    .stDataFrame, .stButton > button, .stSelectbox, .stMultiSelect {{
        transition: all 0.15s ease;
    }}

    /* ===== Plotly Chart Container ===== */
    [data-testid="stPlotlyChart"] {{
        border: 1px solid {theme.border_light};
        border-radius: 12px;
        overflow: hidden;
        background: {theme.bg_secondary};
    }}

    /* ===== Caption / Small Text ===== */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {theme.text_muted} !important;
    }}

    /* ===== Divider ===== */
    hr {{
        border-color: {theme.border_light} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:12px 0 20px 0;border-bottom:1px solid {theme.border_light};margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,{theme.accent_blue},{theme.accent_purple});border-radius:8px;display:flex;align-items:center;justify-content:center;">
                    <span style="color:white;font-size:16px;font-weight:700;">◈</span>
                </div>
                <div>
                    <h2 style="color:{theme.text_primary};margin:0;font-weight:600;font-size:16px;letter-spacing:-0.01em;">信用债风险平台</h2>
                    <p style="color:{theme.text_muted};font-size:11px;margin:2px 0 0 0;letter-spacing:0.02em;">Credit Risk Intelligence</p>
                </div>
            </div>
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

        avg_oas_sidebar = sum(e.weighted_avg_oas * e.total_market_usd for e in exposures) / total_mv if total_mv > 0 else 0
        st.markdown(f"""
        <div style="background:{theme.bg_secondary};border:1px solid {theme.border_light};border-radius:10px;padding:16px 18px;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
            <div style="font-size:10px;color:{theme.text_muted};margin-bottom:14px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Portfolio Summary</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid {theme.bg_tertiary};">
                <span style="color:{theme.text_muted};font-size:12px;">资产规模</span>
                <span style="color:{theme.text_primary};font-weight:600;font-size:14px;">${total_mv/1e9:.1f}B</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid {theme.bg_tertiary};">
                <span style="color:{theme.text_muted};font-size:12px;">发行人</span>
                <span style="color:{theme.text_primary};font-weight:600;font-size:14px;">{total_issuers}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{theme.text_muted};font-size:12px;">加权OAS</span>
                <span style="color:{theme.accent_blue};font-weight:600;font-size:14px;">{avg_oas_sidebar:.0f}bp</span>
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
        <div style="margin-top:32px;padding-top:16px;border-top:1px solid {theme.border_light};">
            <div style="color:{theme.text_light};font-size:10px;text-align:center;letter-spacing:0.05em;">v3.1 · Nordic Edition · 2026</div>
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
