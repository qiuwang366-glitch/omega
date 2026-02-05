"""
Credit Bond Risk - Main Dashboard

Streamlit multi-page application for credit risk monitoring.

Run with: streamlit run dashboard.py
"""

import streamlit as st
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Credit Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import components and modules
from .components.color_scheme import ColorScheme
from .components.alert_table import render_alert_table
from .components.charts import (
    create_concentration_chart,
    create_rating_distribution_chart,
    create_maturity_profile_chart,
    create_sector_concentration_chart,
    create_risk_heatmap,
)

from ..core.config import get_default_config, CreditRiskConfig
from ..core.models import (
    Obligor, CreditExposure, BondPosition, RiskAlert, NewsItem,
)
from ..core.enums import (
    Sector, CreditRating, RatingOutlook, Severity, AlertCategory, AlertStatus, Sentiment,
)
from ..signals.base import SignalContext, SignalRegistry
from ..intelligence.news_analyzer import NewsAnalyzer
from ..intelligence.rag_engine import CreditRAGEngine, VectorStore, RAGConfig

# =============================================================================
# Mock Data Generation (for demo)
# =============================================================================


def generate_mock_data() -> tuple[dict[str, Obligor], list[CreditExposure], list[RiskAlert], list[NewsItem]]:
    """Generate mock data for demonstration"""
    from datetime import date
    import random

    # Mock obligors
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
            obligor_id=oid,
            name_cn=name,
            sector=sector,
            sub_sector=sub,
            province=province,
            rating_internal=rating,
            rating_outlook=outlook,
        )
        obligors[oid] = obligor

        # Generate mock bonds
        bonds = []
        num_bonds = random.randint(2, 5)
        for i in range(num_bonds):
            maturity_years = random.uniform(0.5, 8)
            nominal = random.uniform(50, 300) * 1e6
            bonds.append(BondPosition(
                isin=f"{oid}-BOND-{i+1}",
                obligor_id=oid,
                bond_name=f"{name}债券{i+1}",
                currency="USD",
                maturity_date=date.today() + timedelta(days=int(maturity_years * 365)),
                coupon=random.uniform(3, 6),
                nominal=nominal,
                nominal_usd=nominal,
                book_value_usd=nominal * random.uniform(0.95, 1.02),
                market_value_usd=nominal * random.uniform(0.90, 1.05),
                duration=maturity_years * 0.9,
                oas=random.uniform(80, 400),
            ))

        exposure = CreditExposure.from_positions(obligor, bonds, 50e9)
        exposures.append(exposure)

    # Mock alerts
    alerts = [
        RiskAlert(
            alert_id="ALT001",
            severity=Severity.CRITICAL,
            category=AlertCategory.RATING,
            obligor_id="OBL002",
            obligor_name="某市城建投资",
            signal_name="rating_change",
            message="评级下调至AA-，展望负面",
            metric_value=2.0,
            threshold=1.0,
            status=AlertStatus.PENDING,
        ),
        RiskAlert(
            alert_id="ALT002",
            severity=Severity.WARNING,
            category=AlertCategory.SPREAD,
            obligor_id="OBL006",
            obligor_name="某区县城投",
            signal_name="spread_percentile",
            message="OAS突破历史92%分位",
            metric_value=0.92,
            threshold=0.85,
            status=AlertStatus.INVESTIGATING,
        ),
        RiskAlert(
            alert_id="ALT003",
            severity=Severity.WARNING,
            category=AlertCategory.NEWS,
            obligor_id="OBL005",
            obligor_name="某地方国企",
            signal_name="news_sentiment",
            message="近7天舆情负面 (sentiment: -0.45)",
            metric_value=-0.45,
            threshold=-0.30,
            status=AlertStatus.PENDING,
            ai_summary="近期有关于该企业现金流紧张的报道，建议关注其短期偿债能力。",
        ),
        RiskAlert(
            alert_id="ALT004",
            severity=Severity.CRITICAL,
            category=AlertCategory.CONCENTRATION,
            obligor_id="OBL001",
            obligor_name="某省城投集团",
            signal_name="concentration_single",
            message="单一发行人占比超过5%",
            metric_value=0.052,
            threshold=0.05,
            status=AlertStatus.PENDING,
        ),
    ]

    # Mock news
    news_items = [
        NewsItem(
            news_id="NEWS001",
            timestamp=datetime.now() - timedelta(hours=2),
            source="cls",
            title="某省财政厅发文支持城投平台债务重组",
            content="省财政厅发布指导意见，支持辖内城投平台通过债务重组、资产注入等方式化解债务风险...",
            obligor_ids=["OBL001"],
            summary="省级支持政策出台，利好区域城投",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.6,
        ),
        NewsItem(
            news_id="NEWS002",
            timestamp=datetime.now() - timedelta(hours=5),
            source="bloomberg",
            title="某市城建投资被曝现金流紧张",
            content="据知情人士透露，该公司近期应收账款回款困难，部分项目支出延迟...",
            obligor_ids=["OBL002"],
            summary="现金流压力显现，关注再融资能力",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.7,
        ),
        NewsItem(
            news_id="NEWS003",
            timestamp=datetime.now() - timedelta(days=1),
            source="eastmoney",
            title="美联储议息会议在即，境外中资美元债或承压",
            content="分析师预计美联储将维持高利率，境外中资美元债收益率可能继续上行...",
            obligor_ids=[],
            summary="宏观利率风险提示",
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=-0.1,
        ),
    ]

    return obligors, exposures, alerts, news_items


# =============================================================================
# Session State Initialization
# =============================================================================


def init_session_state():
    """Initialize session state"""
    if "config" not in st.session_state:
        st.session_state.config = get_default_config()

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
    """Render portfolio overview page"""
    scheme = ColorScheme()
    exposures = st.session_state.exposures
    alerts = st.session_state.alerts

    # KPI Row
    total_market = sum(e.total_market_usd for e in exposures)
    total_obligors = len(exposures)
    active_alerts = len([a for a in alerts if a.status == AlertStatus.PENDING])
    critical_alerts = len([a for a in alerts if a.severity == Severity.CRITICAL])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "总市值",
            f"${total_market/1e9:.2f}B",
            help="组合信用债总市值"
        )

    with col2:
        st.metric(
            "发行人数",
            f"{total_obligors}",
            help="持仓发行人总数"
        )

    with col3:
        st.metric(
            "活跃预警",
            f"{active_alerts}",
            delta=f"-{critical_alerts} 严重" if critical_alerts else None,
            delta_color="inverse",
        )

    with col4:
        avg_oas = sum(e.weighted_avg_oas * e.total_market_usd for e in exposures) / total_market
        st.metric(
            "加权OAS",
            f"{avg_oas:.0f}bp",
            help="市值加权平均OAS"
        )

    st.divider()

    # Charts Row 1
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("持仓集中度")
        fig = create_concentration_chart(exposures, top_n=10)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("评级分布")
        fig = create_rating_distribution_chart(exposures)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("行业分布")
        fig = create_sector_concentration_chart(exposures)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("到期分布")
        fig = create_maturity_profile_chart(exposures)
        st.plotly_chart(fig, use_container_width=True)

    # Risk Heatmap
    st.subheader("风险矩阵")
    fig = create_risk_heatmap(exposures)
    st.plotly_chart(fig, use_container_width=True)


def render_alerts_page():
    """Render alerts page"""
    st.subheader("🚨 预警中心")

    alerts = st.session_state.alerts

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        critical = len([a for a in alerts if a.severity == Severity.CRITICAL])
        st.metric("🔴 严重", critical)

    with col2:
        warning = len([a for a in alerts if a.severity == Severity.WARNING])
        st.metric("🟡 警告", warning)

    with col3:
        pending = len([a for a in alerts if a.status == AlertStatus.PENDING])
        st.metric("待处理", pending)

    with col4:
        resolved = len([a for a in alerts if a.status == AlertStatus.RESOLVED])
        st.metric("已解决", resolved)

    st.divider()

    # Alert table
    render_alert_table(alerts, show_filters=True)


def render_news_page():
    """Render news feed page"""
    st.subheader("📰 新闻流")

    news_items = st.session_state.news
    scheme = ColorScheme()

    # News feed
    for news in sorted(news_items, key=lambda x: x.timestamp, reverse=True):
        sentiment_color = {
            Sentiment.POSITIVE: scheme.severity_success,
            Sentiment.NEUTRAL: scheme.text_secondary,
            Sentiment.NEGATIVE: scheme.severity_critical,
        }.get(news.sentiment, scheme.text_secondary)

        sentiment_icon = {
            Sentiment.POSITIVE: "🟢",
            Sentiment.NEUTRAL: "⚪",
            Sentiment.NEGATIVE: "🔴",
        }.get(news.sentiment, "⚪")

        with st.container():
            st.markdown(f"""
            <div style="
                background-color: {scheme.bg_secondary};
                border-left: 3px solid {sentiment_color};
                padding: 12px 16px;
                margin: 8px 0;
                border-radius: 0 8px 8px 0;
            ">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;color:{scheme.text_primary};">
                        {sentiment_icon} {news.title}
                    </span>
                    <span style="color:{scheme.text_muted};font-size:12px;">
                        {news.timestamp.strftime('%m-%d %H:%M')} · {news.source}
                    </span>
                </div>
                <div style="color:{scheme.text_secondary};margin-top:8px;font-size:14px;">
                    {news.summary or news.content[:100] + '...'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if news.obligor_ids:
                obligor_names = [
                    st.session_state.obligors[oid].name_cn
                    for oid in news.obligor_ids
                    if oid in st.session_state.obligors
                ]
                if obligor_names:
                    st.caption(f"关联发行人: {', '.join(obligor_names)}")


def render_chat_page():
    """Render RAG chat interface"""
    st.subheader("💬 AI问答")

    st.info("基于RAG的信用知识库问答（Demo模式）")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("输入问题，例如：云南城投最近有什么风险？"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Generate response (mock for demo)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # Mock RAG response
                if "云南" in prompt:
                    response = """根据近期资料分析：

**云南城投整体情况**：
1. 近期省财政厅出台支持政策，整体信用环境有所改善 [来源1]
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
# Main App
# =============================================================================


def main():
    """Main application entry point"""
    init_session_state()
    scheme = ColorScheme()

    # Custom CSS
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {scheme.bg_primary};
    }}
    .stSidebar {{
        background-color: {scheme.bg_secondary};
    }}
    .stMetric {{
        background-color: {scheme.bg_secondary};
        padding: 16px;
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {scheme.bg_secondary};
        border-radius: 8px;
        padding: 8px 16px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📊 Credit Intelligence")
        st.caption("信用债风险预警平台")

        st.divider()

        # Navigation
        page = st.radio(
            "导航",
            options=["overview", "alerts", "news", "chat"],
            format_func=lambda x: {
                "overview": "📈 组合概览",
                "alerts": "🚨 预警中心",
                "news": "📰 新闻流",
                "chat": "💬 AI问答",
            }[x],
            label_visibility="collapsed",
        )
        st.session_state.active_page = page

        st.divider()

        # Quick stats
        alerts = st.session_state.alerts
        pending_alerts = len([a for a in alerts if a.status == AlertStatus.PENDING])
        critical_alerts = len([a for a in alerts if a.severity == Severity.CRITICAL])

        if critical_alerts > 0:
            st.error(f"🔴 {critical_alerts} 条严重预警待处理")
        elif pending_alerts > 0:
            st.warning(f"🟡 {pending_alerts} 条预警待处理")
        else:
            st.success("✅ 无待处理预警")

        st.divider()

        # Settings
        with st.expander("⚙️ 设置"):
            st.slider(
                "集中度警告阈值 (%)",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.config.concentration.single_obligor_warning * 100,
                step=0.5,
            )
            st.slider(
                "OAS百分位警告阈值",
                min_value=0.7,
                max_value=0.99,
                value=st.session_state.config.spread.percentile_warning,
                step=0.05,
            )

        # Refresh button
        if st.button("🔄 刷新数据", use_container_width=True):
            obligors, exposures, alerts, news = generate_mock_data()
            st.session_state.obligors = obligors
            st.session_state.exposures = exposures
            st.session_state.alerts = alerts
            st.session_state.news = news
            st.rerun()

    # Main content
    st.title("Credit Intelligence Platform")

    # Page routing
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
