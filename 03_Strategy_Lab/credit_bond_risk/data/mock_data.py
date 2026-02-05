"""
Credit Bond Risk - Mock Data Provider

Provides realistic mock data for testing and demonstration purposes.
Includes international obligors (G-SIB, US/EU corporates, EM sovereigns)
alongside China offshore credit exposure.

Usage:
    from data.mock_data import MockDataProvider
    provider = MockDataProvider()
    obligors = provider.get_obligors()
    exposures = provider.get_exposures()
"""

import random
from datetime import datetime, date, timedelta
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    Obligor,
    BondPosition,
    CreditExposure,
    RiskAlert,
    NewsItem,
)
from core.enums import (
    Sector,
    Region,
    CreditRating,
    RatingOutlook,
    Severity,
    AlertCategory,
    AlertStatus,
    Sentiment,
    RATING_SCORE,
)

from .provider import DataProvider, DataProviderConfig


# =============================================================================
# Obligor Templates - Realistic Portfolio Composition
# =============================================================================

# Template format: (id, name_cn, name_en, sector, sub_sector, region, country, province, rating, outlook, ticker, nominal_range)
OBLIGOR_TEMPLATES = [
    # China LGFV
    ("CN001", "云南省城投集团", "Yunnan Provincial Investment", Sector.LGFV, "省级城投", Region.CHINA_OFFSHORE, "CN", "云南", CreditRating.AA, RatingOutlook.STABLE, None, (200, 500)),
    ("CN002", "重庆市城建投资", "Chongqing Urban Construction", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "重庆", CreditRating.AA_MINUS, RatingOutlook.NEGATIVE, None, (150, 350)),
    ("CN003", "贵州省交通投资", "Guizhou Transportation", Sector.LGFV, "省级城投", Region.CHINA_OFFSHORE, "CN", "贵州", CreditRating.AA_MINUS, RatingOutlook.WATCH_NEG, None, (100, 250)),
    ("CN004", "成都兴城投资", "Chengdu Xingcheng Investment", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "四川", CreditRating.AA, RatingOutlook.STABLE, None, (180, 400)),

    # China SOE
    ("CN005", "中国石油化工集团", "Sinopec Group", Sector.SOE, "央企能源", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AAA, RatingOutlook.STABLE, "386 HK", (500, 1000)),
    ("CN006", "中国国家电网", "State Grid Corporation", Sector.SOE, "央企公用事业", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AAA, RatingOutlook.STABLE, None, (400, 800)),
    ("CN007", "中国铁路建设", "China Railway Construction", Sector.SOE, "央企基建", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "1186 HK", (300, 600)),

    # China Financial
    ("CN008", "中国工商银行", "ICBC", Sector.FINANCIAL, "国有大行", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AAA, RatingOutlook.STABLE, "1398 HK", (600, 1200)),
    ("CN009", "招商银行", "China Merchants Bank", Sector.FINANCIAL, "股份制银行", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "3968 HK", (250, 500)),
    ("CN010", "中国人寿保险", "China Life Insurance", Sector.FINANCIAL, "保险", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "2628 HK", (200, 450)),

    # G-SIB Banks
    ("US001", "高盛集团", "Goldman Sachs Group", Sector.G_SIB, "Investment Bank", Region.US, "US", "New York", CreditRating.A, RatingOutlook.STABLE, "GS", (300, 700)),
    ("US002", "摩根大通", "JPMorgan Chase", Sector.G_SIB, "Universal Bank", Region.US, "US", "New York", CreditRating.AA_MINUS, RatingOutlook.STABLE, "JPM", (400, 900)),
    ("US003", "美国银行", "Bank of America", Sector.G_SIB, "Universal Bank", Region.US, "US", "North Carolina", CreditRating.A_PLUS, RatingOutlook.STABLE, "BAC", (350, 750)),
    ("UK001", "汇丰控股", "HSBC Holdings", Sector.G_SIB, "Universal Bank", Region.UK, "GB", "London", CreditRating.AA_MINUS, RatingOutlook.STABLE, "HSBA LN", (450, 850)),
    ("EU001", "法国巴黎银行", "BNP Paribas", Sector.G_SIB, "Universal Bank", Region.EU, "FR", None, CreditRating.A_PLUS, RatingOutlook.STABLE, "BNP FP", (280, 550)),
    ("EU002", "德意志银行", "Deutsche Bank", Sector.G_SIB, "Universal Bank", Region.EU, "DE", "Frankfurt", CreditRating.A_MINUS, RatingOutlook.NEGATIVE, "DBK GR", (200, 400)),
    ("JP001", "三菱日联金融", "MUFG Bank", Sector.G_SIB, "Universal Bank", Region.JAPAN, "JP", "Tokyo", CreditRating.A, RatingOutlook.STABLE, "8306 JP", (350, 650)),

    # US Corporates
    ("US004", "苹果公司", "Apple Inc", Sector.US_CORP, "Technology", Region.US, "US", "California", CreditRating.AA_PLUS, RatingOutlook.STABLE, "AAPL", (500, 1000)),
    ("US005", "微软公司", "Microsoft Corp", Sector.US_CORP, "Technology", Region.US, "US", "Washington", CreditRating.AAA, RatingOutlook.STABLE, "MSFT", (450, 900)),
    ("US006", "埃克森美孚", "Exxon Mobil", Sector.US_CORP, "Energy", Region.US, "US", "Texas", CreditRating.AA, RatingOutlook.STABLE, "XOM", (300, 600)),
    ("US007", "辉瑞制药", "Pfizer Inc", Sector.US_CORP, "Healthcare", Region.US, "US", "New York", CreditRating.A_PLUS, RatingOutlook.STABLE, "PFE", (200, 450)),

    # EU Corporates
    ("EU003", "壳牌公司", "Shell plc", Sector.EU_CORP, "Energy", Region.UK, "GB", "London", CreditRating.AA_MINUS, RatingOutlook.STABLE, "SHEL LN", (350, 700)),
    ("EU004", "西门子集团", "Siemens AG", Sector.EU_CORP, "Industrial", Region.EU, "DE", "Munich", CreditRating.A_PLUS, RatingOutlook.STABLE, "SIE GR", (200, 450)),
    ("EU005", "雀巢公司", "Nestle SA", Sector.EU_CORP, "Consumer", Region.EU, "CH", "Vevey", CreditRating.AA, RatingOutlook.STABLE, "NESN SW", (250, 500)),

    # EM Sovereign & Quasi-Sovereign
    ("BR001", "巴西国家石油", "Petrobras", Sector.EM_SOVEREIGN, "Quasi-Sovereign Oil", Region.LATAM, "BR", "Rio de Janeiro", CreditRating.BBB_MINUS, RatingOutlook.STABLE, "PBR", (200, 450)),
    ("MX001", "墨西哥国家石油", "Pemex", Sector.EM_SOVEREIGN, "Quasi-Sovereign Oil", Region.LATAM, "MX", "Mexico City", CreditRating.BB, RatingOutlook.NEGATIVE, "PEMEX", (150, 350)),
    ("ID001", "印尼国家电力", "PLN Indonesia", Sector.EM_SOVEREIGN, "Quasi-Sovereign Utility", Region.ASIA_EX_CHINA, "ID", "Jakarta", CreditRating.BBB, RatingOutlook.STABLE, None, (100, 250)),

    # Supranational
    ("SUPRA01", "亚洲开发银行", "Asian Development Bank", Sector.SUPRA, "MDB", Region.SUPRANATIONAL, None, None, CreditRating.AAA, RatingOutlook.STABLE, None, (300, 600)),
    ("SUPRA02", "亚投行", "AIIB", Sector.SUPRA, "MDB", Region.SUPRANATIONAL, None, None, CreditRating.AAA, RatingOutlook.STABLE, None, (200, 450)),

    # High Yield
    ("HY001", "某高收益地产", "Evergrande-like Property", Sector.HY, "Real Estate HY", Region.CHINA_OFFSHORE, "CN", "广东", CreditRating.B, RatingOutlook.NEGATIVE, None, (50, 150)),
]


# OAS base spread by rating (bps)
BASE_OAS_BY_RATING = {
    CreditRating.AAA: 20,
    CreditRating.AA_PLUS: 40,
    CreditRating.AA: 60,
    CreditRating.AA_MINUS: 85,
    CreditRating.A_PLUS: 100,
    CreditRating.A: 120,
    CreditRating.A_MINUS: 150,
    CreditRating.BBB_PLUS: 180,
    CreditRating.BBB: 220,
    CreditRating.BBB_MINUS: 280,
    CreditRating.BB_PLUS: 350,
    CreditRating.BB: 400,
    CreditRating.BB_MINUS: 500,
    CreditRating.B_PLUS: 550,
    CreditRating.B: 600,
    CreditRating.B_MINUS: 700,
    CreditRating.CCC: 900,
    CreditRating.NR: 200,
}


# =============================================================================
# Generator Functions
# =============================================================================


def generate_mock_obligors() -> dict[str, Obligor]:
    """
    Generate mock obligor master data.

    Returns:
        Dict mapping obligor_id to Obligor model
    """
    obligors = {}

    for template in OBLIGOR_TEMPLATES:
        oid, name_cn, name_en, sector, sub, region, country, province, rating, outlook, ticker, _ = template

        obligor = Obligor(
            obligor_id=oid,
            name_cn=name_cn,
            name_en=name_en,
            ticker=ticker,
            sector=sector,
            sub_sector=sub,
            region=region,
            country=country,
            province=province,
            rating_internal=rating,
            rating_outlook=outlook,
            rating_external={},
        )
        obligors[oid] = obligor

    return obligors


def generate_mock_positions(
    obligors: dict[str, Obligor],
    seed: int | None = None,
) -> list[BondPosition]:
    """
    Generate mock bond positions for all obligors.

    Args:
        obligors: Dict of Obligor models
        seed: Random seed for reproducibility

    Returns:
        List of BondPosition models
    """
    if seed is not None:
        random.seed(seed)

    positions = []

    # Get nominal ranges from templates
    template_dict = {t[0]: t for t in OBLIGOR_TEMPLATES}

    for oid, obligor in obligors.items():
        template = template_dict.get(oid)
        nominal_range = template[11] if template else (100, 300)

        # Generate 2-6 bonds per obligor
        num_bonds = random.randint(2, 6)

        for i in range(num_bonds):
            maturity_years = random.uniform(0.5, 10)
            nominal = random.uniform(*nominal_range) * 1e6

            # Currency based on region
            if obligor.region in (Region.CHINA_ONSHORE, Region.CHINA_OFFSHORE):
                ccy = random.choice(["USD", "CNH"]) if obligor.region == Region.CHINA_OFFSHORE else "CNY"
            elif obligor.region == Region.US:
                ccy = "USD"
            elif obligor.region in (Region.EU, Region.UK):
                ccy = random.choice(["EUR", "GBP", "USD"])
            elif obligor.region == Region.JAPAN:
                ccy = random.choice(["JPY", "USD"])
            else:
                ccy = "USD"

            # OAS varies by rating
            base_oas = BASE_OAS_BY_RATING.get(obligor.rating_internal, 150)
            oas = base_oas * random.uniform(0.8, 1.3)

            positions.append(BondPosition(
                isin=f"{oid}-{ccy}-{i+1}",
                obligor_id=oid,
                bond_name=f"{obligor.name_en or obligor.name_cn} {maturity_years:.1f}Y {ccy}",
                currency=ccy,
                maturity_date=date.today() + timedelta(days=int(maturity_years * 365)),
                coupon=random.uniform(2, 8),
                nominal=nominal,
                nominal_usd=nominal,
                book_value_usd=nominal * random.uniform(0.95, 1.02),
                market_value_usd=nominal * random.uniform(0.88, 1.05),
                duration=maturity_years * random.uniform(0.85, 0.95),
                oas=oas,
            ))

    return positions


def generate_mock_exposures(
    obligors: dict[str, Obligor] | None = None,
    positions: list[BondPosition] | None = None,
    total_aum: float = 50e9,
) -> list[CreditExposure]:
    """
    Generate mock credit exposures.

    Args:
        obligors: Optional pre-generated obligors
        positions: Optional pre-generated positions
        total_aum: Total AUM for concentration calculations

    Returns:
        List of CreditExposure models
    """
    if obligors is None:
        obligors = generate_mock_obligors()

    if positions is None:
        positions = generate_mock_positions(obligors)

    # Group positions by obligor
    positions_by_obligor: dict[str, list[BondPosition]] = {}
    for pos in positions:
        if pos.obligor_id not in positions_by_obligor:
            positions_by_obligor[pos.obligor_id] = []
        positions_by_obligor[pos.obligor_id].append(pos)

    # Build exposures
    exposures = []
    for oid, obligor in obligors.items():
        obligor_positions = positions_by_obligor.get(oid, [])
        exposure = CreditExposure.from_positions(obligor, obligor_positions, total_aum)
        exposures.append(exposure)

    return exposures


def generate_mock_alerts() -> list[RiskAlert]:
    """
    Generate mock risk alerts with international context.

    Returns:
        List of RiskAlert models
    """
    alerts = [
        RiskAlert(
            alert_id="ALT001",
            severity=Severity.CRITICAL,
            category=AlertCategory.RATING,
            obligor_id="CN002",
            obligor_name="重庆市城建投资",
            signal_name="rating_change",
            message="Moody's downgrade to Ba1, outlook negative | 穆迪下调至Ba1，展望负面",
            metric_value=2.0,
            threshold=1.0,
            status=AlertStatus.PENDING,
        ),
        RiskAlert(
            alert_id="ALT002",
            severity=Severity.WARNING,
            category=AlertCategory.SPREAD,
            obligor_id="CN003",
            obligor_name="贵州省交通投资",
            signal_name="spread_percentile",
            message="OAS widened to 92nd percentile (historical) | OAS突破历史92%分位",
            metric_value=0.92,
            threshold=0.85,
            status=AlertStatus.INVESTIGATING,
        ),
        RiskAlert(
            alert_id="ALT003",
            severity=Severity.CRITICAL,
            category=AlertCategory.RATING,
            obligor_id="EU002",
            obligor_name="Deutsche Bank",
            signal_name="rating_outlook",
            message="S&P revised outlook to Negative citing litigation risk | 标普下调展望至负面，关注诉讼风险",
            metric_value=1.0,
            threshold=0.0,
            status=AlertStatus.PENDING,
        ),
        RiskAlert(
            alert_id="ALT004",
            severity=Severity.WARNING,
            category=AlertCategory.NEWS,
            obligor_id="MX001",
            obligor_name="Pemex",
            signal_name="news_sentiment",
            message="Negative news flow on production decline | 产量下滑负面新闻增加",
            metric_value=-0.55,
            threshold=-0.30,
            status=AlertStatus.PENDING,
            ai_summary="Production continues to decline YoY. Government support uncertain ahead of elections. Refinancing wall in 2025.",
        ),
        RiskAlert(
            alert_id="ALT005",
            severity=Severity.WARNING,
            category=AlertCategory.CONCENTRATION,
            obligor_id="CN005",
            obligor_name="Sinopec Group",
            signal_name="concentration_single",
            message="Single issuer exceeds 4.5% limit | 单一发行人占比超4.5%",
            metric_value=0.048,
            threshold=0.045,
            status=AlertStatus.PENDING,
        ),
        RiskAlert(
            alert_id="ALT006",
            severity=Severity.CRITICAL,
            category=AlertCategory.SPREAD,
            obligor_id="HY001",
            obligor_name="某高收益地产",
            signal_name="spread_zscore",
            message="OAS Z-score > 3.0, distressed levels | OAS Z-score超过3.0，进入困境区间",
            metric_value=3.2,
            threshold=3.0,
            status=AlertStatus.PENDING,
        ),
    ]

    return alerts


def generate_mock_news() -> list[NewsItem]:
    """
    Generate mock news items with international coverage.

    Returns:
        List of NewsItem models
    """
    news_items = [
        NewsItem(
            news_id="NEWS001",
            timestamp=datetime.now() - timedelta(hours=1),
            source="Bloomberg",
            title="Fed Holds Rates Steady, Signals Cuts May Come Later This Year",
            content="The Federal Reserve held interest rates at 5.25%-5.50% but signaled potential cuts in H2...",
            obligor_ids=[],
            summary="Fed on hold but dovish tilt. IG credit should benefit from duration. Monitor HY refinancing.",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.5,
        ),
        NewsItem(
            news_id="NEWS002",
            timestamp=datetime.now() - timedelta(hours=3),
            source="Reuters",
            title="China's Ministry of Finance Announces LGFV Debt Resolution Framework",
            content="中国财政部发布城投债务化解框架，允许省级平台进行债务置换...",
            obligor_ids=["CN001", "CN002", "CN003", "CN004"],
            summary="Major policy support for LGFV sector. Provincial platforms benefit most. Watch implementation.",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.7,
        ),
        NewsItem(
            news_id="NEWS003",
            timestamp=datetime.now() - timedelta(hours=6),
            source="FT",
            title="Deutsche Bank Faces Fresh Concerns Over Commercial Real Estate Exposure",
            content="Deutsche Bank's US CRE portfolio under scrutiny as office vacancies rise...",
            obligor_ids=["EU002"],
            summary="CRE stress continues. Provisioning adequate but sentiment negative. Spreads may widen.",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.6,
        ),
        NewsItem(
            news_id="NEWS004",
            timestamp=datetime.now() - timedelta(hours=8),
            source="Bloomberg",
            title="Pemex Production Drops to Lowest Level Since 1979",
            content="Mexico's state oil company reported another decline in crude output...",
            obligor_ids=["MX001"],
            summary="Structural decline continues. Sovereign support implicit but fiscal constraints remain.",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.7,
        ),
        NewsItem(
            news_id="NEWS005",
            timestamp=datetime.now() - timedelta(hours=12),
            source="IFR",
            title="AIIB Prices $3bn Dual-Tranche Global Bond at Record Tight Spreads",
            content="Asian Infrastructure Investment Bank achieved its tightest spread ever on new issue...",
            obligor_ids=["SUPRA02"],
            summary="Strong demand for AAA supranational paper. Safe haven bid intact.",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.6,
        ),
        NewsItem(
            news_id="NEWS006",
            timestamp=datetime.now() - timedelta(days=1),
            source="Caixin",
            title="贵州省政府召开化债工作推进会",
            content="贵州省召开全省化债攻坚会议，要求各地市加快推进债务化解工作...",
            obligor_ids=["CN003"],
            summary="Provincial government prioritizing debt resolution. Near-term support positive, execution key.",
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=0.2,
        ),
        NewsItem(
            news_id="NEWS007",
            timestamp=datetime.now() - timedelta(days=1, hours=6),
            source="WSJ",
            title="JPMorgan Beats Estimates on Record Net Interest Income",
            content="JPMorgan Chase reported Q4 earnings that exceeded analyst expectations...",
            obligor_ids=["US002"],
            summary="Strong NII tailwind. Capital ratios robust. Credit quality stable.",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.65,
        ),
    ]

    return news_items


def generate_mock_spread_history(
    obligor_id: str,
    days: int = 252,
    base_oas: float | None = None,
) -> dict[str, float]:
    """
    Generate mock spread history for backtesting.

    Args:
        obligor_id: Obligor identifier
        days: Number of trading days
        base_oas: Base OAS level (if None, derived from obligor)

    Returns:
        Dict mapping date string to OAS value
    """
    if base_oas is None:
        # Try to get from obligor templates
        template = next((t for t in OBLIGOR_TEMPLATES if t[0] == obligor_id), None)
        if template:
            rating = template[8]
            base_oas = BASE_OAS_BY_RATING.get(rating, 150)
        else:
            base_oas = 150

    history = {}
    current_date = date.today()
    oas = base_oas

    for i in range(days):
        d = current_date - timedelta(days=i)
        # Skip weekends
        if d.weekday() < 5:
            # Random walk with mean reversion
            change = random.gauss(0, base_oas * 0.02)
            reversion = (base_oas - oas) * 0.05
            oas = max(10, oas + change + reversion)
            history[d.isoformat()] = round(oas, 1)

    return history


# =============================================================================
# Mock Data Provider Implementation
# =============================================================================


class MockDataProvider(DataProvider):
    """
    Mock data provider for testing and demonstration.

    Generates realistic portfolio data including:
    - 30+ obligors across China LGFV, SOE, G-SIB, US/EU corporates
    - 100+ bond positions with realistic pricing
    - Risk alerts and news items
    """

    def __init__(self, config: DataProviderConfig | None = None):
        super().__init__(config)
        self._initialize_data()

    def _initialize_data(self) -> None:
        """Generate and cache all mock data"""
        self._obligors = generate_mock_obligors()
        self._positions = generate_mock_positions(self._obligors)
        self._exposures = generate_mock_exposures(
            self._obligors,
            self._positions,
            self.config.total_aum_usd
        )
        self._alerts = generate_mock_alerts()
        self._news = generate_mock_news()

    def get_obligors(self) -> dict[str, Obligor]:
        return self._obligors

    def get_obligor(self, obligor_id: str) -> Obligor | None:
        return self._obligors.get(obligor_id)

    def get_positions(self, obligor_id: str | None = None) -> list[BondPosition]:
        if obligor_id is None:
            return self._positions
        return [p for p in self._positions if p.obligor_id == obligor_id]

    def get_exposures(self) -> list[CreditExposure]:
        return self._exposures

    def get_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[RiskAlert]:
        alerts = self._alerts

        if status:
            alerts = [a for a in alerts if a.status.value == status]
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]

        # Sort by timestamp descending
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)

        return alerts[:limit]

    def get_news(
        self,
        obligor_id: str | None = None,
        days: int = 7,
        limit: int = 50,
    ) -> list[NewsItem]:
        cutoff = datetime.now() - timedelta(days=days)
        news = [n for n in self._news if n.timestamp >= cutoff]

        if obligor_id:
            news = [n for n in news if obligor_id in n.obligor_ids]

        # Sort by timestamp descending
        news = sorted(news, key=lambda x: x.timestamp, reverse=True)

        return news[:limit]

    def get_spread_history(
        self,
        obligor_id: str,
        days: int = 252,
    ) -> dict[str, float]:
        # Generate on demand
        obligor = self._obligors.get(obligor_id)
        base_oas = None
        if obligor:
            base_oas = BASE_OAS_BY_RATING.get(obligor.rating_internal, 150)

        return generate_mock_spread_history(obligor_id, days, base_oas)

    def refresh_data(self) -> None:
        """Regenerate all mock data"""
        self._initialize_data()
