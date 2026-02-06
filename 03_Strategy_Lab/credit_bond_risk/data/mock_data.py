"""
Credit Bond Risk - Mock Data Provider (Enhanced 2026 Edition)

Provides realistic mock data for testing and demonstration purposes.
Includes international obligors (G-SIB, US/EU corporates, EM sovereigns)
alongside China offshore credit exposure.

Features:
- 45+ obligors with interconnected relationships
- 2026 market scenarios and alerts
- Rich news flow with sentiment analysis
- Obligor relationship graph (guarantees, supply chain, parent-child)

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
# Obligor Templates - Realistic Portfolio Composition (Enhanced 2026)
# =============================================================================

# Template format: (id, name_cn, name_en, sector, sub_sector, region, country, province, rating, outlook, ticker, nominal_range, parent_id, guarantor_ids)
OBLIGOR_TEMPLATES = [
    # =========================================================================
    # China LGFV - Provincial Level (省级平台)
    # =========================================================================
    ("CN001", "云南省城投集团", "Yunnan Provincial Investment", Sector.LGFV, "省级城投", Region.CHINA_OFFSHORE, "CN", "云南", CreditRating.AA, RatingOutlook.STABLE, None, (200, 500), None, []),
    ("CN002", "重庆市城建投资", "Chongqing Urban Construction", Sector.LGFV, "直辖市城投", Region.CHINA_OFFSHORE, "CN", "重庆", CreditRating.AA_MINUS, RatingOutlook.NEGATIVE, None, (150, 350), None, []),
    ("CN003", "贵州省交通投资", "Guizhou Transportation", Sector.LGFV, "省级城投", Region.CHINA_OFFSHORE, "CN", "贵州", CreditRating.AA_MINUS, RatingOutlook.WATCH_NEG, None, (100, 250), None, []),
    ("CN004", "成都兴城投资", "Chengdu Xingcheng Investment", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "四川", CreditRating.AA, RatingOutlook.STABLE, None, (180, 400), None, []),
    ("CN005A", "四川省投资集团", "Sichuan Investment Group", Sector.LGFV, "省级城投", Region.CHINA_OFFSHORE, "CN", "四川", CreditRating.AA_PLUS, RatingOutlook.STABLE, None, (250, 550), None, ["CN004"]),

    # China LGFV - City Level (地市级平台)
    ("CN011", "昆明城投集团", "Kunming Urban Investment", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "云南", CreditRating.AA_MINUS, RatingOutlook.STABLE, None, (80, 180), "CN001", []),
    ("CN012", "曲靖市城投", "Qujing Urban Investment", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "云南", CreditRating.A_PLUS, RatingOutlook.NEGATIVE, None, (50, 120), "CN001", []),
    ("CN013", "遵义道桥建设", "Zunyi Road & Bridge", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "贵州", CreditRating.A, RatingOutlook.WATCH_NEG, None, (60, 150), "CN003", []),
    ("CN014", "六盘水城投", "Liupanshui Urban Investment", Sector.LGFV, "地级市城投", Region.CHINA_OFFSHORE, "CN", "贵州", CreditRating.A_MINUS, RatingOutlook.NEGATIVE, None, (40, 100), "CN003", []),
    ("CN015", "天津城投集团", "Tianjin Urban Investment", Sector.LGFV, "直辖市城投", Region.CHINA_OFFSHORE, "CN", "天津", CreditRating.AA, RatingOutlook.NEGATIVE, None, (200, 450), None, []),
    ("CN016", "山东高速集团", "Shandong Hi-Speed Group", Sector.LGFV, "省级城投", Region.CHINA_OFFSHORE, "CN", "山东", CreditRating.AA_PLUS, RatingOutlook.STABLE, None, (300, 600), None, []),

    # =========================================================================
    # China SOE - Central Enterprises (央企)
    # =========================================================================
    ("CN020", "中国石油化工集团", "Sinopec Group", Sector.SOE, "央企能源", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AAA, RatingOutlook.STABLE, "386 HK", (500, 1000), None, []),
    ("CN021", "中国国家电网", "State Grid Corporation", Sector.SOE, "央企公用事业", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AAA, RatingOutlook.STABLE, None, (400, 800), None, []),
    ("CN022", "中国铁路建设", "China Railway Construction", Sector.SOE, "央企基建", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "1186 HK", (300, 600), None, []),
    ("CN023", "中国中化集团", "Sinochem Group", Sector.SOE, "央企化工", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, None, (250, 500), None, []),
    ("CN024", "中国华能集团", "China Huaneng Group", Sector.SOE, "央企电力", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, None, (280, 550), None, []),
    ("CN025", "中国建筑集团", "China State Construction", Sector.SOE, "央企基建", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "3311 HK", (350, 700), None, []),

    # =========================================================================
    # China Financial Institutions
    # =========================================================================
    ("CN030", "中国工商银行", "ICBC", Sector.FINANCIAL, "国有大行", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AAA, RatingOutlook.STABLE, "1398 HK", (600, 1200), None, []),
    ("CN031", "招商银行", "China Merchants Bank", Sector.FINANCIAL, "股份制银行", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "3968 HK", (250, 500), None, []),
    ("CN032", "中国人寿保险", "China Life Insurance", Sector.FINANCIAL, "保险", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA_PLUS, RatingOutlook.STABLE, "2628 HK", (200, 450), None, []),
    ("CN033", "平安银行", "Ping An Bank", Sector.FINANCIAL, "股份制银行", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA, RatingOutlook.STABLE, "000001 CH", (180, 380), None, []),
    ("CN034", "兴业银行", "Industrial Bank", Sector.FINANCIAL, "股份制银行", Region.CHINA_OFFSHORE, "CN", None, CreditRating.AA, RatingOutlook.STABLE, "601166 CH", (150, 320), None, []),

    # =========================================================================
    # G-SIB Banks (Global Systemically Important Banks)
    # =========================================================================
    ("US001", "高盛集团", "Goldman Sachs Group", Sector.G_SIB, "Investment Bank", Region.US, "US", "New York", CreditRating.A, RatingOutlook.STABLE, "GS", (300, 700), None, []),
    ("US002", "摩根大通", "JPMorgan Chase", Sector.G_SIB, "Universal Bank", Region.US, "US", "New York", CreditRating.AA_MINUS, RatingOutlook.STABLE, "JPM", (400, 900), None, []),
    ("US003", "美国银行", "Bank of America", Sector.G_SIB, "Universal Bank", Region.US, "US", "North Carolina", CreditRating.A_PLUS, RatingOutlook.STABLE, "BAC", (350, 750), None, []),
    ("US004", "花旗集团", "Citigroup", Sector.G_SIB, "Universal Bank", Region.US, "US", "New York", CreditRating.A, RatingOutlook.STABLE, "C", (280, 600), None, []),
    ("UK001", "汇丰控股", "HSBC Holdings", Sector.G_SIB, "Universal Bank", Region.UK, "GB", "London", CreditRating.AA_MINUS, RatingOutlook.STABLE, "HSBA LN", (450, 850), None, []),
    ("EU001", "法国巴黎银行", "BNP Paribas", Sector.G_SIB, "Universal Bank", Region.EU, "FR", None, CreditRating.A_PLUS, RatingOutlook.STABLE, "BNP FP", (280, 550), None, []),
    ("EU002", "德意志银行", "Deutsche Bank", Sector.G_SIB, "Universal Bank", Region.EU, "DE", "Frankfurt", CreditRating.A_MINUS, RatingOutlook.NEGATIVE, "DBK GR", (200, 400), None, []),
    ("JP001", "三菱日联金融", "MUFG Bank", Sector.G_SIB, "Universal Bank", Region.JAPAN, "JP", "Tokyo", CreditRating.A, RatingOutlook.STABLE, "8306 JP", (350, 650), None, []),
    ("UK002", "巴克莱银行", "Barclays", Sector.G_SIB, "Universal Bank", Region.UK, "GB", "London", CreditRating.A, RatingOutlook.STABLE, "BARC LN", (220, 480), None, []),
    ("EU003", "瑞银集团", "UBS Group", Sector.G_SIB, "Universal Bank", Region.EU, "CH", "Zurich", CreditRating.A_PLUS, RatingOutlook.STABLE, "UBSG SW", (300, 600), None, []),

    # =========================================================================
    # US Corporates (Investment Grade)
    # =========================================================================
    ("US010", "苹果公司", "Apple Inc", Sector.US_CORP, "Technology", Region.US, "US", "California", CreditRating.AA_PLUS, RatingOutlook.STABLE, "AAPL", (500, 1000), None, []),
    ("US011", "微软公司", "Microsoft Corp", Sector.US_CORP, "Technology", Region.US, "US", "Washington", CreditRating.AAA, RatingOutlook.STABLE, "MSFT", (450, 900), None, []),
    ("US012", "埃克森美孚", "Exxon Mobil", Sector.US_CORP, "Energy", Region.US, "US", "Texas", CreditRating.AA, RatingOutlook.STABLE, "XOM", (300, 600), None, []),
    ("US013", "辉瑞制药", "Pfizer Inc", Sector.US_CORP, "Healthcare", Region.US, "US", "New York", CreditRating.A_PLUS, RatingOutlook.STABLE, "PFE", (200, 450), None, []),
    ("US014", "强生公司", "Johnson & Johnson", Sector.US_CORP, "Healthcare", Region.US, "US", "New Jersey", CreditRating.AAA, RatingOutlook.STABLE, "JNJ", (350, 700), None, []),
    ("US015", "亚马逊", "Amazon.com", Sector.US_CORP, "Technology", Region.US, "US", "Washington", CreditRating.AA, RatingOutlook.STABLE, "AMZN", (400, 800), None, []),
    ("US016", "特斯拉", "Tesla Inc", Sector.US_CORP, "Auto", Region.US, "US", "Texas", CreditRating.BBB, RatingOutlook.POSITIVE, "TSLA", (150, 350), None, []),

    # =========================================================================
    # EU Corporates
    # =========================================================================
    ("EU010", "壳牌公司", "Shell plc", Sector.EU_CORP, "Energy", Region.UK, "GB", "London", CreditRating.AA_MINUS, RatingOutlook.STABLE, "SHEL LN", (350, 700), None, []),
    ("EU011", "西门子集团", "Siemens AG", Sector.EU_CORP, "Industrial", Region.EU, "DE", "Munich", CreditRating.A_PLUS, RatingOutlook.STABLE, "SIE GR", (200, 450), None, []),
    ("EU012", "雀巢公司", "Nestle SA", Sector.EU_CORP, "Consumer", Region.EU, "CH", "Vevey", CreditRating.AA, RatingOutlook.STABLE, "NESN SW", (250, 500), None, []),
    ("EU013", "路威酩轩", "LVMH", Sector.EU_CORP, "Consumer", Region.EU, "FR", "Paris", CreditRating.A_PLUS, RatingOutlook.STABLE, "MC FP", (180, 400), None, []),
    ("EU014", "大众汽车", "Volkswagen AG", Sector.EU_CORP, "Auto", Region.EU, "DE", "Wolfsburg", CreditRating.A_MINUS, RatingOutlook.NEGATIVE, "VOW GR", (200, 450), None, []),

    # =========================================================================
    # EM Sovereign & Quasi-Sovereign
    # =========================================================================
    ("BR001", "巴西国家石油", "Petrobras", Sector.EM_SOVEREIGN, "Quasi-Sovereign Oil", Region.LATAM, "BR", "Rio de Janeiro", CreditRating.BBB_MINUS, RatingOutlook.STABLE, "PBR", (200, 450), None, []),
    ("MX001", "墨西哥国家石油", "Pemex", Sector.EM_SOVEREIGN, "Quasi-Sovereign Oil", Region.LATAM, "MX", "Mexico City", CreditRating.BB, RatingOutlook.NEGATIVE, "PEMEX", (150, 350), None, []),
    ("ID001", "印尼国家电力", "PLN Indonesia", Sector.EM_SOVEREIGN, "Quasi-Sovereign Utility", Region.ASIA_EX_CHINA, "ID", "Jakarta", CreditRating.BBB, RatingOutlook.STABLE, None, (100, 250), None, []),
    ("SA001", "沙特阿美", "Saudi Aramco", Sector.EM_SOVEREIGN, "Quasi-Sovereign Oil", Region.CEEMEA, "SA", "Dhahran", CreditRating.A, RatingOutlook.STABLE, "2222 AB", (400, 800), None, []),
    ("AE001", "阿布扎比国家石油", "ADNOC", Sector.EM_SOVEREIGN, "Quasi-Sovereign Oil", Region.CEEMEA, "AE", "Abu Dhabi", CreditRating.AA, RatingOutlook.STABLE, None, (250, 500), None, []),

    # =========================================================================
    # Supranational
    # =========================================================================
    ("SUPRA01", "亚洲开发银行", "Asian Development Bank", Sector.SUPRA, "MDB", Region.SUPRANATIONAL, None, None, CreditRating.AAA, RatingOutlook.STABLE, None, (300, 600), None, []),
    ("SUPRA02", "亚投行", "AIIB", Sector.SUPRA, "MDB", Region.SUPRANATIONAL, None, None, CreditRating.AAA, RatingOutlook.STABLE, None, (200, 450), None, []),
    ("SUPRA03", "国际金融公司", "IFC", Sector.SUPRA, "MDB", Region.SUPRANATIONAL, None, None, CreditRating.AAA, RatingOutlook.STABLE, None, (250, 500), None, []),
    ("SUPRA04", "欧洲投资银行", "EIB", Sector.SUPRA, "MDB", Region.SUPRANATIONAL, None, None, CreditRating.AAA, RatingOutlook.STABLE, None, (280, 550), None, []),

    # =========================================================================
    # High Yield / Stressed Credits
    # =========================================================================
    ("HY001", "某高收益地产", "Distressed Property Developer", Sector.HY, "Real Estate HY", Region.CHINA_OFFSHORE, "CN", "广东", CreditRating.B, RatingOutlook.NEGATIVE, None, (50, 150), None, []),
    ("HY002", "某民营钢铁", "Private Steel Company", Sector.HY, "Industrial HY", Region.CHINA_OFFSHORE, "CN", "河北", CreditRating.B_MINUS, RatingOutlook.WATCH_NEG, None, (30, 80), None, []),
    ("HY003", "某科技独角兽", "Tech Unicorn Bond", Sector.HY, "Technology HY", Region.CHINA_OFFSHORE, "CN", "北京", CreditRating.BB, RatingOutlook.STABLE, None, (60, 140), None, []),
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
# Obligor Relationship Graph (Guarantee Chain / Parent-Child / Supply Chain)
# =============================================================================

OBLIGOR_RELATIONSHIPS = {
    # Parent-Child relationships
    "CN011": {"parent": "CN001", "relationship": "子公司", "support_level": "强"},
    "CN012": {"parent": "CN001", "relationship": "子公司", "support_level": "中"},
    "CN013": {"parent": "CN003", "relationship": "子公司", "support_level": "中"},
    "CN014": {"parent": "CN003", "relationship": "子公司", "support_level": "弱"},
    "CN004": {"parent": "CN005A", "relationship": "担保链", "support_level": "强"},

    # Guarantee relationships
    "CN002": {"guarantor": "CN015", "relationship": "互保", "guarantee_amount": 5e9},
    "CN015": {"guarantor": "CN002", "relationship": "互保", "guarantee_amount": 3e9},

    # Supply chain dependencies
    "CN022": {"suppliers": ["CN025"], "relationship": "供应链", "dependency": "高"},
    "CN025": {"customers": ["CN022"], "relationship": "供应链", "dependency": "中"},
}


# =============================================================================
# Generator Functions
# =============================================================================


def generate_mock_obligors() -> dict[str, Obligor]:
    """
    Generate mock obligor master data with enhanced relationships.

    Returns:
        Dict mapping obligor_id to Obligor model
    """
    obligors = {}

    for template in OBLIGOR_TEMPLATES:
        oid = template[0]
        name_cn = template[1]
        name_en = template[2]
        sector = template[3]
        sub = template[4]
        region = template[5]
        country = template[6]
        province = template[7]
        rating = template[8]
        outlook = template[9]
        ticker = template[10]
        # nominal_range = template[11]  # Used in position generation
        parent_id = template[12] if len(template) > 12 else None
        # guarantor_ids = template[13] if len(template) > 13 else []

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
            parent_entity=parent_id,
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
        nominal_range = template[11] if template and len(template) > 11 else (100, 300)

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
    Generate mock risk alerts with 2026 scenarios.

    Returns:
        List of RiskAlert models
    """
    now = datetime.now()

    alerts = [
        # =====================================================================
        # 严重预警 (CRITICAL)
        # =====================================================================
        RiskAlert(
            alert_id="ALT001",
            timestamp=now - timedelta(hours=2),
            severity=Severity.CRITICAL,
            category=AlertCategory.RATING,
            obligor_id="CN002",
            obligor_name="重庆市城建投资",
            signal_name="rating_downgrade",
            message="穆迪下调评级至Ba1，展望负面 | Moody's downgrade to Ba1",
            metric_value=2.0,
            threshold=1.0,
            status=AlertStatus.PENDING,
            ai_summary="重庆城投评级下调主要受区域债务压力影响，建议密切关注化债进展，考虑减持或对冲。",
        ),
        RiskAlert(
            alert_id="ALT002",
            timestamp=now - timedelta(hours=4),
            severity=Severity.CRITICAL,
            category=AlertCategory.SPREAD,
            obligor_id="HY001",
            obligor_name="某高收益地产",
            signal_name="spread_zscore",
            message="OAS Z-Score突破3.5，进入深度困境区间",
            metric_value=3.52,
            threshold=3.0,
            status=AlertStatus.PENDING,
            ai_summary="地产债利差持续走阔，流动性几乎枯竭，存在实质性违约风险。建议立即评估回收率假设。",
        ),
        RiskAlert(
            alert_id="ALT003",
            timestamp=now - timedelta(hours=6),
            severity=Severity.CRITICAL,
            category=AlertCategory.NEWS,
            obligor_id="CN014",
            obligor_name="六盘水城投",
            signal_name="news_negative_burst",
            message="负面新闻集中爆发：财政厅介入调查",
            metric_value=5,
            threshold=3,
            status=AlertStatus.INVESTIGATING,
            ai_summary="六盘水城投被曝虚增收入，省财政厅已介入调查。该主体为贵州交通子公司，需关注担保链传染风险。",
        ),
        RiskAlert(
            alert_id="ALT004",
            timestamp=now - timedelta(hours=8),
            severity=Severity.CRITICAL,
            category=AlertCategory.RATING,
            obligor_id="EU002",
            obligor_name="Deutsche Bank",
            signal_name="rating_outlook",
            message="标普下调展望至负面，关注商业地产敞口",
            metric_value=1.0,
            threshold=0.0,
            status=AlertStatus.PENDING,
            ai_summary="德银美国CRE敞口承压，诉讼准备金可能不足。2026年再融资压力上升。",
        ),

        # =====================================================================
        # 警告 (WARNING)
        # =====================================================================
        RiskAlert(
            alert_id="ALT005",
            timestamp=now - timedelta(hours=1),
            severity=Severity.WARNING,
            category=AlertCategory.SPREAD,
            obligor_id="CN003",
            obligor_name="贵州省交通投资",
            signal_name="spread_percentile",
            message="OAS突破历史92%分位，关注流动性",
            metric_value=0.92,
            threshold=0.85,
            status=AlertStatus.INVESTIGATING,
            ai_summary="贵州交投利差持续走阔，主要受区域化债政策不确定性影响。省级担保仍在，但需监控。",
        ),
        RiskAlert(
            alert_id="ALT006",
            timestamp=now - timedelta(hours=3),
            severity=Severity.WARNING,
            category=AlertCategory.CONCENTRATION,
            obligor_id="CN020",
            obligor_name="中国石油化工集团",
            signal_name="concentration_single",
            message="单一发行人集中度超过4.5%阈值",
            metric_value=0.048,
            threshold=0.045,
            status=AlertStatus.PENDING,
            ai_summary="Sinopec敞口略超内部限额，但作为AAA央企，信用风险可控。建议自然到期减持。",
        ),
        RiskAlert(
            alert_id="ALT007",
            timestamp=now - timedelta(hours=5),
            severity=Severity.WARNING,
            category=AlertCategory.NEWS,
            obligor_id="MX001",
            obligor_name="Pemex",
            signal_name="news_sentiment",
            message="产量持续下滑，负面情绪升温",
            metric_value=-0.55,
            threshold=-0.30,
            status=AlertStatus.PENDING,
            ai_summary="Pemex产量同比继续下滑，政府财政支持意愿存疑。2026年到期债务规模较大。",
        ),
        RiskAlert(
            alert_id="ALT008",
            timestamp=now - timedelta(hours=7),
            severity=Severity.WARNING,
            category=AlertCategory.SPREAD,
            obligor_id="CN012",
            obligor_name="曲靖市城投",
            signal_name="spread_widening",
            message="利差周变动+45bp，超过预警阈值",
            metric_value=45,
            threshold=30,
            status=AlertStatus.PENDING,
            ai_summary="曲靖城投作为云南城投子公司，近期利差异常走阔，可能存在尾部风险传染。",
        ),
        RiskAlert(
            alert_id="ALT009",
            timestamp=now - timedelta(hours=10),
            severity=Severity.WARNING,
            category=AlertCategory.CONCENTRATION,
            obligor_id="CN001",
            obligor_name="云南省城投集团",
            signal_name="province_concentration",
            message="云南省敞口占比超过区域限额",
            metric_value=0.065,
            threshold=0.05,
            status=AlertStatus.INVESTIGATING,
            ai_summary="含下属平台在内，云南省敞口达6.5%，超过5%内部限额。建议调整新增投资方向。",
        ),
        RiskAlert(
            alert_id="ALT010",
            timestamp=now - timedelta(hours=12),
            severity=Severity.WARNING,
            category=AlertCategory.RATING,
            obligor_id="EU014",
            obligor_name="Volkswagen AG",
            signal_name="rating_outlook",
            message="惠誉下调展望至负面，电动化转型承压",
            metric_value=1.0,
            threshold=0.0,
            status=AlertStatus.PENDING,
            ai_summary="大众电动化转型落后于预期，中国市场份额持续下滑。2026年面临较大再融资压力。",
        ),

        # =====================================================================
        # 提示 (INFO)
        # =====================================================================
        RiskAlert(
            alert_id="ALT011",
            timestamp=now - timedelta(hours=9),
            severity=Severity.INFO,
            category=AlertCategory.NEWS,
            obligor_id="CN021",
            obligor_name="中国国家电网",
            signal_name="news_positive",
            message="央行定向降准，支持绿色债券发行",
            metric_value=0.6,
            threshold=0.3,
            status=AlertStatus.RESOLVED,
            ai_summary="国家电网作为绿色债券主力发行人，将直接受益于政策支持。",
        ),
        RiskAlert(
            alert_id="ALT012",
            timestamp=now - timedelta(hours=11),
            severity=Severity.INFO,
            category=AlertCategory.SPREAD,
            obligor_id="SUPRA02",
            obligor_name="亚投行",
            signal_name="spread_tightening",
            message="利差收窄至历史低位，避险资金流入",
            metric_value=15,
            threshold=20,
            status=AlertStatus.RESOLVED,
            ai_summary="AAA超主权债券受益于避险情绪，利差处于历史低位。",
        ),
        RiskAlert(
            alert_id="ALT013",
            timestamp=now - timedelta(days=1),
            severity=Severity.INFO,
            category=AlertCategory.NEWS,
            obligor_id="US002",
            obligor_name="JPMorgan Chase",
            signal_name="news_earnings",
            message="Q4业绩超预期，NII创新高",
            metric_value=0.7,
            threshold=0.3,
            status=AlertStatus.RESOLVED,
            ai_summary="摩根大通Q4业绩强劲，净息差持续扩张，资本充足率稳健。",
        ),
        RiskAlert(
            alert_id="ALT014",
            timestamp=now - timedelta(days=1, hours=3),
            severity=Severity.INFO,
            category=AlertCategory.RATING,
            obligor_id="US016",
            obligor_name="Tesla Inc",
            signal_name="rating_upgrade",
            message="穆迪上调展望至正面，现金流改善",
            metric_value=1.0,
            threshold=0.0,
            status=AlertStatus.RESOLVED,
            ai_summary="特斯拉自由现金流持续改善，债务杠杆下降，评级上调可能性上升。",
        ),
    ]

    return alerts


def generate_mock_news() -> list[NewsItem]:
    """
    Generate mock news items with 2026 market context.

    Returns:
        List of NewsItem models
    """
    now = datetime.now()

    news_items = [
        # =====================================================================
        # 宏观与政策
        # =====================================================================
        NewsItem(
            news_id="NEWS001",
            timestamp=now - timedelta(hours=1),
            source="Bloomberg",
            title="美联储维持利率不变，暗示2026下半年可能降息",
            content="美联储在1月议息会议上维持利率在5.25%-5.50%区间不变，但点阵图显示下半年可能开启降息周期...",
            obligor_ids=[],
            summary="联储按兵不动但偏鸽派，IG信用债将受益于久期。需关注HY再融资压力。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.5,
        ),
        NewsItem(
            news_id="NEWS002",
            timestamp=now - timedelta(hours=2),
            source="财新",
            title="财政部发布城投债务化解新政，允许省级平台开展债务置换",
            content="中国财政部联合人民银行发布《关于进一步加强地方政府融资平台债务管理的指导意见》，明确省级平台可进行债务置换...",
            obligor_ids=["CN001", "CN002", "CN003", "CN004", "CN005A", "CN011", "CN012", "CN013", "CN014", "CN015", "CN016"],
            summary="重大政策利好，省级城投平台直接受益。地市级平台需观察具体执行细则。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.75,
        ),
        NewsItem(
            news_id="NEWS003",
            timestamp=now - timedelta(hours=4),
            source="Reuters",
            title="欧洲央行维持利率，警告通胀下行风险",
            content="欧洲央行在最新货币政策声明中维持利率不变，但强调经济增长放缓和通胀下行风险...",
            obligor_ids=["EU001", "EU002", "EU003", "EU010", "EU011", "EU012", "EU013", "EU014"],
            summary="欧央行偏鸽，欧元区信用债整体受益，但需关注德国制造业疲软。",
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=0.2,
        ),

        # =====================================================================
        # 中国城投
        # =====================================================================
        NewsItem(
            news_id="NEWS004",
            timestamp=now - timedelta(hours=3),
            source="中国债券信息网",
            title="贵州省召开化债攻坚推进会，省财政厅承诺兜底支持",
            content="贵州省政府召开2026年化债攻坚大会，省财政厅表示将对重点平台提供必要支持...",
            obligor_ids=["CN003", "CN013", "CN014"],
            summary="贵州化债决心明确，但执行力度需持续观察。六盘水、遵义等地市平台仍需谨慎。",
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=0.3,
        ),
        NewsItem(
            news_id="NEWS005",
            timestamp=now - timedelta(hours=5),
            source="21世纪经济报道",
            title="云南省财政收入超预期，城投再融资压力缓解",
            content="云南省2025年财政收入同比增长8.2%，超出年初预算目标，为省内城投平台再融资创造有利条件...",
            obligor_ids=["CN001", "CN011", "CN012"],
            summary="云南财政改善对省级平台直接利好，下属地市平台间接受益。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.6,
        ),
        NewsItem(
            news_id="NEWS006",
            timestamp=now - timedelta(hours=8),
            source="经济观察报",
            title="六盘水城投被曝财务造假，省财政厅已介入调查",
            content="知情人士透露，六盘水城投集团涉嫌虚增2024年营业收入约15亿元，贵州省财政厅已成立专项调查组...",
            obligor_ids=["CN014", "CN003"],
            summary="六盘水城投信用事件可能波及贵州交投担保链，需密切关注调查进展。",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.8,
        ),
        NewsItem(
            news_id="NEWS007",
            timestamp=now - timedelta(hours=10),
            source="第一财经",
            title="天津城投获得国开行500亿元授信支持",
            content="天津市与国家开发银行签署战略合作协议，国开行将为天津城投集团提供500亿元专项授信...",
            obligor_ids=["CN015"],
            summary="政策性银行支持增强天津城投流动性，利差有望收窄。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.65,
        ),

        # =====================================================================
        # 国际银行
        # =====================================================================
        NewsItem(
            news_id="NEWS008",
            timestamp=now - timedelta(hours=6),
            source="Financial Times",
            title="德意志银行美国商业地产敞口再受关注，拨备或不足",
            content="Deutsche Bank's US commercial real estate portfolio under renewed scrutiny as office vacancies rise...",
            obligor_ids=["EU002"],
            summary="CRE风险持续发酵，德银拨备可能不足，利差存在走阔压力。",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.65,
        ),
        NewsItem(
            news_id="NEWS009",
            timestamp=now - timedelta(hours=7),
            source="Bloomberg",
            title="摩根大通Q4业绩超预期，净利息收入创历史新高",
            content="JPMorgan Chase reported Q4 earnings that exceeded analyst expectations, with NII reaching record levels...",
            obligor_ids=["US002"],
            summary="强劲NII推动业绩，资本充足率稳健，信用质量稳定。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.7,
        ),
        NewsItem(
            news_id="NEWS010",
            timestamp=now - timedelta(hours=9),
            source="Reuters",
            title="瑞银完成瑞信整合，宣布150亿美元成本节约计划",
            content="UBS Group announced completion of Credit Suisse integration, with CHF 15 billion cost savings...",
            obligor_ids=["EU003"],
            summary="整合进展顺利，成本协同超预期，信用基本面改善。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.55,
        ),

        # =====================================================================
        # 新兴市场
        # =====================================================================
        NewsItem(
            news_id="NEWS011",
            timestamp=now - timedelta(hours=11),
            source="Bloomberg",
            title="Pemex产量跌至1979年以来最低，2026年到期债务承压",
            content="Mexico's state oil company reported another decline in crude output, raising concerns about refinancing...",
            obligor_ids=["MX001"],
            summary="结构性产量下滑持续，政府隐性支持虽在但财政约束收紧。到期墙压力大。",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.7,
        ),
        NewsItem(
            news_id="NEWS012",
            timestamp=now - timedelta(hours=13),
            source="Reuters",
            title="沙特阿美上调亚洲原油官价，看好需求前景",
            content="Saudi Aramco raised its Official Selling Price for Asian customers, signaling confidence in demand...",
            obligor_ids=["SA001"],
            summary="需求前景乐观，阿美基本面稳健，信用风险极低。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.5,
        ),

        # =====================================================================
        # 超主权与央企
        # =====================================================================
        NewsItem(
            news_id="NEWS013",
            timestamp=now - timedelta(hours=12),
            source="IFR",
            title="亚投行以历史最低利差发行30亿美元双币种债券",
            content="Asian Infrastructure Investment Bank achieved its tightest spread ever on new global bond issuance...",
            obligor_ids=["SUPRA02"],
            summary="AAA超主权债券需求强劲，避险情绪推动利差压缩。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.6,
        ),
        NewsItem(
            news_id="NEWS014",
            timestamp=now - timedelta(hours=14),
            source="中国证券报",
            title="国家电网获批发行500亿元绿色债券，央行提供定向支持",
            content="国家电网公司获准发行500亿元绿色债券，人民银行将通过碳减排支持工具提供优惠利率...",
            obligor_ids=["CN021"],
            summary="绿色金融政策持续加码，国家电网作为主力发行人直接受益。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.65,
        ),

        # =====================================================================
        # 高收益/困境债
        # =====================================================================
        NewsItem(
            news_id="NEWS015",
            timestamp=now - timedelta(hours=15),
            source="Caixin",
            title="某高收益地产开发商债务重组谈判陷入僵局",
            content="据悉某大型民营地产开发商与债权人的债务重组谈判再次陷入僵局，回收率预期下调...",
            obligor_ids=["HY001"],
            summary="地产债重组持续艰难，回收率可能低于预期。存量敞口需评估减值。",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.85,
        ),
        NewsItem(
            news_id="NEWS016",
            timestamp=now - timedelta(hours=18),
            source="每日经济新闻",
            title="某民营钢铁企业资金链告急，银行收紧授信",
            content="河北某大型民营钢铁企业被曝资金链紧张，多家银行已收紧或暂停授信...",
            obligor_ids=["HY002"],
            summary="民营钢铁信用事件，行业出清仍在进行。需关注交叉违约条款。",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.75,
        ),

        # =====================================================================
        # 科技与汽车
        # =====================================================================
        NewsItem(
            news_id="NEWS017",
            timestamp=now - timedelta(days=1),
            source="WSJ",
            title="特斯拉自由现金流转正，穆迪上调展望",
            content="Tesla's free cash flow turned positive in Q4, prompting Moody's to revise outlook to positive...",
            obligor_ids=["US016"],
            summary="特斯拉基本面改善，评级上调可能性上升，利差有望收窄。",
            sentiment=Sentiment.POSITIVE,
            sentiment_score=0.6,
        ),
        NewsItem(
            news_id="NEWS018",
            timestamp=now - timedelta(days=1, hours=2),
            source="Handelsblatt",
            title="大众汽车电动化转型落后，中国市场份额持续下滑",
            content="Volkswagen's electric vehicle sales in China fell 15% YoY, losing market share to BYD and Tesla...",
            obligor_ids=["EU014"],
            summary="大众电动化落后竞争对手，中国市场压力加大，信用展望承压。",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-0.5,
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


def get_obligor_relationships() -> dict[str, dict]:
    """
    Get obligor relationship data for graph visualization.

    Returns:
        Dict of relationship metadata
    """
    return OBLIGOR_RELATIONSHIPS


# =============================================================================
# Mock Data Provider Implementation
# =============================================================================


class MockDataProvider(DataProvider):
    """
    Mock data provider for testing and demonstration.

    Generates realistic portfolio data including:
    - 45+ obligors across China LGFV, SOE, G-SIB, US/EU corporates
    - 150+ bond positions with realistic pricing
    - 14+ risk alerts with 2026 scenarios
    - 18+ news items with sentiment analysis
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
        self._relationships = OBLIGOR_RELATIONSHIPS

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

    def get_relationships(self) -> dict[str, dict]:
        """Get obligor relationship data"""
        return self._relationships

    def refresh_data(self) -> None:
        """Regenerate all mock data"""
        self._initialize_data()
