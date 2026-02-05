"""
Credit Bond Risk - Enums

Enumeration types for credit risk classification and status tracking.
"""

from enum import Enum


class Sector(str, Enum):
    """发行人一级行业分类"""
    LGFV = "LGFV"              # 城投平台
    SOE = "SOE"                # 国有企业 (非城投)
    FINANCIAL = "FINANCIAL"    # 金融机构
    CORP = "CORP"              # 民营企业
    SOVEREIGN = "SOVEREIGN"    # 主权/准主权


class SubSector(str, Enum):
    """发行人二级行业分类"""
    # LGFV
    LGFV_PROVINCIAL = "省级城投"
    LGFV_MUNICIPAL = "地级市城投"
    LGFV_DISTRICT = "区县城投"
    LGFV_NATIONAL_NEW_AREA = "国家级新区"

    # SOE
    SOE_CENTRAL = "央企"
    SOE_LOCAL = "地方国企"

    # Financial
    FIN_POLICY_BANK = "政策性银行"
    FIN_STATE_BANK = "国有大行"
    FIN_JOINT_STOCK = "股份制银行"
    FIN_CITY_COMMERCIAL = "城商行"
    FIN_RURAL_COMMERCIAL = "农商行"
    FIN_SECURITIES = "券商"
    FIN_INSURANCE = "保险"
    FIN_AMC = "资产管理"
    FIN_LEASING = "金融租赁"

    # Corp
    CORP_REAL_ESTATE = "房地产"
    CORP_INDUSTRIAL = "工业制造"
    CORP_TECH = "科技"
    CORP_CONSUMER = "消费"
    CORP_UTILITY = "公用事业"
    CORP_OTHER = "其他"


class CreditRating(str, Enum):
    """信用评级 (内部/外部统一映射)"""
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
    CC = "CC"
    C = "C"
    D = "D"
    NR = "NR"  # Not Rated


# 评级数值映射 (用于计算和排序)
RATING_SCORE: dict[CreditRating, int] = {
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
    CreditRating.BB_PLUS: 50,
    CreditRating.BB: 45,
    CreditRating.BB_MINUS: 40,
    CreditRating.B_PLUS: 35,
    CreditRating.B: 30,
    CreditRating.B_MINUS: 25,
    CreditRating.CCC: 20,
    CreditRating.CC: 15,
    CreditRating.C: 10,
    CreditRating.D: 0,
    CreditRating.NR: 50,  # 未评级按BB处理
}


def rating_to_score(rating: CreditRating) -> int:
    """评级转数值分数"""
    return RATING_SCORE.get(rating, 50)


def score_to_rating(score: int) -> CreditRating:
    """数值分数转最接近的评级"""
    closest = min(RATING_SCORE.items(), key=lambda x: abs(x[1] - score))
    return closest[0]


class RatingOutlook(str, Enum):
    """评级展望"""
    POSITIVE = "POSITIVE"      # 正面
    STABLE = "STABLE"          # 稳定
    NEGATIVE = "NEGATIVE"      # 负面
    WATCH_POS = "WATCH_POS"    # 列入观察(正面)
    WATCH_NEG = "WATCH_NEG"    # 列入观察(负面)
    DEVELOPING = "DEVELOPING"  # 发展中


class Severity(str, Enum):
    """预警严重程度"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    """预警类别"""
    CONCENTRATION = "CONCENTRATION"  # 集中度
    RATING = "RATING"                # 评级变动
    SPREAD = "SPREAD"                # 利差异动
    NEWS = "NEWS"                    # 舆情
    MATURITY = "MATURITY"            # 到期压力
    LIQUIDITY = "LIQUIDITY"          # 流动性
    FUNDAMENTAL = "FUNDAMENTAL"      # 基本面


class AlertStatus(str, Enum):
    """预警处理状态"""
    PENDING = "PENDING"              # 待处理
    INVESTIGATING = "INVESTIGATING"  # 调查中
    RESOLVED = "RESOLVED"            # 已解决
    DISMISSED = "DISMISSED"          # 已忽略
    ESCALATED = "ESCALATED"          # 已升级


class Sentiment(str, Enum):
    """情感倾向"""
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class SignalCategory(str, Enum):
    """信号类别"""
    MARKET = "MARKET"              # 市场信号 (价格/利差)
    FUNDAMENTAL = "FUNDAMENTAL"    # 基本面信号 (财务/评级)
    NEWS = "NEWS"                  # 舆情信号
    CONCENTRATION = "CONCENTRATION"  # 集中度信号
    COMPOSITE = "COMPOSITE"        # 复合信号
