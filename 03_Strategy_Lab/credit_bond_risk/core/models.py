"""
Credit Bond Risk - Data Models

Pydantic models for obligors, exposures, alerts, and analysis results.
"""

from datetime import datetime, date
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, computed_field, field_validator

from .enums import (
    Sector,
    SubSector,
    Region,
    CreditRating,
    RatingOutlook,
    Severity,
    AlertCategory,
    AlertStatus,
    Sentiment,
    SignalCategory,
    rating_to_score,
)


# =============================================================================
# Obligor & Bond Models
# =============================================================================


class Obligor(BaseModel):
    """发行人主数据 - 单一视图"""

    obligor_id: str = Field(..., description="唯一标识 (统一社会信用代码或内部ID)")
    name_cn: str = Field(..., description="中文名称")
    name_en: str | None = Field(None, description="英文名称")
    ticker: str | None = Field(None, description="Bloomberg/Reuters Ticker")

    # 分类
    sector: Sector = Field(..., description="一级行业")
    sub_sector: str = Field(..., description="二级行业")
    region: Region = Field(Region.CHINA_OFFSHORE, description="地区")
    country: str | None = Field(None, description="国家 (ISO 3166-1 alpha-2)")
    province: str | None = Field(None, description="省份(中国)/州(美国)")
    city: str | None = Field(None, description="城市")

    # 评级
    rating_external: dict[str, str] = Field(
        default_factory=dict,
        description="外部评级 {'moody': 'Baa1', 'sp': 'BBB+', 'fitch': 'BBB'}"
    )
    rating_internal: CreditRating = Field(..., description="内部评级")
    rating_outlook: RatingOutlook = Field(
        RatingOutlook.STABLE, description="评级展望"
    )

    # 基础财务 (可选，详细财务用 ObligorFinancials)
    total_debt_cny: float | None = Field(None, description="总债务(亿CNY)")
    revenue_cny: float | None = Field(None, description="营收(亿CNY)")

    # AI增强字段
    embedding_vector: list[float] | None = Field(
        None, description="特征向量 (768维)"
    )
    risk_narrative: str | None = Field(
        None, description="LLM生成的风险描述"
    )
    similar_obligors: list[str] | None = Field(
        None, description="相似发行人ID列表"
    )

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def rating_score(self) -> int:
        """评级数值分数 (0-100)"""
        return rating_to_score(self.rating_internal)

    @computed_field
    @property
    def display_name(self) -> str:
        """显示名称"""
        return f"{self.name_cn} ({self.sector.value})"

    class Config:
        use_enum_values = False


class ObligorFinancials(BaseModel):
    """发行人财务数据"""

    obligor_id: str
    report_date: date = Field(..., description="报告日期")

    # 规模指标
    total_assets: float | None = Field(None, description="总资产(亿CNY)")
    total_debt: float | None = Field(None, description="总债务(亿CNY)")
    net_assets: float | None = Field(None, description="净资产(亿CNY)")
    revenue: float | None = Field(None, description="营收(亿CNY)")

    # 盈利指标
    net_profit: float | None = Field(None, description="净利润(亿CNY)")
    ebitda: float | None = Field(None, description="EBITDA(亿CNY)")
    operating_cash_flow: float | None = Field(None, description="经营性现金流(亿CNY)")

    # 杠杆指标
    debt_to_assets: float | None = Field(None, description="资产负债率")
    debt_to_ebitda: float | None = Field(None, description="债务/EBITDA")
    interest_coverage: float | None = Field(None, description="利息覆盖倍数")

    # 流动性指标
    current_ratio: float | None = Field(None, description="流动比率")
    quick_ratio: float | None = Field(None, description="速动比率")
    cash_to_short_debt: float | None = Field(None, description="现金/短期债务")

    @computed_field
    @property
    def leverage_score(self) -> float | None:
        """杠杆综合评分 (0-100, 越高越好)"""
        if self.debt_to_ebitda is None or self.interest_coverage is None:
            return None
        # 简化评分: debt/ebitda < 3 好, > 8 差; coverage > 5 好, < 1 差
        d2e_score = max(0, min(100, (8 - self.debt_to_ebitda) / 5 * 100))
        ic_score = max(0, min(100, self.interest_coverage / 5 * 100))
        return (d2e_score + ic_score) / 2


class BondPosition(BaseModel):
    """单只债券持仓"""

    isin: str = Field(..., description="ISIN代码")
    obligor_id: str = Field(..., description="发行人ID")

    # 债券基本信息
    bond_name: str | None = Field(None, description="债券简称")
    currency: str = Field("USD", description="币种")
    maturity_date: date = Field(..., description="到期日")
    coupon: float = Field(..., description="票面利率")

    # 持仓信息
    nominal: float = Field(..., description="面值(本币)")
    nominal_usd: float = Field(..., description="面值(USD)")
    book_value_usd: float = Field(..., description="账面价值(USD)")
    market_value_usd: float = Field(..., description="市值(USD)")
    accounting_type: str = Field("AC", description="会计分类: AC/FVOCI/FVTPL")

    # 风险指标
    duration: float = Field(..., description="修正久期")
    oas: float | None = Field(None, description="OAS (bps)")
    z_spread: float | None = Field(None, description="Z-Spread (bps)")
    ytm: float | None = Field(None, description="YTM")

    @computed_field
    @property
    def years_to_maturity(self) -> float:
        """剩余期限(年)"""
        days = (self.maturity_date - date.today()).days
        return max(0, days / 365.25)

    @computed_field
    @property
    def credit_dv01(self) -> float:
        """信用DV01 (USD) - 利差变动1bp的损益"""
        return self.market_value_usd * self.duration * 0.0001


# =============================================================================
# Exposure & Aggregation Models
# =============================================================================


class CreditExposure(BaseModel):
    """信用曝光 - 单一发行人维度汇总"""

    obligor: Obligor
    bonds: list[BondPosition] = Field(default_factory=list)

    # 汇总指标
    total_nominal_usd: float = Field(0, description="总面值(USD)")
    total_market_usd: float = Field(0, description="总市值(USD)")
    pct_of_aum: float = Field(0, description="占AUM比例")

    # 加权指标
    weighted_avg_duration: float = Field(0, description="加权平均久期")
    weighted_avg_oas: float = Field(0, description="加权平均OAS")
    credit_dv01_usd: float = Field(0, description="信用DV01(USD)")

    # 到期分布
    maturity_profile: dict[str, float] = Field(
        default_factory=dict,
        description="到期分布 {'0-1Y': 100M, '1-3Y': 200M, ...}"
    )

    @classmethod
    def from_positions(
        cls,
        obligor: Obligor,
        positions: list[BondPosition],
        total_aum: float
    ) -> "CreditExposure":
        """从持仓列表构建曝光汇总"""
        if not positions:
            return cls(obligor=obligor)

        total_nominal = sum(p.nominal_usd for p in positions)
        total_market = sum(p.market_value_usd for p in positions)
        total_dv01 = sum(p.credit_dv01 for p in positions)

        # 市值加权指标
        if total_market > 0:
            weighted_duration = sum(
                p.market_value_usd * p.duration for p in positions
            ) / total_market
            oas_positions = [p for p in positions if p.oas is not None]
            if oas_positions:
                weighted_oas = sum(
                    p.market_value_usd * p.oas for p in oas_positions
                ) / sum(p.market_value_usd for p in oas_positions)
            else:
                weighted_oas = 0
        else:
            weighted_duration = 0
            weighted_oas = 0

        # 到期分布
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


# =============================================================================
# News & Intelligence Models
# =============================================================================


class NewsItem(BaseModel):
    """新闻/公告"""

    news_id: str = Field(..., description="新闻ID")
    timestamp: datetime = Field(..., description="发布时间")
    source: str = Field(..., description="来源: cls/eastmoney/bloomberg/company")
    title: str = Field(..., description="标题")
    content: str = Field(..., description="正文")
    url: str | None = Field(None, description="原文链接")

    # AI增强字段
    obligor_ids: list[str] = Field(
        default_factory=list, description="关联发行人ID (NER提取)"
    )
    summary: str | None = Field(None, description="LLM摘要")
    sentiment: Sentiment | None = Field(None, description="情感倾向")
    sentiment_score: float | None = Field(
        None, description="情感分数 (-1到1)"
    )
    key_events: list[str] | None = Field(None, description="关键事件提取")
    embedding: list[float] | None = Field(None, description="文本向量")

    @field_validator("sentiment_score")
    @classmethod
    def validate_sentiment_score(cls, v: float | None) -> float | None:
        if v is not None and not -1 <= v <= 1:
            raise ValueError("sentiment_score must be between -1 and 1")
        return v


class NewsAnalysisResult(BaseModel):
    """LLM新闻分析结果"""

    summary: str = Field(..., description="一句话摘要")
    sentiment: Sentiment = Field(..., description="情感倾向")
    sentiment_score: float = Field(..., description="情感分数")
    key_events: list[str] = Field(default_factory=list, description="关键事件")
    credit_impact: str | None = Field(None, description="信用影响评估")
    mentioned_entities: list[str] = Field(
        default_factory=list, description="提及的实体"
    )


# =============================================================================
# Signal & Alert Models
# =============================================================================


class SignalResult(BaseModel):
    """信号计算结果"""

    signal_name: str = Field(..., description="信号名称")
    category: SignalCategory = Field(..., description="信号类别")
    obligor_id: str = Field(..., description="发行人ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # 信号值
    value: float = Field(..., description="信号原始值")
    z_score: float | None = Field(None, description="Z-score标准化")
    percentile: float | None = Field(None, description="历史百分位 (0-1)")

    # 阈值
    threshold_warning: float = Field(..., description="警告阈值")
    threshold_critical: float = Field(..., description="严重阈值")

    # 触发判断
    is_triggered: bool = Field(False, description="是否触发预警")
    severity: Severity = Field(Severity.INFO, description="严重程度")

    # 元数据
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="额外信息"
    )

    @classmethod
    def create(
        cls,
        signal_name: str,
        category: SignalCategory,
        obligor_id: str,
        value: float,
        threshold_warning: float,
        threshold_critical: float,
        higher_is_worse: bool = True,
        **kwargs
    ) -> "SignalResult":
        """工厂方法：自动判断触发状态和严重程度"""
        if higher_is_worse:
            is_critical = value >= threshold_critical
            is_warning = value >= threshold_warning
        else:
            is_critical = value <= threshold_critical
            is_warning = value <= threshold_warning

        if is_critical:
            severity = Severity.CRITICAL
            is_triggered = True
        elif is_warning:
            severity = Severity.WARNING
            is_triggered = True
        else:
            severity = Severity.INFO
            is_triggered = False

        return cls(
            signal_name=signal_name,
            category=category,
            obligor_id=obligor_id,
            value=value,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            is_triggered=is_triggered,
            severity=severity,
            **kwargs
        )


class RiskAlert(BaseModel):
    """风险预警"""

    alert_id: str = Field(..., description="预警ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # 预警分类
    severity: Severity = Field(..., description="严重程度")
    category: AlertCategory = Field(..., description="预警类别")

    # 关联信息
    obligor_id: str = Field(..., description="发行人ID")
    obligor_name: str = Field(..., description="发行人名称")
    signal_name: str = Field(..., description="触发信号")

    # 预警内容
    message: str = Field(..., description="预警消息")
    metric_value: float = Field(..., description="指标值")
    threshold: float = Field(..., description="触发阈值")

    # 处理状态
    status: AlertStatus = Field(AlertStatus.PENDING, description="处理状态")
    assigned_to: str | None = Field(None, description="分配给")
    resolution_note: str | None = Field(None, description="处置备注")
    resolved_at: datetime | None = Field(None, description="解决时间")

    # AI增强
    ai_summary: str | None = Field(None, description="AI风险摘要")
    related_news: list[str] | None = Field(None, description="相关新闻ID")

    @classmethod
    def from_signal(
        cls,
        signal: SignalResult,
        obligor_name: str,
        message: str | None = None
    ) -> "RiskAlert":
        """从信号结果创建预警"""
        import uuid

        if message is None:
            message = (
                f"{obligor_name} 触发{signal.signal_name}信号: "
                f"当前值 {signal.value:.2f}, 阈值 {signal.threshold_warning:.2f}"
            )

        # 映射 SignalCategory -> AlertCategory
        category_map = {
            SignalCategory.MARKET: AlertCategory.SPREAD,
            SignalCategory.FUNDAMENTAL: AlertCategory.FUNDAMENTAL,
            SignalCategory.NEWS: AlertCategory.NEWS,
            SignalCategory.CONCENTRATION: AlertCategory.CONCENTRATION,
            SignalCategory.COMPOSITE: AlertCategory.FUNDAMENTAL,
        }

        return cls(
            alert_id=str(uuid.uuid4())[:8],
            severity=signal.severity,
            category=category_map.get(signal.category, AlertCategory.FUNDAMENTAL),
            obligor_id=signal.obligor_id,
            obligor_name=obligor_name,
            signal_name=signal.signal_name,
            message=message,
            metric_value=signal.value,
            threshold=signal.threshold_warning,
        )


# =============================================================================
# RAG & Search Models
# =============================================================================


class RAGResponse(BaseModel):
    """RAG问答响应"""

    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="生成的答案")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="来源文档"
    )
    confidence: float = Field(0.5, description="置信度 (0-1)")
    obligor_context: list[str] | None = Field(
        None, description="涉及的发行人ID"
    )


class SimilarObligor(BaseModel):
    """相似发行人搜索结果"""

    obligor_id: str
    obligor_name: str
    similarity_score: float = Field(..., description="相似度 (0-1)")
    sector: Sector
    rating: CreditRating
    common_factors: list[str] = Field(
        default_factory=list, description="相似因素"
    )
