"""
Credit Bond Risk - Signal Base Classes

Abstract base class and registry for the signal system.
Inspired by JPM Athena's signal library pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from ..core.models import (
    CreditExposure,
    SignalResult,
    NewsItem,
    Obligor,
)
from ..core.enums import SignalCategory, Severity
from ..core.config import CreditRiskConfig


@dataclass
class SignalContext:
    """
    信号计算上下文 - 提供所有信号计算所需的数据

    设计理念：
    - 信号本身是无状态的计算逻辑
    - 所有状态数据通过Context注入
    - 便于测试和回测
    """

    # 持仓数据
    exposures: dict[str, CreditExposure] = field(default_factory=dict)
    total_aum: float = 50_000_000_000  # $50B default

    # 发行人主数据
    obligors: dict[str, Obligor] = field(default_factory=dict)

    # 新闻数据
    news_items: list[NewsItem] = field(default_factory=list)
    news_by_obligor: dict[str, list[NewsItem]] = field(default_factory=dict)

    # 历史利差数据 {obligor_id: {date: oas}}
    spread_history: dict[str, dict[str, float]] = field(default_factory=dict)

    # 评级历史 {obligor_id: [(date, old_rating, new_rating)]}
    rating_history: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)

    # 配置
    config: CreditRiskConfig | None = None

    # 计算时间点
    as_of_date: datetime = field(default_factory=datetime.now)

    def get_exposure(self, obligor_id: str) -> CreditExposure | None:
        """获取单一发行人曝光"""
        return self.exposures.get(obligor_id)

    def get_obligor(self, obligor_id: str) -> Obligor | None:
        """获取发行人信息"""
        return self.obligors.get(obligor_id)

    def get_news(self, obligor_id: str, days: int = 7) -> list[NewsItem]:
        """获取发行人近期新闻"""
        from datetime import timedelta

        cutoff = self.as_of_date - timedelta(days=days)
        news = self.news_by_obligor.get(obligor_id, [])
        return [n for n in news if n.timestamp >= cutoff]

    def get_spread_series(self, obligor_id: str) -> dict[str, float]:
        """获取发行人利差历史"""
        return self.spread_history.get(obligor_id, {})


class Signal(ABC):
    """
    信号抽象基类

    所有风险信号必须实现此接口，确保：
    1. 统一的计算接口
    2. 可组合性（复合信号）
    3. 可追溯性（元数据）
    """

    # 类属性 - 子类必须定义
    name: ClassVar[str]
    category: ClassVar[SignalCategory]
    description: ClassVar[str] = ""

    def __init__(
        self,
        threshold_warning: float,
        threshold_critical: float,
        higher_is_worse: bool = True,
    ):
        """
        Args:
            threshold_warning: 警告阈值
            threshold_critical: 严重阈值
            higher_is_worse: True表示值越高风险越大，False相反
        """
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical
        self.higher_is_worse = higher_is_worse

    @abstractmethod
    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """
        计算单一发行人的信号值

        Args:
            obligor_id: 发行人ID
            context: 计算上下文

        Returns:
            SignalResult with value, severity, and metadata
        """
        pass

    def compute_batch(
        self, obligor_ids: list[str], context: SignalContext
    ) -> list[SignalResult]:
        """
        批量计算多个发行人的信号

        默认实现：循环调用compute()
        子类可覆写以优化性能（向量化计算）
        """
        return [self.compute(oid, context) for oid in obligor_ids]

    def compute_portfolio(self, context: SignalContext) -> list[SignalResult]:
        """
        计算组合内所有发行人的信号

        Returns:
            所有触发预警的SignalResult列表
        """
        all_results = self.compute_batch(list(context.exposures.keys()), context)
        return [r for r in all_results if r.is_triggered]

    def _create_result(
        self,
        obligor_id: str,
        value: float,
        z_score: float | None = None,
        percentile: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SignalResult:
        """
        工厂方法：创建SignalResult并自动判断严重程度
        """
        return SignalResult.create(
            signal_name=self.name,
            category=self.category,
            obligor_id=obligor_id,
            value=value,
            threshold_warning=self.threshold_warning,
            threshold_critical=self.threshold_critical,
            higher_is_worse=self.higher_is_worse,
            z_score=z_score,
            percentile=percentile,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"warning={self.threshold_warning}, "
            f"critical={self.threshold_critical})"
        )


class SignalRegistry:
    """
    信号注册表 - 管理所有可用信号

    用途：
    1. 信号发现和实例化
    2. 配置驱动的信号组合
    3. 批量信号计算
    """

    _signals: ClassVar[dict[str, type[Signal]]] = {}

    @classmethod
    def register(cls, signal_class: type[Signal]) -> type[Signal]:
        """装饰器：注册信号类"""
        cls._signals[signal_class.name] = signal_class
        return signal_class

    @classmethod
    def get(cls, name: str) -> type[Signal] | None:
        """获取信号类"""
        return cls._signals.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        """列出所有已注册信号"""
        return list(cls._signals.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Signal:
        """实例化信号"""
        signal_class = cls.get(name)
        if signal_class is None:
            raise ValueError(f"Unknown signal: {name}")
        return signal_class(**kwargs)

    @classmethod
    def create_from_config(cls, config: CreditRiskConfig) -> list[Signal]:
        """从配置创建所有信号实例"""
        signals = []

        # Concentration signals
        from .concentration import ConcentrationSignal, HHISignal

        signals.append(ConcentrationSignal(
            threshold_warning=config.concentration.single_obligor_warning,
            threshold_critical=config.concentration.single_obligor_critical,
        ))
        signals.append(HHISignal(
            threshold_warning=config.concentration.hhi_warning,
            threshold_critical=config.concentration.hhi_critical,
        ))

        # Spread signals
        from .spread import SpreadPercentileSignal, SpreadZScoreSignal

        signals.append(SpreadPercentileSignal(
            threshold_warning=config.spread.percentile_warning,
            threshold_critical=config.spread.percentile_critical,
            lookback_days=config.spread.lookback_days,
        ))
        signals.append(SpreadZScoreSignal(
            threshold_warning=config.spread.zscore_warning,
            threshold_critical=config.spread.zscore_critical,
        ))

        # News signals (if LLM enabled)
        if config.enable_llm_analysis:
            from .news import NewsSentimentSignal

            signals.append(NewsSentimentSignal(
                threshold_warning=config.news.sentiment_warning,
                threshold_critical=config.news.sentiment_critical,
                lookback_days=config.news.lookback_days,
            ))

        return signals

    @classmethod
    def compute_all(
        cls, signals: list[Signal], context: SignalContext
    ) -> list[SignalResult]:
        """
        计算所有信号并返回触发的预警

        Args:
            signals: 信号实例列表
            context: 计算上下文

        Returns:
            所有触发预警的SignalResult列表
        """
        all_triggered = []
        for signal in signals:
            triggered = signal.compute_portfolio(context)
            all_triggered.extend(triggered)
        return all_triggered
