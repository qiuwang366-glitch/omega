"""
Credit Bond Risk - News/Sentiment Signals

LLM-powered signals for monitoring news and sentiment:
- Aggregate sentiment score
- Negative news volume
- Key event detection
"""

from datetime import timedelta
from typing import ClassVar

import numpy as np

from .base import Signal, SignalContext, SignalRegistry
from ..core.models import SignalResult, NewsItem
from ..core.enums import SignalCategory, Sentiment


@SignalRegistry.register
class NewsSentimentSignal(Signal):
    """
    舆情情感信号

    计算: 近N天新闻的加权平均情感分数
    触发: 情感分数低于阈值 (负面)

    Note: 需要新闻已通过LLM分析获得sentiment_score
    """

    name: ClassVar[str] = "news_sentiment"
    category: ClassVar[SignalCategory] = SignalCategory.NEWS
    description: ClassVar[str] = "新闻情感分数"

    # 新闻来源权重
    DEFAULT_SOURCE_WEIGHTS = {
        "company": 1.5,    # 公司公告权重最高
        "bloomberg": 1.2,
        "cls": 1.0,
        "eastmoney": 0.8,
        "other": 0.5,
    }

    def __init__(
        self,
        threshold_warning: float = -0.3,
        threshold_critical: float = -0.6,
        lookback_days: int = 7,
        source_weights: dict[str, float] | None = None,
    ):
        # 情感分数越低越差，所以higher_is_worse=False
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=False)
        self.lookback_days = lookback_days
        self.source_weights = source_weights or self.DEFAULT_SOURCE_WEIGHTS

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        # 获取近期新闻
        news_items = context.get_news(obligor_id, days=self.lookback_days)

        if not news_items:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,  # 无新闻=中性
                metadata={
                    "reason": "no_news",
                    "lookback_days": self.lookback_days,
                },
            )

        # 过滤有情感分数的新闻
        scored_news = [n for n in news_items if n.sentiment_score is not None]

        if not scored_news:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={
                    "reason": "no_scored_news",
                    "total_news": len(news_items),
                },
            )

        # 计算加权平均情感分数
        total_weight = 0.0
        weighted_sum = 0.0

        for news in scored_news:
            weight = self.source_weights.get(news.source, 0.5)
            weighted_sum += news.sentiment_score * weight  # type: ignore
            total_weight += weight

        avg_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0

        # 统计情感分布
        sentiments = [n.sentiment for n in scored_news if n.sentiment]
        sentiment_dist = {
            "positive": sum(1 for s in sentiments if s == Sentiment.POSITIVE),
            "neutral": sum(1 for s in sentiments if s == Sentiment.NEUTRAL),
            "negative": sum(1 for s in sentiments if s == Sentiment.NEGATIVE),
        }

        # 获取发行人名称
        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return self._create_result(
            obligor_id=obligor_id,
            value=avg_sentiment,
            metadata={
                "news_count": len(scored_news),
                "sentiment_distribution": sentiment_dist,
                "lookback_days": self.lookback_days,
                "recent_headlines": [n.title for n in scored_news[:5]],
                "obligor_name": obligor_name,
            },
        )


@SignalRegistry.register
class NewsVolumeSignal(Signal):
    """
    负面新闻数量信号

    计算: 近N天负面新闻数量
    触发: 数量超过阈值
    """

    name: ClassVar[str] = "news_volume"
    category: ClassVar[SignalCategory] = SignalCategory.NEWS
    description: ClassVar[str] = "负面新闻数量"

    def __init__(
        self,
        threshold_warning: float = 3,
        threshold_critical: float = 5,
        lookback_days: int = 7,
        negative_threshold: float = -0.2,  # sentiment_score阈值
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.lookback_days = lookback_days
        self.negative_threshold = negative_threshold

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        news_items = context.get_news(obligor_id, days=self.lookback_days)

        if not news_items:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_news"},
            )

        # 统计负面新闻
        negative_news = [
            n for n in news_items
            if n.sentiment_score is not None
            and n.sentiment_score < self.negative_threshold
        ]

        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return self._create_result(
            obligor_id=obligor_id,
            value=float(len(negative_news)),
            metadata={
                "total_news": len(news_items),
                "negative_news": len(negative_news),
                "negative_headlines": [n.title for n in negative_news[:5]],
                "lookback_days": self.lookback_days,
                "obligor_name": obligor_name,
            },
        )


@SignalRegistry.register
class NewsEventSignal(Signal):
    """
    关键事件信号

    计算: 检测特定类型的风险事件
    触发: 检测到高风险事件

    事件类型:
    - 违约/展期
    - 评级下调
    - 高管变动
    - 监管处罚
    - 诉讼/仲裁
    """

    name: ClassVar[str] = "news_event"
    category: ClassVar[SignalCategory] = SignalCategory.NEWS
    description: ClassVar[str] = "风险事件检测"

    # 事件关键词和风险分数
    EVENT_KEYWORDS = {
        "default": (["违约", "逾期", "展期", "兑付困难", "未能偿付"], 10),
        "rating": (["评级下调", "列入观察", "负面展望"], 5),
        "management": (["高管离职", "总经理辞职", "董事长变更", "实控人变更"], 3),
        "regulatory": (["监管处罚", "行政处罚", "立案调查", "警示函"], 5),
        "litigation": (["诉讼", "仲裁", "冻结", "查封", "强制执行"], 4),
        "restructure": (["债务重组", "资产重组", "破产", "清算"], 8),
    }

    def __init__(
        self,
        threshold_warning: float = 3,
        threshold_critical: float = 8,
        lookback_days: int = 30,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.lookback_days = lookback_days

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        news_items = context.get_news(obligor_id, days=self.lookback_days)

        if not news_items:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_news"},
            )

        # 检测事件
        detected_events = []
        total_risk_score = 0.0

        for news in news_items:
            text = f"{news.title} {news.content}"
            for event_type, (keywords, score) in self.EVENT_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text:
                        detected_events.append({
                            "type": event_type,
                            "keyword": keyword,
                            "title": news.title,
                            "date": news.timestamp.isoformat(),
                            "score": score,
                        })
                        total_risk_score += score
                        break  # 每类事件每条新闻只计一次

        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return self._create_result(
            obligor_id=obligor_id,
            value=total_risk_score,
            metadata={
                "detected_events": detected_events[:10],
                "event_count": len(detected_events),
                "total_news": len(news_items),
                "lookback_days": self.lookback_days,
                "obligor_name": obligor_name,
            },
        )


@SignalRegistry.register
class NewsTrendSignal(Signal):
    """
    舆情趋势信号

    计算: 近期情感趋势 (近3天 vs 前7天)
    触发: 情感显著恶化
    """

    name: ClassVar[str] = "news_trend"
    category: ClassVar[SignalCategory] = SignalCategory.NEWS
    description: ClassVar[str] = "舆情趋势"

    def __init__(
        self,
        threshold_warning: float = -0.2,  # 情感恶化0.2
        threshold_critical: float = -0.4,  # 情感恶化0.4
        recent_days: int = 3,
        baseline_days: int = 7,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=False)
        self.recent_days = recent_days
        self.baseline_days = baseline_days

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        # 获取近期和基准期新闻
        all_news = context.get_news(
            obligor_id, days=self.recent_days + self.baseline_days
        )

        if len(all_news) < 3:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "insufficient_news"},
            )

        cutoff = context.as_of_date - timedelta(days=self.recent_days)

        recent_news = [n for n in all_news if n.timestamp >= cutoff]
        baseline_news = [n for n in all_news if n.timestamp < cutoff]

        # 计算情感均值
        def avg_sentiment(news_list: list[NewsItem]) -> float | None:
            scores = [n.sentiment_score for n in news_list if n.sentiment_score is not None]
            return float(np.mean(scores)) if scores else None

        recent_sentiment = avg_sentiment(recent_news)
        baseline_sentiment = avg_sentiment(baseline_news)

        if recent_sentiment is None or baseline_sentiment is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "insufficient_scored_news"},
            )

        # 计算趋势 (负数=恶化)
        trend = recent_sentiment - baseline_sentiment

        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return self._create_result(
            obligor_id=obligor_id,
            value=trend,
            metadata={
                "recent_sentiment": recent_sentiment,
                "baseline_sentiment": baseline_sentiment,
                "recent_count": len(recent_news),
                "baseline_count": len(baseline_news),
                "trend_direction": "improving" if trend > 0 else "deteriorating",
                "obligor_name": obligor_name,
            },
        )
