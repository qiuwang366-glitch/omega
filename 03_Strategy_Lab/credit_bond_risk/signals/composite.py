"""
Credit Bond Risk - Composite Signals

Aggregated signals that combine multiple underlying signals:
- Simple composite (any trigger)
- Weighted composite (weighted average)
- Risk score (normalized 0-100)
"""

from typing import ClassVar

import numpy as np

from .base import Signal, SignalContext, SignalRegistry
from ..core.models import SignalResult
from ..core.enums import SignalCategory, Severity


@SignalRegistry.register
class CompositeSignal(Signal):
    """
    复合信号 - 任一子信号触发即触发

    用途: 组合多个信号，任一触发即预警
    """

    name: ClassVar[str] = "composite"
    category: ClassVar[SignalCategory] = SignalCategory.COMPOSITE
    description: ClassVar[str] = "复合信号"

    def __init__(
        self,
        signals: list[Signal],
        threshold_warning: float = 1,  # 1个WARNING
        threshold_critical: float = 1,  # 1个CRITICAL
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.signals = signals

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        # 计算所有子信号
        sub_results = []
        warning_count = 0
        critical_count = 0

        for signal in self.signals:
            result = signal.compute(obligor_id, context)
            sub_results.append(result)

            if result.severity == Severity.WARNING:
                warning_count += 1
            elif result.severity == Severity.CRITICAL:
                critical_count += 1

        # 确定复合信号的严重程度
        if critical_count >= self.threshold_critical:
            severity = Severity.CRITICAL
            is_triggered = True
            value = critical_count
        elif warning_count >= self.threshold_warning:
            severity = Severity.WARNING
            is_triggered = True
            value = warning_count
        else:
            severity = Severity.INFO
            is_triggered = False
            value = 0

        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return SignalResult(
            signal_name=self.name,
            category=self.category,
            obligor_id=obligor_id,
            value=float(value),
            threshold_warning=self.threshold_warning,
            threshold_critical=self.threshold_critical,
            is_triggered=is_triggered,
            severity=severity,
            metadata={
                "sub_signals": [
                    {
                        "name": r.signal_name,
                        "value": r.value,
                        "severity": r.severity.value,
                    }
                    for r in sub_results
                ],
                "warning_count": warning_count,
                "critical_count": critical_count,
                "obligor_name": obligor_name,
            },
        )


@SignalRegistry.register
class WeightedCompositeSignal(Signal):
    """
    加权复合信号 - 子信号加权平均

    用途: 综合多个维度计算风险分数
    """

    name: ClassVar[str] = "weighted_composite"
    category: ClassVar[SignalCategory] = SignalCategory.COMPOSITE
    description: ClassVar[str] = "加权复合信号"

    def __init__(
        self,
        signals: list[Signal],
        weights: list[float] | None = None,
        threshold_warning: float = 0.5,
        threshold_critical: float = 0.8,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.signals = signals

        if weights is None:
            # 默认等权重
            self.weights = [1.0 / len(signals)] * len(signals)
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def _normalize_value(self, result: SignalResult) -> float:
        """将信号值归一化到0-1"""
        if result.percentile is not None:
            return result.percentile

        # 基于阈值归一化
        if result.value <= result.threshold_warning:
            return result.value / result.threshold_warning * 0.5
        elif result.value <= result.threshold_critical:
            return 0.5 + (result.value - result.threshold_warning) / (
                result.threshold_critical - result.threshold_warning
            ) * 0.3
        else:
            return min(1.0, 0.8 + (result.value - result.threshold_critical) / result.threshold_critical * 0.2)

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        sub_results = []
        weighted_sum = 0.0

        for signal, weight in zip(self.signals, self.weights):
            result = signal.compute(obligor_id, context)
            sub_results.append(result)

            normalized = self._normalize_value(result)
            weighted_sum += normalized * weight

        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return self._create_result(
            obligor_id=obligor_id,
            value=weighted_sum,
            metadata={
                "sub_signals": [
                    {
                        "name": r.signal_name,
                        "value": r.value,
                        "weight": w,
                        "normalized": self._normalize_value(r),
                    }
                    for r, w in zip(sub_results, self.weights)
                ],
                "obligor_name": obligor_name,
            },
        )


@SignalRegistry.register
class CreditRiskScoreSignal(Signal):
    """
    综合信用风险评分 (0-100)

    综合考虑:
    - 集中度 (20%)
    - 利差 (25%)
    - 评级 (25%)
    - 舆情 (15%)
    - 到期压力 (15%)
    """

    name: ClassVar[str] = "credit_risk_score"
    category: ClassVar[SignalCategory] = SignalCategory.COMPOSITE
    description: ClassVar[str] = "综合信用风险评分"

    # 各维度权重
    DIMENSION_WEIGHTS = {
        "concentration": 0.20,
        "spread": 0.25,
        "rating": 0.25,
        "news": 0.15,
        "maturity": 0.15,
    }

    def __init__(
        self,
        threshold_warning: float = 60,  # 60分预警
        threshold_critical: float = 80,  # 80分严重
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)

    def _compute_dimension_score(
        self,
        dimension: str,
        obligor_id: str,
        context: SignalContext,
    ) -> tuple[float, dict]:
        """计算单一维度的风险分数 (0-100)"""

        exposure = context.get_exposure(obligor_id)
        obligor = context.get_obligor(obligor_id)

        if dimension == "concentration":
            if exposure is None:
                return 0, {"reason": "no_exposure"}
            # 占比映射到分数
            pct = exposure.pct_of_aum
            score = min(100, pct / 0.05 * 100)  # 5%=100分
            return score, {"pct_of_aum": pct}

        elif dimension == "spread":
            if exposure is None or exposure.weighted_avg_oas is None:
                return 50, {"reason": "no_spread_data"}
            # OAS百分位映射到分数
            # 简化处理：假设200bps=50分，500bps=100分
            oas = exposure.weighted_avg_oas
            score = min(100, max(0, (oas - 100) / 4))
            return score, {"oas": oas}

        elif dimension == "rating":
            if obligor is None:
                return 50, {"reason": "no_obligor"}
            # 评级分数反转 (评级越低分数越高)
            rating_score = obligor.rating_score
            risk_score = 100 - rating_score
            return risk_score, {"rating": obligor.rating_internal.value}

        elif dimension == "news":
            news_items = context.get_news(obligor_id, days=7)
            if not news_items:
                return 30, {"reason": "no_news"}  # 无新闻=低风险

            # 情感分数映射
            scored = [n for n in news_items if n.sentiment_score is not None]
            if not scored:
                return 30, {"reason": "no_scored_news"}

            avg_sentiment = float(np.mean([n.sentiment_score for n in scored]))  # type: ignore
            # -1 -> 100, 0 -> 50, 1 -> 0
            score = (1 - avg_sentiment) / 2 * 100
            return score, {"avg_sentiment": avg_sentiment, "news_count": len(scored)}

        elif dimension == "maturity":
            if exposure is None:
                return 50, {"reason": "no_exposure"}
            # 短期到期占比
            maturity = exposure.maturity_profile
            short_term = maturity.get("0-1Y", 0)
            total = sum(maturity.values())
            if total == 0:
                return 30, {"reason": "no_maturity_data"}
            short_pct = short_term / total
            score = min(100, short_pct / 0.4 * 100)  # 40%=100分
            return score, {"short_term_pct": short_pct}

        return 50, {"reason": "unknown_dimension"}

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        dimension_scores = {}
        total_score = 0.0

        for dimension, weight in self.DIMENSION_WEIGHTS.items():
            score, details = self._compute_dimension_score(
                dimension, obligor_id, context
            )
            dimension_scores[dimension] = {
                "score": score,
                "weight": weight,
                "weighted_score": score * weight,
                "details": details,
            }
            total_score += score * weight

        obligor = context.get_obligor(obligor_id)
        obligor_name = obligor.name_cn if obligor else obligor_id

        return self._create_result(
            obligor_id=obligor_id,
            value=total_score,
            metadata={
                "dimensions": dimension_scores,
                "obligor_name": obligor_name,
            },
        )
