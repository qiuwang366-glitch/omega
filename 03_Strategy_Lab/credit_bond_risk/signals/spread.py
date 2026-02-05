"""
Credit Bond Risk - Spread Signals

Signals for monitoring credit spread movements:
- Historical percentile
- Z-score
- Absolute/relative changes
"""

from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np

from .base import Signal, SignalContext, SignalRegistry
from ..core.models import SignalResult
from ..core.enums import SignalCategory


@SignalRegistry.register
class SpreadPercentileSignal(Signal):
    """
    利差历史百分位信号

    计算: 当前OAS在历史分布中的百分位
    触发: 百分位超过阈值 (e.g., > 90%分位表示利差处于历史高位)
    """

    name: ClassVar[str] = "spread_percentile"
    category: ClassVar[SignalCategory] = SignalCategory.MARKET
    description: ClassVar[str] = "OAS历史百分位"

    def __init__(
        self,
        threshold_warning: float = 0.85,
        threshold_critical: float = 0.95,
        lookback_days: int = 252,
        min_data_points: int = 60,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        exposure = context.get_exposure(obligor_id)

        if exposure is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.5,  # 中性值
                percentile=0.5,
                metadata={"reason": "no_exposure"},
            )

        current_oas = exposure.weighted_avg_oas
        if current_oas is None or current_oas == 0:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.5,
                percentile=0.5,
                metadata={"reason": "no_oas_data"},
            )

        # 获取历史利差数据
        spread_history = context.get_spread_series(obligor_id)

        # 过滤到lookback窗口
        cutoff = context.as_of_date - timedelta(days=self.lookback_days)
        historical_spreads = [
            v for k, v in spread_history.items()
            if datetime.fromisoformat(k) >= cutoff
        ]

        if len(historical_spreads) < self.min_data_points:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.5,
                percentile=0.5,
                metadata={
                    "reason": "insufficient_history",
                    "data_points": len(historical_spreads),
                    "required": self.min_data_points,
                },
            )

        # 计算百分位
        historical_array = np.array(historical_spreads)
        percentile = float(np.sum(historical_array <= current_oas) / len(historical_array))

        return self._create_result(
            obligor_id=obligor_id,
            value=percentile,
            percentile=percentile,
            metadata={
                "current_oas": current_oas,
                "historical_mean": float(np.mean(historical_array)),
                "historical_std": float(np.std(historical_array)),
                "historical_min": float(np.min(historical_array)),
                "historical_max": float(np.max(historical_array)),
                "data_points": len(historical_spreads),
                "obligor_name": exposure.obligor.name_cn,
            },
        )


@SignalRegistry.register
class SpreadZScoreSignal(Signal):
    """
    利差Z-Score信号

    计算: (当前OAS - 历史均值) / 历史标准差
    触发: Z-score超过阈值
    """

    name: ClassVar[str] = "spread_zscore"
    category: ClassVar[SignalCategory] = SignalCategory.MARKET
    description: ClassVar[str] = "OAS Z-Score"

    def __init__(
        self,
        threshold_warning: float = 2.0,
        threshold_critical: float = 3.0,
        lookback_days: int = 252,
        min_data_points: int = 60,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        exposure = context.get_exposure(obligor_id)

        if exposure is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                z_score=0.0,
                metadata={"reason": "no_exposure"},
            )

        current_oas = exposure.weighted_avg_oas
        if current_oas is None or current_oas == 0:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                z_score=0.0,
                metadata={"reason": "no_oas_data"},
            )

        # 获取历史利差数据
        spread_history = context.get_spread_series(obligor_id)

        cutoff = context.as_of_date - timedelta(days=self.lookback_days)
        historical_spreads = [
            v for k, v in spread_history.items()
            if datetime.fromisoformat(k) >= cutoff
        ]

        if len(historical_spreads) < self.min_data_points:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                z_score=0.0,
                metadata={
                    "reason": "insufficient_history",
                    "data_points": len(historical_spreads),
                },
            )

        # 计算Z-score
        historical_array = np.array(historical_spreads)
        mean = float(np.mean(historical_array))
        std = float(np.std(historical_array))

        if std == 0:
            z_score = 0.0
        else:
            z_score = (current_oas - mean) / std

        return self._create_result(
            obligor_id=obligor_id,
            value=z_score,
            z_score=z_score,
            metadata={
                "current_oas": current_oas,
                "historical_mean": mean,
                "historical_std": std,
                "obligor_name": exposure.obligor.name_cn,
            },
        )


@SignalRegistry.register
class SpreadChangeSignal(Signal):
    """
    利差变动信号

    计算: 当前OAS与N天前的变动 (bps)
    触发: 变动超过阈值
    """

    name: ClassVar[str] = "spread_change"
    category: ClassVar[SignalCategory] = SignalCategory.MARKET
    description: ClassVar[str] = "OAS变动"

    def __init__(
        self,
        threshold_warning: float = 30,  # bps
        threshold_critical: float = 80,  # bps
        lookback_days: int = 5,  # 周变动
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.lookback_days = lookback_days

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        exposure = context.get_exposure(obligor_id)

        if exposure is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_exposure"},
            )

        current_oas = exposure.weighted_avg_oas
        if current_oas is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_oas_data"},
            )

        # 获取N天前的利差
        spread_history = context.get_spread_series(obligor_id)
        target_date = context.as_of_date - timedelta(days=self.lookback_days)

        # 找最近的历史数据点
        past_oas = None
        for date_str, oas in sorted(spread_history.items(), reverse=True):
            date = datetime.fromisoformat(date_str)
            if date <= target_date:
                past_oas = oas
                break

        if past_oas is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_historical_data"},
            )

        # 计算变动 (bps)
        change = current_oas - past_oas

        return self._create_result(
            obligor_id=obligor_id,
            value=abs(change),  # 使用绝对值，走阔和收窄都可能是风险
            metadata={
                "current_oas": current_oas,
                "past_oas": past_oas,
                "change_bps": change,
                "lookback_days": self.lookback_days,
                "direction": "widen" if change > 0 else "tighten",
                "obligor_name": exposure.obligor.name_cn,
            },
        )


@SignalRegistry.register
class SpreadVsPeersSignal(Signal):
    """
    利差同业比较信号

    计算: 当前OAS vs 同评级/同行业peers的中位数
    触发: 偏离度超过阈值
    """

    name: ClassVar[str] = "spread_vs_peers"
    category: ClassVar[SignalCategory] = SignalCategory.MARKET
    description: ClassVar[str] = "OAS vs 同业"

    def __init__(
        self,
        threshold_warning: float = 50,  # bps above peers
        threshold_critical: float = 100,  # bps above peers
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        exposure = context.get_exposure(obligor_id)

        if exposure is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_exposure"},
            )

        current_oas = exposure.weighted_avg_oas
        obligor = exposure.obligor

        if current_oas is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_oas_data"},
            )

        # 找到同行业同评级的peers
        peer_spreads = []
        peer_names = []

        for other_id, other_exposure in context.exposures.items():
            if other_id == obligor_id:
                continue

            other_obligor = other_exposure.obligor

            # 匹配条件：同行业 + 评级差距在1档以内
            if (
                other_obligor.sector == obligor.sector
                and abs(other_obligor.rating_score - obligor.rating_score) <= 5
                and other_exposure.weighted_avg_oas is not None
            ):
                peer_spreads.append(other_exposure.weighted_avg_oas)
                peer_names.append(other_obligor.name_cn)

        if len(peer_spreads) < 3:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={
                    "reason": "insufficient_peers",
                    "peer_count": len(peer_spreads),
                },
            )

        # 计算与peers中位数的偏离
        peer_median = float(np.median(peer_spreads))
        deviation = current_oas - peer_median

        return self._create_result(
            obligor_id=obligor_id,
            value=deviation,
            metadata={
                "current_oas": current_oas,
                "peer_median": peer_median,
                "peer_count": len(peer_spreads),
                "peer_names": peer_names[:5],  # 前5个
                "obligor_name": obligor.name_cn,
            },
        )
