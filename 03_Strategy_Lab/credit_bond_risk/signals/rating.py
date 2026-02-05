"""
Credit Bond Risk - Rating Signals

Signals for monitoring credit rating changes:
- Rating downgrades
- Outlook changes
- Watch list additions
"""

from datetime import datetime, timedelta
from typing import ClassVar

from .base import Signal, SignalContext, SignalRegistry
from ..core.models import SignalResult
from ..core.enums import SignalCategory, CreditRating, RatingOutlook, rating_to_score


@SignalRegistry.register
class RatingChangeSignal(Signal):
    """
    评级变动信号

    计算: 最近N天内的评级变动档数
    触发: 下调档数超过阈值
    """

    name: ClassVar[str] = "rating_change"
    category: ClassVar[SignalCategory] = SignalCategory.FUNDAMENTAL
    description: ClassVar[str] = "评级变动"

    def __init__(
        self,
        threshold_warning: float = 1,  # 下调1档
        threshold_critical: float = 2,  # 下调2档
        lookback_days: int = 30,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.lookback_days = lookback_days

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        obligor = context.get_obligor(obligor_id)

        if obligor is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_obligor"},
            )

        # 获取评级历史
        rating_history = context.rating_history.get(obligor_id, [])

        if not rating_history:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_rating_history"},
            )

        # 过滤到lookback窗口内的变动
        cutoff = context.as_of_date - timedelta(days=self.lookback_days)
        recent_changes = [
            (date, old, new)
            for date, old, new in rating_history
            if datetime.fromisoformat(date) >= cutoff
        ]

        if not recent_changes:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_recent_changes"},
            )

        # 计算累计变动档数 (正数=下调，负数=上调)
        total_notches = 0
        change_details = []

        for date, old_rating, new_rating in recent_changes:
            try:
                old_score = rating_to_score(CreditRating(old_rating))
                new_score = rating_to_score(CreditRating(new_rating))
                notches = (old_score - new_score) / 5  # 每5分一档
                total_notches += notches
                change_details.append({
                    "date": date,
                    "from": old_rating,
                    "to": new_rating,
                    "notches": notches,
                })
            except ValueError:
                continue

        return self._create_result(
            obligor_id=obligor_id,
            value=total_notches,
            metadata={
                "current_rating": obligor.rating_internal.value,
                "changes": change_details,
                "lookback_days": self.lookback_days,
                "obligor_name": obligor.name_cn,
            },
        )


@SignalRegistry.register
class OutlookChangeSignal(Signal):
    """
    评级展望信号

    计算: 当前展望状态
    触发: 展望为负面或列入观察(负面)
    """

    name: ClassVar[str] = "outlook_change"
    category: ClassVar[SignalCategory] = SignalCategory.FUNDAMENTAL
    description: ClassVar[str] = "评级展望"

    # 展望风险分数
    OUTLOOK_SCORES = {
        RatingOutlook.POSITIVE: 0,
        RatingOutlook.STABLE: 0,
        RatingOutlook.NEGATIVE: 1,
        RatingOutlook.WATCH_POS: 0.5,
        RatingOutlook.WATCH_NEG: 2,
        RatingOutlook.DEVELOPING: 0.5,
    }

    def __init__(
        self,
        threshold_warning: float = 1,  # NEGATIVE
        threshold_critical: float = 2,  # WATCH_NEG
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        obligor = context.get_obligor(obligor_id)

        if obligor is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_obligor"},
            )

        outlook = obligor.rating_outlook
        score = self.OUTLOOK_SCORES.get(outlook, 0)

        return self._create_result(
            obligor_id=obligor_id,
            value=score,
            metadata={
                "outlook": outlook.value,
                "rating": obligor.rating_internal.value,
                "obligor_name": obligor.name_cn,
            },
        )


@SignalRegistry.register
class RatingVsPeersSignal(Signal):
    """
    评级同业比较信号

    计算: 发行人评级 vs 同行业peers平均评级
    触发: 评级显著低于同业
    """

    name: ClassVar[str] = "rating_vs_peers"
    category: ClassVar[SignalCategory] = SignalCategory.FUNDAMENTAL
    description: ClassVar[str] = "评级 vs 同业"

    def __init__(
        self,
        threshold_warning: float = 10,  # 低于peers 2档
        threshold_critical: float = 15,  # 低于peers 3档
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        obligor = context.get_obligor(obligor_id)

        if obligor is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_obligor"},
            )

        obligor_score = obligor.rating_score

        # 找到同行业peers
        peer_scores = []
        for other_id, other_obligor in context.obligors.items():
            if other_id == obligor_id:
                continue
            if other_obligor.sector == obligor.sector:
                peer_scores.append(other_obligor.rating_score)

        if len(peer_scores) < 3:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={
                    "reason": "insufficient_peers",
                    "peer_count": len(peer_scores),
                },
            )

        # 计算与peers平均的差距 (正数=低于peers)
        import numpy as np
        peer_mean = float(np.mean(peer_scores))
        deviation = peer_mean - obligor_score

        return self._create_result(
            obligor_id=obligor_id,
            value=max(0, deviation),  # 只关心低于peers的情况
            metadata={
                "obligor_rating": obligor.rating_internal.value,
                "obligor_score": obligor_score,
                "peer_mean_score": peer_mean,
                "peer_count": len(peer_scores),
                "obligor_name": obligor.name_cn,
            },
        )


@SignalRegistry.register
class ImpliedRatingSignal(Signal):
    """
    隐含评级信号

    计算: 基于利差的隐含评级 vs 实际评级
    触发: 隐含评级显著低于实际评级 (市场定价更差)
    """

    name: ClassVar[str] = "implied_rating"
    category: ClassVar[SignalCategory] = SignalCategory.MARKET
    description: ClassVar[str] = "隐含评级偏离"

    # 简化的利差-评级映射表 (bps)
    SPREAD_TO_RATING = [
        (50, CreditRating.AAA),
        (80, CreditRating.AA_PLUS),
        (100, CreditRating.AA),
        (130, CreditRating.AA_MINUS),
        (160, CreditRating.A_PLUS),
        (200, CreditRating.A),
        (250, CreditRating.A_MINUS),
        (300, CreditRating.BBB_PLUS),
        (400, CreditRating.BBB),
        (500, CreditRating.BBB_MINUS),
        (600, CreditRating.BB_PLUS),
        (800, CreditRating.BB),
        (1000, CreditRating.BB_MINUS),
    ]

    def __init__(
        self,
        threshold_warning: float = 10,  # 隐含评级低2档
        threshold_critical: float = 15,  # 隐含评级低3档
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)

    def _oas_to_implied_rating(self, oas: float) -> CreditRating:
        """将OAS映射到隐含评级"""
        for spread, rating in self.SPREAD_TO_RATING:
            if oas <= spread:
                return rating
        return CreditRating.B  # 超过1000bps

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

        obligor = exposure.obligor
        actual_rating = obligor.rating_internal
        implied_rating = self._oas_to_implied_rating(current_oas)

        actual_score = rating_to_score(actual_rating)
        implied_score = rating_to_score(implied_rating)

        # 偏离度 (正数=市场定价更差)
        deviation = actual_score - implied_score

        return self._create_result(
            obligor_id=obligor_id,
            value=max(0, deviation),
            metadata={
                "actual_rating": actual_rating.value,
                "implied_rating": implied_rating.value,
                "current_oas": current_oas,
                "deviation_notches": deviation / 5,
                "obligor_name": obligor.name_cn,
            },
        )
