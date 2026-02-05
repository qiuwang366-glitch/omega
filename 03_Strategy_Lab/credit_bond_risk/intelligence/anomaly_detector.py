"""
Credit Bond Risk - Anomaly Detection

Statistical anomaly detection for:
- Spread movements
- Trading volume spikes
- News volume anomalies
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """异常检测结果"""

    obligor_id: str
    metric_name: str
    timestamp: datetime

    # 当前值
    current_value: float

    # 统计特征
    historical_mean: float
    historical_std: float
    z_score: float

    # 判断
    is_anomaly: bool
    anomaly_type: str | None  # "spike", "drop", "regime_change"
    confidence: float

    # 上下文
    metadata: dict[str, Any]


class AnomalyDetector:
    """
    通用异常检测器

    方法:
    1. Z-score (假设正态分布)
    2. IQR (四分位距)
    3. Isolation Forest (多维异常)
    4. CUSUM (累积和控制图)
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        min_data_points: int = 30,
        lookback_days: int = 252,
    ):
        self.z_threshold = z_threshold
        self.min_data_points = min_data_points
        self.lookback_days = lookback_days

    def detect_zscore(
        self,
        values: list[float],
        timestamps: list[datetime] | None = None,
    ) -> list[AnomalyResult]:
        """
        Z-score异常检测

        Args:
            values: 时间序列值
            timestamps: 对应时间戳

        Returns:
            异常结果列表
        """
        if len(values) < self.min_data_points:
            return []

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return []

        anomalies = []
        for i, value in enumerate(values):
            z_score = (value - mean) / std

            if abs(z_score) > self.z_threshold:
                anomaly_type = "spike" if z_score > 0 else "drop"

                anomalies.append(AnomalyResult(
                    obligor_id="",  # 由调用者填充
                    metric_name="",  # 由调用者填充
                    timestamp=timestamps[i] if timestamps else datetime.now(),
                    current_value=value,
                    historical_mean=float(mean),
                    historical_std=float(std),
                    z_score=float(z_score),
                    is_anomaly=True,
                    anomaly_type=anomaly_type,
                    confidence=min(1.0, abs(z_score) / 5),
                    metadata={"index": i},
                ))

        return anomalies

    def detect_iqr(
        self,
        values: list[float],
        multiplier: float = 1.5,
    ) -> list[int]:
        """
        IQR异常检测

        返回异常点的索引
        """
        if len(values) < self.min_data_points:
            return []

        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        anomaly_indices = np.where((arr < lower_bound) | (arr > upper_bound))[0]
        return anomaly_indices.tolist()

    def detect_cusum(
        self,
        values: list[float],
        threshold: float | None = None,
        drift: float | None = None,
    ) -> list[tuple[int, str]]:
        """
        CUSUM (累积和) 异常检测

        用于检测均值漂移

        Returns:
            [(index, "up"/"down"), ...] 变点位置和方向
        """
        if len(values) < self.min_data_points:
            return []

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return []

        # 默认参数
        if threshold is None:
            threshold = 5 * std
        if drift is None:
            drift = std / 2

        # 计算CUSUM
        cusum_pos = np.zeros(len(arr))
        cusum_neg = np.zeros(len(arr))

        for i in range(1, len(arr)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + arr[i] - mean - drift)
            cusum_neg[i] = min(0, cusum_neg[i-1] + arr[i] - mean + drift)

        # 检测变点
        change_points = []

        # 向上变点
        pos_breaches = np.where(cusum_pos > threshold)[0]
        if len(pos_breaches) > 0:
            first_breach = pos_breaches[0]
            change_points.append((int(first_breach), "up"))

        # 向下变点
        neg_breaches = np.where(cusum_neg < -threshold)[0]
        if len(neg_breaches) > 0:
            first_breach = neg_breaches[0]
            change_points.append((int(first_breach), "down"))

        return change_points


class SpreadAnomalyDetector(AnomalyDetector):
    """
    利差异常检测器

    专门用于检测:
    1. 单日大幅走阔
    2. 持续走阔趋势
    3. 相对peers异常
    """

    def detect_spread_anomalies(
        self,
        obligor_id: str,
        spread_history: dict[str, float],
        as_of_date: datetime | None = None,
    ) -> list[AnomalyResult]:
        """
        检测利差异常

        Args:
            obligor_id: 发行人ID
            spread_history: 利差历史 {date_str: oas}
            as_of_date: 基准日期

        Returns:
            异常结果列表
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        # 转换为有序列表
        sorted_data = sorted(
            [(datetime.fromisoformat(k), v) for k, v in spread_history.items()],
            key=lambda x: x[0]
        )

        # 过滤到lookback窗口
        cutoff = as_of_date - timedelta(days=self.lookback_days)
        filtered = [(dt, v) for dt, v in sorted_data if dt >= cutoff]

        if len(filtered) < self.min_data_points:
            return []

        timestamps = [dt for dt, _ in filtered]
        values = [v for _, v in filtered]

        # 1. 水平异常 (当前值 vs 历史)
        level_anomalies = self.detect_zscore(values, timestamps)
        for a in level_anomalies:
            a.obligor_id = obligor_id
            a.metric_name = "spread_level"

        # 2. 变动异常 (日变动)
        if len(values) > 1:
            changes = np.diff(values)
            change_timestamps = timestamps[1:]
            change_anomalies = self.detect_zscore(changes.tolist(), change_timestamps)
            for a in change_anomalies:
                a.obligor_id = obligor_id
                a.metric_name = "spread_change"
        else:
            change_anomalies = []

        # 3. 趋势异常 (CUSUM)
        trend_changes = self.detect_cusum(values)
        trend_anomalies = []
        for idx, direction in trend_changes:
            if idx < len(timestamps):
                trend_anomalies.append(AnomalyResult(
                    obligor_id=obligor_id,
                    metric_name="spread_trend",
                    timestamp=timestamps[idx],
                    current_value=values[idx],
                    historical_mean=float(np.mean(values[:idx])) if idx > 0 else values[idx],
                    historical_std=float(np.std(values[:idx])) if idx > 1 else 0,
                    z_score=0,  # CUSUM不使用z-score
                    is_anomaly=True,
                    anomaly_type=f"trend_{direction}",
                    confidence=0.8,
                    metadata={"cusum_direction": direction},
                ))

        return level_anomalies + change_anomalies + trend_anomalies

    def detect_relative_anomaly(
        self,
        target_spread: float,
        peer_spreads: list[float],
    ) -> AnomalyResult | None:
        """
        相对同业异常检测

        Args:
            target_spread: 目标发行人利差
            peer_spreads: 同业利差列表

        Returns:
            异常结果 (如果是异常)
        """
        if len(peer_spreads) < 3:
            return None

        peer_mean = float(np.mean(peer_spreads))
        peer_std = float(np.std(peer_spreads))

        if peer_std == 0:
            return None

        z_score = (target_spread - peer_mean) / peer_std

        if abs(z_score) > self.z_threshold:
            return AnomalyResult(
                obligor_id="",  # 由调用者填充
                metric_name="spread_vs_peers",
                timestamp=datetime.now(),
                current_value=target_spread,
                historical_mean=peer_mean,
                historical_std=peer_std,
                z_score=z_score,
                is_anomaly=True,
                anomaly_type="rich" if z_score < 0 else "cheap",
                confidence=min(1.0, abs(z_score) / 5),
                metadata={"peer_count": len(peer_spreads)},
            )

        return None


class NewsVolumeAnomalyDetector(AnomalyDetector):
    """
    新闻量异常检测

    检测:
    1. 新闻数量激增
    2. 负面新闻聚集
    """

    def detect_volume_anomaly(
        self,
        obligor_id: str,
        news_counts: dict[str, int],  # {date: count}
        as_of_date: datetime | None = None,
    ) -> list[AnomalyResult]:
        """
        检测新闻量异常

        Args:
            obligor_id: 发行人ID
            news_counts: 每日新闻数量
            as_of_date: 基准日期

        Returns:
            异常结果列表
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        # 转换为有序列表
        sorted_data = sorted(
            [(datetime.fromisoformat(k), v) for k, v in news_counts.items()],
            key=lambda x: x[0]
        )

        if len(sorted_data) < 7:
            return []

        timestamps = [dt for dt, _ in sorted_data]
        values = [v for _, v in sorted_data]

        # 使用较低的阈值 (新闻量波动大)
        anomalies = self.detect_zscore(values, timestamps)

        for a in anomalies:
            a.obligor_id = obligor_id
            a.metric_name = "news_volume"
            # 只关注激增
            if a.anomaly_type != "spike":
                a.is_anomaly = False

        return [a for a in anomalies if a.is_anomaly]
