"""
Credit Bond Risk - Concentration Signals

Signals for monitoring portfolio concentration risk:
- Single obligor concentration
- Top N concentration
- Sector/Province concentration
- HHI (Herfindahl-Hirschman Index)
"""

from typing import ClassVar

import numpy as np

from .base import Signal, SignalContext, SignalRegistry
from ..core.models import SignalResult
from ..core.enums import SignalCategory


@SignalRegistry.register
class ConcentrationSignal(Signal):
    """
    单一发行人集中度信号

    计算: 单一发行人市值 / 组合AUM
    触发: 占比超过阈值
    """

    name: ClassVar[str] = "concentration_single"
    category: ClassVar[SignalCategory] = SignalCategory.CONCENTRATION
    description: ClassVar[str] = "单一发行人持仓占比"

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        exposure = context.get_exposure(obligor_id)

        if exposure is None:
            return self._create_result(
                obligor_id=obligor_id,
                value=0.0,
                metadata={"reason": "no_exposure"},
            )

        pct = exposure.pct_of_aum

        return self._create_result(
            obligor_id=obligor_id,
            value=pct,
            metadata={
                "market_value_usd": exposure.total_market_usd,
                "total_aum": context.total_aum,
                "obligor_name": exposure.obligor.name_cn,
            },
        )

    def compute_batch(
        self, obligor_ids: list[str], context: SignalContext
    ) -> list[SignalResult]:
        """向量化批量计算"""
        results = []
        for oid in obligor_ids:
            results.append(self.compute(oid, context))
        return results


@SignalRegistry.register
class TopNConcentrationSignal(Signal):
    """
    Top N 发行人集中度信号

    计算: Top N发行人合计市值 / 组合AUM
    触发: 合计占比超过阈值
    """

    name: ClassVar[str] = "concentration_top_n"
    category: ClassVar[SignalCategory] = SignalCategory.CONCENTRATION
    description: ClassVar[str] = "Top N发行人合计占比"

    def __init__(
        self,
        threshold_warning: float = 0.40,
        threshold_critical: float = 0.60,
        top_n: int = 10,
    ):
        super().__init__(threshold_warning, threshold_critical, higher_is_worse=True)
        self.top_n = top_n

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """
        注意：这是组合级别信号，obligor_id参数被忽略
        返回的obligor_id为"PORTFOLIO"
        """
        # 按市值排序所有曝光
        exposures = sorted(
            context.exposures.values(),
            key=lambda x: x.total_market_usd,
            reverse=True,
        )

        top_n_exposures = exposures[: self.top_n]
        top_n_total = sum(e.total_market_usd for e in top_n_exposures)
        top_n_pct = top_n_total / context.total_aum if context.total_aum > 0 else 0

        top_n_names = [e.obligor.name_cn for e in top_n_exposures]

        return self._create_result(
            obligor_id="PORTFOLIO",
            value=top_n_pct,
            metadata={
                "top_n": self.top_n,
                "top_n_total_usd": top_n_total,
                "top_n_names": top_n_names,
            },
        )

    def compute_portfolio(self, context: SignalContext) -> list[SignalResult]:
        """组合级别信号只需计算一次"""
        result = self.compute("PORTFOLIO", context)
        return [result] if result.is_triggered else []


@SignalRegistry.register
class HHISignal(Signal):
    """
    HHI (Herfindahl-Hirschman Index) 集中度信号

    计算: sum((market_share_i)^2) for all obligors
    取值范围: 0 (完全分散) - 1 (完全集中)
    """

    name: ClassVar[str] = "concentration_hhi"
    category: ClassVar[SignalCategory] = SignalCategory.CONCENTRATION
    description: ClassVar[str] = "HHI集中度指数"

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """
        组合级别信号，obligor_id参数被忽略
        """
        if not context.exposures or context.total_aum <= 0:
            return self._create_result(
                obligor_id="PORTFOLIO",
                value=0.0,
                metadata={"reason": "no_exposures"},
            )

        # 计算每个发行人的市场份额
        shares = np.array([
            e.total_market_usd / context.total_aum
            for e in context.exposures.values()
        ])

        # HHI = sum(share^2)
        hhi = float(np.sum(shares ** 2))

        # 等效发行人数 = 1/HHI
        equivalent_n = 1 / hhi if hhi > 0 else float("inf")

        return self._create_result(
            obligor_id="PORTFOLIO",
            value=hhi,
            metadata={
                "equivalent_obligors": equivalent_n,
                "total_obligors": len(context.exposures),
            },
        )

    def compute_portfolio(self, context: SignalContext) -> list[SignalResult]:
        """组合级别信号只需计算一次"""
        result = self.compute("PORTFOLIO", context)
        return [result] if result.is_triggered else []


@SignalRegistry.register
class SectorConcentrationSignal(Signal):
    """
    行业集中度信号

    计算: 单一行业市值 / 组合AUM
    触发: 任一行业占比超过阈值
    """

    name: ClassVar[str] = "concentration_sector"
    category: ClassVar[SignalCategory] = SignalCategory.CONCENTRATION
    description: ClassVar[str] = "单一行业占比"

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """
        组合级别信号
        返回占比最高的行业
        """
        if not context.exposures or context.total_aum <= 0:
            return self._create_result(
                obligor_id="PORTFOLIO",
                value=0.0,
                metadata={"reason": "no_exposures"},
            )

        # 按行业汇总
        sector_totals: dict[str, float] = {}
        for exposure in context.exposures.values():
            sector = exposure.obligor.sector.value
            sector_totals[sector] = sector_totals.get(sector, 0) + exposure.total_market_usd

        # 找到占比最高的行业
        max_sector = max(sector_totals, key=sector_totals.get)  # type: ignore
        max_pct = sector_totals[max_sector] / context.total_aum

        # 所有行业占比
        all_sectors = {
            k: v / context.total_aum for k, v in sector_totals.items()
        }

        return self._create_result(
            obligor_id="PORTFOLIO",
            value=max_pct,
            metadata={
                "top_sector": max_sector,
                "all_sectors": all_sectors,
            },
        )

    def compute_portfolio(self, context: SignalContext) -> list[SignalResult]:
        result = self.compute("PORTFOLIO", context)
        return [result] if result.is_triggered else []


@SignalRegistry.register
class ProvinceConcentrationSignal(Signal):
    """
    地区集中度信号 (适用于LGFV)

    计算: 单一省份市值 / 组合AUM (仅计算城投)
    触发: 任一省份占比超过阈值
    """

    name: ClassVar[str] = "concentration_province"
    category: ClassVar[SignalCategory] = SignalCategory.CONCENTRATION
    description: ClassVar[str] = "单一省份城投占比"

    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """组合级别信号"""
        if not context.exposures or context.total_aum <= 0:
            return self._create_result(
                obligor_id="PORTFOLIO",
                value=0.0,
                metadata={"reason": "no_exposures"},
            )

        # 筛选城投，按省份汇总
        province_totals: dict[str, float] = {}
        lgfv_total = 0.0

        for exposure in context.exposures.values():
            if exposure.obligor.sector.value == "LGFV":
                province = exposure.obligor.province or "Unknown"
                province_totals[province] = (
                    province_totals.get(province, 0) + exposure.total_market_usd
                )
                lgfv_total += exposure.total_market_usd

        if not province_totals:
            return self._create_result(
                obligor_id="PORTFOLIO",
                value=0.0,
                metadata={"reason": "no_lgfv_exposures"},
            )

        # 找到占比最高的省份
        max_province = max(province_totals, key=province_totals.get)  # type: ignore
        max_pct = province_totals[max_province] / context.total_aum

        # 所有省份占比
        all_provinces = {
            k: v / context.total_aum for k, v in province_totals.items()
        }

        return self._create_result(
            obligor_id="PORTFOLIO",
            value=max_pct,
            metadata={
                "top_province": max_province,
                "all_provinces": all_provinces,
                "lgfv_total_pct": lgfv_total / context.total_aum,
            },
        )

    def compute_portfolio(self, context: SignalContext) -> list[SignalResult]:
        result = self.compute("PORTFOLIO", context)
        return [result] if result.is_triggered else []
