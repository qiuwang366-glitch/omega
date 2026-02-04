"""
allocation_engine.py - Portfolio Simulation Engine
==================================================
Core business logic for Fixed Income allocation simulation.
Handles NII calculation, FTP arbitrage, and multi-currency P&L.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Add current directory to path for Streamlit Cloud compatibility
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from analytics import FTPCalculator, YieldSurface, FXAnalytics
from config import (
    Currency,
    SimulationParams,
    FTPConfig,
    AccountingBasis,
    YieldCurveRegime,
    get_default_config,
)
from data_provider import MarketDataProvider, MarketDataFactory


# Configure logging
logger = logging.getLogger(__name__)

# Type alias
FloatArray = NDArray[np.float64]


# ============================================================================
# 1. Strategy Configuration
# ============================================================================
class AllocationStrategy(BaseModel):
    """
    Defines an allocation strategy for simulation.

    Captures the 'vintage' concept - timing of purchases matters for profitability.
    """
    name: str = "Default Strategy"
    description: str = ""

    # Target allocations by currency (% of AUM)
    currency_weights: dict[Currency, float] = Field(
        default_factory=lambda: {
            Currency.USD: 0.60,
            Currency.AUD: 0.20,
            Currency.EUR: 0.15,
            Currency.CNH: 0.05,
        }
    )

    # Target duration by currency
    target_durations: dict[Currency, float] = Field(
        default_factory=lambda: {
            Currency.USD: 5.0,
            Currency.AUD: 4.0,
            Currency.EUR: 5.0,
            Currency.CNH: 3.0,
        }
    )

    # Accounting basis by currency
    accounting_basis: dict[Currency, AccountingBasis] = Field(
        default_factory=lambda: {
            Currency.USD: "AC",
            Currency.AUD: "AC",
            Currency.EUR: "FVOCI",
            Currency.CNH: "AC",
        }
    )

    # Reinvestment timing strategy
    reinvest_frequency: str = "M"  # Monthly
    front_load_cuts: bool = True   # Buy before rate cuts
    chase_hikes: bool = True       # Buy after rate hikes

    @property
    def total_weight(self) -> float:
        return sum(self.currency_weights.values())

    def get_duration(self, currency: Currency) -> float:
        return self.target_durations.get(currency, 5.0)

    def get_weight(self, currency: Currency) -> float:
        return self.currency_weights.get(currency, 0.0)


# ============================================================================
# 2. Simulation Results
# ============================================================================
@dataclass
class PeriodResult:
    """Results for a single simulation period."""
    period: int
    date: date
    currency: Currency

    # Yields
    asset_yield: float
    ftp_rate: float
    nii_spread: float

    # Position
    book_value_usd: float
    carry_usd: float

    # FX
    fx_rate: float
    fx_pnl_usd: float

    # Aggregated
    total_nii_usd: float


@dataclass
class SimulationResult:
    """Complete simulation results."""
    strategy_name: str
    start_date: date
    end_date: date
    n_periods: int

    # Time series
    dates: list[date]
    period_results: list[PeriodResult]

    # Summary by currency
    summary_by_currency: dict[Currency, dict[str, float]] = field(default_factory=dict)

    # Aggregate metrics
    total_nii_usd: float = 0.0
    total_fx_pnl_usd: float = 0.0
    avg_spread_bps: float = 0.0
    sharpe_ratio: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        records = []
        for r in self.period_results:
            records.append({
                "period": r.period,
                "date": r.date,
                "currency": r.currency.value,
                "asset_yield": r.asset_yield,
                "ftp_rate": r.ftp_rate,
                "nii_spread": r.nii_spread,
                "book_value_usd": r.book_value_usd,
                "carry_usd": r.carry_usd,
                "fx_rate": r.fx_rate,
                "fx_pnl_usd": r.fx_pnl_usd,
                "total_nii_usd": r.total_nii_usd,
            })
        return pd.DataFrame(records)


# ============================================================================
# 3. Portfolio Simulator
# ============================================================================
class PortfolioSimulator:
    """
    Main simulation engine for Fixed Income allocation.

    Workflow:
    1. Generate yield surfaces for all currencies over horizon
    2. Calculate FTP paths (lagged)
    3. Calculate asset yield paths
    4. Compute NII spreads (vectorized)
    5. Apply FX scenarios
    6. Aggregate results
    """

    def __init__(
        self,
        data_provider: MarketDataProvider,
        config: SimulationParams | None = None,
    ) -> None:
        """
        Initialize simulator.

        Args:
            data_provider: Market data source (CSV or synthetic)
            config: Simulation parameters
        """
        self._provider = data_provider
        self._config = config or get_default_config()

        logger.info(
            f"PortfolioSimulator initialized: "
            f"provider={data_provider.provider_name}, "
            f"horizon={self._config.start_date} to {self._config.end_date}"
        )

    @property
    def config(self) -> SimulationParams:
        return self._config

    def _generate_date_grid(self) -> list[date]:
        """Generate simulation dates based on frequency."""
        freq_map = {"D": 1, "W": 7, "M": 30, "Q": 91}
        delta = timedelta(days=freq_map[self._config.frequency])

        dates = []
        current = self._config.start_date
        while current <= self._config.end_date:
            dates.append(current)
            current += delta

        return dates

    def _get_yield_surface_matrix(
        self,
        currency: Currency,
        dates: list[date],
        target_tenor: float,
    ) -> FloatArray:
        """
        Generate matrix of forward rates over time.

        Returns:
            Array of yields, shape (n_dates,)
        """
        yields = np.zeros(len(dates), dtype=np.float64)

        for i, d in enumerate(dates):
            surface = self._provider.get_yield_surface(currency, d)
            # Get spot yield at target tenor
            rates = surface.get_forward_rates(
                np.array([0.0]),
                np.array([target_tenor])
            )
            yields[i] = rates[0, 0]

        return yields

    def _calculate_ftp_path(
        self,
        currency: Currency,
        dates: list[date],
    ) -> FloatArray:
        """
        Calculate FTP path with lag.

        FTP(t) = 3M yield at (t - lag_months)
        """
        ftp_config = self._config.ftp_config
        lag_days = ftp_config.lag_months * 30

        ftp_rates = np.zeros(len(dates), dtype=np.float64)

        for i, d in enumerate(dates):
            # Look back for FTP reference date
            ref_date = d - timedelta(days=lag_days)
            ref_date = max(ref_date, self._config.start_date - timedelta(days=90))

            surface = self._provider.get_yield_surface(currency, ref_date)
            tenor_years = ftp_config.tenor_months / 12

            rates = surface.get_forward_rates(
                np.array([0.0]),
                np.array([tenor_years])
            )
            ftp_rates[i] = rates[0, 0]

        # Add spread adjustment
        ftp_rates += ftp_config.spread_adjustment_bps / 10000

        return ftp_rates

    def _calculate_fx_path(
        self,
        currency: Currency,
        dates: list[date],
    ) -> FloatArray:
        """Get FX rate path for a currency."""
        if currency == Currency.USD:
            return np.ones(len(dates), dtype=np.float64)

        fx_rates = np.zeros(len(dates), dtype=np.float64)
        for i, d in enumerate(dates):
            fx_rates[i] = self._provider.get_fx_rate(currency, d)

        return fx_rates

    def run_simulation(
        self,
        strategy: AllocationStrategy,
        fx_scenario: str = "base",
    ) -> SimulationResult:
        """
        Run full allocation simulation.

        Steps:
        1. Generate date grid
        2. For each currency in strategy:
           - Generate yield surface matrix (forward rates)
           - Calculate FTP path (lagged)
           - Calculate spreads (vectorized)
           - Apply FX conversion
        3. Aggregate results

        Args:
            strategy: Allocation strategy configuration
            fx_scenario: FX scenario name from config

        Returns:
            SimulationResult with all period results
        """
        logger.info(f"Running simulation: {strategy.name}")

        # Step 1: Generate date grid
        dates = self._generate_date_grid()
        n_periods = len(dates)
        logger.info(f"  Periods: {n_periods} ({self._config.frequency})")

        # Get FX scenario rates
        fx_scenarios = self._config.fx_config.scenarios.get(
            fx_scenario,
            self._config.fx_config.budget_rates
        )
        budget_rates = self._config.fx_config.budget_rates

        # Initialize accumulators
        period_results: list[PeriodResult] = []
        summary_by_ccy: dict[Currency, dict[str, Any]] = {}

        # Step 2: Process each currency
        for currency in strategy.currency_weights.keys():
            weight = strategy.get_weight(currency)
            duration = strategy.get_duration(currency)

            if weight <= 0:
                continue

            logger.info(f"  Processing {currency.value}: weight={weight:.1%}, dur={duration:.1f}Y")

            # Book value in USD
            book_value_usd = self._config.initial_aum_usd * weight

            # === VECTORIZED CALCULATIONS ===

            # Asset yields at target duration
            asset_yields = self._get_yield_surface_matrix(currency, dates, duration)

            # FTP path (lagged)
            ftp_rates = self._calculate_ftp_path(currency, dates)

            # NII spread (vectorized)
            spreads = asset_yields - ftp_rates

            # FX path
            if currency == Currency.USD:
                fx_rates = np.ones(n_periods)
                fx_budget = 1.0
            else:
                fx_rates = self._calculate_fx_path(currency, dates)
                fx_budget = budget_rates.get(currency, 1.0)

            # Carry in local currency (annualized, then scaled to period)
            period_fraction = {"D": 1/365, "W": 1/52, "M": 1/12, "Q": 0.25}[self._config.frequency]
            carry_local = book_value_usd * spreads * period_fraction

            # Convert to USD
            if currency != Currency.USD:
                # For EURUSD, AUDUSD convention: multiply
                carry_usd = carry_local * fx_rates
            else:
                carry_usd = carry_local

            # FX P&L (drag/gain)
            fx_pnl = FXAnalytics.calculate_fx_drag(
                carry_local,
                fx_rates,
                fx_budget,
            )

            # Total NII in USD
            total_nii_usd = carry_usd

            # Store results
            for i in range(n_periods):
                period_results.append(PeriodResult(
                    period=i,
                    date=dates[i],
                    currency=currency,
                    asset_yield=float(asset_yields[i]),
                    ftp_rate=float(ftp_rates[i]),
                    nii_spread=float(spreads[i]),
                    book_value_usd=book_value_usd,
                    carry_usd=float(carry_usd[i]),
                    fx_rate=float(fx_rates[i]),
                    fx_pnl_usd=float(fx_pnl[i]),
                    total_nii_usd=float(total_nii_usd[i]),
                ))

            # Currency summary
            summary_by_ccy[currency] = {
                "weight": weight,
                "book_value_usd": book_value_usd,
                "avg_asset_yield": float(asset_yields.mean()),
                "avg_ftp_rate": float(ftp_rates.mean()),
                "avg_spread_bps": float(spreads.mean() * 10000),
                "total_carry_usd": float(carry_usd.sum()),
                "total_fx_pnl_usd": float(fx_pnl.sum()),
                "total_nii_usd": float(total_nii_usd.sum()),
            }

        # Step 3: Aggregate
        total_nii = sum(s["total_nii_usd"] for s in summary_by_ccy.values())
        total_fx_pnl = sum(s["total_fx_pnl_usd"] for s in summary_by_ccy.values())

        # Weighted average spread
        total_bv = sum(s["book_value_usd"] for s in summary_by_ccy.values())
        avg_spread = sum(
            s["avg_spread_bps"] * s["book_value_usd"]
            for s in summary_by_ccy.values()
        ) / total_bv if total_bv > 0 else 0.0

        # Simple Sharpe (NII vol)
        nii_series = pd.DataFrame(period_results).groupby("period")["total_nii_usd"].sum()
        nii_vol = nii_series.std()
        sharpe = (nii_series.mean() / nii_vol) if nii_vol > 0 else 0.0

        result = SimulationResult(
            strategy_name=strategy.name,
            start_date=self._config.start_date,
            end_date=self._config.end_date,
            n_periods=n_periods,
            dates=dates,
            period_results=period_results,
            summary_by_currency=summary_by_ccy,
            total_nii_usd=total_nii,
            total_fx_pnl_usd=total_fx_pnl,
            avg_spread_bps=avg_spread,
            sharpe_ratio=float(sharpe),
        )

        logger.info(
            f"  Complete: NII=${total_nii/1e6:.1f}MM, "
            f"FX P&L=${total_fx_pnl/1e6:.1f}MM, "
            f"Spread={avg_spread:.0f}bp"
        )

        return result

    def run_scenario_analysis(
        self,
        strategy: AllocationStrategy,
        regimes: list[YieldCurveRegime] | None = None,
        fx_scenarios: list[str] | None = None,
    ) -> dict[str, SimulationResult]:
        """
        Run simulation across multiple scenarios.

        Args:
            strategy: Allocation strategy
            regimes: List of yield curve regimes to test
            fx_scenarios: List of FX scenarios to test

        Returns:
            Dict mapping scenario name to results
        """
        regimes = regimes or [
            YieldCurveRegime.BEAR_FLATTENING,
            YieldCurveRegime.BULL_STEEPENING,
        ]
        fx_scenarios = fx_scenarios or ["base", "usd_strong", "usd_weak"]

        results = {}

        for regime in regimes:
            for fx_scen in fx_scenarios:
                scenario_name = f"{regime.value}_{fx_scen}"
                logger.info(f"Scenario: {scenario_name}")

                # Update provider regime if synthetic
                if hasattr(self._provider, "set_regime"):
                    self._provider.set_regime(regime)  # type: ignore

                result = self.run_simulation(strategy, fx_scenario=fx_scen)
                results[scenario_name] = result

        return results


# ============================================================================
# 4. Optimization Functions
# ============================================================================
def optimize_allocation(
    simulator: PortfolioSimulator,
    target_metric: str = "nii",
) -> AllocationStrategy:
    """
    Simple allocation optimizer using grid search.

    Finds optimal currency weights to maximize NII or minimize volatility.

    Args:
        simulator: Configured simulator
        target_metric: "nii" or "sharpe"

    Returns:
        Optimized AllocationStrategy
    """
    from itertools import product

    # Weight grid (10% increments, sum to 100%)
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    currencies = [Currency.USD, Currency.AUD, Currency.EUR]

    best_result = None
    best_strategy = None
    best_metric = -np.inf

    for weights in product(weight_options, repeat=len(currencies)):
        if abs(sum(weights) - 1.0) > 0.01:
            continue

        strategy = AllocationStrategy(
            name="Optimized",
            currency_weights={
                currencies[i]: weights[i]
                for i in range(len(currencies))
            },
        )

        result = simulator.run_simulation(strategy)

        metric = result.total_nii_usd if target_metric == "nii" else result.sharpe_ratio

        if metric > best_metric:
            best_metric = metric
            best_result = result
            best_strategy = strategy

    logger.info(f"Optimal allocation found: {best_strategy}")
    return best_strategy  # type: ignore


# ============================================================================
# 5. Module Test
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Allocation Engine Test ===\n")

    # Create config and provider
    config = get_default_config()
    provider = MarketDataFactory.create(config, force_synthetic=True)

    # Create simulator
    simulator = PortfolioSimulator(provider, config)

    # Define strategy
    strategy = AllocationStrategy(
        name="2026 Base Case",
        description="Conservative multi-currency allocation",
        currency_weights={
            Currency.USD: 0.55,
            Currency.AUD: 0.25,
            Currency.EUR: 0.15,
            Currency.CNH: 0.05,
        },
        target_durations={
            Currency.USD: 5.0,
            Currency.AUD: 4.0,
            Currency.EUR: 5.0,
            Currency.CNH: 3.0,
        },
    )

    # Run simulation
    result = simulator.run_simulation(strategy)

    # Display results
    print(f"\n{'='*50}")
    print(f"Strategy: {result.strategy_name}")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"{'='*50}\n")

    print("Summary by Currency:")
    print("-" * 50)
    for ccy, summary in result.summary_by_currency.items():
        print(f"\n{ccy.value}:")
        print(f"  Weight: {summary['weight']:.1%}")
        print(f"  Book Value: ${summary['book_value_usd']/1e9:.2f}B")
        print(f"  Avg Asset Yield: {summary['avg_asset_yield']*100:.2f}%")
        print(f"  Avg FTP Rate: {summary['avg_ftp_rate']*100:.2f}%")
        print(f"  Avg Spread: {summary['avg_spread_bps']:.0f}bp")
        print(f"  Total NII: ${summary['total_nii_usd']/1e6:.1f}MM")

    print(f"\n{'='*50}")
    print("AGGREGATE RESULTS")
    print(f"{'='*50}")
    print(f"Total NII (USD): ${result.total_nii_usd/1e6:.1f}MM")
    print(f"Total FX P&L: ${result.total_fx_pnl_usd/1e6:.1f}MM")
    print(f"Avg Spread: {result.avg_spread_bps:.0f}bp")
    print(f"Information Ratio: {result.sharpe_ratio:.2f}")
