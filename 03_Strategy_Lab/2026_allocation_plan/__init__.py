"""
2026 Allocation Plan - Fixed Income Simulation System
======================================================

A high-performance simulation ecosystem for strategic Fixed Income allocation.

Modules:
- config: Pydantic-based configuration and domain models
- analytics: NSS yield curve model and yield surface calculations
- data_provider: Abstract factory for market data with fallback logic
- allocation_engine: Portfolio simulation and NII calculation
- dashboard: Streamlit interactive visualization

Usage:
    from allocation_plan import (
        SimulationParams,
        Currency,
        PortfolioSimulator,
        AllocationStrategy,
        MarketDataFactory,
    )

    config = SimulationParams()
    provider = MarketDataFactory.create(config)
    simulator = PortfolioSimulator(provider, config)

    strategy = AllocationStrategy(name="2026 Plan")
    result = simulator.run_simulation(strategy)

Run Dashboard:
    cd 03_Strategy_Lab/2026_allocation_plan
    streamlit run dashboard.py
"""
import sys
from pathlib import Path

# Add current directory to path for Streamlit Cloud compatibility
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from config import (
    Currency,
    YieldCurveRegime,
    NSSParams,
    FTPConfig,
    FXConfig,
    SimulationParams,
    AllocationTarget,
    PortfolioConfig,
    RegimeShock,
    DEFAULT_NSS_PARAMS,
    REGIME_SHOCKS,
    get_default_config,
)

from analytics import (
    NelsonSiegelSvensson,
    YieldSurface,
    YieldSurfaceGrid,
    FTPCalculator,
    FXAnalytics,
    create_yield_surface_from_params,
    compute_carry_rolldown,
)

from data_provider import (
    MarketDataProvider,
    CSVMarketData,
    SyntheticMarketData,
    HybridMarketData,
    MarketDataFactory,
    YieldCurveData,
    FXRateData,
)

from allocation_engine import (
    AllocationStrategy,
    PortfolioSimulator,
    SimulationResult,
    PeriodResult,
    optimize_allocation,
)


__all__ = [
    # Config
    "Currency",
    "YieldCurveRegime",
    "NSSParams",
    "FTPConfig",
    "FXConfig",
    "SimulationParams",
    "AllocationTarget",
    "PortfolioConfig",
    "RegimeShock",
    "DEFAULT_NSS_PARAMS",
    "REGIME_SHOCKS",
    "get_default_config",
    # Analytics
    "NelsonSiegelSvensson",
    "YieldSurface",
    "YieldSurfaceGrid",
    "FTPCalculator",
    "FXAnalytics",
    "create_yield_surface_from_params",
    "compute_carry_rolldown",
    # Data Provider
    "MarketDataProvider",
    "CSVMarketData",
    "SyntheticMarketData",
    "HybridMarketData",
    "MarketDataFactory",
    "YieldCurveData",
    "FXRateData",
    # Engine
    "AllocationStrategy",
    "PortfolioSimulator",
    "SimulationResult",
    "PeriodResult",
    "optimize_allocation",
]

__version__ = "1.0.0"
