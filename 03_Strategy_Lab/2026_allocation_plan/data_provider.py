"""
data_provider.py - Abstract Factory for Market Data
====================================================
Implements the Dependency Injection pattern for market data sources.
Supports seamless swapping between CSV data and synthetic generators.
"""
from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

# Add current directory to path for Streamlit Cloud compatibility
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from analytics import NelsonSiegelSvensson, YieldSurface
from config import (
    Currency,
    NSSParams,
    SimulationParams,
    DEFAULT_NSS_PARAMS,
    YieldCurveRegime,
    REGIME_SHOCKS,
)


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Type aliases
FloatArray = NDArray[np.float64]
T = TypeVar("T")


# ============================================================================
# 1. Data Models
# ============================================================================
class YieldCurveData(BaseModel):
    """Container for historical yield curve data."""
    date: date
    currency: Currency
    tenors: list[float]  # Years
    yields: list[float]  # Decimal

    class Config:
        arbitrary_types_allowed = True

    def to_array(self) -> tuple[FloatArray, FloatArray]:
        """Convert to numpy arrays."""
        return (
            np.array(self.tenors, dtype=np.float64),
            np.array(self.yields, dtype=np.float64),
        )


class FXRateData(BaseModel):
    """Container for FX rate data."""
    date: date
    base_currency: Currency
    quote_currency: Currency = Currency.USD
    spot_rate: float = Field(gt=0)


class HistoricalDataSet(BaseModel):
    """Collection of historical market data."""
    yield_curves: dict[Currency, list[YieldCurveData]] = Field(default_factory=dict)
    fx_rates: dict[Currency, list[FXRateData]] = Field(default_factory=dict)
    start_date: date | None = None
    end_date: date | None = None

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# 2. Abstract Base Class (Protocol)
# ============================================================================
class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    Implements the Strategy pattern - concrete implementations can be
    swapped without changing client code.
    """

    @abstractmethod
    def get_yield_surface(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldSurface:
        """
        Get yield surface for a currency at a given date.

        Args:
            currency: The currency
            as_of_date: Valuation date

        Returns:
            YieldSurface object
        """
        ...

    @abstractmethod
    def get_yield_curve(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldCurveData:
        """
        Get raw yield curve data.

        Args:
            currency: The currency
            as_of_date: Valuation date

        Returns:
            YieldCurveData object
        """
        ...

    @abstractmethod
    def get_fx_rate(
        self,
        base_currency: Currency,
        as_of_date: date,
    ) -> float:
        """
        Get FX spot rate vs USD.

        Args:
            base_currency: Currency to price
            as_of_date: Valuation date

        Returns:
            Spot rate (base/USD)
        """
        ...

    @abstractmethod
    def get_historical_yields(
        self,
        currency: Currency,
        start_date: date,
        end_date: date,
        tenor: float,
    ) -> pd.DataFrame:
        """
        Get historical yields for a specific tenor.

        Args:
            currency: The currency
            start_date: Start of history
            end_date: End of history
            tenor: Tenor in years

        Returns:
            DataFrame with columns [date, yield]
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging."""
        ...


# ============================================================================
# 3. CSV Market Data Provider
# ============================================================================
class CSVMarketData(MarketDataProvider):
    """
    Market data provider that loads from CSV files.

    Expected file structure:
    - {data_dir}/curves/{currency}_yields.csv
    - {data_dir}/fx/{currency}_spot.csv
    """

    def __init__(self, data_dir: Path) -> None:
        """
        Initialize CSV provider.

        Args:
            data_dir: Root directory for market data files
        """
        self._data_dir = Path(data_dir)
        self._curves_dir = self._data_dir / "curves"
        self._fx_dir = self._data_dir / "fx"
        self._cache: dict[str, Any] = {}

        logger.info(f"CSVMarketData initialized with data_dir={data_dir}")

    @property
    def provider_name(self) -> str:
        return "CSVMarketData"

    def _load_csv_safe(self, path: Path) -> pd.DataFrame | None:
        """Load CSV with error handling."""
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=1).columns else False)
            if df.empty:
                logger.warning(f"Empty file: {path}")
                return None
            return df
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def _get_curve_path(self, currency: Currency) -> Path:
        return self._curves_dir / f"{currency.value.lower()}_yields.csv"

    def _get_fx_path(self, currency: Currency) -> Path:
        return self._fx_dir / f"{currency.value.lower()}_spot.csv"

    def get_yield_curve(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldCurveData:
        """Load yield curve from CSV."""
        cache_key = f"curve_{currency.value}_{as_of_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_curve_path(currency)
        df = self._load_csv_safe(path)

        if df is None:
            raise FileNotFoundError(f"Yield curve data not found for {currency}")

        # Filter to as_of_date or nearest prior date
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[df["date"] <= as_of_date].sort_values("date", ascending=False)

        if df.empty:
            raise ValueError(f"No yield data on or before {as_of_date}")

        latest = df.iloc[0]
        data_date = latest["date"]

        # Extract tenor columns (expect format: 3M, 6M, 1Y, 2Y, etc.)
        tenor_cols = [c for c in df.columns if c not in ["date", "currency"]]
        tenors: list[float] = []
        yields: list[float] = []

        for col in tenor_cols:
            try:
                # Parse tenor string
                if col.endswith("M"):
                    tenor = float(col[:-1]) / 12
                elif col.endswith("Y"):
                    tenor = float(col[:-1])
                else:
                    continue
                tenors.append(tenor)
                yields.append(float(latest[col]) / 100)  # Convert % to decimal
            except (ValueError, KeyError):
                continue

        result = YieldCurveData(
            date=data_date,
            currency=currency,
            tenors=tenors,
            yields=yields,
        )
        self._cache[cache_key] = result
        return result

    def get_yield_surface(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldSurface:
        """
        Build yield surface from CSV data.

        Uses NSS fitting on the observed curve.
        """
        curve_data = self.get_yield_curve(currency, as_of_date)
        tenors, yields = curve_data.to_array()

        # Fit NSS to observed data
        nss = self._fit_nss(tenors, yields, currency)
        return YieldSurface(nss)

    def _fit_nss(
        self,
        tenors: FloatArray,
        yields: FloatArray,
        currency: Currency,
    ) -> NelsonSiegelSvensson:
        """
        Fit NSS model to observed yields.

        Uses scipy optimization with sensible initial guess.
        """
        from scipy.optimize import minimize

        # Initial guess from defaults
        default = DEFAULT_NSS_PARAMS.get(currency, DEFAULT_NSS_PARAMS[Currency.USD])

        def objective(params: FloatArray) -> float:
            nss = NelsonSiegelSvensson(
                beta0=params[0],
                beta1=params[1],
                beta2=params[2],
                beta3=params[3],
                lambda1=params[4],
                lambda2=params[5],
            )
            fitted = nss.yield_at_tenor(tenors)
            return float(np.sum((fitted - yields) ** 2))

        x0 = np.array([
            default.beta0, default.beta1, default.beta2,
            default.beta3, default.lambda1, default.lambda2
        ])

        bounds = [
            (0.0, 0.20),    # beta0
            (-0.10, 0.10),  # beta1
            (-0.10, 0.10),  # beta2
            (-0.10, 0.10),  # beta3
            (0.1, 10.0),    # lambda1
            (0.1, 10.0),    # lambda2
        ]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        return NelsonSiegelSvensson(
            beta0=result.x[0],
            beta1=result.x[1],
            beta2=result.x[2],
            beta3=result.x[3],
            lambda1=result.x[4],
            lambda2=result.x[5],
        )

    def get_fx_rate(
        self,
        base_currency: Currency,
        as_of_date: date,
    ) -> float:
        """Load FX rate from CSV."""
        if base_currency == Currency.USD:
            return 1.0

        path = self._get_fx_path(base_currency)
        df = self._load_csv_safe(path)

        if df is None:
            raise FileNotFoundError(f"FX data not found for {base_currency}")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[df["date"] <= as_of_date].sort_values("date", ascending=False)

        if df.empty:
            raise ValueError(f"No FX data on or before {as_of_date}")

        return float(df.iloc[0]["spot"])

    def get_historical_yields(
        self,
        currency: Currency,
        start_date: date,
        end_date: date,
        tenor: float,
    ) -> pd.DataFrame:
        """Get historical yields for a tenor."""
        path = self._get_curve_path(currency)
        df = self._load_csv_safe(path)

        if df is None:
            raise FileNotFoundError(f"Yield history not found for {currency}")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        # Find closest tenor column
        if tenor < 1:
            tenor_col = f"{int(tenor * 12)}M"
        else:
            tenor_col = f"{int(tenor)}Y"

        if tenor_col not in df.columns:
            raise ValueError(f"Tenor {tenor_col} not found in data")

        return df[["date", tenor_col]].rename(columns={tenor_col: "yield"})


# ============================================================================
# 4. Synthetic Market Data Provider
# ============================================================================
class SyntheticMarketData(MarketDataProvider):
    """
    Synthetic market data generator.

    Uses NSS model to generate yield curves on the fly.
    Useful for backtesting, scenario analysis, and when CSV data is unavailable.
    """

    def __init__(
        self,
        nss_params: dict[Currency, NSSParams] | None = None,
        regime: YieldCurveRegime = YieldCurveRegime.BEAR_FLATTENING,
        random_seed: int | None = 42,
    ) -> None:
        """
        Initialize synthetic provider.

        Args:
            nss_params: NSS parameters by currency (defaults to DEFAULT_NSS_PARAMS)
            regime: Yield curve regime for evolution
            random_seed: Seed for reproducibility
        """
        self._nss_params = nss_params or DEFAULT_NSS_PARAMS.copy()
        self._regime = regime
        self._rng = np.random.default_rng(random_seed)
        self._base_date = date(2025, 12, 31)  # Reference date for evolution

        logger.info(f"SyntheticMarketData initialized with regime={regime.value}")

    @property
    def provider_name(self) -> str:
        return "SyntheticMarketData"

    def _evolve_params(
        self,
        params: NSSParams,
        days_forward: int,
    ) -> NSSParams:
        """
        Evolve NSS parameters forward in time.

        Applies regime-based drift + small random noise.
        """
        # Get regime shock (annual rate)
        shock = REGIME_SHOCKS.get(self._regime)
        if shock is None:
            return params

        # Scale to time period
        years = days_forward / 365.25
        scale = min(years, 1.0)  # Cap at 1 year of drift

        # Apply drift with small noise
        noise_scale = 0.001  # 10bp standard deviation
        noise = self._rng.normal(0, noise_scale, 4)

        return NSSParams(
            beta0=np.clip(params.beta0 + shock.delta_beta0 * scale + noise[0], 0.0, 0.20),
            beta1=np.clip(params.beta1 + shock.delta_beta1 * scale + noise[1], -0.10, 0.10),
            beta2=np.clip(params.beta2 + shock.delta_beta2 * scale + noise[2], -0.10, 0.10),
            beta3=np.clip(params.beta3 + shock.delta_beta3 * scale + noise[3], -0.10, 0.10),
            lambda1=params.lambda1,
            lambda2=params.lambda2,
        )

    def get_yield_curve(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldCurveData:
        """Generate synthetic yield curve."""
        base_params = self._nss_params.get(currency, DEFAULT_NSS_PARAMS[Currency.USD])

        # Evolve parameters from base date
        days_forward = (as_of_date - self._base_date).days
        if days_forward > 0:
            evolved_params = self._evolve_params(base_params, days_forward)
        else:
            evolved_params = base_params

        # Generate curve points
        nss = NelsonSiegelSvensson.from_params(evolved_params)
        tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        yields = nss.yield_at_tenor(np.array(tenors)).tolist()

        return YieldCurveData(
            date=as_of_date,
            currency=currency,
            tenors=tenors,
            yields=yields,
        )

    def get_yield_surface(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldSurface:
        """Generate synthetic yield surface."""
        base_params = self._nss_params.get(currency, DEFAULT_NSS_PARAMS[Currency.USD])

        days_forward = (as_of_date - self._base_date).days
        if days_forward > 0:
            evolved_params = self._evolve_params(base_params, days_forward)
        else:
            evolved_params = base_params

        nss = NelsonSiegelSvensson.from_params(evolved_params)
        return YieldSurface(nss)

    def get_fx_rate(
        self,
        base_currency: Currency,
        as_of_date: date,
    ) -> float:
        """Generate synthetic FX rate."""
        if base_currency == Currency.USD:
            return 1.0

        # Base rates
        base_rates = {
            Currency.EUR: 1.08,
            Currency.AUD: 0.65,
            Currency.CNY: 7.25,
            Currency.CNH: 7.28,
        }

        base = base_rates.get(base_currency, 1.0)
        days = (as_of_date - self._base_date).days

        # Add small random walk
        if days > 0:
            drift = self._rng.normal(0, 0.0001 * days)
            return base * (1 + drift)

        return base

    def get_historical_yields(
        self,
        currency: Currency,
        start_date: date,
        end_date: date,
        tenor: float,
    ) -> pd.DataFrame:
        """Generate synthetic historical yields."""
        dates = pd.date_range(start_date, end_date, freq="B")  # Business days

        yields = []
        for d in dates:
            curve = self.get_yield_curve(currency, d.date())
            # Interpolate to requested tenor
            tenors, ylds = curve.to_array()
            yld = np.interp(tenor, tenors, ylds)
            yields.append(yld)

        return pd.DataFrame({"date": dates.date, "yield": yields})

    def set_regime(self, regime: YieldCurveRegime) -> None:
        """Update the yield curve regime."""
        self._regime = regime
        logger.info(f"Regime updated to {regime.value}")


# ============================================================================
# 5. Smart Factory with Fallback
# ============================================================================
class MarketDataFactory:
    """
    Factory for creating market data providers with automatic fallback.

    Attempts to load CSV data first; if unavailable, falls back to synthetic.
    """

    @staticmethod
    def create(
        config: SimulationParams,
        force_synthetic: bool = False,
    ) -> MarketDataProvider:
        """
        Create appropriate market data provider.

        Args:
            config: Simulation configuration
            force_synthetic: If True, skip CSV attempt

        Returns:
            MarketDataProvider instance
        """
        if force_synthetic or config.use_synthetic_data:
            logger.info("Using synthetic data provider (forced)")
            return SyntheticMarketData(
                nss_params=dict(config.nss_params),
                regime=config.curve_regime,
                random_seed=config.random_seed,
            )

        # Try CSV first
        csv_provider = CSVMarketData(config.data_dir)

        # Test if data is available
        try:
            csv_provider.get_yield_curve(Currency.USD, config.start_date)
            logger.info("CSV data available - using CSVMarketData")
            return csv_provider
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"CSV data unavailable ({e}) - falling back to synthetic")
            return SyntheticMarketData(
                nss_params=dict(config.nss_params),
                regime=config.curve_regime,
                random_seed=config.random_seed,
            )

    @staticmethod
    def create_hybrid(
        config: SimulationParams,
    ) -> MarketDataProvider:
        """
        Create a hybrid provider that tries CSV first, falls back per-request.

        This is useful when some currencies have CSV data and others don't.
        """
        return HybridMarketData(
            csv_provider=CSVMarketData(config.data_dir),
            synthetic_provider=SyntheticMarketData(
                nss_params=dict(config.nss_params),
                regime=config.curve_regime,
                random_seed=config.random_seed,
            ),
        )


class HybridMarketData(MarketDataProvider):
    """
    Hybrid provider that falls back on a per-request basis.
    """

    def __init__(
        self,
        csv_provider: CSVMarketData,
        synthetic_provider: SyntheticMarketData,
    ) -> None:
        self._csv = csv_provider
        self._synthetic = synthetic_provider
        self._fallback_currencies: set[Currency] = set()

    @property
    def provider_name(self) -> str:
        return "HybridMarketData"

    def get_yield_surface(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldSurface:
        if currency in self._fallback_currencies:
            return self._synthetic.get_yield_surface(currency, as_of_date)

        try:
            return self._csv.get_yield_surface(currency, as_of_date)
        except (FileNotFoundError, ValueError):
            logger.warning(f"Falling back to synthetic for {currency}")
            self._fallback_currencies.add(currency)
            return self._synthetic.get_yield_surface(currency, as_of_date)

    def get_yield_curve(
        self,
        currency: Currency,
        as_of_date: date,
    ) -> YieldCurveData:
        if currency in self._fallback_currencies:
            return self._synthetic.get_yield_curve(currency, as_of_date)

        try:
            return self._csv.get_yield_curve(currency, as_of_date)
        except (FileNotFoundError, ValueError):
            self._fallback_currencies.add(currency)
            return self._synthetic.get_yield_curve(currency, as_of_date)

    def get_fx_rate(
        self,
        base_currency: Currency,
        as_of_date: date,
    ) -> float:
        try:
            return self._csv.get_fx_rate(base_currency, as_of_date)
        except (FileNotFoundError, ValueError):
            return self._synthetic.get_fx_rate(base_currency, as_of_date)

    def get_historical_yields(
        self,
        currency: Currency,
        start_date: date,
        end_date: date,
        tenor: float,
    ) -> pd.DataFrame:
        if currency in self._fallback_currencies:
            return self._synthetic.get_historical_yields(currency, start_date, end_date, tenor)

        try:
            return self._csv.get_historical_yields(currency, start_date, end_date, tenor)
        except (FileNotFoundError, ValueError):
            self._fallback_currencies.add(currency)
            return self._synthetic.get_historical_yields(currency, start_date, end_date, tenor)


# ============================================================================
# 6. Module Test
# ============================================================================
if __name__ == "__main__":
    from config import get_default_config

    print("=== Data Provider Module Test ===\n")

    config = get_default_config()

    # Test factory
    provider = MarketDataFactory.create(config, force_synthetic=True)
    print(f"Provider: {provider.provider_name}")

    # Get yield surface
    surface = provider.get_yield_surface(Currency.USD, date(2026, 6, 1))
    print(f"\nYield Surface for USD (2026-06-01):")

    # Sample some rates
    fwd_starts = np.array([0.0, 0.5, 1.0])
    tenors = np.array([2.0, 5.0, 10.0])
    rates = surface.get_forward_rates(fwd_starts, tenors)

    print(f"Forward starts: {fwd_starts}")
    print(f"Tenors: {tenors}")
    print(f"Rates:\n{rates * 100}")

    # Get FX rate
    fx = provider.get_fx_rate(Currency.AUD, date(2026, 6, 1))
    print(f"\nAUD/USD: {fx:.4f}")
