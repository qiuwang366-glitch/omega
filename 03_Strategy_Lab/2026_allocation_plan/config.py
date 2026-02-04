"""
config.py - Simulation Configuration & Domain Models
======================================================
Pydantic-based configuration for the 2026 Allocation Simulation Engine.
All parameters are strictly typed with validation.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# 1. Enums & Type Aliases
# ============================================================================
class Currency(str, Enum):
    """Supported currencies for FX modeling."""
    USD = "USD"
    EUR = "EUR"
    AUD = "AUD"
    CNY = "CNY"
    CNH = "CNH"


class YieldCurveRegime(str, Enum):
    """Yield curve shape regimes for scenario generation."""
    BULL_STEEPENING = "bull_steepening"   # Rates down, short > long
    BULL_FLATTENING = "bull_flattening"   # Rates down, long > short
    BEAR_STEEPENING = "bear_steepening"   # Rates up, short < long
    BEAR_FLATTENING = "bear_flattening"   # Rates up, long < short
    PARALLEL_SHIFT = "parallel_shift"     # Level change only


AccountingBasis = Literal["AC", "FVOCI", "FVTPL"]


# ============================================================================
# 2. NSS Model Parameters
# ============================================================================
class NSSParams(BaseModel):
    """
    Nelson-Siegel-Svensson parameters.

    The NSS yield curve:
    y(τ) = β0 + β1 * [(1-e^(-τ/λ1)) / (τ/λ1)]
              + β2 * [(1-e^(-τ/λ1)) / (τ/λ1) - e^(-τ/λ1)]
              + β3 * [(1-e^(-τ/λ2)) / (τ/λ2) - e^(-τ/λ2)]

    Interpretation:
    - β0: Long-term level (asymptote)
    - β1: Short-term component (level - short rate)
    - β2: Medium-term curvature (hump)
    - β3: Second curvature factor
    - λ1, λ2: Decay parameters (shape of humps)
    """
    beta0: float = Field(ge=0.0, le=0.20, description="Long-term level (0-20%)")
    beta1: float = Field(ge=-0.10, le=0.10, description="Short-term component")
    beta2: float = Field(ge=-0.10, le=0.10, description="Medium-term hump")
    beta3: float = Field(ge=-0.10, le=0.10, description="Second curvature")
    lambda1: float = Field(gt=0.0, le=10.0, description="First decay parameter")
    lambda2: float = Field(gt=0.0, le=10.0, description="Second decay parameter")

    @field_validator("lambda2")
    @classmethod
    def lambda2_differs_from_lambda1(cls, v: float, info) -> float:
        """Ensure λ2 ≠ λ1 to avoid singularity."""
        if "lambda1" in info.data and abs(v - info.data["lambda1"]) < 0.01:
            raise ValueError("lambda2 must differ from lambda1 by at least 0.01")
        return v

    class Config:
        frozen = True


# ============================================================================
# 3. Default NSS Curves by Currency
# ============================================================================
# These are stylized parameters based on late-2025 market conditions
DEFAULT_NSS_PARAMS: dict[Currency, NSSParams] = {
    Currency.USD: NSSParams(
        beta0=0.045,   # ~4.5% long-term
        beta1=-0.010,  # Slight inversion
        beta2=0.005,   # Mild hump
        beta3=0.002,
        lambda1=1.5,
        lambda2=3.0,
    ),
    Currency.EUR: NSSParams(
        beta0=0.025,   # ~2.5% long-term
        beta1=-0.005,
        beta2=0.003,
        beta3=0.001,
        lambda1=1.8,
        lambda2=4.0,
    ),
    Currency.AUD: NSSParams(
        beta0=0.042,   # ~4.2% long-term
        beta1=-0.008,
        beta2=0.006,
        beta3=0.003,
        lambda1=1.2,
        lambda2=2.5,
    ),
    Currency.CNY: NSSParams(
        beta0=0.025,   # ~2.5% long-term
        beta1=0.003,   # Normal slope
        beta2=0.002,
        beta3=0.001,
        lambda1=2.0,
        lambda2=5.0,
    ),
    Currency.CNH: NSSParams(
        beta0=0.028,   # Slight premium to onshore
        beta1=0.004,
        beta2=0.002,
        beta3=0.001,
        lambda1=2.0,
        lambda2=5.0,
    ),
}


# ============================================================================
# 4. FX Configuration
# ============================================================================
class FXScenario(BaseModel):
    """FX scenario for a single currency pair."""
    base_currency: Currency
    quote_currency: Currency = Currency.USD
    spot_rate: float = Field(gt=0.0, description="Current spot rate")
    scenario_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Named scenarios -> rate (e.g., 'bull': 1.10)"
    )

    @property
    def pair_name(self) -> str:
        return f"{self.base_currency.value}/{self.quote_currency.value}"


class FXConfig(BaseModel):
    """Collection of FX scenarios for multi-currency portfolios."""
    budget_rates: dict[Currency, float] = Field(
        default_factory=lambda: {
            Currency.EUR: 1.08,
            Currency.AUD: 0.65,
            Currency.CNY: 7.25,
            Currency.CNH: 7.28,
        },
        description="Budget FX rates (vs USD)"
    )
    scenarios: dict[str, dict[Currency, float]] = Field(
        default_factory=lambda: {
            "base": {Currency.EUR: 1.08, Currency.AUD: 0.65, Currency.CNY: 7.25, Currency.CNH: 7.28},
            "usd_strong": {Currency.EUR: 1.02, Currency.AUD: 0.60, Currency.CNY: 7.50, Currency.CNH: 7.55},
            "usd_weak": {Currency.EUR: 1.15, Currency.AUD: 0.72, Currency.CNY: 7.00, Currency.CNH: 7.02},
        }
    )


# ============================================================================
# 5. Portfolio & Allocation Config
# ============================================================================
class AllocationTarget(BaseModel):
    """Target allocation for a currency/asset class bucket."""
    currency: Currency
    asset_class: str = Field(description="e.g., 'Sovereign', 'SSA', 'IG_Credit'")
    target_weight: float = Field(ge=0.0, le=1.0)
    min_weight: float = Field(ge=0.0, le=1.0)
    max_weight: float = Field(ge=0.0, le=1.0)
    target_duration: float = Field(ge=0.0, le=30.0)
    accounting_basis: AccountingBasis = "AC"

    @model_validator(mode="after")
    def check_weight_bounds(self) -> "AllocationTarget":
        if not (self.min_weight <= self.target_weight <= self.max_weight):
            raise ValueError("Target weight must be within min/max bounds")
        return self


class PortfolioConfig(BaseModel):
    """Configuration for a sub-portfolio."""
    name: str
    initial_aum: float = Field(gt=0, description="Initial AUM in USD")
    allocations: list[AllocationTarget] = Field(default_factory=list)

    @property
    def total_target_weight(self) -> float:
        return sum(a.target_weight for a in self.allocations)


# ============================================================================
# 6. FTP (Funds Transfer Pricing) Configuration
# ============================================================================
class FTPConfig(BaseModel):
    """
    FTP configuration implementing the 'backward-looking' rule.

    Rule: FTP(T) = Average(3M Gov Yield of month T-1)

    This creates an arbitrage opportunity:
    - Hiking cycle: Buy immediately after hike (Asset Yield ↑, FTP flat)
    - Cutting cycle: Front-load before cut (Lock Asset Yield before it drops)
    """
    lag_months: int = Field(default=1, ge=1, le=3, description="FTP lag in months")
    tenor_months: int = Field(default=3, ge=1, le=12, description="Reference tenor for FTP")
    reference_currency: Currency = Currency.USD
    spread_adjustment_bps: float = Field(
        default=0.0,
        description="Additional spread adjustment to FTP (basis points)"
    )


# ============================================================================
# 7. Simulation Parameters (The Control Center)
# ============================================================================
class SimulationParams(BaseModel):
    """
    Master configuration for the allocation simulation.

    This is the single source of truth for all simulation parameters.
    """
    # === Identification ===
    simulation_name: str = Field(default="2026_Allocation_Plan")
    description: str = Field(default="Strategic allocation simulation for 2026")

    # === Time Horizon ===
    start_date: date = Field(default_factory=lambda: date(2026, 1, 1))
    end_date: date = Field(default_factory=lambda: date(2028, 12, 31))
    frequency: Literal["D", "W", "M", "Q"] = Field(
        default="M",
        description="Simulation frequency: Daily/Weekly/Monthly/Quarterly"
    )

    # === AUM & Constraints ===
    initial_aum_usd: float = Field(
        default=60_000_000_000,  # $60B
        gt=0,
        description="Initial AUM in USD"
    )
    max_single_position_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Max single position as % of AUM"
    )

    # === Yield Curve ===
    nss_params: dict[Currency, NSSParams] = Field(
        default_factory=lambda: DEFAULT_NSS_PARAMS.copy()
    )
    curve_regime: YieldCurveRegime = YieldCurveRegime.BEAR_FLATTENING

    # === FTP Settings ===
    ftp_config: FTPConfig = Field(default_factory=FTPConfig)

    # === FX Settings ===
    fx_config: FXConfig = Field(default_factory=FXConfig)

    # === Data Paths ===
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "01_Data_Warehouse"
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "output"
    )

    # === Model Settings ===
    random_seed: int | None = Field(default=42, description="Seed for reproducibility")
    use_synthetic_data: bool = Field(
        default=False,
        description="Force synthetic data generation"
    )

    @property
    def horizon_years(self) -> float:
        """Simulation horizon in years."""
        return (self.end_date - self.start_date).days / 365.25

    @property
    def n_periods(self) -> int:
        """Number of simulation periods."""
        freq_map = {"D": 365, "W": 52, "M": 12, "Q": 4}
        return int(self.horizon_years * freq_map[self.frequency])

    def get_nss_params(self, currency: Currency) -> NSSParams:
        """Get NSS parameters for a currency, with fallback to USD."""
        return self.nss_params.get(currency, self.nss_params[Currency.USD])

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# 8. Regime Shock Parameters
# ============================================================================
class RegimeShock(BaseModel):
    """
    Defines a shock to apply to NSS parameters for scenario analysis.

    Example: Bear Flattening
    - β0 ↑ (rates up)
    - β1 ↓ (more inversion / flattening)
    """
    name: str
    regime: YieldCurveRegime
    delta_beta0: float = Field(default=0.0, description="Change in level")
    delta_beta1: float = Field(default=0.0, description="Change in slope")
    delta_beta2: float = Field(default=0.0, description="Change in curvature")
    delta_beta3: float = Field(default=0.0, description="Change in second hump")

    def apply_to(self, params: NSSParams) -> NSSParams:
        """Apply shock to NSS parameters."""
        return NSSParams(
            beta0=np.clip(params.beta0 + self.delta_beta0, 0.0, 0.20),
            beta1=np.clip(params.beta1 + self.delta_beta1, -0.10, 0.10),
            beta2=np.clip(params.beta2 + self.delta_beta2, -0.10, 0.10),
            beta3=np.clip(params.beta3 + self.delta_beta3, -0.10, 0.10),
            lambda1=params.lambda1,
            lambda2=params.lambda2,
        )


# Pre-defined regime shocks (stylized)
REGIME_SHOCKS: dict[YieldCurveRegime, RegimeShock] = {
    YieldCurveRegime.BULL_STEEPENING: RegimeShock(
        name="Bull Steepening",
        regime=YieldCurveRegime.BULL_STEEPENING,
        delta_beta0=-0.010,  # Rates down 100bp
        delta_beta1=0.005,   # Curve steepens
    ),
    YieldCurveRegime.BULL_FLATTENING: RegimeShock(
        name="Bull Flattening",
        regime=YieldCurveRegime.BULL_FLATTENING,
        delta_beta0=-0.010,
        delta_beta1=-0.003,  # Curve flattens
    ),
    YieldCurveRegime.BEAR_STEEPENING: RegimeShock(
        name="Bear Steepening",
        regime=YieldCurveRegime.BEAR_STEEPENING,
        delta_beta0=0.010,   # Rates up 100bp
        delta_beta1=0.005,   # Curve steepens
    ),
    YieldCurveRegime.BEAR_FLATTENING: RegimeShock(
        name="Bear Flattening",
        regime=YieldCurveRegime.BEAR_FLATTENING,
        delta_beta0=0.010,
        delta_beta1=-0.005,  # Curve flattens
    ),
    YieldCurveRegime.PARALLEL_SHIFT: RegimeShock(
        name="Parallel Shift",
        regime=YieldCurveRegime.PARALLEL_SHIFT,
        delta_beta0=0.005,   # 50bp parallel shift
    ),
}


# ============================================================================
# 9. Utility Functions
# ============================================================================
def get_default_config() -> SimulationParams:
    """Factory function to create default simulation configuration."""
    return SimulationParams()


def load_config_from_file(path: Path) -> SimulationParams:
    """Load configuration from JSON file."""
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return SimulationParams(**data)


if __name__ == "__main__":
    # Quick validation test
    config = get_default_config()
    print(f"Simulation: {config.simulation_name}")
    print(f"Horizon: {config.start_date} to {config.end_date} ({config.horizon_years:.1f}Y)")
    print(f"Periods: {config.n_periods} ({config.frequency})")
    print(f"Initial AUM: ${config.initial_aum_usd / 1e9:.1f}B")
    print(f"\nNSS Params (USD):")
    usd_params = config.get_nss_params(Currency.USD)
    print(f"  β0={usd_params.beta0:.3f}, β1={usd_params.beta1:.3f}")
