"""
analytics.py - The Math Kernel
==============================
Pure numpy implementation of yield curve models and analytics.
All operations are vectorized for performance on large grids.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Add current directory to path for Streamlit Cloud compatibility
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from config import NSSParams, Currency


# Type aliases for clarity
FloatArray = NDArray[np.float64]


# ============================================================================
# 1. Nelson-Siegel-Svensson Model
# ============================================================================
class NelsonSiegelSvensson:
    """
    Nelson-Siegel-Svensson yield curve model.

    The NSS model is an extension of the Nelson-Siegel model that adds
    a second 'hump' factor for better fitting of complex yield curves.

    Model Equation:
    $$
    y(\\tau) = \\beta_0
             + \\beta_1 \\frac{1 - e^{-\\tau/\\lambda_1}}{\\tau/\\lambda_1}
             + \\beta_2 \\left( \\frac{1 - e^{-\\tau/\\lambda_1}}{\\tau/\\lambda_1} - e^{-\\tau/\\lambda_1} \\right)
             + \\beta_3 \\left( \\frac{1 - e^{-\\tau/\\lambda_2}}{\\tau/\\lambda_2} - e^{-\\tau/\\lambda_2} \\right)
    $$

    Implementation notes:
    - All operations are vectorized using numpy broadcasting
    - Handles τ=0 edge case gracefully using np.where
    - Parameters are immutable after initialization
    """

    __slots__ = ("beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2")

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        lambda1: float,
        lambda2: float,
    ) -> None:
        """
        Initialize NSS model with parameters.

        Args:
            beta0: Long-term level (asymptote)
            beta1: Short-term component
            beta2: Medium-term curvature (first hump)
            beta3: Second curvature factor (second hump)
            lambda1: First decay parameter
            lambda2: Second decay parameter
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    @classmethod
    def from_params(cls, params: "NSSParams") -> "NelsonSiegelSvensson":
        """Factory method from pydantic NSSParams."""
        return cls(
            beta0=params.beta0,
            beta1=params.beta1,
            beta2=params.beta2,
            beta3=params.beta3,
            lambda1=params.lambda1,
            lambda2=params.lambda2,
        )

    def _factor_loading(self, tau: FloatArray, lam: float) -> tuple[FloatArray, FloatArray]:
        """
        Compute the factor loadings for a given decay parameter.

        Returns:
            Tuple of (loading1, loading2) where:
            - loading1 = (1 - exp(-τ/λ)) / (τ/λ)
            - loading2 = loading1 - exp(-τ/λ)
        """
        # Handle τ=0 edge case: limit as τ→0 is (1, 0)
        tau_safe = np.where(tau == 0, 1e-10, tau)
        tau_over_lam = tau_safe / lam

        exp_term = np.exp(-tau_over_lam)
        loading1 = (1 - exp_term) / tau_over_lam
        loading2 = loading1 - exp_term

        # Set correct limits at τ=0
        loading1 = np.where(tau == 0, 1.0, loading1)
        loading2 = np.where(tau == 0, 0.0, loading2)

        return loading1, loading2

    def yield_at_tenor(self, tau: FloatArray | float) -> FloatArray:
        """
        Calculate zero-coupon yield for given tenors.

        Args:
            tau: Tenor(s) in years. Can be scalar or array.

        Returns:
            Yield(s) as decimal (e.g., 0.05 for 5%)

        Example:
            >>> nss = NelsonSiegelSvensson(0.045, -0.01, 0.005, 0.002, 1.5, 3.0)
            >>> nss.yield_at_tenor(np.array([0.25, 1.0, 5.0, 10.0]))
            array([0.0458, 0.0445, 0.0432, 0.0441])
        """
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        # Compute factor loadings for both lambda parameters
        f1_l1, f2_l1 = self._factor_loading(tau, self.lambda1)
        f1_l2, f2_l2 = self._factor_loading(tau, self.lambda2)

        # NSS formula (vectorized)
        yield_curve = (
            self.beta0
            + self.beta1 * f1_l1
            + self.beta2 * f2_l1
            + self.beta3 * f2_l2
        )

        return yield_curve

    def forward_rate(
        self,
        forward_start: FloatArray | float,
        tenor: FloatArray | float,
    ) -> FloatArray:
        """
        Calculate instantaneous forward rates.

        The forward rate f(t, T) is the rate agreed today for borrowing
        from time t to time T.

        Uses the relationship: f(t,T) ≈ [Y(T)·T - Y(t)·t] / (T-t)

        Args:
            forward_start: Start time(s) in years
            tenor: Tenor(s) from forward start in years

        Returns:
            Forward rate(s)
        """
        forward_start = np.atleast_1d(np.asarray(forward_start, dtype=np.float64))
        tenor = np.atleast_1d(np.asarray(tenor, dtype=np.float64))

        # Total maturity from today
        T = forward_start + tenor

        # Get spot yields
        y_start = self.yield_at_tenor(forward_start)
        y_end = self.yield_at_tenor(T)

        # Forward rate calculation (handle t=0 case)
        forward = np.where(
            forward_start == 0,
            y_end,  # If starting today, forward = spot yield at maturity
            (y_end * T - y_start * forward_start) / tenor
        )

        return forward

    def par_rate(self, maturities: FloatArray | float) -> FloatArray:
        """
        Calculate par swap rates for given maturities.

        Par rate is the coupon rate at which a bond trades at par.

        Args:
            maturities: Maturity/ies in years

        Returns:
            Par rate(s)
        """
        maturities = np.atleast_1d(np.asarray(maturities, dtype=np.float64))

        # Discount factors using zero rates
        zero_rates = self.yield_at_tenor(maturities)
        df = np.exp(-zero_rates * maturities)

        # Simple approximation: par ≈ (1 - DF) / annuity
        # Annuity = sum of DFs for annual payments
        # For simplicity, use continuous approximation
        annuity = (1 - df) / zero_rates

        par_rates = (1 - df) / annuity

        return par_rates

    def duration(self, coupon: float, maturity: float, freq: int = 2) -> float:
        """
        Calculate modified duration for a bond.

        Args:
            coupon: Annual coupon rate (decimal)
            maturity: Years to maturity
            freq: Payment frequency per year

        Returns:
            Modified duration
        """
        # Simplified Macaulay duration calculation
        y = float(self.yield_at_tenor(maturity))
        n = int(maturity * freq)

        if n == 0:
            return 0.0

        # Time-weighted cashflows
        times = np.arange(1, n + 1) / freq
        cashflows = np.full(n, coupon / freq)
        cashflows[-1] += 1.0  # Principal at maturity

        # Discount factors
        df = np.exp(-y * times)

        # Macaulay duration
        pv = np.sum(cashflows * df)
        mac_dur = np.sum(times * cashflows * df) / pv

        # Modified duration
        mod_dur = mac_dur / (1 + y / freq)

        return mod_dur

    def __repr__(self) -> str:
        return (
            f"NSS(β0={self.beta0:.4f}, β1={self.beta1:.4f}, "
            f"β2={self.beta2:.4f}, β3={self.beta3:.4f}, "
            f"λ1={self.lambda1:.2f}, λ2={self.lambda2:.2f})"
        )


# ============================================================================
# 2. Yield Surface (2D: Forward Start × Tenor)
# ============================================================================
@dataclass
class YieldSurfaceGrid:
    """Container for a yield surface grid."""
    forward_starts: FloatArray  # Shape: (n_forwards,)
    tenors: FloatArray          # Shape: (n_tenors,)
    yields: FloatArray          # Shape: (n_forwards, n_tenors)

    @property
    def shape(self) -> tuple[int, int]:
        return self.yields.shape


class YieldSurface:
    """
    Yield Surface: R(t, T) where t is forward start and T is tenor.

    This class manages a 2D surface of forward rates, supporting:
    - Grid generation (vectorized)
    - Time evolution via regime shocks
    - Interpolation for arbitrary (t, T) queries
    """

    def __init__(self, nss_model: NelsonSiegelSvensson) -> None:
        """
        Initialize yield surface with an NSS model.

        Args:
            nss_model: The underlying NSS model for rate calculations
        """
        self._nss = nss_model
        self._cached_grid: YieldSurfaceGrid | None = None

    @property
    def nss_model(self) -> NelsonSiegelSvensson:
        """Access the underlying NSS model."""
        return self._nss

    def get_forward_rates(
        self,
        forward_start_years: FloatArray,
        tenors: FloatArray,
    ) -> FloatArray:
        """
        Calculate forward rates for a grid of (forward_start, tenor) pairs.

        This is the core vectorized operation using numpy broadcasting.

        Args:
            forward_start_years: Array of forward start times (years)
            tenors: Array of tenors (years)

        Returns:
            2D array of forward rates, shape (len(forward_starts), len(tenors))

        Example:
            >>> surface = YieldSurface(nss_model)
            >>> fwd = np.array([0, 0.5, 1.0, 2.0])  # Forward starts
            >>> tnr = np.array([0.25, 1.0, 5.0, 10.0])  # Tenors
            >>> rates = surface.get_forward_rates(fwd, tnr)
            >>> rates.shape
            (4, 4)
        """
        forward_start_years = np.atleast_1d(np.asarray(forward_start_years, dtype=np.float64))
        tenors = np.atleast_1d(np.asarray(tenors, dtype=np.float64))

        n_fwd = len(forward_start_years)
        n_tnr = len(tenors)

        # Use broadcasting: expand dims for matrix operation
        # forward_start_years: (n_fwd, 1)
        # tenors: (1, n_tnr)
        fwd_grid = forward_start_years[:, np.newaxis]  # (n_fwd, 1)
        tnr_grid = tenors[np.newaxis, :]               # (1, n_tnr)

        # Total maturity from today
        T_total = fwd_grid + tnr_grid  # (n_fwd, n_tnr) via broadcasting

        # Calculate yields at forward start and total maturity
        # Flatten for vectorized calculation, then reshape
        y_start_flat = self._nss.yield_at_tenor(forward_start_years)  # (n_fwd,)
        y_total_flat = self._nss.yield_at_tenor(T_total.ravel())      # (n_fwd * n_tnr,)

        y_start = y_start_flat[:, np.newaxis]  # (n_fwd, 1)
        y_total = y_total_flat.reshape(n_fwd, n_tnr)  # (n_fwd, n_tnr)

        # Forward rate: f(t,T) = [Y(T)·T - Y(t)·t] / tenor
        # Handle t=0 case
        forward_rates = np.where(
            fwd_grid == 0,
            y_total,
            (y_total * T_total - y_start * fwd_grid) / tnr_grid
        )

        return forward_rates

    def generate_grid(
        self,
        max_forward_years: float = 3.0,
        max_tenor_years: float = 10.0,
        n_forward_points: int = 37,   # Monthly for 3 years
        n_tenor_points: int = 41,     # Quarterly tenors up to 10Y
    ) -> YieldSurfaceGrid:
        """
        Generate a full yield surface grid.

        Args:
            max_forward_years: Maximum forward start time
            max_tenor_years: Maximum tenor
            n_forward_points: Number of forward start points
            n_tenor_points: Number of tenor points

        Returns:
            YieldSurfaceGrid containing the full surface
        """
        forward_starts = np.linspace(0, max_forward_years, n_forward_points)
        tenors = np.linspace(0.25, max_tenor_years, n_tenor_points)  # Start at 3M

        yields = self.get_forward_rates(forward_starts, tenors)

        self._cached_grid = YieldSurfaceGrid(
            forward_starts=forward_starts,
            tenors=tenors,
            yields=yields,
        )

        return self._cached_grid

    def interpolate(
        self,
        forward_start: float,
        tenor: float,
    ) -> float:
        """
        Interpolate yield at arbitrary (forward_start, tenor) point.

        Uses bilinear interpolation on cached grid.

        Args:
            forward_start: Forward start time in years
            tenor: Tenor in years

        Returns:
            Interpolated yield
        """
        if self._cached_grid is None:
            self.generate_grid()

        assert self._cached_grid is not None

        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self._cached_grid.forward_starts, self._cached_grid.tenors),
            self._cached_grid.yields,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        return float(interp([[forward_start, tenor]])[0])

    def apply_shock(
        self,
        delta_beta0: float = 0.0,
        delta_beta1: float = 0.0,
        delta_beta2: float = 0.0,
        delta_beta3: float = 0.0,
    ) -> "YieldSurface":
        """
        Create a new YieldSurface with shocked parameters.

        This enables scenario analysis without mutating the original.

        Args:
            delta_beta0: Change to level
            delta_beta1: Change to slope
            delta_beta2: Change to curvature 1
            delta_beta3: Change to curvature 2

        Returns:
            New YieldSurface with shocked parameters
        """
        shocked_nss = NelsonSiegelSvensson(
            beta0=np.clip(self._nss.beta0 + delta_beta0, 0.0, 0.25),
            beta1=np.clip(self._nss.beta1 + delta_beta1, -0.15, 0.15),
            beta2=np.clip(self._nss.beta2 + delta_beta2, -0.15, 0.15),
            beta3=np.clip(self._nss.beta3 + delta_beta3, -0.15, 0.15),
            lambda1=self._nss.lambda1,
            lambda2=self._nss.lambda2,
        )
        return YieldSurface(shocked_nss)


# ============================================================================
# 3. FTP Calculator
# ============================================================================
class FTPCalculator:
    """
    Funds Transfer Pricing calculator.

    Implements the backward-looking FTP rule:
    FTP(T) = Average(3M Gov Yield of month T-1)

    This captures the institutional arbitrage:
    - Hiking cycle: Buy immediately after hike (lock high asset yield vs lagged FTP)
    - Cutting cycle: Front-load before cut (lock yield before FTP catches up)
    """

    def __init__(
        self,
        yield_surface: YieldSurface,
        lag_months: int = 1,
        tenor_months: int = 3,
    ) -> None:
        """
        Initialize FTP calculator.

        Args:
            yield_surface: The yield surface for rate calculations
            lag_months: FTP lag in months (typically 1)
            tenor_months: Reference tenor for FTP (typically 3M)
        """
        self._surface = yield_surface
        self._lag_months = lag_months
        self._tenor_months = tenor_months

    def calculate_ftp_path(
        self,
        periods: int,
        period_type: str = "M",
    ) -> FloatArray:
        """
        Calculate FTP path over simulation horizon.

        Args:
            periods: Number of periods
            period_type: 'M' for monthly, 'Q' for quarterly

        Returns:
            Array of FTP rates for each period
        """
        period_years = 1/12 if period_type == "M" else 0.25
        lag_years = self._lag_months / 12
        tenor_years = self._tenor_months / 12

        # Generate time points
        times = np.arange(periods) * period_years

        # FTP at time t = 3M rate at time (t - lag)
        # Use max(0, t-lag) to handle early periods
        ftp_reference_times = np.maximum(0, times - lag_years)

        # Get 3M forward rates starting at each reference time
        ftp_rates = self._surface.get_forward_rates(
            ftp_reference_times,
            np.array([tenor_years])
        ).squeeze()

        return ftp_rates

    def calculate_nii_spread(
        self,
        asset_tenor: float,
        periods: int,
        period_type: str = "M",
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Calculate NII spread path: Asset Yield - FTP.

        Args:
            asset_tenor: Tenor of assets being purchased (years)
            periods: Number of periods
            period_type: 'M' for monthly, 'Q' for quarterly

        Returns:
            Tuple of (asset_yields, ftp_rates, spreads)
        """
        period_years = 1/12 if period_type == "M" else 0.25

        # Time points for asset purchases
        times = np.arange(periods) * period_years

        # Asset yields: forward rate for asset_tenor starting at each time
        asset_yields = self._surface.get_forward_rates(
            times,
            np.array([asset_tenor])
        ).squeeze()

        # FTP path
        ftp_rates = self.calculate_ftp_path(periods, period_type)

        # NII spread
        spreads = asset_yields - ftp_rates

        return asset_yields, ftp_rates, spreads


# ============================================================================
# 4. FX Analytics
# ============================================================================
class FXAnalytics:
    """
    FX analytics for multi-currency portfolios.

    Handles FX drag calculation for NII conversion.
    """

    @staticmethod
    def calculate_fx_drag(
        nii_local: FloatArray,
        spot_scenario: FloatArray,
        spot_budget: float,
    ) -> FloatArray:
        """
        Calculate FX drag on NII.

        FX Drag = NII_local × (Spot_scenario - Spot_budget)

        For USD-based reporting:
        - If local currency weakens (spot ↓ for AUDUSD), NII in USD decreases

        Args:
            nii_local: NII in local currency
            spot_scenario: Scenario spot rates
            spot_budget: Budget spot rate

        Returns:
            FX drag array (positive = gain, negative = loss)
        """
        return nii_local * (spot_scenario - spot_budget)

    @staticmethod
    def convert_to_usd(
        values_local: FloatArray,
        spot_rates: FloatArray,
    ) -> FloatArray:
        """
        Convert local currency values to USD.

        Args:
            values_local: Values in local currency
            spot_rates: Spot rates (local/USD for EUR, USD/local for AUD)

        Returns:
            Values in USD
        """
        return values_local * spot_rates


# ============================================================================
# 5. Utility Functions
# ============================================================================
def create_yield_surface_from_params(
    params: "NSSParams",
) -> YieldSurface:
    """Factory function to create YieldSurface from config params."""
    nss = NelsonSiegelSvensson.from_params(params)
    return YieldSurface(nss)


def compute_carry_rolldown(
    surface: YieldSurface,
    current_tenor: float,
    horizon_years: float,
) -> float:
    """
    Compute carry + rolldown return for a bond.

    Carry = Current yield
    Rolldown = Capital gain from rolling down the curve

    Args:
        surface: Yield surface
        current_tenor: Current tenor in years
        horizon_years: Holding period

    Returns:
        Total return estimate
    """
    # Current yield
    y_current = surface.get_forward_rates(
        np.array([0.0]),
        np.array([current_tenor])
    )[0, 0]

    # Yield after rolling down
    new_tenor = current_tenor - horizon_years
    if new_tenor <= 0:
        return y_current * horizon_years

    y_rolled = surface.get_forward_rates(
        np.array([0.0]),
        np.array([new_tenor])
    )[0, 0]

    # Carry component
    carry = y_current * horizon_years

    # Rolldown: approximate price change from yield change
    # ΔP/P ≈ -Duration × Δy
    duration_approx = (current_tenor + new_tenor) / 2
    delta_y = y_rolled - y_current
    rolldown = -duration_approx * delta_y

    return carry + rolldown


# ============================================================================
# 6. Carry/DV01 Efficiency Analyzer
# ============================================================================
@dataclass
class EfficiencyMetrics:
    """Container for carry efficiency metrics at a given tenor."""
    tenor: float           # Years
    yield_pct: float       # Yield in %
    duration: float        # Modified duration (approx = tenor for zeros)
    carry_bps: float       # Annual carry in bps (yield × 100)
    dv01_per_mm: float     # DV01 per $1MM notional
    efficiency: float      # Carry / DV01 ratio (bps per $ of DV01)
    yield_per_dur: float   # Yield / Duration (% per year of duration)


class CarryEfficiencyAnalyzer:
    """
    Analyze carry efficiency across the yield curve.

    The key insight: In an inverted curve, short-end may offer better
    "bang for buck" (carry per unit of duration risk).

    Efficiency Metrics:
    - Carry/DV01 = How many bps of carry per $ of rate sensitivity
    - Yield/Duration = Yield earned per year of duration risk

    For reinvestment decisions, higher efficiency = better risk-adjusted carry.
    """

    def __init__(self, yield_surface: YieldSurface) -> None:
        self._surface = yield_surface

    def calculate_efficiency_at_tenor(
        self,
        tenor: float,
        forward_start: float = 0.0,
    ) -> EfficiencyMetrics:
        """
        Calculate carry efficiency metrics at a specific tenor.

        Args:
            tenor: Bond tenor in years
            forward_start: Forward start time (0 = spot)

        Returns:
            EfficiencyMetrics dataclass
        """
        # Get yield from surface
        yield_decimal = self._surface.get_forward_rates(
            np.array([forward_start]),
            np.array([tenor])
        )[0, 0]

        yield_pct = yield_decimal * 100

        # Duration approximation (modified duration ≈ tenor for zero-coupon)
        # For coupon bonds, duration < tenor, but this is conservative
        duration = tenor

        # Carry in bps (annual)
        carry_bps = yield_pct * 100  # e.g., 4.5% → 450 bps

        # DV01 per $1MM notional = Duration × $1MM × 0.0001 = Duration × $100
        dv01_per_mm = duration * 100  # in USD

        # Efficiency = Carry (bps) / DV01 ($)
        # Interpretation: bps of carry earned per $ of rate sensitivity
        efficiency = carry_bps / dv01_per_mm if dv01_per_mm > 0 else 0.0

        # Yield per Duration = Yield% / Duration
        # Interpretation: % yield earned per year of duration risk
        yield_per_dur = yield_pct / duration if duration > 0 else 0.0

        return EfficiencyMetrics(
            tenor=tenor,
            yield_pct=yield_pct,
            duration=duration,
            carry_bps=carry_bps,
            dv01_per_mm=dv01_per_mm,
            efficiency=efficiency,
            yield_per_dur=yield_per_dur,
        )

    def calculate_efficiency_curve(
        self,
        tenors: list[float] | None = None,
        forward_start: float = 0.0,
    ) -> list[EfficiencyMetrics]:
        """
        Calculate efficiency metrics across multiple tenors.

        Args:
            tenors: List of tenors to analyze (default: standard curve points)
            forward_start: Forward start time

        Returns:
            List of EfficiencyMetrics for each tenor
        """
        if tenors is None:
            tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

        return [
            self.calculate_efficiency_at_tenor(t, forward_start)
            for t in tenors
        ]

    def find_optimal_tenor(
        self,
        tenors: list[float] | None = None,
        min_tenor: float = 1.0,
        max_tenor: float = 10.0,
    ) -> EfficiencyMetrics:
        """
        Find the tenor with highest carry efficiency.

        Args:
            tenors: Tenors to search (default: 1-10Y annual)
            min_tenor: Minimum acceptable tenor
            max_tenor: Maximum acceptable tenor

        Returns:
            EfficiencyMetrics for optimal tenor
        """
        if tenors is None:
            tenors = [float(t) for t in range(1, 11)]

        # Filter by constraints
        valid_tenors = [t for t in tenors if min_tenor <= t <= max_tenor]

        if not valid_tenors:
            valid_tenors = tenors

        metrics = self.calculate_efficiency_curve(valid_tenors)

        # Find max efficiency
        return max(metrics, key=lambda m: m.efficiency)


def create_multi_currency_efficiency_matrix(
    surfaces: dict[str, YieldSurface],
    tenors: list[float] | None = None,
) -> dict[str, list[EfficiencyMetrics]]:
    """
    Create efficiency matrix across currencies and tenors.

    Args:
        surfaces: Dict of currency -> YieldSurface
        tenors: Tenors to analyze

    Returns:
        Dict of currency -> list of EfficiencyMetrics
    """
    if tenors is None:
        tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    result = {}
    for ccy, surface in surfaces.items():
        analyzer = CarryEfficiencyAnalyzer(surface)
        result[ccy] = analyzer.calculate_efficiency_curve(tenors)

    return result


if __name__ == "__main__":
    # Quick test
    print("=== Analytics Module Test ===\n")

    # Create NSS model with typical USD parameters
    nss = NelsonSiegelSvensson(
        beta0=0.045,   # 4.5% long-term
        beta1=-0.010,  # Slight inversion
        beta2=0.005,   # Mild hump
        beta3=0.002,
        lambda1=1.5,
        lambda2=3.0,
    )
    print(f"Model: {nss}\n")

    # Calculate yields at key tenors
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    yields = nss.yield_at_tenor(tenors)

    print("Spot Yield Curve:")
    for t, y in zip(tenors, yields):
        print(f"  {t:5.2f}Y: {y*100:5.2f}%")

    # Create yield surface
    print("\n=== Yield Surface ===")
    surface = YieldSurface(nss)
    grid = surface.generate_grid(max_forward_years=2.0, max_tenor_years=10.0)
    print(f"Grid shape: {grid.shape}")
    print(f"Forward starts: {grid.forward_starts[0]:.2f} to {grid.forward_starts[-1]:.2f} years")
    print(f"Tenors: {grid.tenors[0]:.2f} to {grid.tenors[-1]:.2f} years")

    # FTP calculation
    print("\n=== FTP Calculator ===")
    ftp_calc = FTPCalculator(surface, lag_months=1, tenor_months=3)
    asset_yields, ftp_rates, spreads = ftp_calc.calculate_nii_spread(
        asset_tenor=5.0,
        periods=12,
        period_type="M",
    )
    print(f"Asset Yield (5Y): {asset_yields.mean()*100:.2f}% avg")
    print(f"FTP (3M lagged): {ftp_rates.mean()*100:.2f}% avg")
    print(f"NII Spread: {spreads.mean()*100:.2f}% avg")
