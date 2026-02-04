"""
yield_curve_loader.py - Data Loader for Forward Rate Surfaces
==============================================================
Centralized utility for loading yield curve data from the Data Warehouse.
Supports interpolation for non-standard tenors via Linear or Cubic Spline.

Usage:
    from yield_curve_loader import (
        load_forward_surfaces,
        get_surface_by_currency,
        interpolate_rate,
    )

    surfaces = load_forward_surfaces()
    usd_surface = get_surface_by_currency("USD")
    rate = interpolate_rate(usd_surface, tenor_years=2.5, forward_years=0.5)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import (
    interp1d,
    RectBivariateSpline,
    RegularGridInterpolator,
)

# Path configuration
_DATA_WAREHOUSE_PATH = Path(__file__).parent.parent / "db"
_YIELD_CURVES_FILE = _DATA_WAREHOUSE_PATH / "yield_curves_snapshot.json"


@dataclass(frozen=True)
class ForwardRateSurface:
    """
    Container for a forward rate matrix with interpolation support.

    Axes:
    - Rows (tenors): Underlying bond maturity (e.g., 1M, 2Y, 10Y)
    - Columns (forward_starts): When the forward contract starts (e.g., Spot, 3M, 1Y)

    Values are in percentage points (e.g., 4.50 = 4.50%).
    """
    currency: str
    as_of_date: date
    tenor_labels: tuple[str, ...]
    forward_labels: tuple[str, ...]
    rates: NDArray[np.float64]

    @property
    def tenors_years(self) -> NDArray[np.float64]:
        """Convert tenor labels to years."""
        return np.array([_parse_tenor(t) for t in self.tenor_labels])

    @property
    def forwards_years(self) -> NDArray[np.float64]:
        """Convert forward labels to years."""
        return np.array([_parse_tenor(f) for f in self.forward_labels])

    def get_rate(self, tenor: str, forward: str) -> float:
        """Get a specific rate by labels."""
        i = self.tenor_labels.index(tenor)
        j = self.forward_labels.index(forward)
        return float(self.rates[i, j])

    def to_dataframe(self):
        """Convert to pandas DataFrame for display."""
        import pandas as pd
        return pd.DataFrame(
            self.rates,
            index=list(self.tenor_labels),
            columns=list(self.forward_labels),
        )

    def get_spot_curve(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract spot yield curve (first column)."""
        return self.tenors_years, self.rates[:, 0] / 100  # Convert to decimal


def _parse_tenor(label: str) -> float:
    """Parse tenor string to years. 'Spot' -> 0, '3M' -> 0.25, '1Y' -> 1.0"""
    label = label.strip()
    if label.lower() in ("spot", "息票 (spot)", "息票"):
        return 0.0
    if label.endswith("个月") or label.endswith("M") or label.endswith("m"):
        num_str = label.replace("个月", "").replace("M", "").replace("m", "").strip()
        return float(num_str) / 12
    if label.endswith("年") or label.endswith("Y") or label.endswith("y"):
        num_str = label.replace("年", "").replace("Y", "").replace("y", "").strip()
        return float(num_str)
    try:
        return float(label) / 12
    except ValueError:
        return 0.0


def load_forward_surfaces(
    file_path: Path | str | None = None,
) -> dict[str, ForwardRateSurface]:
    """
    Load all forward rate surfaces from the JSON data warehouse.

    Args:
        file_path: Optional path to JSON file. Defaults to standard location.

    Returns:
        Dictionary mapping currency code to ForwardRateSurface.
    """
    path = Path(file_path) if file_path else _YIELD_CURVES_FILE

    if not path.exists():
        raise FileNotFoundError(f"Yield curves file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    surfaces: dict[str, ForwardRateSurface] = {}

    for ccy, surface_data in data.get("surfaces", {}).items():
        # Parse date
        date_str = surface_data.get("as_of_date", "2025-01-31")
        as_of = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Convert rates to numpy array (handle null/None values)
        rates_raw = surface_data["rates"]
        rates = np.array(rates_raw, dtype=np.float64)

        surfaces[ccy] = ForwardRateSurface(
            currency=surface_data["currency"],
            as_of_date=as_of,
            tenor_labels=tuple(surface_data["tenor_labels"]),
            forward_labels=tuple(surface_data["forward_labels"]),
            rates=rates,
        )

    return surfaces


def get_surface_by_currency(
    currency: str,
    file_path: Path | str | None = None,
) -> ForwardRateSurface | None:
    """
    Get a specific currency's forward rate surface.

    Args:
        currency: Currency code (USD, EUR, AUD, CNH)
        file_path: Optional path to JSON file.

    Returns:
        ForwardRateSurface or None if not found.
    """
    surfaces = load_forward_surfaces(file_path)
    return surfaces.get(currency.upper())


def interpolate_rate(
    surface: ForwardRateSurface,
    tenor_years: float,
    forward_years: float = 0.0,
    method: Literal["linear", "cubic"] = "linear",
) -> float:
    """
    Interpolate rate for non-standard tenor/forward using 2D interpolation.

    Args:
        surface: Forward rate surface to interpolate from.
        tenor_years: Target tenor in years (e.g., 2.5 for 2.5Y).
        forward_years: Forward start in years (e.g., 0.5 for 6M forward).
        method: Interpolation method - "linear" or "cubic".

    Returns:
        Interpolated rate in percentage points.

    Raises:
        ValueError: If tenor/forward is outside the surface bounds.
    """
    tenors = surface.tenors_years
    forwards = surface.forwards_years
    rates = surface.rates

    # Handle NaN values by interpolating first
    rates_clean = np.nan_to_num(rates, nan=np.nanmean(rates))

    # Check bounds (with small tolerance for floating point)
    eps = 1e-6
    if tenor_years < tenors.min() - eps or tenor_years > tenors.max() + eps:
        raise ValueError(
            f"Tenor {tenor_years}Y out of bounds [{tenors.min():.2f}, {tenors.max():.2f}]"
        )
    if forward_years < forwards.min() - eps or forward_years > forwards.max() + eps:
        raise ValueError(
            f"Forward {forward_years}Y out of bounds [{forwards.min():.2f}, {forwards.max():.2f}]"
        )

    # Clamp to bounds
    tenor_years = np.clip(tenor_years, tenors.min(), tenors.max())
    forward_years = np.clip(forward_years, forwards.min(), forwards.max())

    if method == "cubic":
        # Use RectBivariateSpline for smooth interpolation
        # Note: kx, ky must be <= min(len(x)-1, 3)
        kx = min(3, len(tenors) - 1)
        ky = min(3, len(forwards) - 1)
        spline = RectBivariateSpline(tenors, forwards, rates_clean, kx=kx, ky=ky)
        result = spline(tenor_years, forward_years)[0, 0]
    else:
        # Use RegularGridInterpolator for linear interpolation
        interp = RegularGridInterpolator(
            (tenors, forwards),
            rates_clean,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        result = interp([[tenor_years, forward_years]])[0]

    return float(result)


def interpolate_spot_rate(
    surface: ForwardRateSurface,
    tenor_years: float,
    method: Literal["linear", "cubic"] = "linear",
) -> float:
    """
    Interpolate spot rate for non-standard tenor using 1D interpolation.

    Args:
        surface: Forward rate surface.
        tenor_years: Target tenor in years.
        method: Interpolation method.

    Returns:
        Interpolated spot rate in percentage points.
    """
    return interpolate_rate(surface, tenor_years, forward_years=0.0, method=method)


def build_new_money_yield_curve(
    surface: ForwardRateSurface,
    target_tenors: list[float],
    forward_months: int = 0,
    method: Literal["linear", "cubic"] = "linear",
) -> dict[float, float]:
    """
    Build a yield curve for new money deployment at specified tenors.

    Args:
        surface: Forward rate surface.
        target_tenors: List of tenors in years (e.g., [2.0, 3.0, 5.0, 7.0, 10.0]).
        forward_months: Forward start in months (e.g., 3 for 3M forward).
        method: Interpolation method.

    Returns:
        Dictionary mapping tenor (years) to yield (percentage).
    """
    forward_years = forward_months / 12.0
    result = {}

    for tenor in target_tenors:
        try:
            rate = interpolate_rate(surface, tenor, forward_years, method)
            result[tenor] = rate
        except ValueError:
            # Skip tenors outside bounds
            continue

    return result


def get_metadata(file_path: Path | str | None = None) -> dict:
    """Get metadata from the yield curves file."""
    path = Path(file_path) if file_path else _YIELD_CURVES_FILE

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("metadata", {})


def update_surface(
    currency: str,
    rates: NDArray[np.float64],
    as_of_date: date | None = None,
    file_path: Path | str | None = None,
) -> None:
    """
    Update a currency's forward rate surface in the JSON file.

    Args:
        currency: Currency code (USD, EUR, AUD, CNH).
        rates: New rate matrix (must match existing shape).
        as_of_date: Date of the new data.
        file_path: Optional path to JSON file.
    """
    path = Path(file_path) if file_path else _YIELD_CURVES_FILE

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ccy = currency.upper()
    if ccy not in data.get("surfaces", {}):
        raise KeyError(f"Currency {ccy} not found in surfaces")

    surface_data = data["surfaces"][ccy]
    expected_shape = (len(surface_data["tenor_labels"]), len(surface_data["forward_labels"]))

    if rates.shape != expected_shape:
        raise ValueError(f"Rate matrix shape {rates.shape} doesn't match expected {expected_shape}")

    # Update rates
    surface_data["rates"] = rates.tolist()
    surface_data["as_of_date"] = (as_of_date or date.today()).isoformat()

    # Update metadata
    data["metadata"]["last_updated"] = date.today().isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# Module Test
# ============================================================================
if __name__ == "__main__":
    print("=== Yield Curve Loader Test ===\n")

    # Load all surfaces
    surfaces = load_forward_surfaces()
    print(f"Loaded {len(surfaces)} surfaces: {list(surfaces.keys())}")

    # Test USD surface
    usd = get_surface_by_currency("USD")
    if usd:
        print(f"\nUSD Surface as of {usd.as_of_date}")
        print(f"Shape: {usd.rates.shape}")
        print(f"Tenor range: {usd.tenors_years.min():.2f}Y - {usd.tenors_years.max():.2f}Y")
        print(f"Forward range: {usd.forwards_years.min():.2f}Y - {usd.forwards_years.max():.2f}Y")

        # Test interpolation
        print("\nInterpolation tests:")
        test_cases = [
            (2.0, 0.0, "2Y Spot"),
            (2.5, 0.0, "2.5Y Spot (interpolated)"),
            (5.0, 0.5, "5Y @ 6M forward"),
            (3.5, 1.0, "3.5Y @ 1Y forward (interpolated)"),
        ]

        for tenor, fwd, label in test_cases:
            rate_linear = interpolate_rate(usd, tenor, fwd, method="linear")
            rate_cubic = interpolate_rate(usd, tenor, fwd, method="cubic")
            print(f"  {label}: Linear={rate_linear:.2f}%, Cubic={rate_cubic:.2f}%")

        # Build new money curve
        print("\nNew Money Yield Curve (Spot):")
        curve = build_new_money_yield_curve(usd, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        for tenor, rate in curve.items():
            print(f"  {tenor:.0f}Y: {rate:.2f}%")

    # Test metadata
    print(f"\nMetadata: {get_metadata()}")
