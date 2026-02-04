"""
forward_rate_data.py - Forward Rate Surface Interface
=====================================================
Interface layer for forward rate data. Data is now loaded from the
centralized Data Warehouse (JSON) instead of being hardcoded.

The actual data resides in: 01_Data_Warehouse/db/yield_curves_snapshot.json
Data loading/interpolation logic: 01_Data_Warehouse/etl_scripts/yield_curve_loader.py

This module provides backward-compatible access to the data through the same
interface (ForwardRateSurface, FORWARD_SURFACES registry, get_forward_surface).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Add Data Warehouse to path for imports
_THIS_DIR = Path(__file__).parent
_DATA_WAREHOUSE_ETL = _THIS_DIR.parent.parent / "01_Data_Warehouse" / "etl_scripts"
if str(_DATA_WAREHOUSE_ETL) not in sys.path:
    sys.path.insert(0, str(_DATA_WAREHOUSE_ETL))

# Import from centralized data loader
from yield_curve_loader import (
    ForwardRateSurface,
    load_forward_surfaces,
    get_surface_by_currency,
    interpolate_rate,
    interpolate_spot_rate,
    build_new_money_yield_curve,
    update_surface,
    get_metadata,
    _parse_tenor,
)


# ============================================================================
# Lazy-loaded Registry (populated on first access)
# ============================================================================
_SURFACES_CACHE: dict[str, ForwardRateSurface] | None = None


def _ensure_surfaces_loaded() -> dict[str, ForwardRateSurface]:
    """Ensure surfaces are loaded from Data Warehouse."""
    global _SURFACES_CACHE
    if _SURFACES_CACHE is None:
        try:
            _SURFACES_CACHE = load_forward_surfaces()
        except FileNotFoundError:
            # Fallback: return empty dict if file not found
            _SURFACES_CACHE = {}
    return _SURFACES_CACHE


def get_forward_surface(currency: str) -> ForwardRateSurface | None:
    """
    Get forward rate surface for a currency.

    This function now loads data from the Data Warehouse instead of
    returning hardcoded values.

    Args:
        currency: Currency code (USD, EUR, AUD, CNH)

    Returns:
        ForwardRateSurface or None if not found.
    """
    surfaces = _ensure_surfaces_loaded()
    return surfaces.get(currency.upper())


def update_usd_surface(
    rates: NDArray[np.float64],
    as_of_date: date | None = None,
) -> None:
    """
    Update USD forward surface with new data.

    This function now persists to the Data Warehouse JSON file.

    Args:
        rates: New rate matrix (same shape as existing)
        as_of_date: Date of the new data
    """
    global _SURFACES_CACHE

    # Update in the JSON file
    update_surface("USD", rates, as_of_date)

    # Invalidate cache to reload on next access
    _SURFACES_CACHE = None


def reload_surfaces() -> dict[str, ForwardRateSurface]:
    """
    Force reload of all surfaces from Data Warehouse.

    Use this after external updates to the JSON file.

    Returns:
        Dictionary of all loaded surfaces.
    """
    global _SURFACES_CACHE
    _SURFACES_CACHE = None
    return _ensure_surfaces_loaded()


# ============================================================================
# Dynamic Registry Property (for backward compatibility)
# ============================================================================
class _ForwardSurfacesProxy(dict):
    """Proxy dict that loads surfaces on first access."""

    def __init__(self):
        super().__init__()
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            surfaces = _ensure_surfaces_loaded()
            self.update(surfaces)
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()


# Backward-compatible registry
FORWARD_SURFACES: dict[str, ForwardRateSurface] = _ForwardSurfacesProxy()


# ============================================================================
# Convenience Aliases (for old code that imported individual surfaces)
# ============================================================================
@property
def USD_FORWARD_SURFACE() -> ForwardRateSurface | None:
    return get_forward_surface("USD")


@property
def EUR_FORWARD_SURFACE() -> ForwardRateSurface | None:
    return get_forward_surface("EUR")


@property
def AUD_FORWARD_SURFACE() -> ForwardRateSurface | None:
    return get_forward_surface("AUD")


@property
def CNH_FORWARD_SURFACE() -> ForwardRateSurface | None:
    return get_forward_surface("CNH")


# ============================================================================
# New Interpolation Interface (Enhanced Functionality)
# ============================================================================
def get_interpolated_rate(
    currency: str,
    tenor_years: float,
    forward_years: float = 0.0,
    method: Literal["linear", "cubic"] = "linear",
) -> float | None:
    """
    Get interpolated rate for any tenor/forward combination.

    This is the new recommended API for accessing rates, as it supports
    non-standard tenors through interpolation.

    Args:
        currency: Currency code (USD, EUR, AUD, CNH)
        tenor_years: Target tenor in years (e.g., 2.5 for 2.5Y)
        forward_years: Forward start in years (e.g., 0.5 for 6M forward)
        method: Interpolation method - "linear" or "cubic"

    Returns:
        Interpolated rate in percentage points, or None if currency not found.

    Example:
        >>> rate = get_interpolated_rate("USD", 2.5, 0.5, method="cubic")
        >>> print(f"2.5Y rate at 6M forward: {rate:.2f}%")
    """
    surface = get_forward_surface(currency)
    if surface is None:
        return None

    try:
        return interpolate_rate(surface, tenor_years, forward_years, method)
    except ValueError:
        return None


def get_new_money_yields(
    currency: str,
    tenors: list[float] | None = None,
    forward_months: int = 0,
    method: Literal["linear", "cubic"] = "linear",
) -> dict[float, float]:
    """
    Get yield estimates for new money deployment.

    Args:
        currency: Currency code
        tenors: Target tenors in years. Defaults to [2, 3, 5, 7, 10].
        forward_months: Forward start in months (0 for spot).
        method: Interpolation method.

    Returns:
        Dictionary mapping tenor (years) to yield (percentage).
    """
    if tenors is None:
        tenors = [2.0, 3.0, 5.0, 7.0, 10.0]

    surface = get_forward_surface(currency)
    if surface is None:
        return {}

    return build_new_money_yield_curve(surface, tenors, forward_months, method)


# ============================================================================
# Module Test
# ============================================================================
if __name__ == "__main__":
    print("=== Forward Rate Data Module (Refactored) ===\n")

    print("Data Source: 01_Data_Warehouse/db/yield_curves_snapshot.json\n")

    # Test backward-compatible interface
    print("Testing backward-compatible interface:")
    print(f"  Available currencies: {list(FORWARD_SURFACES.keys())}")

    usd = get_forward_surface("USD")
    if usd:
        print(f"\nUSD Surface as of {usd.as_of_date}")
        print(f"Shape: {usd.rates.shape}")
        print(f"\nSpot curve (via get_spot_curve):")
        tenors, spots = usd.get_spot_curve()
        for t, s in list(zip(tenors, spots))[:5]:
            print(f"  {t:.2f}Y: {s*100:.2f}%")

        print(f"\nSample forward rates (via get_rate):")
        print(f"  3M starting in 1Y: {usd.get_rate('3M', '1Y'):.2f}%")
        print(f"  2Y starting in 2Y: {usd.get_rate('2Y', '2Y'):.2f}%")

    # Test new interpolation interface
    print("\n--- New Interpolation Interface ---")
    print("Interpolated rates (non-standard tenors):")
    test_cases = [
        ("USD", 2.5, 0.0),
        ("USD", 4.0, 0.5),
        ("EUR", 3.5, 1.0),
        ("AUD", 8.0, 0.25),
    ]

    for ccy, tenor, fwd in test_cases:
        rate = get_interpolated_rate(ccy, tenor, fwd, method="cubic")
        if rate:
            fwd_label = f"@ {fwd*12:.0f}M fwd" if fwd > 0 else "(spot)"
            print(f"  {ccy} {tenor}Y {fwd_label}: {rate:.2f}%")

    # Test new money yields
    print("\nNew Money Yields (USD):")
    yields = get_new_money_yields("USD", [2, 3, 5, 7, 10], forward_months=0)
    for tenor, rate in yields.items():
        print(f"  {tenor:.0f}Y: {rate:.2f}%")

    # Metadata
    print(f"\nData Warehouse Metadata: {get_metadata()}")
