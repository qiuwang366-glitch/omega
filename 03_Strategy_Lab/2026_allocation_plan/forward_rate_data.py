"""
forward_rate_data.py - Real Forward Rate Surface Data
=====================================================
Stores actual market forward rate matrices by currency.
Data source: Bloomberg / Internal treasury system.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ForwardRateSurface:
    """
    Container for a forward rate matrix.

    Axes:
    - Rows (tenors): Underlying bond maturity (e.g., 1M, 2Y, 10Y)
    - Columns (forward_starts): When the forward contract starts (e.g., Spot, 3M, 1Y)

    Values are in percentage points (e.g., 4.50 = 4.50%).
    """
    currency: str
    as_of_date: date
    tenor_labels: tuple[str, ...]      # Row labels: "1M", "2M", "1Y", etc.
    forward_labels: tuple[str, ...]    # Column labels: "Spot", "3M", "1Y", etc.
    rates: NDArray[np.float64]         # Shape: (n_tenors, n_forwards)

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

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for display."""
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
        # Handle Chinese "X个月" or "XM"
        num_str = label.replace("个月", "").replace("M", "").replace("m", "").strip()
        return float(num_str) / 12
    if label.endswith("年") or label.endswith("Y") or label.endswith("y"):
        num_str = label.replace("年", "").replace("Y", "").replace("y", "").strip()
        return float(num_str)
    # Try direct parse as months
    try:
        return float(label) / 12
    except ValueError:
        return 0.0


# ============================================================================
# USD Forward Rate Surface (as of latest update)
# ============================================================================
USD_FORWARD_SURFACE = ForwardRateSurface(
    currency="USD",
    as_of_date=date(2025, 1, 31),  # Update this when data refreshes
    tenor_labels=(
        "1M", "2M", "3M", "4M", "6M",
        "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y",
    ),
    forward_labels=(
        "Spot", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "10Y", "15Y", "30Y",
    ),
    rates=np.array([
        # Spot,   3M,     6M,     1Y,     2Y,     3Y,     4Y,     5Y,    10Y,    15Y,    30Y
        [3.6803, 3.6456, 3.3481, 3.5998, 3.7570, 4.0746, 4.0746, 4.5812, 5.6922, 5.6605, 5.0642],  # 1M
        [3.6933, 3.5566, 3.3245, 3.6256, 3.7837, 4.1043, 4.1043, 4.6156, 5.7374, 5.7056, 5.1030],  # 2M
        [3.6723, 3.5388, 3.3439, 3.6107, 3.7688, 4.0891, 4.0891, 4.5995, 5.7199, 5.6884, 5.0857],  # 3M
        [3.6793, 3.5405, 3.3286, 3.6363, 3.7960, 4.1184, 4.1184, 4.6334, 5.7648, 5.7328, 5.1254],  # 4M
        [3.6350, 3.4846, 3.3782, 3.6273, 3.7868, 4.1100, 4.1100, 4.6259, 5.7607, 5.7287, 5.1184],  # 6M
        [3.4848, 3.4845, 3.5016, 3.6585, 3.8268, 4.1440, 4.1496, 4.6641, 5.7924, 5.7761, 5.1458],  # 1Y
        [3.5737, 3.5920, 3.6177, 3.7413, 3.9816, 4.1474, 4.4011, 4.6717, 5.7845, 5.7761, 5.1393],  # 2Y
        [3.6476, 3.6909, 3.7359, 3.8698, 4.0348, 4.3122, 4.4900, 4.7631, 5.7819, 5.7761, 5.1371],  # 3Y
        [3.8376, 3.8863, 3.9377, 4.0690, 4.2731, 4.5006, 4.6647, 4.8683, 5.7826, 5.7537, 5.1378],  # 5Y
        [4.0530, 4.0964, 4.1435, 4.2595, 4.4475, 4.6387, 4.8635, 5.0967, 5.7810, 5.5998, 5.1363],  # 7Y
        [4.2765, 4.3475, 4.3987, 4.5175, 4.7250, 4.9229, 5.0888, 5.2647, 5.7702, 5.4854, 5.1365],  # 10Y
        [4.8541, 4.8739, 4.9003, 4.9634, 5.0720, 5.1751, 5.2577, 5.3454, 5.5362, 5.3549, 5.1359],  # 20Y
        [4.9050, 4.9211, 4.9428, 4.9943, 5.0835, 5.1679, 5.2358, 5.3080, 5.4636, 5.3145, 5.1360],  # 30Y
    ], dtype=np.float64),
)


# ============================================================================
# Placeholder surfaces for other currencies (update with real data)
# ============================================================================
EUR_FORWARD_SURFACE = ForwardRateSurface(
    currency="EUR",
    as_of_date=date(2025, 1, 31),
    tenor_labels=("1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
    forward_labels=("Spot", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"),
    rates=np.array([
        # Spot,   3M,     6M,     1Y,     2Y,     3Y,     5Y,    10Y
        [2.85,  2.75,   2.65,   2.55,   2.50,   2.55,   2.70,   3.00],  # 1M
        [2.80,  2.70,   2.60,   2.52,   2.48,   2.53,   2.68,   2.98],  # 3M
        [2.75,  2.65,   2.58,   2.50,   2.47,   2.52,   2.67,   2.97],  # 6M
        [2.65,  2.58,   2.52,   2.48,   2.48,   2.55,   2.70,   3.00],  # 1Y
        [2.55,  2.52,   2.50,   2.52,   2.58,   2.68,   2.85,   3.15],  # 2Y
        [2.58,  2.55,   2.55,   2.60,   2.70,   2.82,   3.00,   3.28],  # 3Y
        [2.72,  2.70,   2.72,   2.80,   2.95,   3.10,   3.25,   3.45],  # 5Y
        [2.90,  2.90,   2.95,   3.05,   3.18,   3.30,   3.42,   3.55],  # 7Y
        [3.10,  3.12,   3.18,   3.28,   3.40,   3.50,   3.58,   3.65],  # 10Y
        [3.35,  3.38,   3.42,   3.50,   3.58,   3.65,   3.70,   3.72],  # 20Y
        [3.25,  3.28,   3.32,   3.40,   3.48,   3.55,   3.60,   3.62],  # 30Y
    ], dtype=np.float64),
)


AUD_FORWARD_SURFACE = ForwardRateSurface(
    currency="AUD",
    as_of_date=date(2025, 1, 31),
    tenor_labels=("1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
    forward_labels=("Spot", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"),
    rates=np.array([
        # Spot,   3M,     6M,     1Y,     2Y,     3Y,     5Y,    10Y
        [4.10,  4.05,   3.95,   3.85,   3.90,   4.10,   4.40,   5.00],  # 1M
        [4.05,  4.00,   3.90,   3.82,   3.88,   4.08,   4.38,   4.98],  # 3M
        [4.00,  3.95,   3.88,   3.80,   3.88,   4.10,   4.42,   5.02],  # 6M
        [3.90,  3.88,   3.85,   3.82,   3.95,   4.18,   4.52,   5.12],  # 1Y
        [3.85,  3.85,   3.88,   3.95,   4.15,   4.40,   4.75,   5.32],  # 2Y
        [3.90,  3.92,   3.98,   4.10,   4.35,   4.60,   4.95,   5.48],  # 3Y
        [4.10,  4.15,   4.22,   4.38,   4.65,   4.90,   5.20,   5.62],  # 5Y
        [4.32,  4.38,   4.48,   4.65,   4.90,   5.12,   5.38,   5.72],  # 7Y
        [4.55,  4.62,   4.72,   4.88,   5.10,   5.28,   5.50,   5.78],  # 10Y
        [4.85,  4.90,   4.98,   5.10,   5.25,   5.38,   5.55,   5.75],  # 20Y
        [4.80,  4.85,   4.92,   5.02,   5.18,   5.30,   5.48,   5.68],  # 30Y
    ], dtype=np.float64),
)


CNH_FORWARD_SURFACE = ForwardRateSurface(
    currency="CNH",
    as_of_date=date(2025, 1, 31),
    tenor_labels=("1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
    forward_labels=("Spot", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"),
    rates=np.array([
        # Spot,   3M,     6M,     1Y,     2Y,     3Y,     5Y,    10Y
        [1.85,  1.88,   1.92,   2.00,   2.15,   2.30,   2.55,   3.00],  # 1M
        [1.82,  1.85,   1.90,   1.98,   2.13,   2.28,   2.53,   2.98],  # 3M
        [1.80,  1.85,   1.92,   2.02,   2.18,   2.35,   2.60,   3.05],  # 6M
        [1.85,  1.92,   2.02,   2.15,   2.35,   2.55,   2.82,   3.25],  # 1Y
        [2.00,  2.10,   2.22,   2.40,   2.65,   2.88,   3.15,   3.55],  # 2Y
        [2.15,  2.25,   2.38,   2.58,   2.85,   3.08,   3.35,   3.72],  # 3Y
        [2.40,  2.50,   2.65,   2.85,   3.12,   3.35,   3.58,   3.88],  # 5Y
        [2.60,  2.70,   2.85,   3.05,   3.30,   3.52,   3.72,   3.98],  # 7Y
        [2.80,  2.90,   3.05,   3.25,   3.48,   3.68,   3.85,   4.05],  # 10Y
        [3.10,  3.18,   3.30,   3.48,   3.68,   3.85,   3.98,   4.12],  # 20Y
        [3.15,  3.22,   3.35,   3.52,   3.72,   3.88,   4.00,   4.12],  # 30Y
    ], dtype=np.float64),
)


# ============================================================================
# Registry for easy access
# ============================================================================
FORWARD_SURFACES: dict[str, ForwardRateSurface] = {
    "USD": USD_FORWARD_SURFACE,
    "EUR": EUR_FORWARD_SURFACE,
    "AUD": AUD_FORWARD_SURFACE,
    "CNH": CNH_FORWARD_SURFACE,
}


def get_forward_surface(currency: str) -> ForwardRateSurface | None:
    """Get forward rate surface for a currency."""
    return FORWARD_SURFACES.get(currency.upper())


def update_usd_surface(
    rates: NDArray[np.float64],
    as_of_date: date | None = None,
) -> None:
    """
    Update USD forward surface with new data.

    Args:
        rates: New rate matrix (same shape as existing)
        as_of_date: Date of the new data
    """
    global USD_FORWARD_SURFACE, FORWARD_SURFACES

    USD_FORWARD_SURFACE = ForwardRateSurface(
        currency="USD",
        as_of_date=as_of_date or date.today(),
        tenor_labels=USD_FORWARD_SURFACE.tenor_labels,
        forward_labels=USD_FORWARD_SURFACE.forward_labels,
        rates=rates,
    )
    FORWARD_SURFACES["USD"] = USD_FORWARD_SURFACE


# ============================================================================
# Module Test
# ============================================================================
if __name__ == "__main__":
    print("=== Forward Rate Surface Data ===\n")

    usd = get_forward_surface("USD")
    if usd:
        print(f"USD Surface as of {usd.as_of_date}")
        print(f"Shape: {usd.rates.shape}")
        print(f"\nSpot curve:")
        tenors, spots = usd.get_spot_curve()
        for t, s in zip(tenors, spots):
            print(f"  {t:.2f}Y: {s*100:.2f}%")

        print(f"\nSample forward rates:")
        print(f"  3M starting in 1Y: {usd.get_rate('3M', '1Y'):.2f}%")
        print(f"  2Y starting in 2Y: {usd.get_rate('2Y', '2Y'):.2f}%")
        print(f"  10Y starting in 5Y: {usd.get_rate('10Y', '5Y'):.2f}%")

        print(f"\nFull DataFrame:")
        print(usd.to_dataframe())
