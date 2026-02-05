"""
Credit Bond Risk - Signal System

JPM Athena-style signal library for credit risk monitoring.
All risk indicators are abstracted as composable Signal objects.
"""

from .base import Signal, SignalContext, SignalRegistry
from .concentration import ConcentrationSignal, HHISignal, SectorConcentrationSignal
from .spread import SpreadPercentileSignal, SpreadZScoreSignal, SpreadChangeSignal
from .rating import RatingChangeSignal, OutlookChangeSignal
from .news import NewsSentimentSignal, NewsVolumeSignal
from .composite import CompositeSignal, WeightedCompositeSignal

__all__ = [
    # Base
    "Signal",
    "SignalContext",
    "SignalRegistry",
    # Concentration
    "ConcentrationSignal",
    "HHISignal",
    "SectorConcentrationSignal",
    # Spread
    "SpreadPercentileSignal",
    "SpreadZScoreSignal",
    "SpreadChangeSignal",
    # Rating
    "RatingChangeSignal",
    "OutlookChangeSignal",
    # News
    "NewsSentimentSignal",
    "NewsVolumeSignal",
    # Composite
    "CompositeSignal",
    "WeightedCompositeSignal",
]
