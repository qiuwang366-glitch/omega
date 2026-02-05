"""
Credit Bond Risk - Core Module

Core data models, configuration, and enums for the credit risk monitoring system.
"""

from .enums import (
    Sector,
    SubSector,
    Region,
    CreditRating,
    RatingOutlook,
    Severity,
    AlertCategory,
    AlertStatus,
    Sentiment,
    SignalCategory,
)
from .models import (
    Obligor,
    BondPosition,
    CreditExposure,
    NewsItem,
    NewsAnalysisResult,
    SignalResult,
    RiskAlert,
    ObligorFinancials,
)
from .config import (
    CreditRiskConfig,
    ConcentrationConfig,
    SpreadConfig,
    NewsConfig,
    LLMConfig,
    get_default_config,
)

__all__ = [
    # Enums
    "Sector",
    "SubSector",
    "Region",
    "CreditRating",
    "RatingOutlook",
    "Severity",
    "AlertCategory",
    "AlertStatus",
    "Sentiment",
    "SignalCategory",
    # Models
    "Obligor",
    "BondPosition",
    "CreditExposure",
    "NewsItem",
    "NewsAnalysisResult",
    "SignalResult",
    "RiskAlert",
    "ObligorFinancials",
    # Config
    "CreditRiskConfig",
    "ConcentrationConfig",
    "SpreadConfig",
    "NewsConfig",
    "LLMConfig",
    "get_default_config",
]
