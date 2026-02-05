"""
Credit Bond Risk Intelligence Platform

信用债风险预警系统 - 结合传统信用分析与LLM智能分析能力

Modules:
    core: 配置、数据模型、枚举
    signals: Athena风格信号系统
    intelligence: LLM增强分析 (新闻摘要、RAG、向量搜索)
    ui: Streamlit Dashboard

Quick Start:
    # 启动Dashboard
    cd 03_Strategy_Lab/credit_bond_risk
    streamlit run ui/dashboard.py

Example Usage:
    from credit_bond_risk.core import (
        CreditRiskConfig,
        Obligor,
        CreditExposure,
        get_default_config,
    )
    from credit_bond_risk.signals import (
        SignalContext,
        SignalRegistry,
        ConcentrationSignal,
        SpreadPercentileSignal,
    )
    from credit_bond_risk.intelligence import (
        NewsAnalyzer,
        CreditRAGEngine,
    )

    # Initialize config
    config = get_default_config()

    # Create signals from config
    signals = SignalRegistry.create_from_config(config)

    # Build context with your data
    context = SignalContext(
        exposures=your_exposures,
        total_aum=50e9,
    )

    # Compute all signals
    triggered_alerts = SignalRegistry.compute_all(signals, context)

Author: Project Omega
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Project Omega"

# Core exports
from .core import (
    # Enums
    Sector,
    SubSector,
    CreditRating,
    RatingOutlook,
    Severity,
    AlertCategory,
    AlertStatus,
    Sentiment,
    # Models
    Obligor,
    BondPosition,
    CreditExposure,
    NewsItem,
    NewsAnalysisResult,
    SignalResult,
    RiskAlert,
    ObligorFinancials,
    # Config
    CreditRiskConfig,
    get_default_config,
)

# Signal exports
from .signals import (
    Signal,
    SignalContext,
    SignalRegistry,
    ConcentrationSignal,
    HHISignal,
    SpreadPercentileSignal,
    SpreadZScoreSignal,
    NewsSentimentSignal,
    CompositeSignal,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Enums
    "Sector",
    "SubSector",
    "CreditRating",
    "RatingOutlook",
    "Severity",
    "AlertCategory",
    "AlertStatus",
    "Sentiment",
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
    "get_default_config",
    # Signals
    "Signal",
    "SignalContext",
    "SignalRegistry",
    "ConcentrationSignal",
    "HHISignal",
    "SpreadPercentileSignal",
    "SpreadZScoreSignal",
    "NewsSentimentSignal",
    "CompositeSignal",
]
