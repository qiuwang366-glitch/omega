"""
Credit Bond Risk - Intelligence Module

LLM-powered analysis capabilities:
- News analysis and summarization
- RAG-based Q&A
- Embedding and similarity search
- Anomaly detection
"""

from .news_analyzer import NewsAnalyzer, BatchNewsProcessor
from .rag_engine import CreditRAGEngine, RAGConfig
from .embeddings import EmbeddingService, ObligorEmbedding
from .anomaly_detector import AnomalyDetector, SpreadAnomalyDetector

__all__ = [
    # News
    "NewsAnalyzer",
    "BatchNewsProcessor",
    # RAG
    "CreditRAGEngine",
    "RAGConfig",
    # Embeddings
    "EmbeddingService",
    "ObligorEmbedding",
    # Anomaly
    "AnomalyDetector",
    "SpreadAnomalyDetector",
]
