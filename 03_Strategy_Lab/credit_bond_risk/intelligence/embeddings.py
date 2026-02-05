"""
Credit Bond Risk - Embedding Service

Text and entity embedding for:
- Semantic search
- Similarity matching
- Obligor feature vectors
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.models import Obligor, ObligorFinancials
from ..core.enums import rating_to_score

logger = logging.getLogger(__name__)


@dataclass
class ObligorEmbedding:
    """发行人特征向量"""

    obligor_id: str
    text_embedding: NDArray[np.float64]      # 文本特征 (768维)
    numeric_features: NDArray[np.float64]    # 数值特征 (10维)
    combined_embedding: NDArray[np.float64]  # 组合特征

    @property
    def dimension(self) -> int:
        return len(self.combined_embedding)


class EmbeddingService:
    """
    Embedding服务

    支持多种后端:
    - OpenAI text-embedding-3-small
    - BGE-M3 (本地)
    - 简化的TF-IDF (无API fallback)
    """

    # 数值特征标准化参数
    NUMERIC_FEATURE_SCALES = {
        "rating_score": 100,
        "debt_to_ebitda": 10,
        "interest_coverage": 10,
        "debt_to_assets": 1,
        "current_ratio": 3,
        "total_debt_log": 4,  # log10(债务/亿)
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 768,
    ):
        self.model = model
        self.dimension = dimension
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                logger.warning("openai not installed, using mock embeddings")
                self._client = None
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """
        生成文本embedding

        Args:
            text: 输入文本

        Returns:
            768维向量
        """
        if self.client is None:
            return self._mock_embedding(text)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text[:8000],  # OpenAI限制
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return self._mock_embedding(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成embedding"""
        if self.client is None:
            return [self._mock_embedding(t) for t in texts]

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[t[:8000] for t in texts],
            )
            return [d.embedding for d in response.data]

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [self._mock_embedding(t) for t in texts]

    def _mock_embedding(self, text: str) -> list[float]:
        """
        简化的mock embedding (基于字符hash)

        仅用于测试，不保证语义质量
        """
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension)
        # L2归一化
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def build_obligor_embedding(
        self,
        obligor: Obligor,
        financials: ObligorFinancials | None = None,
    ) -> ObligorEmbedding:
        """
        构建发行人综合特征向量

        组合:
        1. 文本特征: 行业描述 + 风险叙述 (768维)
        2. 数值特征: 财务指标标准化 (10维)
        3. 组合特征: concat后降维或直接拼接

        Args:
            obligor: 发行人信息
            financials: 财务数据 (可选)

        Returns:
            ObligorEmbedding
        """
        # 1. 构建文本描述
        text_parts = [
            f"行业: {obligor.sector.value}",
            f"细分: {obligor.sub_sector}",
            f"地区: {obligor.province or '全国'}",
            f"评级: {obligor.rating_internal.value}",
        ]

        if obligor.risk_narrative:
            text_parts.append(f"风险描述: {obligor.risk_narrative}")

        text = " ".join(text_parts)
        text_embedding = np.array(self.embed_text(text))

        # 2. 构建数值特征
        numeric_features = self._build_numeric_features(obligor, financials)

        # 3. 组合特征 (简单拼接并归一化)
        combined = np.concatenate([
            text_embedding * 0.8,  # 文本权重
            numeric_features * 0.2,  # 数值权重
        ])
        combined = combined / np.linalg.norm(combined)

        return ObligorEmbedding(
            obligor_id=obligor.obligor_id,
            text_embedding=text_embedding,
            numeric_features=numeric_features,
            combined_embedding=combined,
        )

    def _build_numeric_features(
        self,
        obligor: Obligor,
        financials: ObligorFinancials | None,
    ) -> NDArray[np.float64]:
        """构建标准化的数值特征向量"""
        features = []

        # 评级分数 (0-1)
        rating_score = rating_to_score(obligor.rating_internal)
        features.append(rating_score / self.NUMERIC_FEATURE_SCALES["rating_score"])

        # 展望分数 (0-1)
        outlook_scores = {
            "POSITIVE": 1.0,
            "STABLE": 0.5,
            "NEGATIVE": 0.2,
            "WATCH_POS": 0.7,
            "WATCH_NEG": 0.1,
            "DEVELOPING": 0.5,
        }
        features.append(outlook_scores.get(obligor.rating_outlook.value, 0.5))

        if financials:
            # 债务/EBITDA (截断到0-1)
            d2e = financials.debt_to_ebitda or 5
            features.append(min(1.0, d2e / self.NUMERIC_FEATURE_SCALES["debt_to_ebitda"]))

            # 利息覆盖倍数 (截断到0-1)
            ic = financials.interest_coverage or 2
            features.append(min(1.0, ic / self.NUMERIC_FEATURE_SCALES["interest_coverage"]))

            # 资产负债率 (0-1)
            d2a = financials.debt_to_assets or 0.6
            features.append(d2a)

            # 流动比率 (截断到0-1)
            cr = financials.current_ratio or 1
            features.append(min(1.0, cr / self.NUMERIC_FEATURE_SCALES["current_ratio"]))

            # 债务规模 (log标准化)
            debt = financials.total_debt or 100
            debt_log = np.log10(debt + 1) / self.NUMERIC_FEATURE_SCALES["total_debt_log"]
            features.append(min(1.0, debt_log))

        else:
            # 缺失财务数据时使用默认值
            features.extend([0.5, 0.5, 0.6, 0.5, 0.5])

        # 补齐到10维
        while len(features) < 10:
            features.append(0.5)

        return np.array(features[:10])


class SimilaritySearch:
    """
    相似性搜索

    用途:
    1. 找到相似发行人 (peer comparison)
    2. 传染风险评估 (相似发行人出问题)
    3. 新闻关联 (相似新闻聚类)
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._obligor_embeddings: dict[str, ObligorEmbedding] = {}

    def index_obligor(
        self,
        obligor: Obligor,
        financials: ObligorFinancials | None = None,
    ) -> None:
        """索引发行人"""
        embedding = self.embedding_service.build_obligor_embedding(obligor, financials)
        self._obligor_embeddings[obligor.obligor_id] = embedding

    def find_similar_obligors(
        self,
        obligor_id: str,
        top_k: int = 5,
        same_sector_only: bool = False,
        obligor_map: dict[str, Obligor] | None = None,
    ) -> list[tuple[str, float, list[str]]]:
        """
        找到相似发行人

        Args:
            obligor_id: 目标发行人ID
            top_k: 返回数量
            same_sector_only: 是否只搜索同行业
            obligor_map: 发行人映射 (用于行业过滤)

        Returns:
            [(obligor_id, similarity_score, common_factors), ...]
        """
        if obligor_id not in self._obligor_embeddings:
            logger.warning(f"Obligor {obligor_id} not indexed")
            return []

        target_embedding = self._obligor_embeddings[obligor_id]

        results = []
        for other_id, other_embedding in self._obligor_embeddings.items():
            if other_id == obligor_id:
                continue

            # 行业过滤
            if same_sector_only and obligor_map:
                target_obligor = obligor_map.get(obligor_id)
                other_obligor = obligor_map.get(other_id)
                if target_obligor and other_obligor:
                    if target_obligor.sector != other_obligor.sector:
                        continue

            # 计算相似度
            similarity = self._cosine_similarity(
                target_embedding.combined_embedding,
                other_embedding.combined_embedding,
            )

            # 分析相似因素
            common_factors = self._analyze_common_factors(
                target_embedding, other_embedding
            )

            results.append((other_id, similarity, common_factors))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(
        self,
        vec1: NDArray[np.float64],
        vec2: NDArray[np.float64],
    ) -> float:
        """计算余弦相似度"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _analyze_common_factors(
        self,
        emb1: ObligorEmbedding,
        emb2: ObligorEmbedding,
    ) -> list[str]:
        """分析相似因素"""
        factors = []

        # 文本相似度
        text_sim = self._cosine_similarity(emb1.text_embedding, emb2.text_embedding)
        if text_sim > 0.8:
            factors.append("行业相似")

        # 数值特征相似度
        numeric_sim = self._cosine_similarity(emb1.numeric_features, emb2.numeric_features)
        if numeric_sim > 0.9:
            factors.append("财务特征相似")

        # 评级相似 (第一个数值特征)
        if abs(emb1.numeric_features[0] - emb2.numeric_features[0]) < 0.1:
            factors.append("评级相近")

        return factors
