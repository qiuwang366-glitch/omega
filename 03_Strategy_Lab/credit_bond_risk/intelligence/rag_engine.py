"""
Credit Bond Risk - RAG Engine

Retrieval-Augmented Generation for credit knowledge Q&A:
- Vector store management
- Semantic search
- Context-aware answer generation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.models import NewsItem, Obligor, RAGResponse
from ..core.config import LLMConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    """RAG引擎配置"""

    # 检索参数
    top_k: int = Field(10, description="检索文档数量")
    similarity_threshold: float = Field(0.6, description="相似度阈值")

    # 上下文参数
    max_context_length: int = Field(4000, description="最大上下文长度")
    include_metadata: bool = Field(True, description="是否包含元数据")

    # 生成参数
    answer_max_tokens: int = Field(1000, description="答案最大token")
    include_sources: bool = Field(True, description="是否引用来源")


@dataclass
class Document:
    """向量存储中的文档"""

    doc_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # 来源信息
    source_type: str = "news"  # news, filing, research
    source_id: str | None = None  # news_id, filing_id, etc.
    obligor_id: str | None = None
    timestamp: datetime | None = None


class VectorStore:
    """
    向量存储抽象层

    支持多种后端:
    - DuckDB + VSS扩展
    - ChromaDB
    - 内存存储 (测试用)
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._store: dict[str, Document] = {}
        self._embeddings: list[tuple[str, list[float]]] = []

    def add_document(self, doc: Document) -> None:
        """添加文档"""
        self._store[doc.doc_id] = doc
        if doc.embedding:
            self._embeddings.append((doc.doc_id, doc.embedding))

    def add_documents(self, docs: list[Document]) -> None:
        """批量添加文档"""
        for doc in docs:
            self.add_document(doc)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_obligor: str | None = None,
        filter_source_type: str | None = None,
    ) -> list[tuple[Document, float]]:
        """
        向量相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter_obligor: 过滤发行人ID
            filter_source_type: 过滤来源类型

        Returns:
            [(Document, similarity_score), ...]
        """
        import numpy as np

        if not self._embeddings:
            return []

        query = np.array(query_embedding)
        results = []

        for doc_id, embedding in self._embeddings:
            doc = self._store.get(doc_id)
            if doc is None:
                continue

            # 应用过滤条件
            if filter_obligor and doc.obligor_id != filter_obligor:
                continue
            if filter_source_type and doc.source_type != filter_source_type:
                continue

            # 计算余弦相似度
            emb = np.array(embedding)
            similarity = float(np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb)))
            results.append((doc, similarity))

        # 排序并返回Top-K
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_document(self, doc_id: str) -> Document | None:
        """获取单个文档"""
        return self._store.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id in self._store:
            del self._store[doc_id]
            self._embeddings = [(d, e) for d, e in self._embeddings if d != doc_id]
            return True
        return False

    def count(self) -> int:
        """文档数量"""
        return len(self._store)


class CreditRAGEngine:
    """
    信用知识库RAG引擎

    功能:
    - 自然语言问答
    - 发行人相关信息检索
    - 上下文感知的答案生成

    示例问题:
    - "云南城投最近有什么风险事件？"
    - "哪些LGFV的债务率超过300%？"
    - "比较一下重庆和成都城投的信用资质"
    """

    SYSTEM_PROMPT = """你是专业的信用债分析师，基于提供的资料回答问题。

回答原则：
1. 只基于提供的资料回答，不要编造信息
2. 如果资料不足以回答，明确说明
3. 引用具体的新闻或数据来源
4. 保持专业、简洁的风格
5. 涉及风险判断时要谨慎，给出依据"""

    QA_PROMPT = """基于以下资料回答问题。

## 相关资料
{context}

## 问题
{question}

## 回答要求
- 直接回答问题，不要重复问题
- 如有多条相关信息，综合分析
- 标注信息来源 [来源X]
- 如果信息不足，说明还需要哪些数据"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_config: LLMConfig | None = None,
        rag_config: RAGConfig | None = None,
    ):
        self.vector_store = vector_store
        self.llm_config = llm_config or LLMConfig()
        self.rag_config = rag_config or RAGConfig()
        self._llm_client = None
        self._embedding_service = None

    @property
    def llm_client(self):
        """Lazy load LLM client"""
        if self._llm_client is None:
            try:
                from anthropic import Anthropic
                self._llm_client = Anthropic()
            except ImportError:
                logger.warning("anthropic not installed, using mock")
                from .news_analyzer import MockLLMClient
                self._llm_client = MockLLMClient()
        return self._llm_client

    @property
    def embedding_service(self):
        """Lazy load embedding service"""
        if self._embedding_service is None:
            from .embeddings import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def query(
        self,
        question: str,
        obligor_id: str | None = None,
        source_type: str | None = None,
    ) -> RAGResponse:
        """
        RAG问答

        Args:
            question: 用户问题
            obligor_id: 可选，限定发行人
            source_type: 可选，限定来源类型

        Returns:
            RAGResponse with answer and sources
        """
        # 1. 生成问题的embedding
        query_embedding = self.embedding_service.embed_text(question)

        # 2. 向量检索相关文档
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.rag_config.top_k,
            filter_obligor=obligor_id,
            filter_source_type=source_type,
        )

        # 过滤低相似度结果
        filtered_results = [
            (doc, score)
            for doc, score in search_results
            if score >= self.rag_config.similarity_threshold
        ]

        if not filtered_results:
            return RAGResponse(
                question=question,
                answer="抱歉，未找到相关资料。请尝试更具体的问题或检查发行人名称。",
                sources=[],
                confidence=0.0,
            )

        # 3. 构建上下文
        context = self._build_context(filtered_results)

        # 4. 生成答案
        answer = self._generate_answer(question, context)

        # 5. 提取涉及的发行人
        obligor_ids = list(set(
            doc.obligor_id
            for doc, _ in filtered_results
            if doc.obligor_id
        ))

        return RAGResponse(
            question=question,
            answer=answer,
            sources=[
                {
                    "doc_id": doc.doc_id,
                    "content_preview": doc.content[:200],
                    "source_type": doc.source_type,
                    "similarity": score,
                    "obligor_id": doc.obligor_id,
                    "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
                }
                for doc, score in filtered_results[:5]  # 只返回前5个来源
            ],
            confidence=self._estimate_confidence(filtered_results),
            obligor_context=obligor_ids,
        )

    def _build_context(
        self,
        search_results: list[tuple[Document, float]],
    ) -> str:
        """构建LLM上下文"""
        context_parts = []
        total_length = 0

        for i, (doc, score) in enumerate(search_results, 1):
            # 格式化文档
            part = f"[来源{i}] "
            if doc.timestamp:
                part += f"({doc.timestamp.strftime('%Y-%m-%d')}) "
            if doc.metadata.get("title"):
                part += f"{doc.metadata['title']}\n"
            part += doc.content

            # 检查长度限制
            if total_length + len(part) > self.rag_config.max_context_length:
                # 截断
                remaining = self.rag_config.max_context_length - total_length
                if remaining > 100:
                    part = part[:remaining] + "..."
                else:
                    break

            context_parts.append(part)
            total_length += len(part)

        return "\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """调用LLM生成答案"""
        prompt = self.QA_PROMPT.format(
            context=context,
            question=question,
        )

        try:
            response = self.llm_client.messages.create(
                model=self.llm_config.model_primary,
                max_tokens=self.rag_config.answer_max_tokens,
                temperature=self.llm_config.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"答案生成失败: {e}"

    def _estimate_confidence(
        self,
        search_results: list[tuple[Document, float]],
    ) -> float:
        """估计答案置信度"""
        if not search_results:
            return 0.0

        # 基于相似度分数估计置信度
        avg_similarity = sum(score for _, score in search_results) / len(search_results)
        top_similarity = search_results[0][1]

        # 综合考虑平均相似度和最高相似度
        confidence = 0.6 * top_similarity + 0.4 * avg_similarity

        return min(1.0, confidence)

    def index_news(self, news_items: list[NewsItem]) -> int:
        """
        将新闻索引到向量存储

        Args:
            news_items: 新闻列表

        Returns:
            成功索引的文档数
        """
        count = 0
        for news in news_items:
            # 构建文档内容
            content = f"{news.title}\n\n{news.content}"
            if news.summary:
                content = f"摘要: {news.summary}\n\n{content}"

            # 生成embedding
            embedding = self.embedding_service.embed_text(content)

            # 创建文档
            doc = Document(
                doc_id=f"news_{news.news_id}",
                content=content,
                embedding=embedding,
                metadata={
                    "title": news.title,
                    "source": news.source,
                    "sentiment": news.sentiment.value if news.sentiment else None,
                    "sentiment_score": news.sentiment_score,
                },
                source_type="news",
                source_id=news.news_id,
                obligor_id=news.obligor_ids[0] if news.obligor_ids else None,
                timestamp=news.timestamp,
            )

            self.vector_store.add_document(doc)
            count += 1

        logger.info(f"Indexed {count} news documents")
        return count

    def index_obligor_profile(self, obligor: Obligor) -> None:
        """
        将发行人简介索引到向量存储
        """
        content = f"""发行人: {obligor.name_cn}
行业: {obligor.sector.value} / {obligor.sub_sector}
地区: {obligor.province or 'N/A'}, {obligor.city or 'N/A'}
评级: {obligor.rating_internal.value} (展望: {obligor.rating_outlook.value})
外部评级: {obligor.rating_external}
"""

        if obligor.risk_narrative:
            content += f"\n风险描述:\n{obligor.risk_narrative}"

        embedding = self.embedding_service.embed_text(content)

        doc = Document(
            doc_id=f"obligor_{obligor.obligor_id}",
            content=content,
            embedding=embedding,
            metadata={
                "name": obligor.name_cn,
                "sector": obligor.sector.value,
                "rating": obligor.rating_internal.value,
            },
            source_type="obligor_profile",
            source_id=obligor.obligor_id,
            obligor_id=obligor.obligor_id,
            timestamp=obligor.updated_at,
        )

        self.vector_store.add_document(doc)
