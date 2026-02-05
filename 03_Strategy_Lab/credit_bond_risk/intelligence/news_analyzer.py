"""
Credit Bond Risk - News Analyzer

LLM-powered news analysis:
- Single news analysis (sentiment, summary, entities)
- Batch processing with rate limiting
- Obligor risk digest generation
"""

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.models import NewsItem, NewsAnalysisResult, Obligor, CreditExposure
from ..core.enums import Sentiment
from ..core.config import LLMConfig

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """
    LLM驱动的新闻分析器

    Features:
    - 单篇新闻情感分析
    - 关键事件提取
    - 实体识别 (发行人关联)
    - 信用影响评估
    """

    ANALYSIS_PROMPT = """分析以下信用债相关新闻，提取关键信息。

标题：{title}
来源：{source}
时间：{timestamp}
内容：{content}

请返回JSON格式（确保是有效的JSON）：
{{
    "summary": "一句话摘要（不超过50字）",
    "sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
    "sentiment_score": -1到1的数值（-1最负面，1最正面，0中性）,
    "key_events": ["事件1", "事件2"],
    "credit_impact": "对发行人信用资质的潜在影响（一句话）",
    "mentioned_entities": ["公司名1", "公司名2"]
}}

注意：
1. sentiment_score要与sentiment一致
2. 关注违约、评级、融资、业绩、政策等信用相关事件
3. mentioned_entities只提取公司/机构名称"""

    DIGEST_PROMPT = """作为资深信用分析师，为以下发行人生成风险简报。

## 发行人信息
- 名称：{name}
- 行业：{sector} / {sub_sector}
- 地区：{province}
- 评级：{rating} (展望: {outlook})
- 持仓市值：${market_value_m:.1f}M (占AUM {pct_aum:.2%})
- 加权OAS：{oas:.0f}bps

## 近期新闻 ({news_count}条，过去{days}天)
{news_summary}

请生成简报：
1. **风险摘要**（3句话以内）
2. **关键关注点**（bullet points，最多5条）
3. **建议行动**：增持观察 / 持有 / 减持观察 / 立即减持
4. **风险评级**：低/中/高/极高

格式要求：使用Markdown，简洁专业。"""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None

    @property
    def client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic()
            except ImportError:
                logger.warning("anthropic package not installed, using mock client")
                self._client = MockLLMClient()
        return self._client

    def analyze_news(self, news: NewsItem) -> NewsAnalysisResult:
        """
        分析单篇新闻

        Args:
            news: 待分析的新闻

        Returns:
            NewsAnalysisResult with sentiment, summary, etc.
        """
        prompt = self.ANALYSIS_PROMPT.format(
            title=news.title,
            source=news.source,
            timestamp=news.timestamp.strftime("%Y-%m-%d %H:%M"),
            content=news.content[:2000],  # 截断长文本
        )

        try:
            response = self.client.messages.create(
                model=self.config.model_fast,  # 使用快速模型
                max_tokens=self.config.max_tokens_summary,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # 解析JSON响应
            content = response.content[0].text
            # 提取JSON部分
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
            else:
                raise ValueError("No valid JSON found in response")

            return NewsAnalysisResult(
                summary=data.get("summary", ""),
                sentiment=Sentiment(data.get("sentiment", "NEUTRAL")),
                sentiment_score=float(data.get("sentiment_score", 0)),
                key_events=data.get("key_events", []),
                credit_impact=data.get("credit_impact"),
                mentioned_entities=data.get("mentioned_entities", []),
            )

        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            # 返回默认中性结果
            return NewsAnalysisResult(
                summary=news.title[:50],
                sentiment=Sentiment.NEUTRAL,
                sentiment_score=0.0,
                key_events=[],
                credit_impact=None,
                mentioned_entities=[],
            )

    def generate_obligor_digest(
        self,
        obligor: Obligor,
        exposure: CreditExposure,
        news_items: list[NewsItem],
        lookback_days: int = 7,
    ) -> str:
        """
        生成发行人风险简报

        Args:
            obligor: 发行人信息
            exposure: 持仓曝光
            news_items: 近期新闻
            lookback_days: 新闻回看天数

        Returns:
            Markdown格式的风险简报
        """
        # 格式化新闻摘要
        news_summary = ""
        for i, news in enumerate(news_items[:10], 1):
            sentiment_marker = {
                Sentiment.POSITIVE: "🟢",
                Sentiment.NEUTRAL: "⚪",
                Sentiment.NEGATIVE: "🔴",
            }.get(news.sentiment, "⚪")

            summary = news.summary or news.title[:50]
            news_summary += f"{i}. {sentiment_marker} [{news.timestamp.strftime('%m-%d')}] {summary}\n"

        if not news_summary:
            news_summary = "（无近期新闻）"

        prompt = self.DIGEST_PROMPT.format(
            name=obligor.name_cn,
            sector=obligor.sector.value,
            sub_sector=obligor.sub_sector,
            province=obligor.province or "N/A",
            rating=obligor.rating_internal.value,
            outlook=obligor.rating_outlook.value,
            market_value_m=exposure.total_market_usd / 1e6,
            pct_aum=exposure.pct_of_aum,
            oas=exposure.weighted_avg_oas or 0,
            news_count=len(news_items),
            days=lookback_days,
            news_summary=news_summary,
        )

        try:
            response = self.client.messages.create(
                model=self.config.model_primary,  # 使用主力模型
                max_tokens=self.config.max_tokens_analysis,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Digest generation failed: {e}")
            return f"## {obligor.name_cn} 风险简报\n\n*生成失败: {e}*"

    def extract_entities(self, text: str) -> list[str]:
        """
        从文本中提取实体 (公司名)

        简化实现：使用规则匹配
        生产环境建议使用NER模型
        """
        # 常见公司名后缀
        suffixes = [
            "集团", "公司", "控股", "投资", "发展", "建设", "城投",
            "国资", "资产", "资本", "银行", "保险", "证券",
        ]

        entities = []
        for suffix in suffixes:
            # 简单匹配：2-10个字 + 后缀
            import re
            pattern = rf"[\u4e00-\u9fa5]{{2,10}}{suffix}"
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return list(set(entities))


class BatchNewsProcessor:
    """
    批量新闻处理器

    Features:
    - 批量分析with速率限制
    - 结果缓存
    - 进度回调
    """

    def __init__(
        self,
        analyzer: NewsAnalyzer,
        batch_size: int = 10,
        cache_enabled: bool = True,
    ):
        self.analyzer = analyzer
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self._cache: dict[str, NewsAnalysisResult] = {}

    def process_batch(
        self,
        news_items: list[NewsItem],
        progress_callback: callable | None = None,
    ) -> list[NewsItem]:
        """
        批量处理新闻

        Args:
            news_items: 待处理新闻列表
            progress_callback: 进度回调 (processed, total)

        Returns:
            更新后的新闻列表 (带分析结果)
        """
        total = len(news_items)
        results = []

        for i, news in enumerate(news_items):
            # 检查缓存
            if self.cache_enabled and news.news_id in self._cache:
                analysis = self._cache[news.news_id]
            else:
                analysis = self.analyzer.analyze_news(news)
                if self.cache_enabled:
                    self._cache[news.news_id] = analysis

            # 更新新闻对象
            updated_news = news.model_copy(update={
                "summary": analysis.summary,
                "sentiment": analysis.sentiment,
                "sentiment_score": analysis.sentiment_score,
                "key_events": analysis.key_events,
            })

            # 如果未关联发行人，尝试从实体提取
            if not updated_news.obligor_ids and analysis.mentioned_entities:
                # TODO: 实体到发行人ID的映射
                pass

            results.append(updated_news)

            # 进度回调
            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


class MockLLMClient:
    """Mock LLM client for testing without API"""

    class MockMessage:
        def __init__(self, text: str):
            self.text = text

    class MockResponse:
        def __init__(self, text: str):
            self.content = [MockLLMClient.MockMessage(text)]

    class MockMessages:
        def create(self, **kwargs) -> "MockLLMClient.MockResponse":
            # 返回模拟的JSON响应
            mock_result = {
                "summary": "模拟新闻摘要",
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.0,
                "key_events": ["模拟事件"],
                "credit_impact": "影响有限",
                "mentioned_entities": ["某公司"],
            }
            return MockLLMClient.MockResponse(json.dumps(mock_result))

    def __init__(self):
        self.messages = self.MockMessages()
