# Credit Bond Risk Intelligence Platform

> "Credit risk is not about the expected loss, but the unexpected one."
> "信用风险的本质不在于预期损失，而在于那些意想不到的尾部事件。"

---

## 1. Executive Vision (愿景)

本平台旨在构建**下一代信用风险预警系统**，区别于传统的规则引擎（Rule-Based）和纯统计模型（Statistical），我们采用**AI-Native + Signal-Composable**的架构理念，将：

- **传统信用分析**（评级迁移、利差监控、集中度管理）
- **现代AI能力**（LLM语义理解、RAG知识检索、向量相似度）
- **量化信号框架**（可组合、可回测、可解释的Signal对象）

三者深度融合，打造一个能够**早于市场感知信用风险**的智能预警平台。

### 核心差异化

| 传统系统 | 本平台 |
|----------|--------|
| 硬编码规则 `if oas > 500: alert()` | 信号对象 `SpreadSignal(threshold=95%ile)` |
| 人工阅读新闻 | LLM批量摘要 + 情感分析 |
| 搜索靠关键词 | RAG语义检索 "云南城投近期风险？" |
| 发行人孤立分析 | 向量空间找Peer Group |
| 报表静态展示 | 实时Dashboard + Alert Workflow |

---

## 2. Design Philosophy (设计哲学)

### 2.1 BlackRock Aladdin: Unified Risk View

**核心理念**：所有风险数据汇聚到**单一发行人视图**，消除信息孤岛。

```
┌─────────────────────────────────────────────────────────────┐
│                    Obligor Single View                       │
├─────────────────────────────────────────────────────────────┤
│  持仓数据        市场数据        新闻舆情        评级变动    │
│  (Positions)    (Market)       (News)         (Rating)      │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                            │                                 │
│                    ┌───────▼───────┐                        │
│                    │  Obligor View │                        │
│                    │  - 曝光汇总    │                        │
│                    │  - 风险信号    │                        │
│                    │  - AI摘要      │                        │
│                    │  - 历史事件    │                        │
│                    └───────┬───────┘                        │
│                            │                                 │
│                    ┌───────▼───────┐                        │
│                    │    Actions    │                        │
│                    │  - Alert      │                        │
│                    │  - Investigate│                        │
│                    │  - Report     │                        │
│                    └───────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

**落地方式**：
- `CreditExposure.from_positions()` 方法自动聚合持仓
- `Obligor` 模型包含AI增强字段（embedding_vector, risk_narrative, similar_obligors）
- Dashboard以发行人为中心展示所有维度数据

### 2.2 JPM Athena: Signal Library Pattern

**核心理念**：所有风险指标统一抽象为**可组合、可测试、可配置**的Signal对象。

```python
# 信号抽象基类
class Signal(ABC):
    name: ClassVar[str]
    category: ClassVar[SignalCategory]

    @abstractmethod
    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """计算单一发行人的信号值"""
        pass
```

**设计优势**：

| 特性 | 说明 |
|------|------|
| **Composable** | 信号可组合为复合信号 `CompositeSignal([spread, rating, news])` |
| **Configurable** | 阈值通过Pydantic配置注入，非硬编码 |
| **Testable** | 信号是纯函数，易于单元测试和回测 |
| **Traceable** | SignalResult包含完整元数据（value, threshold, metadata） |
| **Extensible** | 新增信号只需继承Signal基类 |

**内置信号矩阵**：

| Signal | Category | 触发条件 | 数据依赖 |
|--------|----------|----------|----------|
| `ConcentrationSignal` | CONCENTRATION | 单一发行人占比 > 2% AUM | positions |
| `HHISignal` | CONCENTRATION | HHI > 0.10 (Warning) / 0.18 (Critical) | positions |
| `SpreadPercentileSignal` | MARKET | OAS突破历史85/95%分位 | spread_history |
| `SpreadZScoreSignal` | MARKET | Z-score > 2.0 (Warning) / 3.0 (Critical) | spread_history |
| `RatingSignal` | FUNDAMENTAL | 评级下调或展望转负 | rating_history |
| `NewsSentimentSignal` | NEWS | 7天情感均值 < -0.3 (Warning) / -0.6 (Critical) | news_items |
| `MaturityPressureSignal` | FUNDAMENTAL | 12M内到期量 > 总债务25% | positions |

### 2.3 LLM-Native Architecture

**核心理念**：AI不是"附加功能"，而是**系统的核心智能层**。

#### 三层AI增强

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: RAG Q&A                          │
│  "云南城投最近有什么风险？" → 向量检索 → 上下文构建 → LLM回答  │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: News Intelligence               │
│  批量新闻 → Claude Haiku → 摘要 + 情感 + 事件提取            │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Embedding & Similarity          │
│  发行人/新闻文本 → text-embedding-3-small → 向量空间         │
│  相似发行人识别 / 语义搜索 / 异常检测                         │
└─────────────────────────────────────────────────────────────┘
```

#### LLM使用策略

| 场景 | 模型 | 理由 |
|------|------|------|
| 新闻批量摘要 | Claude Haiku | 成本低，速度快，适合批处理 |
| 深度风险分析 | Claude Sonnet | 推理能力强，生成质量高 |
| RAG问答 | Claude Sonnet | 需要理解复杂上下文 |
| 文本向量化 | text-embedding-3-small | OpenAI最优性价比 |

#### RAG Engine 设计

```python
class CreditRAGEngine:
    """
    信用知识库RAG引擎

    工作流程:
    1. 用户提问 → Embedding
    2. 向量检索相关文档 (news, filings, obligor_profiles)
    3. 构建上下文 (相似度过滤 + 长度截断)
    4. LLM生成答案 (带来源引用)

    示例问题:
    - "云南城投最近有什么风险事件？"
    - "哪些LGFV的债务率超过300%？"
    - "比较一下重庆和成都城投的信用资质"
    """
```

### 2.4 ML Ops: Feature Store Pattern

**核心理念**：发行人数据向量化，支持**相似性搜索**和**异常检测**。

```python
class Obligor(BaseModel):
    # ... 基础字段 ...

    # AI增强字段 (Feature Store)
    embedding_vector: list[float] | None  # 768维特征向量
    risk_narrative: str | None            # LLM生成的风险描述
    similar_obligors: list[str] | None    # 相似发行人ID列表
```

**应用场景**：

1. **Peer Group识别**：找出与目标发行人最相似的N个发行人，用于横向比较
2. **异常检测**：在向量空间中识别outlier（特征突变的发行人）
3. **知识迁移**：当某发行人出现风险事件时，自动预警其相似发行人

---

## 3. Architecture Deep Dive

### 3.1 Module Structure

```
credit_bond_risk/
│
├── core/                    # 核心层 - 配置与数据模型
│   ├── config.py            # Pydantic配置中心 (10+ 子配置)
│   │   ├── ConcentrationConfig    # 集中度阈值
│   │   ├── SpreadConfig           # 利差阈值
│   │   ├── RatingConfig           # 评级阈值
│   │   ├── NewsConfig             # 舆情阈值
│   │   ├── MaturityConfig         # 到期压力阈值
│   │   ├── LLMConfig              # LLM模型配置
│   │   ├── VectorStoreConfig      # 向量存储配置
│   │   ├── DataSourceConfig       # 数据源配置
│   │   └── AlertConfig            # 预警通知配置
│   │
│   ├── models.py            # Pydantic数据模型
│   │   ├── Obligor                # 发行人主数据
│   │   ├── ObligorFinancials      # 发行人财务数据
│   │   ├── BondPosition           # 单只债券持仓
│   │   ├── CreditExposure         # 信用曝光聚合
│   │   ├── NewsItem               # 新闻/公告
│   │   ├── NewsAnalysisResult     # LLM分析结果
│   │   ├── SignalResult           # 信号计算结果
│   │   ├── RiskAlert              # 风险预警
│   │   ├── RAGResponse            # RAG问答响应
│   │   └── SimilarObligor         # 相似发行人
│   │
│   └── enums.py             # 枚举定义
│       ├── Sector / SubSector     # 行业分类
│       ├── Region                 # 地区分类
│       ├── CreditRating           # 信用评级
│       ├── RatingOutlook          # 评级展望
│       ├── Severity               # 严重程度
│       ├── AlertCategory          # 预警类别
│       ├── AlertStatus            # 预警状态
│       ├── Sentiment              # 情感倾向
│       └── SignalCategory         # 信号类别
│
├── data/                    # 数据层 - 数据获取与解析 (NEW)
│   ├── __init__.py          # 统一导出接口
│   ├── provider.py          # 抽象数据接口 (Strategy Pattern)
│   │   ├── DataProvider (ABC)     # 数据提供者抽象基类
│   │   ├── DataProviderConfig     # 数据源配置
│   │   └── get_data_provider()    # 工厂函数
│   │
│   ├── mock_data.py         # Mock数据生成器
│   │   ├── MockDataProvider       # Mock数据提供者实现
│   │   ├── generate_mock_obligors()   # 生成发行人
│   │   ├── generate_mock_exposures()  # 生成曝光
│   │   ├── generate_mock_alerts()     # 生成预警
│   │   └── generate_mock_news()       # 生成新闻
│   │
│   ├── market_data.py       # 市场数据服务 (BBG/Wind)
│   │   ├── MarketDataService      # 统一市场数据接口
│   │   ├── BloombergFetcher       # Bloomberg数据 (xbbg)
│   │   ├── WindFetcher            # Wind数据 (WindPy)
│   │   ├── PriceQuote             # 实时报价模型
│   │   └── SpreadHistory          # 历史利差数据
│   │
│   ├── news_fetcher.py      # 新闻聚合服务
│   │   ├── NewsAggregator         # 多源新闻聚合器
│   │   ├── BloombergFetcher       # Bloomberg RSS
│   │   ├── ReutersFetcher         # Reuters RSS
│   │   ├── FTFetcher              # FT RSS
│   │   ├── WSJFetcher             # WSJ RSS
│   │   ├── CLSFetcher             # 财联社 API
│   │   ├── EastMoneyFetcher       # 东方财富 API
│   │   ├── CaixinFetcher          # 财新 API
│   │   ├── RawNewsItem            # 原始新闻数据
│   │   └── AnalyzedNewsItem       # 分析后新闻
│   │
│   └── filing_parser.py     # 公告/财报解析器
│       ├── FilingParser           # 统一解析接口
│       ├── PDFFilingParser        # PDF解析 (pdfplumber)
│       ├── EDGARParser            # SEC EDGAR解析
│       ├── ParsedFiling           # 解析结果模型
│       ├── FinancialMetrics       # 财务指标提取
│       └── DebtMaturity           # 到期债务提取
│
├── signals/                 # 信号层 - Athena风格信号系统
│   ├── base.py              # Signal抽象基类 + SignalContext + SignalRegistry
│   ├── concentration.py     # 集中度信号
│   │   ├── ConcentrationSignal    # 单一发行人集中度
│   │   └── HHISignal              # HHI指数
│   ├── spread.py            # 利差信号
│   │   ├── SpreadPercentileSignal # 历史百分位
│   │   └── SpreadZScoreSignal     # Z-score标准化
│   ├── rating.py            # 评级信号
│   │   └── RatingMigrationSignal  # 评级迁移
│   ├── news.py              # 舆情信号
│   │   └── NewsSentimentSignal    # LLM情感分析
│   └── composite.py         # 复合信号
│       └── CompositeSignal        # 多信号聚合
│
├── intelligence/            # AI智能层 - LLM增强分析
│   ├── embeddings.py        # 文本向量化服务
│   │   └── EmbeddingService       # OpenAI Embedding封装
│   ├── news_analyzer.py     # LLM新闻分析
│   │   └── NewsAnalyzer           # Claude新闻摘要+情感
│   ├── rag_engine.py        # RAG检索增强生成
│   │   ├── VectorStore            # 向量存储抽象
│   │   ├── Document               # 文档模型
│   │   └── CreditRAGEngine        # RAG引擎主类
│   └── anomaly_detector.py  # 异常检测
│       └── AnomalyDetector        # 统计异常检测
│
├── ui/                      # 展示层 - Streamlit Dashboard
│   ├── dashboard.py         # 主Dashboard (多页面)
│   └── components/          # 可复用UI组件
│       ├── charts.py            # Plotly可视化
│       ├── alert_table.py       # 预警表格
│       ├── obligor_card.py      # 发行人卡片
│       └── color_scheme.py      # 配色方案
│
├── scripts/                 # 运维脚本
│   ├── init_db.py           # 初始化信用数据库
│   └── sync_news.py         # 定时同步新闻 (uses data.news_fetcher)
│
└── app.py                   # Streamlit独立入口
```

### 3.2 Data Layer Design (NEW)

数据层采用**Strategy Pattern**实现数据源的可插拔设计，支持多种数据来源无缝切换。

#### 3.2.1 DataProvider 抽象接口

```python
from data import get_data_provider, DataProviderType

# Mock数据 (测试/演示)
provider = get_data_provider(DataProviderType.MOCK)

# 数据库 (生产环境)
provider = get_data_provider(DataProviderType.DATABASE)

# 使用统一API
obligors = provider.get_obligors()           # 发行人主数据
exposures = provider.get_exposures()         # 信用曝光
alerts = provider.get_alerts(status="PENDING")  # 预警
news = provider.get_news(days=7)             # 新闻
```

#### 3.2.2 Market Data Service

```python
from data.market_data import MarketDataService, MarketDataSource

service = MarketDataService()

# 获取实时报价 (自动选择可用数据源)
quote = service.get_quote("US912828ZT05")
print(f"OAS: {quote.oas}bp, Duration: {quote.duration}")

# 获取历史利差
history = service.get_spread_history("US912828ZT05", days=252)
zscore = history.get_zscore(quote.oas)
percentile = history.get_percentile(quote.oas)
```

#### 3.2.3 News Aggregator

```python
from data.news_fetcher import NewsAggregator, NewsSource

aggregator = NewsAggregator()

# 获取所有来源新闻
news = aggregator.fetch_all(days=7)

# 仅国际新闻
news = aggregator.fetch_all(include_domestic=False)

# 单一来源
news = aggregator.fetch_source(NewsSource.BLOOMBERG)

# LLM分析
analyzed = aggregator.analyze(news, use_llm=True)
```

#### 3.2.4 Filing Parser

```python
from data.filing_parser import FilingParser, FilingType

parser = FilingParser()

# 解析年报PDF
result = parser.parse_file("annual_report.pdf", FilingType.ANNUAL_REPORT)
print(f"Revenue: {result.financials.revenue}")
print(f"Debt/EBITDA: {result.financials.debt_to_ebitda}")

# 解析SEC EDGAR
result = parser.parse_url("https://sec.gov/...", FilingType.SEC_10K)
```

#### 3.2.5 设计优势

| 特性 | 说明 |
|------|------|
| **可插拔** | 数据源通过配置切换，代码无需修改 |
| **可测试** | Mock Provider支持单元测试和演示 |
| **可扩展** | 新增数据源只需实现抽象接口 |
| **缓存** | 内置缓存层，减少重复请求 |
| **类型安全** | 所有数据模型使用Pydantic强类型 |

### 3.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                               │
├─────────────────────────────────────────────────────────────────────┤
│  Bloomberg     SharePoint      RSS/API         Rating Agencies       │
│  (Prices)      (Positions)     (News)          (Ratings)            │
└──────┬────────────┬─────────────┬────────────────┬──────────────────┘
       │            │             │                │
       ▼            ▼             ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Spread   │  │ Position │  │ News     │  │ Rating   │            │
│  │ History  │  │ Daily    │  │ Items    │  │ History  │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       └─────────────┴──────┬──────┴─────────────┘                   │
│                            │                                         │
│                    ┌───────▼───────┐                                │
│                    │ SignalContext │                                │
│                    │ (统一数据上下文) │                                │
│                    └───────┬───────┘                                │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          SIGNAL LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │Concentration│ │   Spread   │ │   Rating   │ │    News    │       │
│  │   Signal   │ │   Signal   │ │   Signal   │ │   Signal   │       │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘       │
│        │              │              │              │                │
│        └──────────────┴──────┬───────┴──────────────┘                │
│                              │                                       │
│                      ┌───────▼───────┐                              │
│                      │ SignalResult  │                              │
│                      │ (is_triggered)│                              │
│                      └───────┬───────┘                              │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          ALERT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  SignalResult → RiskAlert → [PENDING] → [INVESTIGATING] → [RESOLVED]│
│                                │                                     │
│                         ┌──────▼──────┐                             │
│                         │ RAG辅助调查  │                             │
│                         │ (自动拉取新闻)│                             │
│                         └──────┬──────┘                             │
│                                │                                     │
│                         ┌──────▼──────┐                             │
│                         │   通知分发   │                             │
│                         │ Email/WeChat│                             │
│                         └─────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration System

### 4.1 Pydantic Config Hierarchy

```python
class CreditRiskConfig(BaseModel):
    """信用风险监控主配置 - 所有配置的聚合入口"""

    # 子配置 (嵌套BaseModel)
    concentration: ConcentrationConfig
    spread: SpreadConfig
    rating: RatingConfig
    news: NewsConfig
    maturity: MaturityConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
    data_source: DataSourceConfig
    alert: AlertConfig

    # 全局设置
    total_aum_usd: float = 50_000_000_000  # $50B
    base_currency: str = "USD"

    # 功能开关
    enable_llm_analysis: bool = True
    enable_embedding_search: bool = True
    enable_anomaly_detection: bool = True
```

### 4.2 Key Thresholds Reference

| Config | Parameter | Warning | Critical | Unit |
|--------|-----------|---------|----------|------|
| Concentration | single_obligor | 2% | 5% | % AUM |
| Concentration | top_n (10) | 40% | 60% | % AUM |
| Concentration | sector | 30% | 50% | % AUM |
| Concentration | province | 20% | 35% | % AUM |
| Concentration | hhi | 0.10 | 0.18 | Index |
| Spread | percentile | 85% | 95% | %ile |
| Spread | daily_change | 20 | 50 | bps |
| Spread | weekly_change | 30 | 80 | bps |
| Spread | zscore | 2.0 | 3.0 | σ |
| Rating | downgrade_notches | 1 | 2 | notches |
| News | sentiment | -0.3 | -0.6 | [-1,1] |
| News | negative_count | 3 | 5 | count/7d |
| Maturity | short_term_ratio | 25% | 40% | % debt |

---

## 5. Data Models Deep Dive

### 5.1 Obligor (发行人主数据)

```python
class Obligor(BaseModel):
    """
    发行人单一视图 - 汇聚所有维度的发行人数据

    设计原则:
    1. 唯一标识: obligor_id (统一社会信用代码或内部ID)
    2. 分类体系: sector → sub_sector → province → city
    3. 评级体系: external (三大) + internal + outlook
    4. AI增强: embedding + narrative + similar_obligors
    """

    obligor_id: str           # 唯一标识
    name_cn: str              # 中文名称
    name_en: str | None       # 英文名称

    # 分类 (支持中国离岸市场特色)
    sector: Sector            # LGFV / SOE / FINANCIAL / CORP
    sub_sector: str           # 省级城投 / 股份行 / 央企...
    region: Region            # CHINA_OFFSHORE / CHINA_ONSHORE / ...
    province: str | None
    city: str | None

    # 评级 (支持多来源)
    rating_external: dict[str, str]  # {"moody": "Baa1", "sp": "BBB+"}
    rating_internal: CreditRating
    rating_outlook: RatingOutlook

    # AI增强字段
    embedding_vector: list[float] | None  # 768维
    risk_narrative: str | None            # LLM生成
    similar_obligors: list[str] | None    # Top-N相似

    # Computed Fields
    @computed_field
    def rating_score(self) -> int:
        """评级数值化 (0-100, 便于排序/聚合)"""
        return rating_to_score(self.rating_internal)
```

### 5.2 CreditExposure (信用曝光聚合)

```python
class CreditExposure(BaseModel):
    """
    单一发行人的信用曝光汇总

    设计原则:
    1. 自动聚合: from_positions() 工厂方法
    2. 市值加权: duration, oas 按市值加权
    3. 到期分布: 0-1Y / 1-3Y / 3-5Y / 5-10Y / 10Y+
    """

    obligor: Obligor
    bonds: list[BondPosition]

    total_nominal_usd: float
    total_market_usd: float
    pct_of_aum: float              # 关键: 集中度计算依据

    weighted_avg_duration: float
    weighted_avg_oas: float
    credit_dv01_usd: float

    maturity_profile: dict[str, float]  # 到期分布

    @classmethod
    def from_positions(cls, obligor, positions, total_aum) -> "CreditExposure":
        """从持仓列表自动构建曝光汇总"""
        # ... 自动计算加权指标和到期分布
```

### 5.3 SignalResult (信号计算结果)

```python
class SignalResult(BaseModel):
    """
    信号计算结果 - 标准化输出

    设计原则:
    1. 完整性: 包含value, threshold, severity, metadata
    2. 自动判断: create() 工厂方法自动判断触发状态
    3. 可追溯: metadata可存储任意调试信息
    """

    signal_name: str
    category: SignalCategory
    obligor_id: str
    timestamp: datetime

    # 信号值
    value: float
    z_score: float | None
    percentile: float | None

    # 阈值
    threshold_warning: float
    threshold_critical: float

    # 触发判断 (自动计算)
    is_triggered: bool
    severity: Severity  # INFO / WARNING / CRITICAL

    # 元数据 (可扩展)
    metadata: dict[str, Any]

    @classmethod
    def create(cls, ..., higher_is_worse: bool = True) -> "SignalResult":
        """工厂方法: 自动判断触发状态和严重程度"""
```

---

## 6. Signal System Implementation

### 6.1 Signal Base Class

```python
class Signal(ABC):
    """
    信号抽象基类 - 所有信号的统一接口

    设计原则:
    1. 无状态: 信号本身是纯计算逻辑，状态通过SignalContext注入
    2. 可配置: 阈值通过构造函数注入，支持配置驱动
    3. 批量友好: compute_batch() 支持向量化优化
    """

    name: ClassVar[str]               # 信号名称
    category: ClassVar[SignalCategory] # 信号类别

    def __init__(self, threshold_warning, threshold_critical, higher_is_worse=True):
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical
        self.higher_is_worse = higher_is_worse

    @abstractmethod
    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """计算单一发行人的信号值"""
        pass

    def compute_batch(self, obligor_ids: list[str], context: SignalContext) -> list[SignalResult]:
        """批量计算 - 子类可覆写以优化性能"""
        return [self.compute(oid, context) for oid in obligor_ids]

    def compute_portfolio(self, context: SignalContext) -> list[SignalResult]:
        """计算组合内所有触发预警的发行人"""
        all_results = self.compute_batch(list(context.exposures.keys()), context)
        return [r for r in all_results if r.is_triggered]
```

### 6.2 SignalContext (数据上下文)

```python
@dataclass
class SignalContext:
    """
    信号计算上下文 - 提供所有信号计算所需的数据

    设计原则:
    1. 数据注入: 所有数据通过Context提供，信号无需关心数据来源
    2. 便于测试: 可以用Mock数据构造Context进行单元测试
    3. 便于回测: 可以用历史数据构造Context进行历史回测
    """

    # 持仓数据
    exposures: dict[str, CreditExposure]
    total_aum: float

    # 发行人主数据
    obligors: dict[str, Obligor]

    # 新闻数据
    news_items: list[NewsItem]
    news_by_obligor: dict[str, list[NewsItem]]

    # 历史利差数据
    spread_history: dict[str, dict[str, float]]

    # 评级历史
    rating_history: dict[str, list[tuple[str, str, str]]]

    # 配置
    config: CreditRiskConfig | None

    # 计算时间点 (支持历史回测)
    as_of_date: datetime
```

### 6.3 SignalRegistry (信号注册表)

```python
class SignalRegistry:
    """
    信号注册表 - 管理所有可用信号

    用途:
    1. 信号发现: list_all() 列出所有已注册信号
    2. 配置驱动: create_from_config() 从配置实例化信号
    3. 批量计算: compute_all() 计算所有信号
    """

    @classmethod
    def register(cls, signal_class: type[Signal]) -> type[Signal]:
        """装饰器: 注册信号类"""

    @classmethod
    def create_from_config(cls, config: CreditRiskConfig) -> list[Signal]:
        """从配置创建所有信号实例"""

    @classmethod
    def compute_all(cls, signals: list[Signal], context: SignalContext) -> list[SignalResult]:
        """计算所有信号并返回触发的预警"""
```

---

## 7. LLM Integration Patterns

### 7.1 News Analyzer

```python
class NewsAnalyzer:
    """
    LLM新闻分析器

    功能:
    1. 单篇摘要: 新闻 → 一句话摘要 + 情感分类
    2. 批量处理: 多篇新闻 → 使用Haiku降低成本
    3. 事件提取: 识别关键事件 (违约、降级、流动性危机...)

    设计原则:
    1. 成本控制: 批量用Haiku, 深度分析用Sonnet
    2. 响应缓存: 相同新闻不重复调用
    3. 敏感脱敏: 发送前脱敏处理
    """

    SYSTEM_PROMPT = """你是专业的信用债分析师。
    分析以下新闻，输出JSON格式：
    - summary: 一句话摘要
    - sentiment: POSITIVE/NEUTRAL/NEGATIVE
    - sentiment_score: -1到1的分数
    - key_events: 关键事件列表
    - credit_impact: 对信用的影响评估
    """
```

### 7.2 RAG Engine

```python
class CreditRAGEngine:
    """
    信用知识库RAG引擎

    工作流程:
    1. Query Embedding: 问题 → text-embedding-3-small → 向量
    2. Vector Search: 向量 → 检索相关文档 (news, filings, profiles)
    3. Context Building: 相似度过滤 + 长度截断 + 格式化
    4. Answer Generation: 上下文 + 问题 → Claude Sonnet → 答案
    5. Source Citation: 答案中引用来源 [来源1] [来源2]

    示例问题:
    - "云南城投最近有什么风险事件？"
    - "哪些LGFV的债务率超过300%？"
    - "总结一下恒大系发行人的近期新闻"
    """

    def query(self, question: str, obligor_id: str | None = None) -> RAGResponse:
        # 1. 生成问题embedding
        query_embedding = self.embedding_service.embed_text(question)

        # 2. 向量检索
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.rag_config.top_k,
            filter_obligor=obligor_id,
        )

        # 3. 构建上下文
        context = self._build_context(search_results)

        # 4. LLM生成答案
        answer = self._generate_answer(question, context)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=[...],
            confidence=self._estimate_confidence(search_results),
        )
```

---

## 8. Development Guidelines

### 8.1 Coding Standards

```python
# Type Hints Required (Python 3.10+)
def compute_spread_zscore(
    obligor_id: str,
    spread_history: dict[str, float],
    current_oas: float,
) -> float | None:
    """计算利差Z-score"""
    ...

# Pydantic for All Config/Input Validation
class SpreadConfig(BaseModel):
    percentile_warning: float = Field(0.85, ge=0, le=1)
    lookback_days: int = Field(252, ge=30)

# Vectorization Over Loops
# Bad:
for i in range(len(spreads)):
    zscore = (spreads[i] - mean) / std

# Good:
zscores = (np.array(spreads) - mean) / std
```

### 8.2 Logging Convention

```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debug info")      # 开发调试
logger.info("Normal operation")          # 正常运行
logger.warning("Unexpected but handled") # 异常但已处理
logger.error("Error occurred")           # 错误
```

### 8.3 LLM Usage Guidelines

```python
# 1. 批量处理时使用Haiku降低成本
for batch in batched(news_items, batch_size=10):
    results = analyzer.analyze_batch(batch, model="haiku")

# 2. 缓存LLM响应
@lru_cache(maxsize=1000)
def analyze_news(news_id: str) -> NewsAnalysisResult:
    ...

# 3. 设置合理的max_tokens
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=500,  # 摘要不需要太长
    ...
)

# 4. 敏感数据脱敏
def sanitize_for_llm(text: str) -> str:
    """移除敏感信息后再发送给LLM"""
    text = re.sub(r'\d{18}', '[ID]', text)  # 统一社会信用代码
    text = re.sub(r'\d+亿', '[X]亿', text)   # 具体金额
    return text
```

---

## 9. Integration Points

### 9.1 With Main Allocation Engine

```python
# 共享数据库
portfolio_db_path = "01_Data_Warehouse/db/portfolio.duckdb"

# 读取持仓数据
positions = duckdb.sql("""
    SELECT * FROM positions_daily
    WHERE date = (SELECT MAX(date) FROM positions_daily)
""").df()

# 共享Security Master
security_master = duckdb.sql("""
    SELECT * FROM security_master
""").df()
```

### 9.2 Dashboard Navigation

```python
# 主Dashboard可跳转至信用风险页面
if st.button("查看信用风险详情"):
    st.switch_page("03_Strategy_Lab/credit_bond_risk/app.py")
```

---

## 10. Roadmap

- [x] **Phase 1**: Core models & config (Pydantic v2)
- [x] **Phase 2**: Signal system framework (Athena pattern)
- [x] **Phase 3**: Intelligence layer (LLM + RAG + Embeddings)
- [x] **Phase 4**: Dashboard MVP (Streamlit multi-page)
- [x] **Phase 5**: International issuers support
- [x] **Phase 6**: Data layer refactoring (Provider pattern, News/Market/Filing modules)
- [ ] **Phase 7**: Alert workflow automation
- [ ] **Phase 8**: Mobile notifications (企业微信)
- [ ] **Phase 9**: Historical backtesting framework
- [ ] **Phase 10**: Real-time streaming alerts

---

## 11. Quick Reference

### Commands

```bash
# 启动Dashboard
cd 03_Strategy_Lab/credit_bond_risk && streamlit run app.py

# 初始化数据库
python scripts/init_db.py

# 同步新闻
python scripts/sync_news.py
```

### Key Files

| File | Purpose |
|------|---------|
| `core/config.py` | 所有配置定义 |
| `core/models.py` | 所有数据模型 |
| `data/provider.py` | 数据提供者抽象接口 |
| `data/mock_data.py` | Mock数据生成器 |
| `data/market_data.py` | 市场数据服务 (BBG/Wind) |
| `data/news_fetcher.py` | 新闻聚合服务 |
| `data/filing_parser.py` | 公告/财报解析器 |
| `signals/base.py` | Signal抽象基类 |
| `intelligence/rag_engine.py` | RAG问答引擎 |
| `ui/dashboard.py` | 主Dashboard |

---

*Last Updated: 2026-02*
*Version: 2.1 (Data Layer Refactoring)*
