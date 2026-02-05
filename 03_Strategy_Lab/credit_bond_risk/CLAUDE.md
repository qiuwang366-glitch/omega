# Credit Bond Risk Intelligence Platform

> "Credit risk is not about the expected loss, but the unexpected one."

## 1. Module Overview

**Purpose**: 存量信用债发行人风险预警系统，结合传统信用分析与LLM智能分析能力。

**Design Philosophy**:
| 来源 | 理念 | 落地方式 |
|------|------|----------|
| **BlackRock Aladdin** | Unified Risk View | 多数据源 → 统一发行人视图 → 预警 → 调查 → 行动 |
| **JPM Athena** | Signal Library | 风险信号标准化为可组合的 Signal 对象 |
| **LLM Native** | RAG + Summarization | 新闻/公告 → Embedding → 检索 → 摘要生成 |
| **ML Ops** | Feature Store | 发行人特征向量化，支持相似性搜索和异常检测 |

---

## 2. Architecture

```
credit_bond_risk/
│
├── core/                    # 核心层 - 配置与数据模型
│   ├── config.py            # Pydantic配置中心
│   ├── models.py            # 数据模型 (Obligor, Bond, Alert, News)
│   └── enums.py             # 枚举定义 (Rating, Sector, Severity)
│
├── data/                    # 数据层 - 数据获取与解析
│   ├── provider.py          # 抽象数据接口
│   ├── market_data.py       # BBG/Wind价格数据
│   ├── news_fetcher.py      # 新闻抓取 (RSS/API)
│   └── filing_parser.py     # 公告/财报解析
│
├── intelligence/            # AI智能层 - LLM增强分析
│   ├── embeddings.py        # 文本向量化
│   ├── news_analyzer.py     # LLM新闻摘要+情感分析
│   ├── rag_engine.py        # RAG检索增强生成
│   ├── anomaly_detector.py  # Spread/Volume异常检测
│   └── similarity_search.py # 相似发行人识别
│
├── signals/                 # 信号层 - Athena风格信号系统
│   ├── base.py              # Signal抽象基类
│   ├── concentration.py     # 集中度信号
│   ├── rating.py            # 评级迁移信号
│   ├── spread.py            # 利差异动信号
│   ├── news.py              # 舆情信号 (LLM驱动)
│   └── composite.py         # 复合信号聚合
│
├── engine/                  # 引擎层 - 监控与预警
│   ├── monitoring.py        # 实时监控主循环
│   ├── alerting.py          # 预警生成+分发
│   └── workflow.py          # Alert → Investigation → Action
│
├── analytics/               # 分析层 - 传统信用分析
│   ├── credit_metrics.py    # PD/LGD/EL计算
│   ├── migration_matrix.py  # 评级迁移矩阵
│   └── stress_testing.py    # 信用压力测试
│
├── reporting/               # 报告层
│   ├── obligor_report.py    # 单一发行人深度报告
│   ├── portfolio_report.py  # 组合信用报告
│   └── templates/           # Jinja2模板
│
├── ui/                      # 展示层
│   ├── dashboard.py         # Streamlit主入口
│   ├── pages/               # 多页面应用
│   │   ├── overview.py      # 组合概览
│   │   ├── obligor_drill.py # 发行人深挖
│   │   ├── alerts.py        # 预警中心
│   │   ├── news_feed.py     # 新闻流
│   │   └── chat.py          # RAG问答界面
│   └── components/          # 可复用UI组件
│
└── scripts/                 # 运维脚本
    ├── init_db.py           # 初始化信用数据库
    ├── sync_news.py         # 定时同步新闻
    └── backfill_embeddings.py
```

---

## 3. Core Concepts

### 3.1 Signal System (信号系统)

所有风险指标统一抽象为 `Signal` 对象：

```python
class Signal(ABC):
    """信号抽象基类"""
    name: str
    category: Literal["MARKET", "FUNDAMENTAL", "NEWS", "CONCENTRATION"]

    @abstractmethod
    def compute(self, obligor_id: str, context: SignalContext) -> SignalResult:
        """计算单一发行人的信号值"""
        pass
```

**内置信号类型**:
| Signal | Category | 触发条件 |
|--------|----------|----------|
| `ConcentrationSignal` | CONCENTRATION | 单一发行人占比 > 2% AUM |
| `RatingSignal` | FUNDAMENTAL | 评级下调或展望转负 |
| `SpreadSignal` | MARKET | OAS突破历史90%分位 |
| `NewsSentimentSignal` | NEWS | 7天内情感均值 < -0.5 |
| `MaturityPressureSignal` | FUNDAMENTAL | 12M内到期量 > 总债务30% |

### 3.2 Intelligence Layer (AI增强)

**LLM 用途**:
1. **新闻摘要**: 批量新闻 → 单句摘要 + 情感分类
2. **风险简报**: 发行人数据 + 新闻 → 结构化风险报告
3. **RAG问答**: "云南城投最近有什么风险事件？"
4. **实体识别**: 新闻文本 → 关联发行人ID

**Embedding 用途**:
1. **相似发行人**: 基于特征向量找peer group
2. **新闻检索**: 语义搜索相关历史新闻
3. **异常检测**: 特征空间中的outlier识别

### 3.3 Alert Workflow (预警工作流)

```
Signal Triggered
      │
      ▼
┌─────────────┐
│ Alert Queue │  ← 写入数据库，状态: PENDING
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Triage    │  ← 规则引擎判断严重程度
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Investigation│  ← RAG辅助：自动拉取相关新闻/财报
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Action    │  ← 通知(Email/企业微信) + 记录处置
└─────────────┘
```

---

## 4. Data Models

### 4.1 Obligor (发行人)

```python
class Obligor(BaseModel):
    obligor_id: str           # 唯一标识 (可用统一社会信用代码)
    name_cn: str
    name_en: str | None

    # 分类
    sector: Sector            # LGFV, SOE, FINANCIAL, CORP
    sub_sector: str           # 省级城投/股份行/央企...
    province: str | None
    city: str | None

    # 评级
    rating_external: dict[str, str]  # {"moody": "Baa1", "sp": "BBB+"}
    rating_internal: str
    rating_outlook: RatingOutlook

    # AI增强字段
    embedding_vector: list[float] | None
    risk_narrative: str | None
    similar_obligors: list[str] | None
```

### 4.2 CreditExposure (信用曝光)

```python
class CreditExposure(BaseModel):
    obligor: Obligor
    bonds: list[BondPosition]

    total_nominal_usd: float
    total_market_usd: float
    pct_of_aum: float
    weighted_avg_oas: float
    credit_dv01_usd: float
    maturity_profile: dict[str, float]
```

### 4.3 RiskAlert (风险预警)

```python
class RiskAlert(BaseModel):
    alert_id: str
    timestamp: datetime
    severity: Severity        # INFO, WARNING, CRITICAL
    category: AlertCategory   # CONCENTRATION, RATING, SPREAD, NEWS
    obligor_id: str
    signal_name: str

    message: str
    metric_value: float
    threshold: float

    status: AlertStatus       # PENDING, INVESTIGATING, RESOLVED, DISMISSED
    assigned_to: str | None
    resolution_note: str | None
```

---

## 5. Tech Stack

| 组件 | 选型 | 理由 |
|------|------|------|
| **Database** | DuckDB | 与主项目一致，OLAP性能好 |
| **Vector Store** | DuckDB + vss 或 ChromaDB | 向量检索 |
| **LLM** | Claude 3.5 Sonnet / Haiku | 中文理解强 |
| **Embedding** | text-embedding-3-small / bge-m3 | 多语言 |
| **UI** | Streamlit | 与主项目一致 |
| **Scheduler** | APScheduler | 轻量级定时任务 |
| **Validation** | Pydantic v2 | 数据校验 |

---

## 6. Configuration

所有配置通过 `core/config.py` 的 Pydantic 模型管理：

```python
class CreditRiskConfig(BaseModel):
    # 集中度阈值
    concentration_warning: float = 0.02      # 2% AUM
    concentration_critical: float = 0.05     # 5% AUM
    top_n_limit: int = 10                    # Top N发行人上限

    # 利差阈值
    spread_percentile_warning: float = 0.85  # 85%分位
    spread_percentile_critical: float = 0.95 # 95%分位
    spread_lookback_days: int = 252          # 1年历史

    # 舆情阈值
    news_sentiment_warning: float = -0.3
    news_sentiment_critical: float = -0.6
    news_lookback_days: int = 7

    # LLM配置
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_model_fast: str = "claude-3-5-haiku-20241022"
    embedding_model: str = "text-embedding-3-small"

    # 数据源
    news_sources: list[str] = ["cls", "eastmoney", "bloomberg"]
```

---

## 7. Quick Start

```bash
# 启动Dashboard
cd 03_Strategy_Lab/credit_bond_risk
streamlit run ui/dashboard.py

# 初始化数据库
python scripts/init_db.py

# 同步新闻 (手动)
python scripts/sync_news.py

# 回填Embedding
python scripts/backfill_embeddings.py
```

---

## 8. Integration with Main System

本模块与主系统 `2026_allocation_plan` 的集成点：

1. **共享数据库**: 读取 `positions_daily` 表获取持仓
2. **共享Security Master**: 扩展 `security_master` 表增加信用字段
3. **配置继承**: 可从主配置继承货币、账户等设置
4. **Dashboard链接**: 主Dashboard可跳转至信用风险页面

---

## 9. Development Guidelines

### 9.1 Coding Standards
- Type hints required (Python 3.10+)
- Pydantic for all config/input validation
- Numpy vectorization, avoid loops
- Docstrings: module + class + method level
- Logging: `logger = logging.getLogger(__name__)`

### 9.2 LLM Usage Guidelines
- 批量处理时使用 Haiku 降低成本
- 缓存LLM响应避免重复调用
- 设置合理的 `max_tokens` 限制
- 敏感数据脱敏后再发送

### 9.3 Testing
- Unit tests for signals: `tests/test_signals.py`
- Integration tests for LLM: mock responses
- E2E tests for dashboard: Streamlit testing

---

## 10. Roadmap

- [x] Phase 1: Core models & config
- [x] Phase 2: Signal system framework
- [x] Phase 3: Intelligence layer (LLM)
- [x] Phase 4: Dashboard MVP
- [ ] Phase 5: News fetcher integration
- [ ] Phase 6: Alert workflow automation
- [ ] Phase 7: Mobile notifications (企业微信)

---

*Last Updated: 2024-01*
