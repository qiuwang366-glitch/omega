# Project Omega - Directory Structure

```
Project_Omega_2026/
│
├── CLAUDE.md                   # AI 交互指南 (System Prompt for LLM)
├── README.md                   # 项目说明文档
├── content.md                  # 目录结构说明 (本文件)
├── requirements.txt            # Python 依赖清单
├── .gitignore                  # Git 忽略规则
│
├── 00_Config/                  # 【配置中心】
│   ├── secrets.yaml            # API Keys (Bloomberg, Sharepoint Auth) - *GitIgnore*
│   ├── universe_mapping.csv    # 内部代码 vs Bloomberg Ticker 映射表
│   └── curve_config.json       # 关键期限点定义 (Key Rates Def)
│
├── 01_Data_Warehouse/          # 【数据中台】DuckDB + Parquet 架构
│   ├── raw_landing/            # 原始数据暂存 (Excel/CSV/JSON)
│   │   └── position20251231.csv# 持仓快照文件
│   ├── db/                     # 数据库文件
│   │   ├── portfolio.duckdb    # 主OLAP数据库 (~1.3MB)
│   │   └── yield_curves_snapshot.json  # 收益率曲线快照
│   └── etl_scripts/            # Python ETL 脚本
│       ├── init_db.py          # 数据库初始化
│       ├── ingest_sharepoint.py# 自动拉取 Sharepoint Excel
│       ├── ingest_bloomberg.py # 通过 BLPAPI/xbbg 拉取数据
│       ├── clean_and_load.py   # 数据清洗与入库
│       └── yield_curve_loader.py # 收益率曲线加载器 (Nelson-Siegel插值)
│
├── 03_Strategy_Lab/            # 【策略实验室】研判与配置方案
│   ├── 2026_Investment_Plan/   # 2026 投资计划
│   │   ├── hypothesis.md       # 核心假设 (宏观剧本)
│   │   ├── 2026_Subportfolio_Profile.md # 子组合分析报告
│   │   └── analyze_subportfolios.py     # 子组合分析脚本
│   │
│   ├── 2026_allocation_plan/   # 2026 资产配置引擎 (核心模块)
│   │   ├── __init__.py         # 模块入口
│   │   ├── config.py           # 配置定义 (Pydantic Models)
│   │   ├── analytics.py        # 分析工具 (Nelson-Siegel, FX Analytics)
│   │   ├── data_provider.py    # 市场数据抽象层
│   │   ├── position_loader.py  # 持仓文件解析
│   │   ├── allocation_engine.py# 配置引擎核心逻辑
│   │   ├── forward_rate_data.py# 远期利率数据处理
│   │   ├── visualizations.py   # Plotly可视化组件
│   │   └── dashboard.py        # Streamlit 交互式仪表盘
│   │
│   └── credit_bond_risk/       # 【信用风险智能平台】LLM增强 (6,000+ LOC)
│       ├── CLAUDE.md           # 模块专属AI交互指南
│       ├── app.py              # Streamlit 独立入口
│       ├── __init__.py         # 模块入口
│       │
│       ├── core/               # 核心层 - 配置与数据模型
│       │   ├── __init__.py
│       │   ├── config.py       # Pydantic配置中心 (10+ 子配置)
│       │   ├── models.py       # 数据模型 (Obligor, Bond, Alert, News)
│       │   └── enums.py        # 枚举定义 (Rating, Sector, Severity)
│       │
│       ├── signals/            # 信号层 - Athena风格信号系统
│       │   ├── __init__.py
│       │   ├── base.py         # Signal抽象基类 + SignalRegistry
│       │   ├── concentration.py # 集中度信号 (Single/TopN/Sector/HHI)
│       │   ├── rating.py       # 评级迁移信号
│       │   ├── spread.py       # 利差异动信号 (Percentile/Z-Score)
│       │   ├── news.py         # 舆情信号 (LLM驱动)
│       │   └── composite.py    # 复合信号聚合
│       │
│       ├── intelligence/       # AI智能层 - LLM增强分析
│       │   ├── __init__.py
│       │   ├── embeddings.py   # 文本向量化 (OpenAI/BGE)
│       │   ├── news_analyzer.py# LLM新闻摘要+情感分析
│       │   ├── rag_engine.py   # RAG检索增强生成
│       │   └── anomaly_detector.py # Spread/Volume异常检测
│       │
│       ├── ui/                 # 展示层 - Streamlit Dashboard
│       │   ├── __init__.py
│       │   ├── dashboard.py    # 主Dashboard (多页面)
│       │   └── components/     # 可复用UI组件
│       │       ├── __init__.py
│       │       ├── charts.py       # Plotly可视化
│       │       ├── alert_table.py  # 预警表格
│       │       ├── obligor_card.py # 发行人卡片
│       │       └── color_scheme.py # 配色方案
│       │
│       └── scripts/            # 运维脚本
│           ├── init_db.py      # 初始化信用数据库
│           └── sync_news.py    # 定时同步新闻
│
└── 05_Dashboard_UI/            # 【可视化】Web 界面入口
    └── app.py                  # Streamlit 启动入口 (聚合)
```

---

## Key Modules (核心模块说明)

### 01_Data_Warehouse
- **Purpose**: 数据中台，负责原始数据的摄取、清洗和存储
- **Tech**: DuckDB (OLAP) + Parquet (时序数据)
- **Key Scripts**:
  - `yield_curve_loader.py`: 基于 Nelson-Siegel-Svensson 模型的收益率曲线插值
  - `init_db.py`: 数据库Schema初始化

### 03_Strategy_Lab/2026_allocation_plan
- **Purpose**: 2026 年度资产配置的核心计算引擎
- **LOC**: ~230KB (9 Python模块)
- **Components**:
  - `config.py`: 定义货币、账户、FTP 等配置参数
  - `analytics.py`: 收益率曲面、FX 分析、FTP 计算器
  - `allocation_engine.py`: NII 优化、多币种配置逻辑
  - `dashboard.py`: 情景分析可视化 (Streamlit)

### 03_Strategy_Lab/credit_bond_risk
- **Purpose**: 存量信用债发行人风险预警系统
- **LOC**: ~6,165 lines (27 Python files)
- **Design Philosophy**:

| 来源 | 理念 | 落地方式 |
|------|------|----------|
| **BlackRock Aladdin** | Unified Risk View | 多数据源 → 单一发行人视图 → 预警 → 行动 |
| **JPM Athena** | Signal Library | 风险信号标准化为可组合的 Signal 对象 |
| **LLM Native** | RAG + Summarization | 新闻 → Embedding → 检索 → Claude 摘要 |
| **ML Ops** | Feature Store | 发行人特征向量化，支持相似性搜索 |

- **Key Subsystems**:
  - `signals/`: Concentration / Rating / Spread / News 四类信号
  - `intelligence/`: RAG Engine + News Analyzer + Embeddings
  - `ui/`: Multi-page Streamlit Dashboard

### 05_Dashboard_UI
- **Purpose**: Web 端入口，整合各模块的可视化展示
- **Tech**: Streamlit

---

## Quick Commands (常用操作)

```bash
# 启动 2026 资产配置仪表盘
cd 03_Strategy_Lab/2026_allocation_plan && streamlit run dashboard.py

# 启动 信用风险智能仪表盘
cd 03_Strategy_Lab/credit_bond_risk && streamlit run app.py

# 初始化主数据库
cd 01_Data_Warehouse/etl_scripts && python init_db.py

# 初始化信用风险数据库
cd 03_Strategy_Lab/credit_bond_risk && python scripts/init_db.py

# 同步新闻数据
cd 03_Strategy_Lab/credit_bond_risk && python scripts/sync_news.py

# 子组合分析
cd 03_Strategy_Lab/2026_Investment_Plan && python analyze_subportfolios.py
```
