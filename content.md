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
│   └── 2026_allocation_plan/   # 2026 资产配置引擎 (核心模块)
│       ├── __init__.py         # 模块入口
│       ├── config.py           # 配置定义 (Pydantic Models)
│       ├── analytics.py        # 分析工具 (Nelson-Siegel, FX Analytics)
│       ├── data_provider.py    # 市场数据抽象层
│       ├── allocation_engine.py# 配置引擎核心逻辑
│       ├── forward_rate_data.py# 远期利率数据处理
│       └── dashboard.py        # Streamlit 交互式仪表盘
│
└── 05_Dashboard_UI/            # 【可视化】Web 界面入口
    └── app.py                  # Streamlit 启动入口
```

## Key Modules (核心模块说明)

### 01_Data_Warehouse
- **Purpose**: 数据中台，负责原始数据的摄取、清洗和存储
- **Tech**: DuckDB (OLAP) + Parquet (时序数据)
- **Key Scripts**:
  - `yield_curve_loader.py`: 基于 Nelson-Siegel-Svensson 模型的收益率曲线插值

### 03_Strategy_Lab/2026_allocation_plan
- **Purpose**: 2026 年度资产配置的核心计算引擎
- **Components**:
  - `config.py`: 定义货币、账户、FTP 等配置参数
  - `analytics.py`: 收益率曲面、FX 分析、FTP 计算器
  - `allocation_engine.py`: NII 优化、多币种配置逻辑
  - `dashboard.py`: 情景分析可视化 (Streamlit)

### 05_Dashboard_UI
- **Purpose**: Web 端入口，整合各模块的可视化展示
- **Tech**: Streamlit
