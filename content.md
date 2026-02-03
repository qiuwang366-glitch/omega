# Project_Omega_2026/
│
├── CLAUDE.md                   # AI 交互指南
├── requirements.txt            # 依赖 (duckdb, polars, xbbg, office365-rest-python-client, quantlib-python)
│
├── 00_Config/                  # 【配置中心】
│   ├── secrets.yaml            # API Keys (Bloomberg, Sharepoint Auth) - *GitIgnore*
│   ├── universe_mapping.csv    # 内部代码 vs Bloomberg Ticker 映射表
│   └── curve_config.json       # 关键期限点定义 (Key Rates Def)
│
├── 01_Data_Warehouse/          # 【数据中台】DuckDB + Parquet 架构
│   ├── raw_landing/            # 原始数据暂存 (Excel/CSV/JSON)
│   ├── db/                     # 数据库文件
│   │   ├── portfolio.duckdb    # 主数据库 (SQL Interface)
│   │   └── market_data/        # 大规模时序数据 (.parquet, 按月/品种分区)
│   └── etl_scripts/            # Python ETL 脚本
│       ├── ingest_sharepoint.py# 自动拉取 Sharepoint Excel
│       ├── ingest_bloomberg.py # 通过 BLPAPI/xbbg 拉取数据
│       └── clean_and_load.py   # 数据清洗与入库
│
├── 02_Quant_Engine/            # 【计算内核】(因数据变细，可上 QuantLib)
│   ├── pricing/                # 现金流折现引擎 (Yield Calculation)
│   └── attribution/            # Brinson / Campisi 归因模型
│
├── 03_Strategy_Lab/          # 【研判】策略研发与假设验证
│   ├── 2026_Allocation/      # 2026 资产配置方案 (SAA/TAA)
│   │   ├── hypothesis.md     # 核心假设 (e.g., "美联储 QT 结束后的利率曲线陡峭化")
│   │   └── simulation/       # 蒙特卡洛模拟代码
│   ├── relative_value/       # 相对价值策略 (Curve Spreads, Basis Trade)
│   └── tail_hedging/         # 尾部对冲方案 (VIX Call, OTM Puts)
│
├── 04_Knowledge_Graph/       # 【知识库】RAG 系统的核心 (Obsidian/Markdown)
│   ├── research_notes/       # 卖方研报精读与笔记 (Tag: #Bearish #Duration)
│   ├── meeting_minutes/      # 内部会议纪要 (政治博弈与资金流向记录)
│   └── global_macro_logs/    # 每日宏观复盘日志 (我们的 Daily Review)
│
├── 05_Dashboard_UI/          # 【驾驶舱】Web 界面 (Streamlit/Next.js)
│   ├── components/           # 图表组件 (Yield Curve Visualizer, PnL Attribution)
│   ├── pages/                # 页面逻辑 (Overview, Risk Monitor, Scenario Analysis)
│   └── app.py                # 启动入口
│
└── 06_Output_Delivery/       # 【交付】汇报与展示
    ├── investment_memos/     # 投资备忘录 (给 IC 投决会看)
    ├── board_presentation/   # 董事会/高管汇报 PPT 逻辑
    └── weekly_monitor/       # 周度监控自动生成