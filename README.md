# Project Omega: Institutional Fixed Income Quant System
> **Portable Alpha Infrastructure for the $50B+ Asset Manager**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Database](https://img.shields.io/badge/Data-DuckDB%20%7C%20Parquet-yellow.svg)
![Architecture](https://img.shields.io/badge/Architecture-Quant--Macro%20Hybrid-red.svg)
![Status](https://img.shields.io/badge/Status-Active%20(2026%20Plan)-success.svg)

## 📖 Executive Summary (项目概述)

**Project Omega** 是一套专为大规模固收投资（$50B+ AUM）设计的**一体化投研与风控系统**。

本项目旨在解决传统银行/资管体系中“数据孤岛”与“系统僵化”的痛点，通过现代化的数据技术栈（Modern Data Stack）和 AI 增强的工作流，实现：
1.  **影子账本 (Shadow PMS)**：独立于核心银行系统的持仓与现金流计算引擎。
2.  **宏观对冲 (Macro Hedging)**：基于利率二阶导数（Convexity/Gamma）和宏观因子（G10 Rates/FX）的策略生成。
3.  **可携带 Alpha (Portable Alpha)**：将投资经理的决策逻辑代码化、SaaS 化，使其不依赖于特定机构的 IT 设施。

---

## 🏗️ System Architecture (系统架构)

本系统采用 **"Local-First, Cloud-Ready"** 架构，核心是一个文件夹即可打包带走的“口袋彭博”。

```text
Project_Omega_2026/
├── 01_Data_Warehouse/   # [Data Layer] 基于 DuckDB 的高性能 OLAP 数据中台
│   ├── db/              # .duckdb 单文件数据库 (Portable SQL Engine)
│   └── etl_scripts/     # 针对 Bloomberg/Sharepoint 的清洗脚本
│
├── 02_Quant_Engine/     # [Logic Layer] 金融数学内核
│   ├── pricing/         # 现金流折现、OIS 曲线构建 (QuantLib/Scipy)
│   └── risk/            # 风险归因 (Brinson) 与压力测试
│
├── 03_Strategy_Lab/     # [Decision Layer] 策略研发实验室
│   ├── 2026_Allocation/ # 2026 年度资产配置回测
│   └── hypothesis.md    # 核心投资假设与宏观剧本
│
├── 05_Dashboard_UI/     # [Presentation Layer] 交互式驾驶舱
│   └── app.py           # 基于 Streamlit 的动态情景分析工具
│
└── CLAUDE.md            # [AI Kernel] LLM 智能参谋的系统指令集 (System Prompt)
```

🦁 Strategic Framework (投资宪法)
本系统的所有代码与模型均遵循以下机构级约束 (Institutional Constraints)：

1. 核心账户逻辑 (Accounting Regime)
AC / FVOCI 主导：80% 持仓以净利息收入 (NII) 为考核目标。

忽略 OCI 波动：只要不发生信用减值，FVOCI 的市值波动（MTM）不影响核心利润表。

FVTPL 卫星仓位：仅在此账户进行高频交易或 Total Return 策略。

2. 资金成本套利 (The FTP Lag Arbitrage)
规则：当月 FTP (资金成本) = 上月 3M 国债均值。

Alpha 来源：

加息周期 (Hiking)：利用 FTP 滞后，在加息当月 aggressively 加仓（资产收益率跳升，资金成本未变）。

降息周期 (Cutting)：在降息前抢跑（Front-loading），避免陷入资产收益率下跌但 FTP 居高不下的“有毒窗口”。

3. 无限流动性假设 (Liquidity Assumption)
假设资金端供应无限 (Unlimited Funding)，投资边界仅受制于：

Spread > 0 (相对于滞后 FTP)

RWA (风险加权资产) 约束

FX P&L Buffer (即时汇率折算后的安全垫)

📊 Key Modules (核心功能模块)
A. Data Warehouse (数据中台)
自动化清洗来自 Sharepoint (Excel) 和 Bloomberg AIM 的脏数据。

Feature Engineering: 自动计算 Implied FX Rate，并基于 security_master 补全久期与凸性数据。

B. 2026 Strategy Lab (2026 战略推演)
USD SSA Strategy: "The Convexity Bridge"

目标：在美联储降息前，将组合久期从 2.45 拉长至 4.0+。

手段：利用无限子弹，在 5Y-7Y 区间进行 Aggressive Front-loading。

AUD Rates Strategy: "The Lagged Arbitrage"

目标：博弈 RBA 政策与 AUD 汇率。

风控：建立 CCS (Cross-Currency Swap) 监控机制，当 AUD_Yield * FX_Scenario < USD_Cost 时触发对冲。

C. Dashboard (指挥官驾驶舱)
提供 Streamlit 界面，支持实时调节：

USD/AUD 投入规模 (Firepower)

目标建仓收益率 (Entry Yield)

汇率压力测试 (FX Stress Test)

可视化输出：NII 瀑布图、FX 盈亏平衡热力图。

🚀 Getting Started (快速上手)
1. 环境配置
确保已安装 Python 3.10+，然后安装依赖：

Bash
pip install duckdb pandas plotly streamlit office365-rest-python-client xbbg
2. 数据初始化 (ETL)
将原始持仓文件 (position20251231.csv) 放入 01_Data_Warehouse/raw_landing/，然后运行：

Bash
cd 01_Data_Warehouse/etl_scripts
python init_db_v2.py
此步骤将生成 portfolio.duckdb 数据库文件。

3. 启动驾驶舱
Bash
cd 05_Dashboard_UI
streamlit run app.py
🤖 AI Interaction Protocol (AI 交互协议)
本项目集成了 AI 参谋模式。在与 Claude/ChatGPT 交互时，请遵循以下流程：

Context Injection: 始终确保 AI 读取了根目录下的 CLAUDE.md。

Dual Persona:

询问市场时，AI 是 Macro CIO (关注二阶导数、流动性)。

询问职场/汇报时，AI 是 Political Strategist (关注合规、话术)。

Command Triggers:

输入 "复盘"：触发每日总结模式。

输入 "Stress Test"：触发情景分析代码生成。

⚠️ Disclaimer
Institutional Use Only: 本系统参数基于 $50B+ 机构资产负债表设定（如固定 FTP），不适用于个人零售投资者。

Data Privacy: 所有上传至 GitHub 的代码均已脱敏，不包含真实交易对手方信息或未公开的内部头寸。

Project Omega Building the bridge between Math, Macro, and Management.

© 2026 Managed by the CIO Office.
