# Project Omega: Institutional Fixed Income Quant System
> **Portable Alpha Infrastructure for the $50B+ Asset Manager**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Database](https://img.shields.io/badge/Data-DuckDB%20%7C%20Parquet-yellow.svg)
![Architecture](https://img.shields.io/badge/Architecture-Quant--Macro%20Hybrid-red.svg)
![Status](https://img.shields.io/badge/Status-Active%20(2026%20Plan)-success.svg)

## Executive Summary

**Project Omega** 是一套专为大规模固收投资（$50B+ AUM）设计的**一体化投研与风控系统**。

本项目旨在解决传统银行/资管体系中"数据孤岛"与"系统僵化"的痛点，通过现代化的数据技术栈（Modern Data Stack）和 AI 增强的工作流，实现：
1.  **影子账本 (Shadow PMS)**：独立于核心银行系统的持仓与现金流计算引擎
2.  **宏观对冲 (Macro Hedging)**：基于利率二阶导数（Convexity/Gamma）和宏观因子的策略生成
3.  **可携带 Alpha (Portable Alpha)**：将投资经理的决策逻辑代码化，使其不依赖于特定机构的 IT 设施

---

## System Architecture

本系统采用 **"Local-First, Cloud-Ready"** 架构，核心是一个文件夹即可打包带走的"口袋彭博"。

```
Project_Omega_2026/
├── 00_Config/               # [Config] API Keys, Ticker Mapping, Curve Config
├── 01_Data_Warehouse/       # [Data Layer] DuckDB + ETL Scripts
│   ├── db/                  # Database files
│   ├── raw_landing/         # Raw data staging
│   └── etl_scripts/         # Data ingestion & transformation
│
├── 03_Strategy_Lab/         # [Decision Layer] Strategy & Allocation Engine
│   ├── 2026_Investment_Plan/# Investment hypothesis & analysis
│   └── 2026_allocation_plan/# Core allocation engine (Pydantic + Streamlit)
│
├── 05_Dashboard_UI/         # [Presentation Layer] Streamlit Entry Point
│
├── CLAUDE.md                # [AI Kernel] LLM System Prompt
├── README.md                # Project Documentation
├── content.md               # Directory Structure Reference
└── requirements.txt         # Python Dependencies
```

## Strategic Framework (投资宪法)

### 1. 核心账户逻辑 (Accounting Regime)
* **AC / FVOCI 主导**：80% 持仓以净利息收入 (NII) 为考核目标
* **忽略 OCI 波动**：只要不发生信用减值，FVOCI 的市值波动不影响核心利润表
* **FVTPL 卫星仓位**：仅在此账户进行 Total Return 策略

### 2. 资金成本套利 (The FTP Lag Arbitrage)
* **规则**：当月 FTP (资金成本) = 上月 3M 国债均值
* **Alpha 来源**：
  * **加息周期 (Hiking)**：利用 FTP 滞后，在加息当月 aggressively 加仓
  * **降息周期 (Cutting)**：在降息前抢跑（Front-loading）

### 3. 流动性假设 (Liquidity Assumption)
- 假设资金端供应无限 (Unlimited Funding)，投资边界仅受制于：
  - Spread > 0 (相对于滞后 FTP)
  - RWA (风险加权资产) 约束
  - FX P&L Buffer (汇率折算安全垫)

## Key Modules

### A. Data Warehouse (`01_Data_Warehouse/`)
- 自动化清洗来自 Sharepoint (Excel) 和 Bloomberg AIM 的数据
- `yield_curve_loader.py`: 基于 Nelson-Siegel-Svensson 模型的收益率曲线插值
- 支持 DuckDB (SQL Interface) 和 Parquet (时序数据)

### B. 2026 Allocation Engine (`03_Strategy_Lab/2026_allocation_plan/`)
- `config.py`: Pydantic-based 配置管理（货币、账户、FTP 参数）
- `analytics.py`: 收益率曲面、FX 分析、FTP 计算器
- `allocation_engine.py`: 多币种 NII 优化逻辑
- `dashboard.py`: Streamlit 交互式情景分析

### C. Dashboard (`05_Dashboard_UI/`)
- Streamlit 入口，支持实时调节：
  - USD/AUD 投入规模 (Firepower)
  - 目标建仓收益率 (Entry Yield)
  - 汇率压力测试 (FX Stress Test)

## Getting Started

### 1. 环境配置
```bash
# Ensure Python 3.10+
pip install -r requirements.txt
```

### 2. 数据初始化 (ETL)
```bash
# Place raw position file in raw_landing/
cd 01_Data_Warehouse/etl_scripts
python init_db.py
```

### 3. 启动配置仪表盘
```bash
cd 03_Strategy_Lab/2026_allocation_plan
streamlit run dashboard.py
```

## AI Interaction Protocol

本项目集成了 AI 参谋模式。在与 Claude/ChatGPT 交互时：

1. **Context Injection**: 确保 AI 读取了根目录下的 `CLAUDE.md`
2. **Dual Persona**:
   - 询问市场时，AI 是 Macro CIO (关注二阶导数、流动性)
   - 询问职场/汇报时，AI 是 Political Strategist (关注合规、话术)
3. **Command Triggers**:
   - 输入 "复盘"：触发每日总结模式
   - 输入 "Stress Test"：触发情景分析代码生成

## Disclaimer

- **Institutional Use Only**: 本系统参数基于 $50B+ 机构资产负债表设定，不适用于个人零售投资者
- **Data Privacy**: 所有上传至 GitHub 的代码均已脱敏，不包含真实交易信息

---

**Project Omega** - Building the bridge between Math, Macro, and Management.

*Managed by the CIO Office*
