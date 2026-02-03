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
