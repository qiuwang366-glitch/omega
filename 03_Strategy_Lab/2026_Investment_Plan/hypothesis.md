# 2026 Strategic Hypothesis: "The Unconstrained Yield Grab"
> Status: Draft | Reviewer: CIO | Base Date: 2025-12-31
> **New Constraints**: Unlimited Funding | Spot FX P&L Recognition

## 1. Executive Summary
We are shifting from a "Reinvestment" mindset to a **"Capacity & Carry Maximization"** mindset.
* **The Paradigm Shift**: With unlimited funding and Fixed FTP, the only constraint is **Market Depth** and **P&L Volatility**.
* **The Core Objective**: 
    * **USD SSA**: **Maximal Expansion**. Aggressively front-load duration before Fed cuts. Target Yield > FTP + 30bps.
    * **AUD Rates**: **Opportunistic Carry with FX Buffer**. Since NII is converted at Spot, we must demand a higher nominal yield to buffer against AUD depreciation risks.

---

## 2. Portfolio Specific Strategy

### A. USD SSA (The "Infinite" Carry Machine)
* **Strategy**: **Aggressive Front-Loading**.
* **Logic**: 
    * Since we don't need to wait for maturities to generate cash, we should **buy the 5Y-7Y sector NOW** while yields are still elevated (pre-Fed pivot).
    * **Target Size**: Limited only by Tier 1 issuer supply. Aim to double the SSA book if spreads remain attractive.
* **The Trade**:
    * **Curve Location**: Overweight **7Y**. This point offers the best roll-down and convexity protection if rates drop.
    * **Supply Absorption**: Be the "Anchor Investor" in new primary issuances from IBRD/EIB to demand better allocations.

### B. AUD Rates (The FX-Sensitive Yield)
* **Strategy**: **Conditional Expansion based on "FX-Adjusted Carry"**.
* **The Accounting Trap**: 
    * KPI = `AUD_Interest_Income * Spot_FX_Rate`.
    * *Risk*: If AUD drops 10% against USD, our reported NII drops 10%, effectively crushing our spread over USD funding cost.
* **The "Breakeven" Rule**: 
    * We only expand AUD exposure if: 
      $$(AUD\_Yield \times E[FX_{2026}]) > (USD\_FTP) + Risk\_Premium$$
* **Execution Logic**:
    * **Bull Case (AUD Rally)**: Unhedged ACGBs are the "Double Alpha" (Higher Yield + FX Gain).
    * **Bear Case (AUD Depr)**: We must use **Cross-Currency Swaps (CCS)** to turn AUD assets into synthetic USD assets, locking in the FX rate for income calculation purposes.

---

## 3. Quantitative Signals & Triggers

### 3.1 The "Green Light" Matrix (USD SSA)
Since funding is unlimited, we buy whenever this condition is met:
* **Condition**: `Asset_Yield > (Fixed_FTP + 20bps)` AND `ASW_Spread > Historic_Median`.
* **Action**: **Immediate Buy**. Do not time the absolute bottom. Volume is the priority.

### 3.2 The FX P&L Buffer (AUD Rates)
* **Metric**: **"FX-Implied Yield Drag"**
* **Scenario**:
    * If we budget AUD/USD = 0.65, but it drops to 0.60, our income drops 7.7%.
    * *Requirement*: The AUD Nominal Yield must be at least **50bps higher** than equivalent USD assets to compensate for this translation volatility risk (unless we hedge).

---

## 4. Scenario Stress Tests (P&L Impact)

* **Scenario A: Fed Cuts Aggressively, AUD Rallies**
    * *USD Book*: Massive Capital Gains (Duration play).
    * *AUD Book*: **"The Golden Scenario"**. High nominal yield + FX Translation Gain boosting the KPI.
    * *Action*: Maximize unhedged AUD duration.

* **Scenario B: Strong USD (Trump Trade / Risk Off)**
    * *USD Book*: Carry is stable (locked spread).
    * *AUD Book*: **"The Income Trap"**. AUD yield stays constant, but converted USD income plummets.
    * *Mitigation*: **Portfolio B must have a CCS Overlay**. If AUD/USD breaks below 0.62, strictly swap all coupon flows into USD.

---

## 5. Execution Roadmap (Q1 2026)
1.  **"Big Bang" Deployment (USD)**: Deploy $2B+ into USD SSA 5-10Y in Jan/Feb to lock pre-cut yields. Do not wait for maturities.
2.  **AUD "Step-In"**:
    * Buy ACGBs **only on dip**.
    * Set up a **Coupon Swap** facility to hedge the P&L translation risk if needed.
Update Content:

### Tactical Execution (The "Lag" Trade):
* AUD (The Hike Call): If RBA hikes in Feb, deploy maximum capital in Feb Spot. The spread is artificially widened by the lag.
* USD (The Cut Defense): Stop buying immediately after the first Fed cut is priced in, wait for FTP to reset lower next month. Do not buy in the "Gap Month".

> *Signed: CIO Office / Unconstrained Mandate*
