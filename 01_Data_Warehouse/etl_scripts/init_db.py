import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 0. Config
# ==========================================
DB_PATH = '../db/portfolio.duckdb'
CSV_PATH = '../raw_landing/position20251231.csv'  # 假设你把 CSV 放在 raw_landing
SNAPSHOT_DATE = '2025-12-31'

# ==========================================
# 1. Logic Helpers
# ==========================================
def clean_num(x):
    """Excel 字符串转浮点数"""
    if pd.isna(x) or str(x).strip() in ['-', '']: return 0.0
    if isinstance(x, (int, float)): return float(x)
    return float(str(x).replace(',', '').replace(' ', ''))

def get_carry_base(row):
    """
    核心业务逻辑：
    - AC/FVOCI: 基于摊余成本 (Book Value) 算 Carry
    - FVTPL: 基于市值 (Market Value) 算 Carry (如有不同可调整)
    """
    if row['accounting_type'] in ['AC', 'FVOCI']:
        return row['book_value_usd']
    else:
        # FVTPL 默认 fallback 到市值，如果市值为空则用 Book Value
        return row['market_value_usd'] if row['market_value_usd'] != 0 else row['book_value_usd']

# ==========================================
# 2. ETL Process
# ==========================================
print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# A. 重命名与清洗
col_map = {
    'ISIN': 'isin', 'TICKER': 'ticker', '债券名称': 'security_name',
    '分类1': 'sector_l1', '分类2': 'sector_l2', 'CCY': 'currency',
    'AccSection': 'accounting_raw', 'Portfolio': 'portfolio_id',
    'Nominal（USD）': 'nominal_usd', 'Nominal（原币）': 'nominal_local',
    '摊余成本（USD）': 'book_value_usd', 'Maket Value（USD）': 'market_value_usd',
    'EffectiveYield': 'yield_effective', 'FTP Rate': 'ftp_rate',
    'Duration': 'duration_modified', 'Maturity': 'maturity_date'
}
df.rename(columns=col_map, inplace=True)

# 清洗数值
num_cols = ['nominal_usd', 'nominal_local', 'book_value_usd', 'market_value_usd', 
            'yield_effective', 'ftp_rate', 'duration_modified']
for col in num_cols:
    df[col] = df[col].apply(clean_num)

# B. 业务逻辑计算
# B1. 会计分类
acc_map = {'HTM': 'AC', 'AFS': 'FVOCI', 'HFT': 'FVTPL', 'Trading': 'FVTPL'}
df['accounting_type'] = df['accounting_raw'].map(acc_map).fillna('OTHER')

# B2. 汇率 (FX Rate)
# 逻辑：如果 Currency 是 USD，汇率为 1。否则用 Nominal(USD)/Nominal(Local)
df['fx_rate'] = df.apply(
    lambda x: 1.0 if x['currency'] == 'USD' else 
    (x['nominal_usd'] / x['nominal_local'] if x['nominal_local'] != 0 else 0.0), 
    axis=1
)

# B3. Carry 计算
df['carry_base_usd'] = df.apply(get_carry_base, axis=1)
# Annualized Carry = Base * Yield%. (Yield is usually percentage like 5.5, so divide by 100)
df['carry_annual_usd'] = df['carry_base_usd'] * (df['yield_effective'] / 100.0)
df['carry_daily_usd'] = df['carry_annual_usd'] / 365.0

# 日期处理
df['snapshot_date'] = pd.to_datetime(SNAPSHOT_DATE).date()
# 处理 Excel 日期格式可能不一致的问题 (DD/MM/YYYY vs YYYY-MM-DD)
df['maturity_date'] = pd.to_datetime(df['maturity_date'], dayfirst=True, errors='coerce').dt.date

# ==========================================
# 3. DuckDB Operations
# ==========================================
conn = duckdb.connect(DB_PATH)

# 建表：Security Master (保持不变)
conn.execute("""
CREATE TABLE IF NOT EXISTS security_master (
    isin VARCHAR PRIMARY KEY,
    ticker VARCHAR,
    name VARCHAR,
    currency VARCHAR(3),
    sector_l1 VARCHAR,
    sector_l2 VARCHAR,
    maturity_date DATE,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# 建表：Positions (增加 FX 和 Carry 字段)
conn.execute("""
CREATE TABLE IF NOT EXISTS positions_daily (
    snapshot_date DATE,
    isin VARCHAR,
    portfolio_id VARCHAR,
    accounting_type VARCHAR,
    
    -- 核心敞口
    nominal_usd DOUBLE,
    nominal_local DOUBLE,
    book_value_usd DOUBLE,
    market_value_usd DOUBLE,
    
    -- 价格与收益率
    fx_rate DOUBLE,
    yield_effective DOUBLE,
    ftp_rate DOUBLE,
    duration DOUBLE,
    
    -- PnL Metrics
    carry_base_usd DOUBLE,      -- 用于计算 Carry 的基数 (AC=Book, FVTPL=MV)
    carry_annual_usd DOUBLE,    -- 年化 Carry
    
    PRIMARY KEY (snapshot_date, isin, portfolio_id)
);
""")

print("Updating Database...")

# 1. 更新 Master
conn.execute("""
INSERT OR IGNORE INTO security_master 
SELECT DISTINCT 
    isin, ticker, security_name, currency, sector_l1, sector_l2, maturity_date, CURRENT_TIMESTAMP
FROM df WHERE isin IS NOT NULL
""")

# 2. 插入持仓 (先删后插，防止重复运行报错)
conn.execute(f"DELETE FROM positions_daily WHERE snapshot_date = '{SNAPSHOT_DATE}'")
conn.execute("""
INSERT INTO positions_daily 
SELECT 
    snapshot_date, isin, portfolio_id, accounting_type,
    nominal_usd, nominal_local, book_value_usd, market_value_usd,
    fx_rate, yield_effective, ftp_rate, duration_modified,
    carry_base_usd, carry_annual_usd
FROM df WHERE isin IS NOT NULL
""")

# 验证输出
print("\nValidation Report (Top 3 FX Rates):")
print(conn.execute("SELECT currency, fx_rate, COUNT(*) as cnt FROM positions_daily WHERE currency != 'USD' GROUP BY 1,2 ORDER BY 3 DESC LIMIT 3").df())

print("\nValidation Report (Carry by Type):")
print(conn.execute("SELECT accounting_type, SUM(carry_annual_usd)/1e6 as total_carry_mm FROM positions_daily GROUP BY 1").df())

conn.close()
print("✅ Done.")
