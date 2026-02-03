import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. 配置与映射 (Configuration)
# ==========================================
DB_PATH = 'portfolio.duckdb'
CSV_PATH = 'position20251231.csv'
SNAPSHOT_DATE = '2025-12-31'  # 基于文件名

# 会计分类映射表
ACC_MAPPING = {
    'HTM': 'AC',
    'AFS': 'FVOCI',
    'HFT': 'FVTPL',
    'Trading': 'FVTPL'
}

# ==========================================
# 2. 清洗函数 (Cleaning Utils)
# ==========================================
def clean_currency_str(x):
    """处理 ' 40,000,000.00 ' 这种带空格和逗号的数字"""
    if pd.isna(x) or str(x).strip() == '-':
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    # 去除空格和逗号
    clean_str = str(x).replace(',', '').strip()
    try:
        return float(clean_str)
    except:
        return 0.0

def parse_date(x):
    """统一日期格式"""
    if pd.isna(x): return None
    try:
        # 尝试 DD/MM/YYYY
        return pd.to_datetime(x, dayfirst=True).date()
    except:
        return None

# ==========================================
# 3. 数据加载与清洗 (Load & Clean)
# ==========================================
print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# 重命名核心列 (Mapping Analyst Headers -> System Headers)
col_map = {
    'ISIN': 'isin',
    'TICKER': 'ticker',
    '债券名称': 'security_name',
    '分类1': 'sector_l1',
    '分类2': 'sector_l2',
    'CCY': 'currency',
    'Maturity': 'maturity_date',
    'Portfolio': 'portfolio_id',
    'AccSection': 'accounting_raw',
    'Nominal（USD）': 'nominal_usd',
    'Nominal（原币）': 'nominal_local',
    '摊余成本（USD）': 'book_value_usd',
    'Maket Value（USD）': 'market_value_usd',
    'EffectiveYield': 'yield_effective',
    'FTP Rate': 'ftp_rate',
    'Duration': 'duration_modified'
}
df.rename(columns=col_map, inplace=True)

# 应用清洗逻辑
num_cols = ['nominal_usd', 'nominal_local', 'book_value_usd', 'market_value_usd', 'ftp_rate']
for col in num_cols:
    df[col] = df[col].apply(clean_currency_str)

df['snapshot_date'] = pd.to_datetime(SNAPSHOT_DATE).date()
df['accounting_type'] = df['accounting_raw'].map(ACC_MAPPING).fillna('OTHER')

# ==========================================
# 4. 入库 (DuckDB Execution)
# ==========================================
conn = duckdb.connect(DB_PATH)

# A. 创建 Security Master (如果不存在)
conn.execute("""
CREATE TABLE IF NOT EXISTS security_master (
    isin VARCHAR PRIMARY KEY,
    ticker VARCHAR,
    name VARCHAR,
    sector_l1 VARCHAR,
    sector_l2 VARCHAR,
    currency VARCHAR(3),
    maturity_date DATE,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# B. 创建 Positions 表
conn.execute("""
CREATE TABLE IF NOT EXISTS positions_daily (
    snapshot_date DATE,
    isin VARCHAR,
    portfolio_id VARCHAR,
    accounting_type VARCHAR,
    nominal_usd DOUBLE,
    nominal_local DOUBLE,
    book_value_usd DOUBLE,
    market_value_usd DOUBLE,
    yield_effective DOUBLE,
    ftp_rate DOUBLE,
    duration DOUBLE,
    PRIMARY KEY (snapshot_date, isin, portfolio_id)
);
""")

# C. 写入数据 (使用 SQL 也就是 DuckDB 的强项)
# 1. 更新 Master (这里用 INSERT OR IGNORE 简单处理，实际可用 Upsert)
print("Updating Security Master...")
conn.execute("""
INSERT OR IGNORE INTO security_master 
SELECT DISTINCT 
    isin, ticker, security_name, sector_l1, sector_l2, currency, 
    strptime(maturity_date, '%d/%m/%Y') as maturity_date, -- 假设 CSV 是 DD/MM/YYYY
    CURRENT_TIMESTAMP
FROM df
WHERE isin IS NOT NULL
""")

# 2. 写入 Position
print("Inserting Positions...")
conn.execute("""
INSERT INTO positions_daily 
SELECT 
    snapshot_date, isin, portfolio_id, accounting_type,
    nominal_usd, nominal_local, book_value_usd, market_value_usd,
    yield_effective, ftp_rate, duration_modified
FROM df
WHERE isin IS NOT NULL
""")

# 验证
res = conn.execute("SELECT accounting_type, SUM(nominal_usd)/1e8 as total_bn FROM positions_daily GROUP BY 1").df()
print("\nValidation (AUM by Accounting Type):")
print(res)

conn.close()
print(f"\n✅ Database initialized at {DB_PATH}")
