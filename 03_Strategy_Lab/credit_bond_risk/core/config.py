"""
Credit Bond Risk - Configuration

Pydantic-based configuration for the credit risk monitoring system.
All thresholds, model settings, and feature flags are defined here.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ConcentrationConfig(BaseModel):
    """集中度监控配置"""

    # 单一发行人限额
    single_obligor_warning: float = Field(
        0.02, description="单一发行人占比警告阈值 (2%)"
    )
    single_obligor_critical: float = Field(
        0.05, description="单一发行人占比严重阈值 (5%)"
    )

    # Top N 集中度
    top_n: int = Field(10, description="监控Top N发行人")
    top_n_warning: float = Field(
        0.40, description="Top N合计占比警告阈值 (40%)"
    )
    top_n_critical: float = Field(
        0.60, description="Top N合计占比严重阈值 (60%)"
    )

    # 行业集中度
    sector_warning: float = Field(
        0.30, description="单一行业占比警告阈值 (30%)"
    )
    sector_critical: float = Field(
        0.50, description="单一行业占比严重阈值 (50%)"
    )

    # 地区集中度
    province_warning: float = Field(
        0.20, description="单一省份占比警告阈值 (20%)"
    )
    province_critical: float = Field(
        0.35, description="单一省份占比严重阈值 (35%)"
    )

    # HHI指数
    hhi_warning: float = Field(0.10, description="HHI警告阈值")
    hhi_critical: float = Field(0.18, description="HHI严重阈值")


class SpreadConfig(BaseModel):
    """利差监控配置"""

    # 历史百分位
    percentile_warning: float = Field(
        0.85, description="OAS历史百分位警告阈值 (85%)"
    )
    percentile_critical: float = Field(
        0.95, description="OAS历史百分位严重阈值 (95%)"
    )

    # 绝对值变动
    daily_change_warning: int = Field(
        20, description="日变动警告阈值 (bps)"
    )
    daily_change_critical: int = Field(
        50, description="日变动严重阈值 (bps)"
    )
    weekly_change_warning: int = Field(
        30, description="周变动警告阈值 (bps)"
    )
    weekly_change_critical: int = Field(
        80, description="周变动严重阈值 (bps)"
    )

    # Z-score
    zscore_warning: float = Field(2.0, description="Z-score警告阈值")
    zscore_critical: float = Field(3.0, description="Z-score严重阈值")

    # 历史窗口
    lookback_days: int = Field(252, description="历史回看天数 (1年)")
    min_data_points: int = Field(60, description="最少数据点数")


class RatingConfig(BaseModel):
    """评级监控配置"""

    # 评级变动
    downgrade_notches_warning: int = Field(
        1, description="评级下调档数警告阈值"
    )
    downgrade_notches_critical: int = Field(
        2, description="评级下调档数严重阈值"
    )

    # 展望变动
    outlook_negative_warning: bool = Field(
        True, description="展望转负面是否触发警告"
    )
    watch_negative_critical: bool = Field(
        True, description="列入观察(负面)是否触发严重"
    )

    # 监控窗口
    lookback_days: int = Field(30, description="评级变动回看天数")


class NewsConfig(BaseModel):
    """舆情监控配置"""

    # 情感阈值
    sentiment_warning: float = Field(
        -0.3, description="情感分数警告阈值"
    )
    sentiment_critical: float = Field(
        -0.6, description="情感分数严重阈值"
    )

    # 负面新闻数量
    negative_count_warning: int = Field(
        3, description="负面新闻数量警告阈值"
    )
    negative_count_critical: int = Field(
        5, description="负面新闻数量严重阈值"
    )

    # 时间窗口
    lookback_days: int = Field(7, description="新闻回看天数")

    # 来源权重
    source_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "bloomberg": 1.2,
            "company": 1.5,  # 公司公告权重最高
            "cls": 1.0,
            "eastmoney": 0.8,
            "other": 0.5,
        },
        description="新闻来源权重"
    )


class MaturityConfig(BaseModel):
    """到期压力监控配置"""

    # 短期到期占比
    short_term_months: int = Field(12, description="短期定义(月)")
    short_term_warning: float = Field(
        0.25, description="短期到期占比警告阈值 (25%)"
    )
    short_term_critical: float = Field(
        0.40, description="短期到期占比严重阈值 (40%)"
    )

    # 再融资压力
    refinancing_coverage_warning: float = Field(
        1.5, description="现金/短期债务警告阈值"
    )
    refinancing_coverage_critical: float = Field(
        1.0, description="现金/短期债务严重阈值"
    )


class LLMConfig(BaseModel):
    """LLM配置"""

    # 模型选择
    model_primary: str = Field(
        "claude-sonnet-4-20250514",
        description="主力模型 (摘要/分析)"
    )
    model_fast: str = Field(
        "claude-3-5-haiku-20241022",
        description="快速模型 (批量处理)"
    )
    model_embedding: str = Field(
        "text-embedding-3-small",
        description="Embedding模型"
    )

    # 参数
    max_tokens_summary: int = Field(500, description="摘要最大token")
    max_tokens_analysis: int = Field(2000, description="分析最大token")
    temperature: float = Field(0.3, description="生成温度")

    # 缓存
    cache_ttl_hours: int = Field(24, description="LLM缓存有效期(小时)")

    # 成本控制
    daily_budget_usd: float = Field(10.0, description="每日预算(USD)")
    batch_size: int = Field(10, description="批处理大小")


class VectorStoreConfig(BaseModel):
    """向量存储配置"""

    # 存储类型
    store_type: Literal["duckdb_vss", "chroma", "memory"] = Field(
        "duckdb_vss", description="向量存储类型"
    )

    # DuckDB VSS配置
    db_path: str = Field(
        "data/credit_vectors.duckdb",
        description="DuckDB文件路径"
    )

    # ChromaDB配置
    chroma_persist_dir: str = Field(
        "data/chroma",
        description="ChromaDB持久化目录"
    )
    chroma_collection: str = Field(
        "credit_news",
        description="ChromaDB collection名称"
    )

    # 检索参数
    similarity_top_k: int = Field(10, description="相似性检索数量")
    similarity_threshold: float = Field(0.7, description="相似度阈值")


class DataSourceConfig(BaseModel):
    """数据源配置"""

    # 新闻源
    news_sources: list[str] = Field(
        default_factory=lambda: ["cls", "eastmoney", "company"],
        description="启用的新闻源"
    )

    # Bloomberg
    bbg_enabled: bool = Field(False, description="是否启用Bloomberg")

    # 内部数据库
    portfolio_db_path: str = Field(
        "01_Data_Warehouse/db/portfolio.duckdb",
        description="组合数据库路径"
    )

    # 刷新频率
    news_sync_interval_minutes: int = Field(30, description="新闻同步间隔(分钟)")
    market_data_interval_minutes: int = Field(60, description="行情同步间隔(分钟)")


class AlertConfig(BaseModel):
    """预警配置"""

    # 通知渠道
    notify_email: bool = Field(True, description="邮件通知")
    notify_wechat: bool = Field(False, description="企业微信通知")

    # 通知规则
    email_on_critical: bool = Field(True, description="严重预警发邮件")
    email_on_warning: bool = Field(False, description="警告预警发邮件")
    digest_hour: int = Field(8, description="每日汇总发送时间")

    # 静默规则
    silence_resolved_hours: int = Field(24, description="已解决预警静默时间")
    max_alerts_per_obligor: int = Field(5, description="同一发行人最大预警数")


class CreditRiskConfig(BaseModel):
    """信用风险监控主配置"""

    # 子配置
    concentration: ConcentrationConfig = Field(
        default_factory=ConcentrationConfig
    )
    spread: SpreadConfig = Field(default_factory=SpreadConfig)
    rating: RatingConfig = Field(default_factory=RatingConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    maturity: MaturityConfig = Field(default_factory=MaturityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)

    # 全局设置
    total_aum_usd: float = Field(
        50_000_000_000,  # $50B
        description="总AUM (USD)"
    )
    base_currency: str = Field("USD", description="基准货币")

    # 功能开关
    enable_llm_analysis: bool = Field(True, description="启用LLM分析")
    enable_embedding_search: bool = Field(True, description="启用向量搜索")
    enable_anomaly_detection: bool = Field(True, description="启用异常检测")

    # 日志
    log_level: str = Field("INFO", description="日志级别")

    class Config:
        json_schema_extra = {
            "example": {
                "total_aum_usd": 50_000_000_000,
                "enable_llm_analysis": True,
                "concentration": {
                    "single_obligor_warning": 0.02,
                    "single_obligor_critical": 0.05,
                },
            }
        }


def get_default_config() -> CreditRiskConfig:
    """获取默认配置"""
    return CreditRiskConfig()


def load_config_from_file(path: str) -> CreditRiskConfig:
    """从JSON文件加载配置"""
    import json
    from pathlib import Path

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        return CreditRiskConfig.model_validate(data)
    return get_default_config()


def save_config_to_file(config: CreditRiskConfig, path: str) -> None:
    """保存配置到JSON文件"""
    from pathlib import Path

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config.model_dump_json(indent=2))
