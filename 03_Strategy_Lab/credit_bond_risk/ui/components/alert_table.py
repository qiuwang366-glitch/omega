"""
Credit Bond Risk - Alert Table Component

Renders a table of risk alerts.
"""

import streamlit as st
import pandas as pd

from ...core.models import RiskAlert
from ...core.enums import Severity, AlertStatus
from .color_scheme import ColorScheme


def render_alert_table(
    alerts: list[RiskAlert],
    show_filters: bool = True,
    max_rows: int = 20,
) -> list[RiskAlert]:
    """
    Render an interactive alert table

    Args:
        alerts: List of alerts to display
        show_filters: Whether to show filter controls
        max_rows: Maximum rows to display

    Returns:
        Filtered list of alerts (for downstream use)
    """
    scheme = ColorScheme()

    if not alerts:
        st.info("暂无预警")
        return []

    # Filters
    filtered_alerts = alerts

    if show_filters:
        col1, col2, col3 = st.columns(3)

        with col1:
            severity_filter = st.multiselect(
                "严重程度",
                options=["CRITICAL", "WARNING", "INFO"],
                default=["CRITICAL", "WARNING"],
            )
            if severity_filter:
                filtered_alerts = [
                    a for a in filtered_alerts
                    if a.severity.value in severity_filter
                ]

        with col2:
            category_filter = st.multiselect(
                "类别",
                options=list(set(a.category.value for a in alerts)),
                default=None,
            )
            if category_filter:
                filtered_alerts = [
                    a for a in filtered_alerts
                    if a.category.value in category_filter
                ]

        with col3:
            status_filter = st.multiselect(
                "状态",
                options=["PENDING", "INVESTIGATING", "RESOLVED", "DISMISSED"],
                default=["PENDING", "INVESTIGATING"],
            )
            if status_filter:
                filtered_alerts = [
                    a for a in filtered_alerts
                    if a.status.value in status_filter
                ]

    # Sort by severity and timestamp
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    filtered_alerts.sort(
        key=lambda a: (severity_order.get(a.severity.value, 3), -a.timestamp.timestamp())
    )

    # Limit rows
    display_alerts = filtered_alerts[:max_rows]

    # Build table data
    table_data = []
    for alert in display_alerts:
        severity_color = ColorScheme.get_severity_color(alert.severity.value)
        severity_icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(
            alert.severity.value, "⚪"
        )

        table_data.append({
            "severity_icon": severity_icon,
            "时间": alert.timestamp.strftime("%m-%d %H:%M"),
            "发行人": alert.obligor_name,
            "类别": alert.category.value,
            "消息": alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
            "指标": f"{alert.metric_value:.2f}",
            "阈值": f"{alert.threshold:.2f}",
            "状态": alert.status.value,
            "alert_id": alert.alert_id,
        })

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Custom CSS for table
    st.markdown(f"""
    <style>
    .alert-table {{
        font-size: 14px;
    }}
    .alert-critical {{
        background-color: {scheme.severity_critical}20 !important;
    }}
    .alert-warning {{
        background-color: {scheme.severity_warning}20 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Render table
    if not df.empty:
        # Hide alert_id column in display
        display_columns = [c for c in df.columns if c != "alert_id"]

        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "severity_icon": st.column_config.TextColumn("", width=30),
                "时间": st.column_config.TextColumn("时间", width=100),
                "发行人": st.column_config.TextColumn("发行人", width=120),
                "类别": st.column_config.TextColumn("类别", width=100),
                "消息": st.column_config.TextColumn("消息", width=200),
                "指标": st.column_config.TextColumn("指标", width=80),
                "阈值": st.column_config.TextColumn("阈值", width=80),
                "状态": st.column_config.TextColumn("状态", width=100),
            },
        )

    # Summary stats
    st.caption(
        f"显示 {len(display_alerts)}/{len(filtered_alerts)} 条预警 | "
        f"🔴 {sum(1 for a in filtered_alerts if a.severity == Severity.CRITICAL)} 严重 | "
        f"🟡 {sum(1 for a in filtered_alerts if a.severity == Severity.WARNING)} 警告"
    )

    return filtered_alerts


def render_alert_detail(alert: RiskAlert) -> None:
    """
    Render detailed view of a single alert

    Args:
        alert: Alert to display
    """
    scheme = ColorScheme()
    severity_color = ColorScheme.get_severity_color(alert.severity.value)

    st.markdown(f"""
    <div style="
        background-color: {scheme.bg_secondary};
        border-left: 4px solid {severity_color};
        padding: 16px;
        border-radius: 0 8px 8px 0;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:18px;font-weight:600;color:{scheme.text_primary};">
                {alert.obligor_name}
            </span>
            <span style="
                background-color:{severity_color}20;
                color:{severity_color};
                padding:4px 12px;
                border-radius:4px;
            ">
                {alert.severity.value}
            </span>
        </div>
        <div style="color:{scheme.text_secondary};margin:8px 0;">
            {alert.category.value} · {alert.signal_name} · {alert.timestamp.strftime('%Y-%m-%d %H:%M')}
        </div>
        <div style="color:{scheme.text_primary};margin:12px 0;line-height:1.6;">
            {alert.message}
        </div>
        <div style="color:{scheme.text_muted};font-size:14px;">
            指标值: {alert.metric_value:.4f} | 阈值: {alert.threshold:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI Summary
    if alert.ai_summary:
        st.markdown("**AI 分析**")
        st.markdown(alert.ai_summary)

    # Related news
    if alert.related_news:
        st.markdown("**相关新闻**")
        for news_id in alert.related_news[:3]:
            st.text(f"• {news_id}")

    # Actions
    st.markdown("**处置操作**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("开始调查", key=f"investigate_{alert.alert_id}"):
            st.success("状态已更新为: 调查中")

    with col2:
        if st.button("标记已解决", key=f"resolve_{alert.alert_id}"):
            st.success("状态已更新为: 已解决")

    with col3:
        if st.button("忽略", key=f"dismiss_{alert.alert_id}"):
            st.success("状态已更新为: 已忽略")
