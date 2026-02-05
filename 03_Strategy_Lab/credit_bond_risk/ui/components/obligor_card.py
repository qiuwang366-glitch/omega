"""
Credit Bond Risk - Obligor Card Component

Renders a summary card for a single obligor.
"""

import streamlit as st

from ...core.models import Obligor, CreditExposure
from ...core.enums import Severity, RatingOutlook
from .color_scheme import ColorScheme


def render_obligor_card(
    obligor: Obligor,
    exposure: CreditExposure | None = None,
    alerts: list | None = None,
    expanded: bool = False,
) -> None:
    """
    Render an obligor summary card

    Args:
        obligor: Obligor data
        exposure: Optional exposure data
        alerts: Optional list of active alerts
        expanded: Whether to show expanded view
    """
    scheme = ColorScheme()

    # Determine card border color based on alerts
    if alerts:
        max_severity = max(a.severity for a in alerts)
        border_color = ColorScheme.get_severity_color(max_severity.value)
    else:
        border_color = scheme.bg_tertiary

    # Rating color
    rating_color = ColorScheme.get_rating_color(obligor.rating_internal.value)

    # Outlook indicator
    outlook_icons = {
        RatingOutlook.POSITIVE: "🔼",
        RatingOutlook.STABLE: "➖",
        RatingOutlook.NEGATIVE: "🔽",
        RatingOutlook.WATCH_POS: "👁️🔼",
        RatingOutlook.WATCH_NEG: "👁️🔽",
        RatingOutlook.DEVELOPING: "🔄",
    }
    outlook_icon = outlook_icons.get(obligor.rating_outlook, "➖")

    # Card CSS
    card_style = f"""
    <style>
    .obligor-card {{
        background-color: {scheme.bg_secondary};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }}
    .obligor-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}
    .obligor-name {{
        font-size: 18px;
        font-weight: 600;
        color: {scheme.text_primary};
    }}
    .obligor-rating {{
        background-color: {rating_color}20;
        color: {rating_color};
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
    }}
    .obligor-meta {{
        color: {scheme.text_secondary};
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .obligor-metrics {{
        display: flex;
        gap: 24px;
        margin-top: 12px;
    }}
    .metric {{
        text-align: center;
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 600;
        color: {scheme.text_primary};
    }}
    .metric-label {{
        font-size: 12px;
        color: {scheme.text_muted};
    }}
    </style>
    """

    # Build metrics section
    metrics_html = ""
    if exposure:
        metrics_html = f"""
        <div class="obligor-metrics">
            <div class="metric">
                <div class="metric-value">${exposure.total_market_usd/1e6:.1f}M</div>
                <div class="metric-label">市值</div>
            </div>
            <div class="metric">
                <div class="metric-value">{exposure.pct_of_aum:.2%}</div>
                <div class="metric-label">占比</div>
            </div>
            <div class="metric">
                <div class="metric-value">{exposure.weighted_avg_duration:.1f}Y</div>
                <div class="metric-label">久期</div>
            </div>
            <div class="metric">
                <div class="metric-value">{exposure.weighted_avg_oas:.0f}bp</div>
                <div class="metric-label">OAS</div>
            </div>
        </div>
        """

    # Alert badges
    alert_html = ""
    if alerts:
        alert_badges = []
        for alert in alerts[:3]:
            color = ColorScheme.get_severity_color(alert.severity.value)
            alert_badges.append(
                f'<span style="background:{color}20;color:{color};'
                f'padding:2px 8px;border-radius:4px;font-size:12px;">'
                f'{alert.category.value}</span>'
            )
        alert_html = f'<div style="margin-top:8px;">{"  ".join(alert_badges)}</div>'

    # Render card
    card_html = f"""
    {card_style}
    <div class="obligor-card">
        <div class="obligor-header">
            <span class="obligor-name">{obligor.name_cn}</span>
            <span class="obligor-rating">{obligor.rating_internal.value} {outlook_icon}</span>
        </div>
        <div class="obligor-meta">
            {obligor.sector.value} · {obligor.sub_sector} · {obligor.province or 'N/A'}
        </div>
        {metrics_html}
        {alert_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    # Expanded view
    if expanded and exposure:
        with st.expander("详细信息", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**持仓明细**")
                for bond in exposure.bonds[:5]:
                    st.text(f"• {bond.isin}: ${bond.market_value_usd/1e6:.1f}M, {bond.years_to_maturity:.1f}Y")

            with col2:
                st.markdown("**到期分布**")
                for bucket, value in exposure.maturity_profile.items():
                    if value > 0:
                        st.text(f"• {bucket}: ${value/1e6:.1f}M")
