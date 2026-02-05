"""
Credit Bond Risk - Chart Components

Plotly charts for credit risk visualization.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

from ...core.models import CreditExposure, Obligor
from ...core.enums import Sector, CreditRating
from .color_scheme import ColorScheme, get_premium_layout


def create_concentration_chart(
    exposures: list[CreditExposure],
    top_n: int = 15,
    chart_type: str = "bar",
) -> go.Figure:
    """
    Create concentration chart (top obligors by exposure)

    Args:
        exposures: List of credit exposures
        top_n: Number of top obligors to show
        chart_type: "bar" or "treemap"

    Returns:
        Plotly figure
    """
    scheme = ColorScheme()

    # Sort by market value
    sorted_exposures = sorted(
        exposures, key=lambda x: x.total_market_usd, reverse=True
    )[:top_n]

    names = [e.obligor.name_cn for e in sorted_exposures]
    values = [e.total_market_usd / 1e6 for e in sorted_exposures]
    pcts = [e.pct_of_aum for e in sorted_exposures]
    sectors = [e.obligor.sector.value for e in sorted_exposures]

    colors = [ColorScheme.get_sector_color(s) for s in sectors]

    if chart_type == "bar":
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"${v:.0f}M ({p:.1%})" for v, p in zip(values, pcts)],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>市值: $%{x:.0f}M<br>占比: %{customdata:.2%}<extra></extra>",
            customdata=pcts,
        ))

        fig.update_layout(
            **get_premium_layout("Top发行人持仓", height=max(400, top_n * 30)),
            yaxis=dict(
                autorange="reversed",
                tickfont={"size": 12},
            ),
            xaxis=dict(title="市值 (百万美元)"),
            showlegend=False,
        )

    else:  # treemap
        fig = px.treemap(
            names=names,
            parents=[""] * len(names),
            values=values,
            color=sectors,
            color_discrete_map={
                "LGFV": scheme.sector_lgfv,
                "SOE": scheme.sector_soe,
                "FINANCIAL": scheme.sector_financial,
                "CORP": scheme.sector_corp,
            },
        )
        fig.update_layout(**get_premium_layout("持仓分布", height=500))

    return fig


def create_spread_history_chart(
    spread_history: dict[str, float],
    obligor_name: str = "",
    current_value: float | None = None,
    warning_threshold: float | None = None,
    critical_threshold: float | None = None,
) -> go.Figure:
    """
    Create spread history line chart

    Args:
        spread_history: {date_str: oas_value}
        obligor_name: Name for title
        current_value: Current OAS to highlight
        warning_threshold: Warning level line
        critical_threshold: Critical level line

    Returns:
        Plotly figure
    """
    scheme = ColorScheme()

    # Sort by date
    sorted_data = sorted(spread_history.items(), key=lambda x: x[0])
    dates = [datetime.fromisoformat(d) for d, _ in sorted_data]
    values = [v for _, v in sorted_data]

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="lines",
        name="OAS",
        line=dict(color=scheme.accent_blue, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>OAS: %{y:.0f}bp<extra></extra>",
    ))

    # Thresholds
    if warning_threshold and dates:
        fig.add_hline(
            y=warning_threshold,
            line_dash="dash",
            line_color=scheme.severity_warning,
            annotation_text="Warning",
            annotation_position="right",
        )

    if critical_threshold and dates:
        fig.add_hline(
            y=critical_threshold,
            line_dash="dash",
            line_color=scheme.severity_critical,
            annotation_text="Critical",
            annotation_position="right",
        )

    # Current value marker
    if current_value and dates:
        fig.add_trace(go.Scatter(
            x=[dates[-1]],
            y=[current_value],
            mode="markers",
            marker=dict(size=12, color=scheme.accent_orange),
            name="当前值",
            hovertemplate="当前: %{y:.0f}bp<extra></extra>",
        ))

    title = f"{obligor_name} OAS历史" if obligor_name else "OAS历史"
    fig.update_layout(
        **get_premium_layout(title, height=350),
        xaxis=dict(title="日期"),
        yaxis=dict(title="OAS (bps)"),
    )

    return fig


def create_rating_distribution_chart(
    exposures: list[CreditExposure],
) -> go.Figure:
    """
    Create rating distribution pie chart

    Args:
        exposures: List of credit exposures

    Returns:
        Plotly figure
    """
    scheme = ColorScheme()

    # Aggregate by rating
    rating_totals: dict[str, float] = {}
    for exp in exposures:
        rating = exp.obligor.rating_internal.value
        rating_totals[rating] = rating_totals.get(rating, 0) + exp.total_market_usd

    # Sort by rating quality (AAA first)
    rating_order = [r.value for r in CreditRating]
    sorted_ratings = sorted(
        rating_totals.items(),
        key=lambda x: rating_order.index(x[0]) if x[0] in rating_order else 999
    )

    labels = [r for r, _ in sorted_ratings]
    values = [v / 1e6 for _, v in sorted_ratings]
    colors = [ColorScheme.get_rating_color(r) for r in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>$%{value:.0f}M<br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        **get_premium_layout("评级分布", height=400),
        showlegend=False,
        annotations=[{
            "text": f"${sum(values):.0f}M",
            "showarrow": False,
            "font": {"size": 20, "color": scheme.text_primary},
        }],
    )

    return fig


def create_maturity_profile_chart(
    exposures: list[CreditExposure],
    by_rating: bool = False,
) -> go.Figure:
    """
    Create maturity profile stacked bar chart

    Args:
        exposures: List of credit exposures
        by_rating: Whether to color by rating

    Returns:
        Plotly figure
    """
    scheme = ColorScheme()

    # Aggregate maturity profiles
    buckets = ["0-1Y", "1-3Y", "3-5Y", "5-10Y", "10Y+"]
    bucket_totals = {b: 0 for b in buckets}

    if by_rating:
        # Separate by rating
        rating_buckets: dict[str, dict[str, float]] = {}
        for exp in exposures:
            rating = exp.obligor.rating_internal.value
            if rating not in rating_buckets:
                rating_buckets[rating] = {b: 0 for b in buckets}
            for bucket, value in exp.maturity_profile.items():
                if bucket in buckets:
                    rating_buckets[rating][bucket] += value

        fig = go.Figure()
        for rating, data in rating_buckets.items():
            values = [data[b] / 1e6 for b in buckets]
            fig.add_trace(go.Bar(
                name=rating,
                x=buckets,
                y=values,
                marker_color=ColorScheme.get_rating_color(rating),
                hovertemplate=f"<b>{rating}</b><br>%{{x}}: $%{{y:.0f}}M<extra></extra>",
            ))
        fig.update_layout(barmode="stack")

    else:
        # Simple total
        for exp in exposures:
            for bucket, value in exp.maturity_profile.items():
                if bucket in buckets:
                    bucket_totals[bucket] += value

        values = [bucket_totals[b] / 1e6 for b in buckets]

        fig = go.Figure(data=[go.Bar(
            x=buckets,
            y=values,
            marker_color=[
                scheme.accent_green,
                scheme.accent_blue,
                scheme.accent_purple,
                scheme.accent_orange,
                scheme.accent_red,
            ],
            text=[f"${v:.0f}M" for v in values],
            textposition="outside",
            hovertemplate="%{x}: $%{y:.0f}M<extra></extra>",
        )])

    fig.update_layout(
        **get_premium_layout("到期分布", height=350),
        xaxis=dict(title="到期期限"),
        yaxis=dict(title="面值 (百万美元)"),
    )

    return fig


def create_sector_concentration_chart(
    exposures: list[CreditExposure],
) -> go.Figure:
    """
    Create sector concentration donut chart

    Args:
        exposures: List of credit exposures

    Returns:
        Plotly figure
    """
    scheme = ColorScheme()

    # Aggregate by sector
    sector_totals: dict[str, float] = {}
    for exp in exposures:
        sector = exp.obligor.sector.value
        sector_totals[sector] = sector_totals.get(sector, 0) + exp.total_market_usd

    labels = list(sector_totals.keys())
    values = [v / 1e6 for v in sector_totals.values()]
    colors = [ColorScheme.get_sector_color(s) for s in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>$%{value:.0f}M<br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        **get_premium_layout("行业分布", height=350),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def create_risk_heatmap(
    exposures: list[CreditExposure],
) -> go.Figure:
    """
    Create risk heatmap (rating vs duration)

    Args:
        exposures: List of credit exposures

    Returns:
        Plotly figure
    """
    scheme = ColorScheme()

    # Define buckets
    rating_buckets = ["AAA/AA", "A", "BBB", "BB/B"]
    duration_buckets = ["0-2Y", "2-5Y", "5-10Y", "10Y+"]

    # Initialize matrix
    matrix = np.zeros((len(rating_buckets), len(duration_buckets)))

    # Populate matrix
    for exp in exposures:
        # Rating bucket
        rating = exp.obligor.rating_internal.value
        if "AAA" in rating or "AA" in rating:
            r_idx = 0
        elif rating.startswith("A"):
            r_idx = 1
        elif "BBB" in rating:
            r_idx = 2
        else:
            r_idx = 3

        # Duration bucket
        dur = exp.weighted_avg_duration
        if dur <= 2:
            d_idx = 0
        elif dur <= 5:
            d_idx = 1
        elif dur <= 10:
            d_idx = 2
        else:
            d_idx = 3

        matrix[r_idx, d_idx] += exp.total_market_usd / 1e6

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=duration_buckets,
        y=rating_buckets,
        colorscale=[
            [0, scheme.bg_secondary],
            [0.5, scheme.accent_blue],
            [1, scheme.accent_orange],
        ],
        text=[[f"${v:.0f}M" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="评级: %{y}<br>久期: %{x}<br>金额: $%{z:.0f}M<extra></extra>",
    ))

    fig.update_layout(
        **get_premium_layout("风险矩阵 (评级 × 久期)", height=350),
        xaxis=dict(title="久期", side="bottom"),
        yaxis=dict(title="评级", autorange="reversed"),
    )

    return fig
