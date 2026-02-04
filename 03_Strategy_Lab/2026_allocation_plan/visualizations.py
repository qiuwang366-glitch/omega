"""
visualizations.py - Premium Visualization Components
======================================================
High-end visualization components for institutional-grade dashboards.
Inspired by BlackRock Aladdin, Bloomberg PORT, and top-tier HF risk systems.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# 1. Premium Color Schemes
# ============================================================================
class ColorScheme:
    """Premium color schemes for institutional dashboards."""

    # Primary palette - Dark mode
    DARK = {
        "bg_primary": "#0d1117",
        "bg_secondary": "#161b22",
        "bg_card": "#21262d",
        "text_primary": "#f0f6fc",
        "text_secondary": "#8b949e",
        "accent_blue": "#58a6ff",
        "accent_green": "#3fb950",
        "accent_red": "#f85149",
        "accent_yellow": "#d29922",
        "accent_purple": "#a371f7",
        "accent_cyan": "#39c5cf",
        "grid": "#30363d",
    }

    # Gradient palettes
    MATURITY_GRADIENT = [
        "#1a5f7a",  # Near-term (cool blue)
        "#2e8b57",  # 1-2Y (green)
        "#4682b4",  # 3-5Y (steel blue)
        "#6b5b95",  # 5-7Y (purple)
        "#d35400",  # 7-10Y (orange)
        "#c0392b",  # 10Y+ (red)
    ]

    CURRENCY_COLORS = {
        "USD": "#58a6ff",
        "EUR": "#3fb950",
        "AUD": "#d29922",
        "CNY": "#f85149",
        "CNH": "#a371f7",
        "GBP": "#39c5cf",
    }

    CATEGORY_COLORS = {
        "Rates": "#58a6ff",
        "Credit": "#3fb950",
        "Govs": "#d29922",
        "SSA": "#a371f7",
        "China Sovereign": "#f85149",
    }


# ============================================================================
# 2. Premium Layout Templates
# ============================================================================
def get_premium_layout(
    title: str = "",
    height: int = 500,
    show_legend: bool = True,
    dark_mode: bool = True,
) -> dict[str, Any]:
    """Get premium Plotly layout configuration."""
    colors = ColorScheme.DARK

    return {
        "title": {
            "text": title,
            "font": {"size": 18, "color": colors["text_primary"], "family": "Inter, SF Pro Display, -apple-system"},
            "x": 0.02,
            "xanchor": "left",
        },
        "paper_bgcolor": colors["bg_primary"],
        "plot_bgcolor": colors["bg_secondary"],
        "font": {
            "color": colors["text_secondary"],
            "family": "Inter, SF Pro Display, -apple-system",
            "size": 12,
        },
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "height": height,
        "showlegend": show_legend,
        "legend": {
            "bgcolor": "rgba(33, 38, 45, 0.8)",
            "bordercolor": colors["grid"],
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "xaxis": {
            "gridcolor": colors["grid"],
            "zerolinecolor": colors["grid"],
            "tickfont": {"size": 11},
        },
        "yaxis": {
            "gridcolor": colors["grid"],
            "zerolinecolor": colors["grid"],
            "tickfont": {"size": 11},
        },
    }


# ============================================================================
# 3. Maturity Analysis Visualizations
# ============================================================================
def create_maturity_waterfall(
    df: pd.DataFrame,
    value_col: str = "NominalYi",
    label_col: str = "MaturityYear",
    title: str = "Maturity Profile by Year",
) -> go.Figure:
    """
    Create premium waterfall chart for maturity distribution.

    Args:
        df: DataFrame with maturity data
        value_col: Column name for values
        label_col: Column name for labels/years

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    colors = ColorScheme.DARK
    gradient = ColorScheme.MATURITY_GRADIENT

    # Sort by year
    df = df.sort_values(label_col)

    # Create gradient colors based on position
    n = len(df)
    bar_colors = [gradient[min(i, len(gradient) - 1)] for i in range(n)]

    fig = go.Figure()

    # Main bars
    fig.add_trace(go.Bar(
        x=df[label_col].astype(str),
        y=df[value_col],
        marker=dict(
            color=bar_colors,
            line=dict(color=colors["text_secondary"], width=0.5),
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Nominal: %{y:.2f}亿美元<br>"
            "<extra></extra>"
        ),
        text=[f"{v:.1f}" for v in df[value_col]],
        textposition="outside",
        textfont=dict(size=10, color=colors["text_primary"]),
    ))

    # Add cumulative line
    cumulative = df[value_col].cumsum()
    fig.add_trace(go.Scatter(
        x=df[label_col].astype(str),
        y=cumulative,
        mode="lines+markers",
        name="Cumulative",
        line=dict(color=colors["accent_cyan"], width=3, dash="dot"),
        marker=dict(size=8, symbol="diamond", color=colors["accent_cyan"]),
        yaxis="y2",
        hovertemplate="Cumulative: %{y:.2f}亿美元<extra></extra>",
    ))

    layout = get_premium_layout(title, height=450)
    layout.update({
        "barmode": "relative",
        "xaxis": {**layout["xaxis"], "title": {"text": "Maturity Year", "font": {"size": 12}}},
        "yaxis": {**layout["yaxis"], "title": {"text": "Nominal (亿美元)", "font": {"size": 12}}},
        "yaxis2": {
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(0,0,0,0)",
            "title": {"text": "Cumulative (亿美元)", "font": {"size": 12}},
            "tickfont": {"size": 11, "color": colors["accent_cyan"]},
        },
    })

    fig.update_layout(**layout)

    return fig


def create_maturity_heatmap(
    pivot_df: pd.DataFrame,
    title: str = "Maturity by Currency × Year",
) -> go.Figure:
    """
    Create premium heatmap for maturity by currency and year.

    Args:
        pivot_df: Pivot table with years as index, currencies as columns

    Returns:
        Plotly Figure
    """
    if pivot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    colors = ColorScheme.DARK

    # Custom colorscale (dark blue to gold)
    colorscale = [
        [0.0, "#0d1117"],
        [0.2, "#1a365d"],
        [0.4, "#2563eb"],
        [0.6, "#3b82f6"],
        [0.8, "#d97706"],
        [1.0, "#f59e0b"],
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.astype(str).tolist(),
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="亿美元", side="right"),
            tickfont=dict(size=10),
            len=0.8,
        ),
        hovertemplate=(
            "<b>%{y}</b> - %{x}<br>"
            "Nominal: %{z:.2f}亿美元<br>"
            "<extra></extra>"
        ),
        text=[[f"{v:.1f}" if v > 0.5 else "" for v in row] for row in pivot_df.values],
        texttemplate="%{text}",
        textfont=dict(size=9, color=colors["text_primary"]),
    ))

    layout = get_premium_layout(title, height=400)
    layout.update({
        "xaxis": {**layout["xaxis"], "title": {"text": "Currency", "font": {"size": 12}}},
        "yaxis": {**layout["yaxis"], "title": {"text": "Year", "font": {"size": 12}}, "autorange": "reversed"},
    })

    fig.update_layout(**layout)

    return fig


def create_maturity_sunburst(
    df: pd.DataFrame,
    title: str = "Maturity Distribution",
) -> go.Figure:
    """
    Create sunburst chart for maturity distribution by year and currency.

    Args:
        df: DataFrame with CCY, MaturityYear, NominalYi columns

    Returns:
        Plotly Figure
    """
    colors = ColorScheme.DARK
    ccy_colors = ColorScheme.CURRENCY_COLORS

    if df.empty or "CCY" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Aggregate data
    agg = df.groupby(["CCY", "MaturityYear"]).agg({
        "NominalUSD": "sum"
    }).reset_index()
    agg["NominalYi"] = agg["NominalUSD"] / 100_000_000

    # Build hierarchical structure
    labels = ["Total"]
    parents = [""]
    values = [agg["NominalYi"].sum()]
    marker_colors = [colors["accent_blue"]]

    # Add currency level
    ccy_totals = agg.groupby("CCY")["NominalYi"].sum()
    for ccy in ccy_totals.index:
        labels.append(ccy)
        parents.append("Total")
        values.append(ccy_totals[ccy])
        marker_colors.append(ccy_colors.get(ccy, colors["accent_purple"]))

    # Add year level under each currency
    for _, row in agg.iterrows():
        labels.append(f"{row['CCY']}-{int(row['MaturityYear'])}")
        parents.append(row["CCY"])
        values.append(row["NominalYi"])
        marker_colors.append(ccy_colors.get(row["CCY"], colors["accent_purple"]))

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=marker_colors, line=dict(width=1, color=colors["bg_primary"])),
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>%{value:.2f}亿美元<br>%{percentParent:.1%}<extra></extra>",
        textfont=dict(size=11, color=colors["text_primary"]),
    ))

    layout = get_premium_layout(title, height=500)
    fig.update_layout(**layout)

    return fig


# ============================================================================
# 4. Duration & Risk Visualizations
# ============================================================================
def create_duration_profile_chart(
    df: pd.DataFrame,
    title: str = "Duration Profile by Maturity Bucket",
) -> go.Figure:
    """
    Create combined bar + scatter chart for duration profile.

    Args:
        df: DataFrame with Bucket, NominalYi, AvgDuration columns

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    colors = ColorScheme.DARK
    gradient = ColorScheme.MATURITY_GRADIENT

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for nominal
    n = len(df)
    bar_colors = [gradient[min(i, len(gradient) - 1)] for i in range(n)]

    fig.add_trace(
        go.Bar(
            x=df["Bucket"],
            y=df["NominalYi"],
            name="Nominal (亿美元)",
            marker=dict(color=bar_colors, line=dict(width=0.5, color=colors["text_secondary"])),
            hovertemplate="<b>%{x}</b><br>Nominal: %{y:.2f}亿<extra></extra>",
        ),
        secondary_y=False,
    )

    # Scatter for duration
    fig.add_trace(
        go.Scatter(
            x=df["Bucket"],
            y=df["AvgDuration"],
            name="Avg Duration",
            mode="lines+markers",
            line=dict(color=colors["accent_cyan"], width=3),
            marker=dict(size=10, symbol="diamond", color=colors["accent_cyan"]),
            hovertemplate="<b>%{x}</b><br>Duration: %{y:.2f}Y<extra></extra>",
        ),
        secondary_y=True,
    )

    layout = get_premium_layout(title, height=400)
    fig.update_layout(**layout)

    fig.update_yaxes(
        title_text="Nominal (亿美元)",
        secondary_y=False,
        gridcolor=colors["grid"],
    )
    fig.update_yaxes(
        title_text="Duration (Years)",
        secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(color=colors["accent_cyan"]),
    )

    return fig


def create_key_rate_duration_chart(
    krd_dict: dict[str, float],
    title: str = "Key Rate Duration Contribution",
) -> go.Figure:
    """
    Create KRD contribution bar chart.

    Args:
        krd_dict: Dict mapping tenor -> KRD value

    Returns:
        Plotly Figure
    """
    colors = ColorScheme.DARK

    tenors = list(krd_dict.keys())
    values = list(krd_dict.values())

    # Color gradient based on tenor
    n = len(tenors)
    bar_colors = [f"hsl({180 + i * 15}, 70%, 50%)" for i in range(n)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tenors,
        y=values,
        marker=dict(
            color=bar_colors,
            line=dict(width=0.5, color=colors["text_secondary"]),
        ),
        hovertemplate="<b>%{x}</b><br>KRD: %{y:.3f}<extra></extra>",
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=colors["text_primary"]),
    ))

    layout = get_premium_layout(title, height=350)
    layout.update({
        "xaxis": {**layout["xaxis"], "title": {"text": "Tenor", "font": {"size": 12}}},
        "yaxis": {**layout["yaxis"], "title": {"text": "Duration Contribution", "font": {"size": 12}}},
    })

    fig.update_layout(**layout)

    return fig


# ============================================================================
# 5. Currency Breakdown Visualizations
# ============================================================================
def create_currency_donut(
    df: pd.DataFrame,
    title: str = "Portfolio by Currency",
) -> go.Figure:
    """
    Create premium donut chart for currency breakdown.

    Args:
        df: DataFrame with CCY, NominalYi, WeightPct columns

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    colors = ColorScheme.DARK
    ccy_colors = ColorScheme.CURRENCY_COLORS

    marker_colors = [ccy_colors.get(ccy, colors["accent_purple"]) for ccy in df["CCY"]]

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=df["CCY"],
        values=df["NominalYi"],
        hole=0.55,
        marker=dict(
            colors=marker_colors,
            line=dict(color=colors["bg_primary"], width=2),
        ),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=12, color=colors["text_primary"]),
        hovertemplate="<b>%{label}</b><br>%{value:.2f}亿美元<br>%{percent}<extra></extra>",
        pull=[0.02] * len(df),
    ))

    # Center text
    total_yi = df["NominalYi"].sum()
    fig.add_annotation(
        text=f"<b>{total_yi:.1f}</b><br><span style='font-size:12px'>亿美元</span>",
        x=0.5, y=0.5,
        font=dict(size=20, color=colors["text_primary"]),
        showarrow=False,
    )

    layout = get_premium_layout(title, height=400, show_legend=False)
    fig.update_layout(**layout)

    return fig


# ============================================================================
# 6. Rolldown & Carry Analysis
# ============================================================================
def create_rolldown_chart(
    forward_surface: Any,  # ForwardRateSurface
    entry_tenors: list[float] | None = None,
    holding_periods: list[float] | None = None,
    title: str = "Rolldown Analysis",
) -> go.Figure:
    """
    Create rolldown analysis chart showing carry + rolldown by entry tenor.

    Args:
        forward_surface: ForwardRateSurface object
        entry_tenors: List of entry tenors in years
        holding_periods: List of holding periods in years

    Returns:
        Plotly Figure
    """
    if entry_tenors is None:
        entry_tenors = [2, 3, 5, 7, 10]
    if holding_periods is None:
        holding_periods = [0.25, 0.5, 1.0]  # 3M, 6M, 1Y

    colors = ColorScheme.DARK

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Rolldown by Entry Tenor", "Total Return (Carry + Rolldown)"],
        horizontal_spacing=0.12,
    )

    spot_rates = forward_surface.rates[:, 0] if hasattr(forward_surface, 'rates') else []
    tenors_years = forward_surface.tenors_years if hasattr(forward_surface, 'tenors_years') else entry_tenors

    # Calculate rolldown for each entry tenor and holding period
    for hp_idx, hp in enumerate(holding_periods):
        rolldowns = []
        for tenor in entry_tenors:
            # Find closest tenor in surface
            tenor_idx = min(range(len(tenors_years)), key=lambda i: abs(tenors_years[i] - tenor))
            rolled_tenor_idx = min(range(len(tenors_years)), key=lambda i: abs(tenors_years[i] - (tenor - hp)))

            if rolled_tenor_idx >= 0 and tenor_idx < len(spot_rates) and rolled_tenor_idx < len(spot_rates):
                entry_yield = spot_rates[tenor_idx]
                exit_yield = spot_rates[rolled_tenor_idx]
                # Rolldown = (Entry Yield - Exit Yield) * Duration approx
                rolldown = (entry_yield - exit_yield) * min(tenor, 5) / 100  # Simplified
                rolldowns.append(rolldown * 100)  # Convert to bp
            else:
                rolldowns.append(0)

        hp_label = f"{int(hp * 12)}M" if hp < 1 else f"{int(hp)}Y"

        fig.add_trace(
            go.Bar(
                x=[f"{t}Y" for t in entry_tenors],
                y=rolldowns,
                name=f"{hp_label} Hold",
                marker=dict(opacity=0.8),
            ),
            row=1, col=1,
        )

    # Total return chart (Carry component)
    carries = []
    for tenor in entry_tenors:
        tenor_idx = min(range(len(tenors_years)), key=lambda i: abs(tenors_years[i] - tenor))
        if tenor_idx < len(spot_rates):
            carries.append(spot_rates[tenor_idx])
        else:
            carries.append(0)

    fig.add_trace(
        go.Bar(
            x=[f"{t}Y" for t in entry_tenors],
            y=carries,
            name="Carry (Spot Yield)",
            marker=dict(color=colors["accent_green"]),
        ),
        row=1, col=2,
    )

    layout = get_premium_layout(title, height=400)
    fig.update_layout(**layout)
    fig.update_layout(barmode="group")

    return fig


def create_forward_curve_comparison(
    forward_surface: Any,
    forward_points: list[int] | None = None,
    title: str = "Forward Curve Comparison",
) -> go.Figure:
    """
    Create forward curve comparison chart.

    Args:
        forward_surface: ForwardRateSurface object
        forward_points: List of forward start months to compare

    Returns:
        Plotly Figure
    """
    if forward_points is None:
        forward_points = [0, 3, 6, 12]  # Spot, 3M, 6M, 1Y forward

    colors = ColorScheme.DARK

    if not hasattr(forward_surface, 'rates') or not hasattr(forward_surface, 'tenor_labels'):
        fig = go.Figure()
        fig.add_annotation(text="Forward surface data not available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = go.Figure()

    forward_labels = forward_surface.forward_labels
    tenor_labels = forward_surface.tenor_labels
    rates = forward_surface.rates

    color_scale = [colors["accent_blue"], colors["accent_green"], colors["accent_yellow"], colors["accent_red"]]

    for idx, fwd_months in enumerate(forward_points):
        # Find column index for this forward point
        fwd_label = "Spot" if fwd_months == 0 else f"{fwd_months}M"
        if fwd_label in forward_labels:
            col_idx = forward_labels.index(fwd_label)
            curve_rates = rates[:, col_idx]

            fig.add_trace(go.Scatter(
                x=tenor_labels,
                y=curve_rates,
                mode="lines+markers",
                name=fwd_label,
                line=dict(color=color_scale[idx % len(color_scale)], width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{fwd_label}</b><br>Tenor: %{{x}}<br>Yield: %{{y:.2f}}%<extra></extra>",
            ))

    layout = get_premium_layout(title, height=400)
    layout.update({
        "xaxis": {**layout["xaxis"], "title": {"text": "Tenor", "font": {"size": 12}}},
        "yaxis": {**layout["yaxis"], "title": {"text": "Yield (%)", "font": {"size": 12}}},
    })

    fig.update_layout(**layout)

    return fig


# ============================================================================
# 7. Concentration Risk Visualizations
# ============================================================================
def create_concentration_treemap(
    df: pd.DataFrame,
    title: str = "Position Concentration",
) -> go.Figure:
    """
    Create treemap for position concentration analysis.

    Args:
        df: DataFrame with BondName, NominalYi, CCY columns

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    colors = ColorScheme.DARK
    ccy_colors = ColorScheme.CURRENCY_COLORS

    # Limit to top positions
    df = df.nlargest(30, "NominalYi" if "NominalYi" in df.columns else "NominalUSD")

    if "NominalYi" not in df.columns:
        df["NominalYi"] = df["NominalUSD"] / 100_000_000

    labels = df["BondName"].tolist() if "BondName" in df.columns else df.index.tolist()
    values = df["NominalYi"].tolist()
    parents = df["CCY"].tolist() if "CCY" in df.columns else ["" for _ in labels]

    # Build hierarchical data
    all_labels = list(set(parents)) + labels
    all_parents = ["" for _ in set(parents)] + parents
    all_values = [df[df["CCY"] == p]["NominalYi"].sum() for p in set(parents)] + values

    marker_colors = [ccy_colors.get(p, colors["accent_purple"]) for p in set(parents)]
    marker_colors += [ccy_colors.get(p, colors["accent_purple"]) for p in parents]

    fig = go.Figure(go.Treemap(
        labels=all_labels,
        parents=all_parents,
        values=all_values,
        marker=dict(colors=marker_colors, line=dict(width=1, color=colors["bg_primary"])),
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:.2f}亿",
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>%{value:.2f}亿美元<extra></extra>",
    ))

    layout = get_premium_layout(title, height=500)
    fig.update_layout(**layout)

    return fig


# ============================================================================
# 8. Premium Table Styling
# ============================================================================
def style_dataframe_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply premium styling to pandas DataFrame for Streamlit display.

    Note: Returns styled DataFrame for use with st.dataframe()
    """
    # This is for reference - actual styling in Streamlit uses different methods
    return df


# ============================================================================
# 9. Enhanced 3D Surface
# ============================================================================
def create_premium_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "Yield Surface",
    x_label: str = "Forward Start (Years)",
    y_label: str = "Tenor (Years)",
    z_label: str = "Yield (%)",
) -> go.Figure:
    """
    Create premium 3D surface plot with enhanced aesthetics.

    Args:
        x: X-axis values (forward start)
        y: Y-axis values (tenor)
        z: Z-axis values (yields - 2D array)
        title: Chart title

    Returns:
        Plotly Figure
    """
    colors = ColorScheme.DARK

    # Premium colorscale
    colorscale = [
        [0.0, "#0d47a1"],   # Deep blue
        [0.25, "#1976d2"],  # Blue
        [0.5, "#42a5f5"],   # Light blue
        [0.75, "#ffd54f"],  # Gold
        [1.0, "#ff6f00"],   # Orange
    ]

    fig = go.Figure(data=[
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=z_label, side="right", font=dict(size=12)),
                tickfont=dict(size=10),
                len=0.7,
                thickness=15,
            ),
            hovertemplate=(
                f"{x_label}: %{{x:.2f}}<br>"
                f"{y_label}: %{{y:.2f}}<br>"
                f"{z_label}: %{{z:.2f}}<br>"
                "<extra></extra>"
            ),
            lighting=dict(
                ambient=0.8,
                diffuse=0.9,
                specular=0.3,
                roughness=0.5,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
            ),
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=colors["text_primary"])),
        paper_bgcolor=colors["bg_primary"],
        font=dict(color=colors["text_secondary"], size=11),
        scene=dict(
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=12)),
                backgroundcolor=colors["bg_secondary"],
                gridcolor=colors["grid"],
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text=y_label, font=dict(size=12)),
                backgroundcolor=colors["bg_secondary"],
                gridcolor=colors["grid"],
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text=z_label, font=dict(size=12)),
                backgroundcolor=colors["bg_secondary"],
                gridcolor=colors["grid"],
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=0.9),
            ),
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=50),
    )

    return fig


# ============================================================================
# 10. CSS Styles for Streamlit
# ============================================================================
PREMIUM_CSS = """
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0d1117;
    }

    /* Card-like containers */
    .premium-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Table styling */
    .dataframe {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #8b949e;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #f0f6fc;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        border: none;
        border-radius: 6px;
        color: white;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #21262d;
        border-radius: 6px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #161b22;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
</style>
"""


def get_premium_css() -> str:
    """Return premium CSS styles for Streamlit."""
    return PREMIUM_CSS
