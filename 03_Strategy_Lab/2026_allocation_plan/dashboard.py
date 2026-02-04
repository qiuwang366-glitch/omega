"""
dashboard.py - Streamlit Interactive Dashboard
===============================================
3D Yield Surface visualization and allocation scenario analysis.
The Cockpit for the 2026 Allocation Planning System.

Run with: streamlit run dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for Streamlit Cloud compatibility
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

# Local imports (use absolute imports for Streamlit compatibility)
from analytics import NelsonSiegelSvensson, YieldSurface
from config import (
    Currency,
    SimulationParams,
    YieldCurveRegime,
    NSSParams,
    DEFAULT_NSS_PARAMS,
    get_default_config,
    SubPortfolioProfile,
    DEFAULT_SUBPORTFOLIOS,
    MONTHS_2026,
    MONTHS_CN,
    MM_TO_YI,
)
from data_provider import MarketDataFactory, SyntheticMarketData
from allocation_engine import (
    PortfolioSimulator,
    AllocationStrategy,
    SimulationResult,
)
from forward_rate_data import (
    ForwardRateSurface,
    get_forward_surface,
    get_interpolated_rate,
    get_new_money_yields,
    FORWARD_SURFACES,
)


# ============================================================================
# 1. Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Project Omega | 2026 Allocation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# 2. Session State Initialization
# ============================================================================
def init_session_state() -> None:
    """Initialize session state variables."""
    if "config" not in st.session_state:
        st.session_state.config = get_default_config()

    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None

    if "nss_params" not in st.session_state:
        st.session_state.nss_params = DEFAULT_NSS_PARAMS.copy()

    # Sub-portfolio investment plans (user-editable)
    if "subportfolio_plans" not in st.session_state:
        st.session_state.subportfolio_plans = {
            key: profile.model_copy(deep=True)
            for key, profile in DEFAULT_SUBPORTFOLIOS.items()
        }

    # Global allocation strategy state (for sticky header sync)
    if "active_strategy" not in st.session_state:
        st.session_state.active_strategy = "match_maturity"  # Default strategy

    # Portfolio duration inputs by currency (for yield impact analysis)
    if "portfolio_durations" not in st.session_state:
        st.session_state.portfolio_durations = {
            "USD": 3.5,
            "EUR": 4.0,
            "AUD": 3.2,
            "CNH": 2.5,
        }

    # New money yield estimates by currency
    if "new_money_yields" not in st.session_state:
        st.session_state.new_money_yields = {
            "USD": 4.50,
            "EUR": 2.90,
            "AUD": 4.50,
            "CNH": 2.20,
        }


init_session_state()


# ============================================================================
# 3. Helper Functions
# ============================================================================
@st.cache_data(ttl=300)
def generate_yield_surface_data(
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
    max_forward: float = 3.0,
    max_tenor: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate yield surface grid for plotting."""
    nss = NelsonSiegelSvensson(beta0, beta1, beta2, beta3, lambda1, lambda2)
    surface = YieldSurface(nss)

    grid = surface.generate_grid(
        max_forward_years=max_forward,
        max_tenor_years=max_tenor,
        n_forward_points=25,
        n_tenor_points=30,
    )

    return grid.forward_starts, grid.tenors, grid.yields * 100  # Convert to %


def create_surface_plot(
    forward_starts: np.ndarray,
    tenors: np.ndarray,
    yields: np.ndarray,
    title: str = "Yield Surface",
) -> go.Figure:
    """Create 3D surface plot with Plotly."""
    fig = go.Figure(data=[
        go.Surface(
            x=tenors,           # X = Tenor
            y=forward_starts,   # Y = Forward Start
            z=yields,           # Z = Yield
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Yield (%)", side="right"),
            ),
            hovertemplate=(
                "Tenor: %{x:.1f}Y<br>"
                "Forward: %{y:.1f}Y<br>"
                "Yield: %{z:.2f}%<br>"
                "<extra></extra>"
            ),
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title=dict(text="Tenor (Years)", font=dict(size=14))),
            yaxis=dict(title=dict(text="Forward Start (Years)", font=dict(size=14))),
            zaxis=dict(title=dict(text="Yield (%)", font=dict(size=14))),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
    )

    return fig


def create_real_forward_surface_plot(
    surface: ForwardRateSurface,
    title: str = "Forward Rate Surface",
) -> go.Figure:
    """Create 3D surface plot from real forward rate data."""
    # Create meshgrid for plotting
    tenors = surface.tenors_years
    forwards = surface.forwards_years

    fig = go.Figure(data=[
        go.Surface(
            x=forwards,          # X = Forward Start
            y=tenors,            # Y = Tenor (Maturity)
            z=surface.rates,     # Z = Rate (already in %)
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Yield (%)", side="right"),
            ),
            hovertemplate=(
                "Forward: %{x:.2f}Y<br>"
                "Tenor: %{y:.2f}Y<br>"
                "Rate: %{z:.2f}%<br>"
                "<extra></extra>"
            ),
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title=dict(text="Forward Start (Years)", font=dict(size=14))),
            yaxis=dict(title=dict(text="Tenor (Years)", font=dict(size=14))),
            zaxis=dict(title=dict(text="Yield (%)", font=dict(size=14))),
            camera=dict(
                eye=dict(x=1.8, y=-1.8, z=0.8),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
    )

    return fig


def create_curve_comparison_plot(
    params_list: list[tuple[str, NSSParams]],
) -> go.Figure:
    """Create yield curve comparison chart."""
    tenors = np.linspace(0.1, 30, 100)

    fig = go.Figure()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    for i, (name, params) in enumerate(params_list):
        nss = NelsonSiegelSvensson.from_params(params)
        yields = nss.yield_at_tenor(tenors) * 100

        fig.add_trace(go.Scatter(
            x=tenors,
            y=yields,
            mode="lines",
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Yield Curve Comparison",
        xaxis=dict(title="Tenor (Years)", gridcolor="#eee"),
        yaxis=dict(title="Yield (%)", gridcolor="#eee"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400,
        hovermode="x unified",
    )

    return fig


def create_nii_heatmap(
    result: SimulationResult,
) -> go.Figure:
    """Create NII heatmap by currency and time."""
    df = result.to_dataframe()

    # Pivot for heatmap
    pivot = df.pivot_table(
        index="currency",
        columns="period",
        values="nii_spread",
        aggfunc="mean",
    ) * 10000  # Convert to bps

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"M{i+1}" for i in range(pivot.shape[1])],
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        colorbar=dict(title="Spread (bps)"),
        hovertemplate=(
            "Period: %{x}<br>"
            "Currency: %{y}<br>"
            "Spread: %{z:.0f}bp<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="NII Spread by Currency & Period",
        xaxis=dict(title="Period"),
        yaxis=dict(title="Currency"),
        height=300,
    )

    return fig


def create_fx_sensitivity_heatmap(
    base_nii: float,
    currencies: list[Currency],
    fx_shocks: list[float],
    weights: dict[Currency, float],
) -> go.Figure:
    """Create FX sensitivity heatmap."""
    # Simulate FX impact
    sensitivity = np.zeros((len(currencies) - 1, len(fx_shocks)))

    non_usd = [c for c in currencies if c != Currency.USD]

    for i, ccy in enumerate(non_usd):
        weight = weights.get(ccy, 0.0)
        for j, shock in enumerate(fx_shocks):
            # FX drag = weight * shock
            sensitivity[i, j] = base_nii * weight * shock / 1e6  # In MM

    fig = go.Figure(data=go.Heatmap(
        z=sensitivity,
        x=[f"{s:+.0%}" for s in fx_shocks],
        y=[c.value for c in non_usd],
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="P&L (MM)"),
        hovertemplate=(
            "FX Shock: %{x}<br>"
            "Currency: %{y}<br>"
            "Impact: $%{z:.1f}MM<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="FX Sensitivity Analysis",
        xaxis=dict(title="FX Shock (vs Budget)"),
        yaxis=dict(title="Currency"),
        height=250,
    )

    return fig


def mm_to_yi(value_mm: float) -> float:
    """Convert USD MM to 亿美元 (1亿 = 100 MM)."""
    return value_mm / MM_TO_YI


def yi_to_mm(value_yi: float) -> float:
    """Convert 亿美元 to USD MM."""
    return value_yi * MM_TO_YI


def create_maturity_investment_chart(
    profiles: dict[str, SubPortfolioProfile],
) -> go.Figure:
    """
    Create combined maturity, reinvestment, and additional investment chart.

    Features:
    - Bar chart for monthly maturities/investments
    - Line chart on secondary Y-axis for cumulative investment plan
    - Legend positioned outside the plot area (right side)
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["USD SSA 美元超主权债", "AUD Rates 澳元利率债"],
        vertical_spacing=0.18,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    colors = {
        "maturity": "#E74C3C",      # Red for maturities (outflow)
        "reinvest": "#27AE60",      # Green for reinvestments
        "additional": "#3498DB",    # Blue for additional investments
        "cumulative": "#9B59B6",    # Purple for cumulative line
    }

    month_labels = [MONTHS_CN[m] for m in MONTHS_2026]

    for idx, (key, profile) in enumerate(profiles.items(), start=1):
        # Convert to 亿美元
        maturities_yi = [mm_to_yi(profile.maturity_schedule.get(m, 0.0)) for m in MONTHS_2026]
        reinvest_yi = [mm_to_yi(profile.investment_plan.get(m, 0.0)) for m in MONTHS_2026]
        additional_yi = [mm_to_yi(profile.additional_investment.get(m, 0.0)) for m in MONTHS_2026]

        # Cumulative investment plan (reinvest + additional)
        total_invest_yi = [r + a for r, a in zip(reinvest_yi, additional_yi)]
        cumulative_yi = np.cumsum(total_invest_yi).tolist()

        # Maturity (shown as negative - outflow)
        fig.add_trace(
            go.Bar(
                name="到期量",
                x=month_labels,
                y=[-m for m in maturities_yi],
                marker_color=colors["maturity"],
                hovertemplate="%{x}<br>到期: %{customdata:.2f}亿美元<extra></extra>",
                customdata=maturities_yi,
                showlegend=(idx == 1),
                legendgroup="maturity",
            ),
            row=idx, col=1,
            secondary_y=False,
        )

        # Reinvestment (到期再投)
        fig.add_trace(
            go.Bar(
                name="到期再投",
                x=month_labels,
                y=reinvest_yi,
                marker_color=colors["reinvest"],
                hovertemplate="%{x}<br>到期再投: %{y:.2f}亿美元<extra></extra>",
                showlegend=(idx == 1),
                legendgroup="reinvest",
            ),
            row=idx, col=1,
            secondary_y=False,
        )

        # Additional investment (追加投资)
        fig.add_trace(
            go.Bar(
                name="追加投资",
                x=month_labels,
                y=additional_yi,
                marker_color=colors["additional"],
                hovertemplate="%{x}<br>追加投资: %{y:.2f}亿美元<extra></extra>",
                showlegend=(idx == 1),
                legendgroup="additional",
            ),
            row=idx, col=1,
            secondary_y=False,
        )

        # Cumulative investment line (secondary Y-axis)
        fig.add_trace(
            go.Scatter(
                name="累计投放",
                x=month_labels,
                y=cumulative_yi,
                mode="lines+markers",
                line=dict(color=colors["cumulative"], width=3, dash="solid"),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate="%{x}<br>累计投放: %{y:.2f}亿美元<extra></extra>",
                showlegend=(idx == 1),
                legendgroup="cumulative",
            ),
            row=idx, col=1,
            secondary_y=True,
        )

        # Add zero line for bars
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=idx, col=1)

    fig.update_layout(
        title=dict(text="2026年到期墙与投资计划", font=dict(size=18)),
        barmode="relative",
        height=600,
        # Legend positioned outside to the right
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        margin=dict(r=150),  # Extra right margin for legend
    )

    # Update Y-axes labels
    fig.update_yaxes(title_text="金额 (亿美元)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计 (亿美元)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="金额 (亿美元)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计 (亿美元)", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="月份", row=2, col=1)

    return fig


def create_cashflow_waterfall(
    profile: SubPortfolioProfile,
) -> go.Figure:
    """Create waterfall chart showing cumulative cash position."""
    month_labels = [MONTHS_CN[m] for m in MONTHS_2026]

    # Convert to 亿美元
    maturities_yi = [mm_to_yi(profile.maturity_schedule.get(m, 0.0)) for m in MONTHS_2026]
    reinvest_yi = [mm_to_yi(profile.investment_plan.get(m, 0.0)) for m in MONTHS_2026]
    additional_yi = [mm_to_yi(profile.additional_investment.get(m, 0.0)) for m in MONTHS_2026]

    # Net cashflow = maturities - reinvest - additional (positive = cash in hand)
    net_flows = [m - r - a for m, r, a in zip(maturities_yi, reinvest_yi, additional_yi)]
    cumulative = np.cumsum(net_flows)

    fig = go.Figure()

    # Net flow bars
    fig.add_trace(go.Bar(
        name="净现金流",
        x=month_labels,
        y=net_flows,
        marker_color=["#27AE60" if x >= 0 else "#E74C3C" for x in net_flows],
        hovertemplate="%{x}<br>净现金流: %{y:.2f}亿美元<extra></extra>",
    ))

    # Cumulative line
    fig.add_trace(go.Scatter(
        name="累计",
        x=month_labels,
        y=cumulative,
        mode="lines+markers",
        line=dict(color="#9B59B6", width=3),
        marker=dict(size=8),
        hovertemplate="%{x}<br>累计: %{y:.2f}亿美元<extra></extra>",
    ))

    fig.update_layout(
        title=f"{profile.name}: 净现金流与累计头寸",
        xaxis=dict(title="月份"),
        yaxis=dict(title="金额 (亿美元)"),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def create_subportfolio_summary_table(
    profiles: dict[str, SubPortfolioProfile],
) -> pd.DataFrame:
    """Create summary table for sub-portfolios in Chinese with 亿美元 units."""
    records = []
    for key, p in profiles.items():
        total_mat_yi = mm_to_yi(p.total_maturing_2026)
        total_reinv_yi = mm_to_yi(p.total_planned_investment)
        total_add_yi = mm_to_yi(p.total_additional_investment)
        total_deploy_yi = total_reinv_yi + total_add_yi
        aum_yi = mm_to_yi(p.aum_usd_mm)
        carry_yi = mm_to_yi(p.annual_carry_usd_mm)

        records.append({
            "组合": p.name,
            "币种": p.currency.value,
            "规模(亿美元)": f"{aum_yi:.1f}",
            "持仓数": p.n_positions,
            "收益率(%)": f"{p.wtd_avg_yield * 100:.2f}",
            "久期": f"{p.wtd_avg_duration:.2f}",
            "年化利息(亿美元)": f"{carry_yi:.2f}",
            "2026到期(亿美元)": f"{total_mat_yi:.2f}",
            "到期占比": f"{p.maturity_pct_of_aum * 100:.1f}%",
            "到期再投(亿美元)": f"{total_reinv_yi:.2f}",
            "追加投资(亿美元)": f"{total_add_yi:.2f}",
            "总投放(亿美元)": f"{total_deploy_yi:.2f}",
            "缺口(亿美元)": f"{total_mat_yi - total_deploy_yi:.2f}",
        })
    return pd.DataFrame(records)


def create_yield_impact_analysis(
    profiles: dict[str, SubPortfolioProfile],
    current_yield_estimate: float = 0.045,
) -> pd.DataFrame:
    """Create yield pickup analysis for new investments."""
    records = []
    for key, p in profiles.items():
        # Calculate weighted average exit yield for maturities
        total_mat = p.total_maturing_2026
        if total_mat > 0:
            weighted_exit_yield = sum(
                p.maturity_schedule.get(m, 0.0) * (p.maturity_exit_yields.get(m) or 0.0)
                for m in MONTHS_2026
                if p.maturity_exit_yields.get(m) is not None
            )
            maturing_with_yield = sum(
                p.maturity_schedule.get(m, 0.0)
                for m in MONTHS_2026
                if p.maturity_exit_yields.get(m) is not None
            )
            avg_exit_yield = weighted_exit_yield / maturing_with_yield if maturing_with_yield > 0 else 0.0
        else:
            avg_exit_yield = 0.0

        yield_pickup = current_yield_estimate - avg_exit_yield

        records.append({
            "组合": p.name,
            "当前组合收益率(%)": f"{p.wtd_avg_yield * 100:.2f}",
            "到期平均收益率(%)": f"{avg_exit_yield * 100:.2f}",
            "新投预估收益率(%)": f"{current_yield_estimate * 100:.2f}",
            "收益率提升(bp)": f"{yield_pickup * 10000:.0f}",
            "到期量(亿美元)": f"{mm_to_yi(total_mat):.2f}",
            "预估NII提升(万元)": f"{mm_to_yi(total_mat) * yield_pickup * 10000:.0f}",
        })
    return pd.DataFrame(records)


def create_enhanced_yield_impact_analysis(
    profiles: dict[str, SubPortfolioProfile],
    yield_estimates: dict[str, float],
    durations: dict[str, float],
    rate_change_bp: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced yield impact analysis with currency granularity and duration.

    Separates USD vs Non-USD buckets and calculates capital gain/loss impact
    using the formula: Delta P = -D * Delta y

    Args:
        profiles: Sub-portfolio profiles
        yield_estimates: New money yield estimates by currency (e.g., {"USD": 4.5, "AUD": 4.5})
        durations: Portfolio duration by currency
        rate_change_bp: Expected rate change in basis points (for capital gain/loss calc)

    Returns:
        Tuple of (summary_df, detail_df)
    """
    records = []
    summary_records = []

    # Group by USD vs Non-USD
    usd_profiles = {k: v for k, v in profiles.items() if v.currency.value == "USD"}
    non_usd_profiles = {k: v for k, v in profiles.items() if v.currency.value != "USD"}

    for group_name, group_profiles in [("USD", usd_profiles), ("Non-USD", non_usd_profiles)]:
        group_total_mat = 0.0
        group_total_nii_impact = 0.0
        group_total_cap_impact = 0.0

        for key, p in group_profiles.items():
            ccy = p.currency.value
            new_yield = yield_estimates.get(ccy, 4.5) / 100  # Convert from % to decimal
            duration = durations.get(ccy, 3.5)

            # Calculate yield metrics
            total_mat = p.total_maturing_2026
            if total_mat > 0:
                weighted_exit_yield = sum(
                    p.maturity_schedule.get(m, 0.0) * (p.maturity_exit_yields.get(m) or 0.0)
                    for m in MONTHS_2026
                    if p.maturity_exit_yields.get(m) is not None
                )
                maturing_with_yield = sum(
                    p.maturity_schedule.get(m, 0.0)
                    for m in MONTHS_2026
                    if p.maturity_exit_yields.get(m) is not None
                )
                avg_exit_yield = weighted_exit_yield / maturing_with_yield if maturing_with_yield > 0 else 0.0
            else:
                avg_exit_yield = 0.0

            yield_pickup = new_yield - avg_exit_yield
            nii_impact = mm_to_yi(total_mat) * yield_pickup * 10000  # In 万元

            # Capital Gain/Loss: Delta P = -D * Delta y * Notional
            # Delta y is in decimal (rate_change_bp / 10000)
            rate_change_decimal = rate_change_bp / 10000
            cap_gain_loss = -duration * rate_change_decimal * mm_to_yi(total_mat)  # In 亿美元

            group_total_mat += total_mat
            group_total_nii_impact += nii_impact
            group_total_cap_impact += cap_gain_loss

            records.append({
                "分组": group_name,
                "组合": p.name,
                "币种": ccy,
                "久期": f"{duration:.2f}",
                "到期量(亿)": f"{mm_to_yi(total_mat):.2f}",
                "退出收益率(%)": f"{avg_exit_yield * 100:.2f}",
                "新投收益率(%)": f"{new_yield * 100:.2f}",
                "收益率变动(bp)": f"{yield_pickup * 10000:.0f}",
                "NII影响(万)": f"{nii_impact:.0f}",
                "资本损益(亿)": f"{cap_gain_loss:.2f}" if rate_change_bp != 0 else "-",
            })

        if group_profiles:
            summary_records.append({
                "分组": group_name,
                "组合数": len(group_profiles),
                "总到期(亿)": f"{mm_to_yi(group_total_mat):.2f}",
                "NII总影响(万)": f"{group_total_nii_impact:.0f}",
                "资本损益(亿)": f"{group_total_cap_impact:.2f}" if rate_change_bp != 0 else "-",
            })

    return pd.DataFrame(summary_records), pd.DataFrame(records)


# ============================================================================
# 4. Sidebar Controls
# ============================================================================
def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return current settings."""
    st.sidebar.title("🎛️ Control Panel")

    # === Yield Curve Parameters ===
    st.sidebar.header("Yield Curve (NSS)")

    currency = st.sidebar.selectbox(
        "Currency",
        options=[c.value for c in Currency],
        index=0,
    )
    currency_enum = Currency(currency)

    default_params = DEFAULT_NSS_PARAMS.get(currency_enum, DEFAULT_NSS_PARAMS[Currency.USD])

    col1, col2 = st.sidebar.columns(2)

    with col1:
        beta0 = st.slider(
            "β₀ (Level)",
            min_value=0.0,
            max_value=0.10,
            value=float(default_params.beta0),
            step=0.005,
            format="%.3f",
        )
        beta2 = st.slider(
            "β₂ (Curvature)",
            min_value=-0.05,
            max_value=0.05,
            value=float(default_params.beta2),
            step=0.002,
            format="%.3f",
        )

    with col2:
        beta1 = st.slider(
            "β₁ (Slope)",
            min_value=-0.05,
            max_value=0.05,
            value=float(default_params.beta1),
            step=0.002,
            format="%.3f",
        )
        beta3 = st.slider(
            "β₃ (Hump 2)",
            min_value=-0.05,
            max_value=0.05,
            value=float(default_params.beta3),
            step=0.002,
            format="%.3f",
        )

    lambda1 = st.sidebar.slider("λ₁ (Decay 1)", 0.5, 5.0, float(default_params.lambda1), 0.1)
    lambda2 = st.sidebar.slider("λ₂ (Decay 2)", 1.0, 8.0, float(default_params.lambda2), 0.1)

    # === Regime Selection ===
    st.sidebar.header("Scenario Settings")

    regime = st.sidebar.selectbox(
        "Yield Regime",
        options=[r.value for r in YieldCurveRegime],
        index=3,  # Bear Flattening default
    )

    fx_scenario = st.sidebar.selectbox(
        "FX Scenario",
        options=["base", "usd_strong", "usd_weak"],
        index=0,
    )

    # === Allocation Weights ===
    st.sidebar.header("Allocation Weights")

    weight_usd = st.sidebar.slider("USD %", 0, 100, 55)
    weight_aud = st.sidebar.slider("AUD %", 0, 100, 25)
    weight_eur = st.sidebar.slider("EUR %", 0, 100, 15)
    weight_cnh = st.sidebar.slider("CNH %", 0, 100, 5)

    total = weight_usd + weight_aud + weight_eur + weight_cnh
    if total != 100:
        st.sidebar.warning(f"⚠️ Weights sum to {total}% (should be 100%)")

    # === Run Button ===
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

    return {
        "currency": currency_enum,
        "beta0": beta0,
        "beta1": beta1,
        "beta2": beta2,
        "beta3": beta3,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "regime": YieldCurveRegime(regime),
        "fx_scenario": fx_scenario,
        "weights": {
            Currency.USD: weight_usd / 100,
            Currency.AUD: weight_aud / 100,
            Currency.EUR: weight_eur / 100,
            Currency.CNH: weight_cnh / 100,
        },
        "run_simulation": run_simulation,
    }


# ============================================================================
# 5. Main Content
# ============================================================================
def apply_strategy(strategy_key: str, profiles: dict[str, SubPortfolioProfile]) -> None:
    """Apply allocation strategy to all profiles and update session state."""
    for key, profile in profiles.items():
        if strategy_key == "match_maturity":
            profile.investment_plan = profile.maturity_schedule.copy()
        elif strategy_key == "frontload_q1":
            total = profile.total_maturing_2026
            profile.investment_plan = {m: 0.0 for m in MONTHS_2026}
            profile.investment_plan["Jan"] = total * 0.4
            profile.investment_plan["Feb"] = total * 0.35
            profile.investment_plan["Mar"] = total * 0.25
        elif strategy_key == "even_distribution":
            monthly = profile.total_maturing_2026 / 12
            profile.investment_plan = {m: monthly for m in MONTHS_2026}
        elif strategy_key == "clear_all":
            profile.investment_plan = {m: 0.0 for m in MONTHS_2026}
            profile.additional_investment = {m: 0.0 for m in MONTHS_2026}

        st.session_state.subportfolio_plans[key] = profile


def render_sticky_header() -> str | None:
    """
    Render sticky header with quick allocation strategy selector.

    Returns the selected strategy key if changed, None otherwise.
    """
    # Custom CSS for sticky header
    st.markdown("""
    <style>
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .strategy-btn {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0 0.25rem;
        border-radius: 4px;
        font-size: 0.85rem;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header container
    header_cols = st.columns([3, 1, 1, 1, 1, 1])

    with header_cols[0]:
        st.markdown("**快捷配置策略**")

    strategies = [
        ("match_maturity", "🔄 匹配到期", "每月到期再投 = 到期量"),
        ("frontload_q1", "⚡ 前置Q1", "40% 1月 / 35% 2月 / 25% 3月"),
        ("even_distribution", "📅 均匀分布", "平均分配到12个月"),
        ("clear_all", "🗑️ 清空", "清空所有投资计划"),
    ]

    selected_strategy = None
    for i, (key, label, tooltip) in enumerate(strategies):
        with header_cols[i + 1]:
            is_active = st.session_state.active_strategy == key
            button_type = "primary" if is_active else "secondary"
            if st.button(label, help=tooltip, key=f"header_strategy_{key}", type=button_type):
                selected_strategy = key
                st.session_state.active_strategy = key

    return selected_strategy


def render_main_content(settings: dict[str, Any]) -> None:
    """Render main dashboard content."""

    # === Header ===
    st.title("🏛️ Project Omega | 2026 Allocation Cockpit")
    st.markdown(
        "> *Strategic Fixed Income Allocation Simulator* | "
        f"AUM: **$60B** | Horizon: **2026-2028**"
    )

    # === Sticky Strategy Header ===
    st.markdown("---")
    selected_strategy = render_sticky_header()

    # Apply strategy if changed
    if selected_strategy:
        profiles = st.session_state.subportfolio_plans
        apply_strategy(selected_strategy, profiles)
        st.rerun()

    # === Tabs ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Yield Surface",
        "🎯 2026 Allocation",
        "💹 Simulation",
        "📊 FTP Analysis",
        "📋 Reports",
    ])

    # === Tab 1: Yield Surface ===
    with tab1:
        st.header("3D Yield Surface Visualization")

        # Data source toggle
        data_source = st.radio(
            "Data Source",
            ["Market Data (Real)", "NSS Model (Synthetic)"],
            horizontal=True,
        )

        if data_source == "Market Data (Real)":
            # === Real Forward Rate Surface ===
            ccy_str = settings["currency"].value
            real_surface = get_forward_surface(ccy_str)

            if real_surface is not None:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = create_real_forward_surface_plot(
                        real_surface,
                        f"{ccy_str} Forward Rate Surface (as of {real_surface.as_of_date})",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Surface Info")
                    st.markdown(f"""
                    - **Currency**: {real_surface.currency}
                    - **As of**: {real_surface.as_of_date}
                    - **Tenors**: {len(real_surface.tenor_labels)} points
                    - **Forwards**: {len(real_surface.forward_labels)} points
                    """)

                    st.subheader("Spot Curve (Key Rates)")
                    df_spot = real_surface.to_dataframe()[["Spot"]].rename(columns={"Spot": "Yield (%)"})
                    st.dataframe(df_spot, use_container_width=True)

                # Full forward rate matrix
                st.subheader("Forward Rate Matrix (%)")
                st.dataframe(
                    real_surface.to_dataframe(),
                    use_container_width=True,
                )

            else:
                st.warning(f"No market data available for {ccy_str}. Available: {list(FORWARD_SURFACES.keys())}")

        else:
            # === NSS Synthetic Surface ===
            col1, col2 = st.columns([2, 1])

            with col1:
                # Generate and plot surface
                fwd, tnr, ylds = generate_yield_surface_data(
                    settings["beta0"],
                    settings["beta1"],
                    settings["beta2"],
                    settings["beta3"],
                    settings["lambda1"],
                    settings["lambda2"],
                )

                fig = create_surface_plot(fwd, tnr, ylds, f"{settings['currency'].value} Yield Surface (NSS)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("NSS Parameters")
                st.json({
                    "β₀ (Level)": f"{settings['beta0']:.3f}",
                    "β₁ (Slope)": f"{settings['beta1']:.3f}",
                    "β₂ (Curvature)": f"{settings['beta2']:.3f}",
                    "β₃ (Hump 2)": f"{settings['beta3']:.3f}",
                    "λ₁": f"{settings['lambda1']:.1f}",
                    "λ₂": f"{settings['lambda2']:.1f}",
                })

                st.subheader("Key Rates")
                nss = NelsonSiegelSvensson(
                    settings["beta0"],
                    settings["beta1"],
                    settings["beta2"],
                    settings["beta3"],
                    settings["lambda1"],
                    settings["lambda2"],
                )
                key_tenors = [0.25, 1.0, 2.0, 5.0, 10.0]
                rates = nss.yield_at_tenor(np.array(key_tenors)) * 100

                for t, r in zip(key_tenors, rates):
                    label = f"{int(t*12)}M" if t < 1 else f"{int(t)}Y"
                    st.metric(label, f"{r:.2f}%")

        # Curve comparison (always shown)
        st.markdown("---")
        st.subheader("Multi-Currency Spot Curve Comparison")
        params_list = [
            (ccy.value, params)
            for ccy, params in DEFAULT_NSS_PARAMS.items()
        ]
        fig_comp = create_curve_comparison_plot(params_list)
        st.plotly_chart(fig_comp, use_container_width=True)

    # === Tab 2: 2026 Allocation Plan ===
    with tab2:
        st.header("2026年子组合配置计划")

        st.markdown("""
        > **再投资风险管理**: 跟踪到期墙，规划到期再投与追加投资节奏。
        > 下方可编辑每月的投资计划，单位为**亿美元**（1亿 = 100 MM USD）。
        """)

        # Get current plans from session state
        profiles = st.session_state.subportfolio_plans

        # Summary table
        st.subheader("组合概览")
        summary_df = create_subportfolio_summary_table(profiles)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Main visualization
        st.subheader("到期墙与投资计划")
        fig_maturity = create_maturity_investment_chart(profiles)
        st.plotly_chart(fig_maturity, use_container_width=True)

        # Enhanced Yield Impact Analysis with Currency & Duration Granularity
        st.subheader("收益率影响分析 (增强版)")

        st.markdown("""
        > **分析维度**: USD vs Non-USD 分组 | 久期敏感性 | 资本损益估算
        > $$\\Delta P \\approx -D \\times \\Delta y$$ (久期近似公式)
        """)

        # Input section for yield estimates and durations
        with st.expander("📊 参数设置 (新投收益率 & 组合久期)", expanded=True):
            col_params1, col_params2 = st.columns(2)

            with col_params1:
                st.markdown("**新投预估收益率 (%)**")
                yield_cols = st.columns(4)
                yield_estimates = {}
                for i, ccy in enumerate(["USD", "EUR", "AUD", "CNH"]):
                    with yield_cols[i]:
                        default_yield = st.session_state.new_money_yields.get(ccy, 4.5)
                        yield_estimates[ccy] = st.number_input(
                            f"{ccy}",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(default_yield),
                            step=0.1,
                            key=f"yield_{ccy}",
                            format="%.2f",
                        )
                        st.session_state.new_money_yields[ccy] = yield_estimates[ccy]

            with col_params2:
                st.markdown("**组合久期 (年)**")
                dur_cols = st.columns(4)
                durations = {}
                for i, ccy in enumerate(["USD", "EUR", "AUD", "CNH"]):
                    with dur_cols[i]:
                        default_dur = st.session_state.portfolio_durations.get(ccy, 3.5)
                        durations[ccy] = st.number_input(
                            f"{ccy}",
                            min_value=0.0,
                            max_value=15.0,
                            value=float(default_dur),
                            step=0.1,
                            key=f"dur_{ccy}",
                            format="%.2f",
                        )
                        st.session_state.portfolio_durations[ccy] = durations[ccy]

            # Rate change scenario for capital gain/loss
            st.markdown("**利率变动情景 (bp)**")
            rate_change_bp = st.slider(
                "预期利率变动",
                min_value=-100,
                max_value=100,
                value=0,
                step=5,
                key="rate_change_slider",
                help="正值=加息, 负值=降息. 用于计算资本损益: ΔP = -D × Δy",
            )

        # Display enhanced analysis
        summary_df, detail_df = create_enhanced_yield_impact_analysis(
            profiles, yield_estimates, durations, rate_change_bp
        )

        col_sum, col_det = st.columns([1, 2])
        with col_sum:
            st.markdown("**分组汇总 (USD vs Non-USD)**")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with col_det:
            st.markdown("**组合明细**")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

        # Forward rate based new money yields (from Data Warehouse)
        st.markdown("---")
        st.markdown("**市场隐含新投收益率 (Forward Rate Surface)**")

        fwd_col1, fwd_col2 = st.columns([1, 3])
        with fwd_col1:
            fwd_months = st.selectbox(
                "Forward Start",
                options=[0, 3, 6, 12],
                format_func=lambda x: "Spot" if x == 0 else f"{x}M Forward",
                key="fwd_months_select",
            )
            target_tenor = st.selectbox(
                "Target Tenor",
                options=[2.0, 3.0, 5.0, 7.0, 10.0],
                format_func=lambda x: f"{int(x)}Y",
                key="target_tenor_select",
                index=2,  # Default 5Y
            )

        with fwd_col2:
            # Get rates from forward surface
            fwd_rates_data = []
            for ccy in ["USD", "EUR", "AUD", "CNH"]:
                rate = get_interpolated_rate(ccy, target_tenor, fwd_months / 12, method="cubic")
                if rate is not None:
                    fwd_rates_data.append({
                        "币种": ccy,
                        f"{int(target_tenor)}Y @ {'Spot' if fwd_months == 0 else str(fwd_months) + 'M Fwd'} (%)": f"{rate:.2f}",
                    })

            if fwd_rates_data:
                st.dataframe(pd.DataFrame(fwd_rates_data), use_container_width=True, hide_index=True)
            else:
                st.info("Forward rate data not available. Check Data Warehouse.")

        # Editable investment plan
        st.markdown("---")
        st.subheader("编辑投资计划")

        for key, profile in profiles.items():
            aum_yi = mm_to_yi(profile.aum_usd_mm)
            mat_yi = mm_to_yi(profile.total_maturing_2026)

            with st.expander(f"📌 {profile.name} ({profile.currency.value})", expanded=True):
                st.markdown(f"""
                **当前规模**: {aum_yi:.1f}亿美元 |
                **2026到期**: {mat_yi:.2f}亿美元 ({profile.maturity_pct_of_aum*100:.1f}%)
                """)

                # Two sections: 到期再投 and 追加投资
                tab_reinv, tab_add = st.tabs(["📥 到期再投", "📈 追加投资"])

                with tab_reinv:
                    st.caption("到期再投：将到期资金重新投入市场")
                    cols = st.columns(6)
                    updated_reinv = profile.investment_plan.copy()

                    for i, month in enumerate(MONTHS_2026):
                        col_idx = i % 6
                        maturity_amt_yi = mm_to_yi(profile.maturity_schedule.get(month, 0.0))
                        current_reinv_yi = mm_to_yi(profile.investment_plan.get(month, 0.0))

                        with cols[col_idx]:
                            if maturity_amt_yi > 0:
                                st.caption(f"{MONTHS_CN[month]}: 到期{maturity_amt_yi:.2f}亿")
                            else:
                                st.caption(f"{MONTHS_CN[month]}")

                            new_val_yi = st.number_input(
                                f"再投({month})",
                                min_value=0.0,
                                max_value=50.0,
                                value=float(current_reinv_yi),
                                step=0.5,
                                key=f"reinv_{key}_{month}",
                                label_visibility="collapsed",
                                format="%.2f",
                            )
                            updated_reinv[month] = yi_to_mm(new_val_yi)

                        if col_idx == 5 and i < 11:
                            cols = st.columns(6)

                    profile.investment_plan = updated_reinv

                with tab_add:
                    st.caption("追加投资：新增资金配置（非到期再投）")
                    cols = st.columns(6)
                    updated_add = profile.additional_investment.copy()

                    for i, month in enumerate(MONTHS_2026):
                        col_idx = i % 6
                        current_add_yi = mm_to_yi(profile.additional_investment.get(month, 0.0))

                        with cols[col_idx]:
                            st.caption(f"{MONTHS_CN[month]}")

                            new_val_yi = st.number_input(
                                f"追加({month})",
                                min_value=0.0,
                                max_value=50.0,
                                value=float(current_add_yi),
                                step=0.5,
                                key=f"add_{key}_{month}",
                                label_visibility="collapsed",
                                format="%.2f",
                            )
                            updated_add[month] = yi_to_mm(new_val_yi)

                        if col_idx == 5 and i < 11:
                            cols = st.columns(6)

                    profile.additional_investment = updated_add

                # Update session state
                st.session_state.subportfolio_plans[key] = profile

        # Cashflow analysis
        st.markdown("---")
        st.subheader("净现金流分析")
        col1, col2 = st.columns(2)

        with col1:
            usd_profile = profiles.get("USD_SSA")
            if usd_profile:
                fig_cf1 = create_cashflow_waterfall(usd_profile)
                st.plotly_chart(fig_cf1, use_container_width=True)

        with col2:
            aud_profile = profiles.get("AUD_Rates")
            if aud_profile:
                fig_cf2 = create_cashflow_waterfall(aud_profile)
                st.plotly_chart(fig_cf2, use_container_width=True)

        # Note: Quick allocation buttons moved to sticky header at top of page
        st.markdown("---")
        st.info("💡 **快捷配置策略** 已移至页面顶部导航栏，全局同步所有组件。")

        # Summary metrics
        st.markdown("---")
        st.subheader("投资计划汇总")

        total_mat_yi = sum(mm_to_yi(p.total_maturing_2026) for p in profiles.values())
        total_reinv_yi = sum(mm_to_yi(p.total_planned_investment) for p in profiles.values())
        total_add_yi = sum(mm_to_yi(p.total_additional_investment) for p in profiles.values())
        total_deploy_yi = total_reinv_yi + total_add_yi

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("2026总到期", f"{total_mat_yi:.2f}亿美元")
        with m2:
            st.metric("到期再投", f"{total_reinv_yi:.2f}亿美元")
        with m3:
            st.metric("追加投资", f"{total_add_yi:.2f}亿美元", delta=f"+{total_add_yi:.2f}" if total_add_yi > 0 else None)
        with m4:
            gap = total_mat_yi - total_deploy_yi
            st.metric("资金缺口", f"{gap:.2f}亿美元", delta=f"{-gap:.2f}" if gap != 0 else "0", delta_color="inverse")

    # === Tab 3: Simulation ===
    with tab3:
        st.header("Allocation Simulation")

        if settings["run_simulation"]:
            with st.spinner("Running simulation..."):
                # Create config and provider
                config = get_default_config()
                config.curve_regime = settings["regime"]

                provider = SyntheticMarketData(
                    regime=settings["regime"],
                    random_seed=42,
                )

                simulator = PortfolioSimulator(provider, config)

                strategy = AllocationStrategy(
                    name="Dashboard Strategy",
                    currency_weights=settings["weights"],
                )

                result = simulator.run_simulation(strategy, fx_scenario=settings["fx_scenario"])
                st.session_state.simulation_result = result

            st.success("✅ Simulation complete!")

        result = st.session_state.simulation_result

        if result is not None:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total NII",
                    f"${result.total_nii_usd/1e6:.1f}MM",
                    help="Net Interest Income over simulation horizon",
                )

            with col2:
                st.metric(
                    "FX P&L",
                    f"${result.total_fx_pnl_usd/1e6:.1f}MM",
                    delta_color="normal" if result.total_fx_pnl_usd >= 0 else "inverse",
                )

            with col3:
                st.metric(
                    "Avg Spread",
                    f"{result.avg_spread_bps:.0f}bp",
                )

            with col4:
                st.metric(
                    "Info Ratio",
                    f"{result.sharpe_ratio:.2f}",
                )

            # NII Heatmap
            st.subheader("NII Spread Distribution")
            fig_heatmap = create_nii_heatmap(result)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Currency breakdown
            st.subheader("Currency Breakdown")

            df_summary = pd.DataFrame([
                {
                    "Currency": ccy.value,
                    "Weight": f"{summary['weight']:.1%}",
                    "Book Value": f"${summary['book_value_usd']/1e9:.1f}B",
                    "Avg Yield": f"{summary['avg_asset_yield']*100:.2f}%",
                    "Avg FTP": f"{summary['avg_ftp_rate']*100:.2f}%",
                    "Spread (bp)": f"{summary['avg_spread_bps']:.0f}",
                    "NII ($MM)": f"{summary['total_nii_usd']/1e6:.1f}",
                }
                for ccy, summary in result.summary_by_currency.items()
            ])

            st.dataframe(df_summary, use_container_width=True, hide_index=True)

        else:
            st.info("👈 Configure settings in the sidebar and click **Run Simulation**")

    # === Tab 4: FTP Analysis ===
    with tab4:
        st.header("FTP Arbitrage Analysis")

        st.markdown("""
        **The FTP Rule**: $FTP(T) = Average(3M\\;Yield_{T-1\\;month})$

        This creates timing arbitrage:
        - **Hiking Cycle**: Buy immediately after hike (Asset Yield ↑, FTP flat)
        - **Cutting Cycle**: Front-load before cut (Lock yield before FTP catches up)
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hiking Cycle Strategy")
            st.markdown("""
            ```
            Month 0: Fed hikes 25bp
            Month 0: Asset Yield = 4.75% (new)
            Month 0: FTP = 4.50% (lagged)
            ➡️ Spread = +25bp (Profit!)

            Action: Aggressive deployment
            ```
            """)

        with col2:
            st.subheader("Cutting Cycle Strategy")
            st.markdown("""
            ```
            Month 0: Fed signals cut
            Month 0: Asset Yield = 4.50% (current)
            Month 0: FTP = 4.50% (same)

            Month 1: Fed cuts 25bp
            Month 1: Asset Yield = 4.25% (new)
            Month 1: FTP = 4.50% (lagged!)
            ➡️ Spread = -25bp (Loss!)

            Action: Front-load purchases BEFORE cut
            ```
            """)

        # FX Sensitivity
        st.subheader("FX Sensitivity Grid")

        if st.session_state.simulation_result is not None:
            fx_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10]
            fig_fx = create_fx_sensitivity_heatmap(
                st.session_state.simulation_result.total_nii_usd,
                list(settings["weights"].keys()),
                fx_shocks,
                settings["weights"],
            )
            st.plotly_chart(fig_fx, use_container_width=True)

    # === Tab 5: Reports ===
    with tab5:
        st.header("Export & Reports")

        if st.session_state.simulation_result is not None:
            result = st.session_state.simulation_result

            # Download DataFrame
            df = result.to_dataframe()

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Data (CSV)",
                data=csv,
                file_name=f"simulation_{result.strategy_name}_{date.today()}.csv",
                mime="text/csv",
            )

            # Summary report
            st.subheader("Executive Summary")

            report = f"""
## 2026 Allocation Simulation Report
**Generated**: {date.today()}
**Strategy**: {result.strategy_name}
**Period**: {result.start_date} to {result.end_date}

### Key Metrics
| Metric | Value |
|--------|-------|
| Total NII | ${result.total_nii_usd/1e6:.1f}MM |
| FX P&L | ${result.total_fx_pnl_usd/1e6:.1f}MM |
| Average Spread | {result.avg_spread_bps:.0f}bp |
| Information Ratio | {result.sharpe_ratio:.2f} |

### Currency Allocation
"""
            for ccy, summary in result.summary_by_currency.items():
                report += f"- **{ccy.value}**: {summary['weight']:.0%} weight, ${summary['total_nii_usd']/1e6:.1f}MM NII\n"

            st.markdown(report)

            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"report_{date.today()}.md",
                mime="text/markdown",
            )

        else:
            st.info("Run a simulation first to generate reports.")


# ============================================================================
# 6. Main Entry Point
# ============================================================================
def main() -> None:
    """Main dashboard entry point."""
    settings = render_sidebar()
    render_main_content(settings)


if __name__ == "__main__":
    main()
