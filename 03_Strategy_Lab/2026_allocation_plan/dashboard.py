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
)
from data_provider import MarketDataFactory, SyntheticMarketData
from allocation_engine import (
    PortfolioSimulator,
    AllocationStrategy,
    SimulationResult,
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
            xaxis=dict(title="Tenor (Years)", titlefont=dict(size=14)),
            yaxis=dict(title="Forward Start (Years)", titlefont=dict(size=14)),
            zaxis=dict(title="Yield (%)", titlefont=dict(size=14)),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
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
def render_main_content(settings: dict[str, Any]) -> None:
    """Render main dashboard content."""

    # === Header ===
    st.title("🏛️ Project Omega | 2026 Allocation Cockpit")
    st.markdown(
        "> *Strategic Fixed Income Allocation Simulator* | "
        f"AUM: **$60B** | Horizon: **2026-2028**"
    )

    # === Tabs ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Yield Surface",
        "💹 Simulation",
        "🎯 FTP Analysis",
        "📊 Reports",
    ])

    # === Tab 1: Yield Surface ===
    with tab1:
        st.header("3D Yield Surface Visualization")

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

            fig = create_surface_plot(fwd, tnr, ylds, f"{settings['currency'].value} Yield Surface")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Current Parameters")
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

        # Curve comparison
        st.subheader("Multi-Currency Comparison")
        params_list = [
            (ccy.value, params)
            for ccy, params in DEFAULT_NSS_PARAMS.items()
        ]
        fig_comp = create_curve_comparison_plot(params_list)
        st.plotly_chart(fig_comp, use_container_width=True)

    # === Tab 2: Simulation ===
    with tab2:
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

    # === Tab 3: FTP Analysis ===
    with tab3:
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

    # === Tab 4: Reports ===
    with tab4:
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
