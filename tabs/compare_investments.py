from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions import load_compare_series


def _compute_metrics(df: pd.DataFrame):
    if df.empty or "value" not in df.columns:
        return float("nan"), float("nan")

    # period return
    start_val = df["value"].iloc[0]
    end_val = df["value"].iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        period_return = float("nan")
    else:
        period_return = end_val / start_val - 1.0

    # volatility
    rets = df["value"].pct_change().dropna()
    vol = float(rets.std()) if not rets.empty else float("nan")

    return period_return, vol


def render_compare_tab(data: pd.DataFrame):
    st.subheader("Compare Historical Investments")

    GOLD = "#DEB64B"
    BG = "#0E1117"

    # Load S&P500 (spx), bonds, CPI
    spx, bonds, cpi = load_compare_series()

    # Gold currencies
    gold_currencies = [c for c in data.columns if c != "date"]
    gold_currency = st.selectbox(
    "Gold currency",
    gold_currencies,
    index=0,
    key="gold_currency_compare",  
)


    # Investment options
    investment_options = ["Gold price"]
    if spx is not None:
        investment_options.append("S&P 500")
    if bonds is not None:
        investment_options.append("Bond price")
    if cpi is not None:
        investment_options.append("Inflation (CPI)")

    if len(investment_options) < 2:
        st.info("Not enough data to compare.")
        return

    st.markdown("#### 1. Select date range")

    global_min = data["date"].min().date()
    global_max = data["date"].max().date()

    default_start = (
        date(global_max.year - 5, global_max.month, global_max.day)
        if global_max.year - 5 >= global_min.year
        else global_min
    )

    start_dt, end_dt = st.date_input(
        "Period",
        value=(default_start, global_max),
        min_value=global_min,
        max_value=global_max,
    )

    if isinstance(start_dt, (list, tuple)):
        start_dt, end_dt = start_dt

    if start_dt > end_dt:
        st.error("Start date must be before end date.")
        return

    st.markdown("#### 2. Choose two investments")

    col1, col2 = st.columns(2)
    with col1:
        inv1 = st.selectbox("Investment 1", investment_options, index=0)
    with col2:
        inv2 = st.selectbox(
            "Investment 2",
            investment_options,
            index=min(1, len(investment_options)-1),
        )

    if inv1 == inv2:
        st.warning("Choose two different investments.")
        return

    # Map names → data series
    def get_series(name: str) -> pd.DataFrame:
        if name == "Gold price":
            return (
                data[["date", gold_currency]]
                .rename(columns={gold_currency: "value"})
                .dropna()
                .sort_values("date")
                .reset_index(drop=True)
            )
        elif name == "S&P 500":
            return spx.copy() if spx is not None else pd.DataFrame(columns=["date", "value"])
        elif name == "Bond price":
            return bonds.copy() if bonds is not None else pd.DataFrame(columns=["date", "value"])
        elif name == "Inflation (CPI)":
            return cpi.copy() if cpi is not None else pd.DataFrame(columns=["date", "value"])
        return pd.DataFrame(columns=["date", "value"])

    s1 = get_series(inv1)
    s2 = get_series(inv2)

    if s1.empty or s2.empty:
        st.error("One of the selected investments has no data.")
        return

    # Overlapping window calculations
    s1_min, s1_max = s1["date"].min().date(), s1["date"].max().date()
    s2_min, s2_max = s2["date"].min().date(), s2["date"].max().date()

    overlap_min = max(s1_min, s2_min)
    overlap_max = min(s1_max, s2_max)

    if end_dt > overlap_max:
        st.error(f"End date too far. Latest common date: {overlap_max}")
        return

    actual_start = max(start_dt, overlap_min)
    actual_end = end_dt

    if actual_start > actual_end:
        st.error("No overlapping data in this range.")
        return

    mask1 = (s1["date"].dt.date >= actual_start) & (s1["date"].dt.date <= actual_end)
    mask2 = (s2["date"].dt.date >= actual_start) & (s2["date"].dt.date <= actual_end)

    w1 = s1.loc[mask1].reset_index(drop=True)
    w2 = s2.loc[mask2].reset_index(drop=True)

    if w1.empty or w2.empty:
        st.error("No data in this period.")
        return

    st.markdown(
        f"**Effective window:** {actual_start} → {actual_end}"
    )

    # Normalize to 100
    w1_norm = w1.copy()
    w2_norm = w2.copy()
    w1_norm["index"] = w1_norm["value"] / w1_norm["value"].iloc[0] * 100
    w2_norm["index"] = w2_norm["value"] / w2_norm["value"].iloc[0] * 100

    # Metrics
    ret1, vol1 = _compute_metrics(w1)
    ret2, vol2 = _compute_metrics(w2)

    # Plot
    fig = go.Figure()

    # Trace 1
    t1 = go.Scatter(
        x=w1_norm["date"],
        y=w1_norm["index"],
        mode="lines",
        name=inv1,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
    )
    if inv1 == "Gold price":
        t1.update(line=dict(color=GOLD, width=2.5))
    fig.add_trace(t1)

    # Trace 2
    t2 = go.Scatter(
        x=w2_norm["date"],
        y=w2_norm["index"],
        mode="lines",
        name=inv2,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
    )
    if inv2 == "Gold price":
        t2.update(line=dict(color=GOLD, width=2.5))
    fig.add_trace(t2)

    fig.update_layout(
        height=520,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),

        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=GOLD),
        hoverlabel=dict(font=dict(color=GOLD)),

        xaxis=dict(
            title="Date",
            rangeslider=dict(
                visible=True,
                bgcolor=BG,
                bordercolor="rgba(255,255,255,0.12)",
                thickness=0.10,
            ),
            title_font=dict(color=GOLD),
            tickfont=dict(color=GOLD),
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.12)",
        ),
        yaxis=dict(
            title="Indexed start = 100",
            title_font=dict(color=GOLD),
            tickfont=dict(color=GOLD),
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.12)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics side-by-side
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {inv1}")
        st.metric("Total return", f"{ret1:+.2%}" if pd.notna(ret1) else "—")
        st.metric("Volatility (σ)", f"{vol1:.2%}" if pd.notna(vol1) else "—")

    with c2:
        st.markdown(f"### {inv2}")
        st.metric("Total return", f"{ret2:+.2%}" if pd.notna(ret2) else "—")
        st.metric("Volatility (σ)", f"{vol2:.2%}" if pd.notna(vol2) else "—")
