from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions import (
    compute_drawdown,
    detect_regimes_coarse,
    window_stats,
    human_pct,
)


def render_gold_prices_tab(data: pd.DataFrame):
    st.subheader("Price Explorer")

    min_dt = data["date"].min().date()
    max_dt = data["date"].max().date()

    currency_cols = [c for c in data.columns if c != "date" and pd.api.types.is_numeric_dtype(data[c])]
    if not currency_cols:
        st.error("No numeric currency columns found in the CSV.")
        st.stop()

    c1, c2, c3 = st.columns([1.3, 1.3, 2.4])
    with c1:
        currency = st.selectbox("Currency column", currency_cols, index=0)
    with c2:
        bear_thr = st.number_input("Bear threshold (≤)", value=-0.20, step=0.01, format="%.2f")
    with c3:
        min_days = st.slider("Min days per regime", 15, 180, 60, step=5)

    preset = st.radio("Period", ["MAX", "YTD", "1Y", "5Y", "10Y", "Custom"],
                      horizontal=True, index=3)
    today = max_dt
    if preset == "MAX":
        start_dt, end_dt = min_dt, max_dt
    elif preset == "YTD":
        start_dt, end_dt = date(today.year, 1, 1), today
    elif preset == "1Y":
        start_dt, end_dt = today - timedelta(days=365), today
    elif preset == "5Y":
        start_dt, end_dt = today - timedelta(days=365*5), today
    elif preset == "10Y":
        start_dt, end_dt = today - timedelta(days=365*10), today
    else:
        start_dt, end_dt = data["date"].quantile(0.6).date(), today

    if preset == "Custom":
        start_dt, end_dt = st.date_input(
            "Custom date range",
            value=(start_dt, end_dt),
            min_value=min_dt,
            max_value=max_dt
        )
        if isinstance(start_dt, (list, tuple)):
            start_dt, end_dt = start_dt

    work = data[["date", currency]].rename(columns={currency: "price"}).dropna(subset=["price"])
    mask = (work["date"].dt.date >= start_dt) & (work["date"].dt.date <= end_dt)
    win = work.loc[mask].reset_index(drop=True)

    if win.empty:
        st.warning("No data in the selected range. Adjust your dates.")
        st.stop()

    dd_df = compute_drawdown(win)
    regimes_df, seg_df = detect_regimes_coarse(win, bear_threshold=bear_thr, min_days=min_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=win["date"], y=win["price"], mode="lines",
        name=f"Price ({currency})",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
    ))

    # bear shading
    bear_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = None
    for i in range(len(regimes_df)):
        d = regimes_df.loc[i, "date"]
        label = regimes_df.loc[i, "regime"]
        if label == "Bear" and cur_start is None:
            cur_start = d
        if label != "Bear" and cur_start is not None:
            bear_spans.append((cur_start, regimes_df.loc[i-1, "date"]))
            cur_start = None
    if cur_start is not None:
        bear_spans.append((cur_start, regimes_df.loc[len(regimes_df)-1, "date"]))

    for a, b in bear_spans:
        fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.10, line_width=0, layer="below")

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    mdd, peak_dt, trough_dt, peak_px, trough_px, mdd_abs = window_stats(win)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Max drawdown", human_pct(mdd))
    col2.metric("Max drawdown (abs)", f"{mdd_abs:,.2f} {currency}" if mdd_abs is not None else "—")
    col3.metric("Peak before MDD", peak_dt.strftime("%Y-%m-%d") if peak_dt else "—")
    col4.metric("Trough of MDD", trough_dt.strftime("%Y-%m-%d") if trough_dt else "—")
    if peak_dt and trough_dt:
        col5.metric("Peak→Trough days", (trough_dt.date() - peak_dt.date()).days)
    else:
        col5.metric("Peak→Trough days", "—")

    st.markdown("### Regimes in window")
    if not seg_df.empty:
        pretty = seg_df.copy()
        pretty["Return"] = pretty["Return"].map(human_pct)
        pretty["Max Drawdown in Segment"] = pretty["Max Drawdown in Segment"].map(human_pct)
        st.dataframe(pretty, use_container_width=True, hide_index=True)
    else:
        st.write("No regimes identified in this window.")

    st.caption("Bear/Bull regimes use drawdown≤bear threshold and a minimum duration filter to reduce noise.")
