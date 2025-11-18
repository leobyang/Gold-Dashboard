from datetime import date, timedelta
from typing import Optional, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions import (
    load_macro_series,
    join_on_date,
    yoy,
)


def _plot_pair(macro_currency: str,
               gold_series: pd.DataFrame,
               other: Optional[pd.DataFrame],
               other_name: str,
               y2_pct: bool = False,
               y2_range: Optional[List[float]] = None):
    if other is None:
        st.info(f"No {other_name} data.")
        return

    df_gold = gold_series.rename(columns={macro_currency: "Gold"})
    merged = join_on_date(df_gold, other)
    dmax = merged["date"].max().date()
    dmin = merged["date"].min().date()

    colw = st.columns(3)
    with colw[0]:
        window = st.selectbox("Window", ["MAX","YTD","1Y","5Y","10Y"], index=0, key=f"{other_name}_win")
    with colw[1]:
        roll_days = st.selectbox("Rolling corr window", [60, 126, 252], index=1, key=f"{other_name}_roll")
    with colw[2]:
        show_corr = st.toggle("Show rolling correlations", value=True, key=f"{other_name}_corr")

    if window == "YTD":
        dmin_w = date(dmax.year, 1, 1)
    elif window == "1Y":
        dmin_w = dmax - timedelta(days=365)
    elif window == "5Y":
        dmin_w = dmax - timedelta(days=365*5)
    elif window == "10Y":
        dmin_w = dmax - timedelta(days=365*10)
    else:
        dmin_w = dmin

    mask = (merged["date"].dt.date >= dmin_w) & (merged["date"].dt.date <= dmax)
    mwin = merged.loc[mask].dropna(subset=["Gold"]).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mwin["date"], y=mwin["Gold"], mode="lines",
        name=f"Gold ({macro_currency})",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
    ))

    right_name = other_name
    if "value" in mwin.columns:
        y2 = mwin["value"] * 100 if y2_pct else mwin["value"]
    else:
        y2 = pd.Series(dtype=float)

    if not y2.empty:
        fig.add_trace(go.Scatter(
            x=mwin["date"], y=y2, mode="lines",
            name=right_name + (" (%)" if y2_pct else ""),
            yaxis="y2"
        ))

    fig.update_layout(
        height=520, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title=f"Gold ({macro_currency})"),
        yaxis2=dict(
            title=f"{right_name}" + (" / %" if y2_pct else ""),
            overlaying="y", side="right",
            range=y2_range if y2_range is not None else None,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_corr and not y2.empty:
        s = mwin["Gold"].rolling(roll_days, min_periods=roll_days).corr(y2)
        cf = go.Figure()
        cf.add_trace(go.Scatter(x=mwin["date"], y=s, mode="lines",
                                name=f"Corr(Gold, {right_name})"))
        cf.update_layout(
            height=320, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Rolling correlation", range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(cf, use_container_width=True)


def render_macro_tab(data: pd.DataFrame):
    st.subheader("CPI · US Dollar · Real Interest Rate vs Gold")

    cpi, dxy, real = load_macro_series()
    gold_currencies = [c for c in data.columns if c != "date"]
    macro_currency = st.selectbox("Gold currency", gold_currencies, index=0)

    tab_usd, tab_cpi, tab_real = st.tabs(["Gold vs USD", "Gold vs CPI", "Gold vs Real Rate"])

    with tab_usd:
        if dxy is not None:
            dxy_usd = dxy.rename(columns={"value": "USD Index"})
            _plot_pair(macro_currency, data[["date", macro_currency]], dxy_usd,
                       "USD Index", y2_pct=False, y2_range=None)
        else:
            st.info("No USD Index data.")

    with tab_cpi:
        if cpi is not None:
            cpi_y = yoy(cpi)[["date", "yoy"]].rename(columns={"yoy": "value"})
            max_pct = st.slider("Max % scale (CPI YoY)", 5, 50, 20, key="cpi_max")
            _plot_pair(macro_currency, data[["date", macro_currency]], cpi_y,
                       "CPI YoY", y2_pct=True, y2_range=[0, max_pct])
        else:
            st.info("No CPI data.")

    with tab_real:
        if real is not None:
            real_pct = real.rename(columns={"value": "value"})
            max_pct_r = st.slider("Max % scale (Real Rate)", 5, 50, 20, key="real_max")
            _plot_pair(macro_currency, data[["date", macro_currency]], real_pct,
                       "Real Rate", y2_pct=True, y2_range=[0, max_pct_r])
        else:
            st.info("No Real Rate data.")
