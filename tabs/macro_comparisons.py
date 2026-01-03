from datetime import date, timedelta
from typing import Optional, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions import (
    load_macro_series,
    join_on_date,
)


def _plot_pair(
    macro_currency: str,
    gold_series: pd.DataFrame,
    other: Optional[pd.DataFrame],
    other_name: str,
    y2_pct: bool = False,
):
    if other is None:
        st.info(f"No {other_name} data.")
        return
    
    GOLD = "#DEB64B"
    BG = "#0E1117"

    # Align gold and macro series on date
    df_gold = gold_series.rename(columns={macro_currency: "Gold"})
    merged = join_on_date(df_gold, other)

    dmax = merged["date"].max().date()
    dmin = merged["date"].min().date()

    # Controls for window and rolling correlation
    colw = st.columns(3)
    with colw[0]:
        window = st.selectbox(
            "Window",
            ["MAX", "YTD", "1Y", "5Y", "10Y"],
            index=0,
            key=f"{other_name}_win",
        )
    with colw[1]:
        roll_days = st.selectbox(
            "Rolling corr window",
            [60, 126, 252],
            index=1,
            key=f"{other_name}_roll",
        )
    with colw[2]:
        show_corr = st.toggle(
            "Show rolling correlations",
            value=True,
            key=f"{other_name}_corr",
        )

    # Date windowing
    if window == "YTD":
        dmin_w = date(dmax.year, 1, 1)
    elif window == "1Y":
        dmin_w = dmax - timedelta(days=365)
    elif window == "5Y":
        dmin_w = dmax - timedelta(days=365 * 5)
    elif window == "10Y":
        dmin_w = dmax - timedelta(days=365 * 10)
    else:  # MAX
        dmin_w = dmin

    mask = (merged["date"].dt.date >= dmin_w) & (merged["date"].dt.date <= dmax)
    mwin = (
        merged.loc[mask]
        .dropna(subset=["Gold"])
        .reset_index(drop=True)
    )

    fig = go.Figure()
    fig.add_trace(
    go.Scatter(
        x=mwin["date"],
        y=mwin["Gold"],
        mode="lines",
        name=f"Gold ({macro_currency})",
        line=dict(color=GOLD, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
    )
    )


    right_name = other_name
    if "value" in mwin.columns:
        y2 = mwin["value"] * 100 if y2_pct else mwin["value"]
    else:
        y2 = pd.Series(dtype=float)

    # Plot macro series on right axis
    if not y2.empty:
        fig.add_trace(
            go.Scatter(
                x=mwin["date"],
                y=y2,
                mode="lines",
                name=right_name + (" (%)" if y2_pct else ""),
                yaxis="y2",
            )
        )

    # NOTE: no fixed range on yaxis2 -> Plotly autoscale, so you always see full series
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
        title=f"Gold ({macro_currency})",
        title_font=dict(color=GOLD),
        tickfont=dict(color=GOLD),
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
    ),
    yaxis2=dict(
        title=f"{right_name}" + (" / %" if y2_pct else ""),
        overlaying="y",
        side="right",
        title_font=dict(color=GOLD),
        tickfont=dict(color=GOLD),
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
    ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Rolling correlation plot
    if show_corr and not y2.empty:
        s = mwin["Gold"].rolling(roll_days, min_periods=roll_days).corr(y2)
        cf = go.Figure()
        cf.add_trace(
            go.Scatter(
                x=mwin["date"],
                y=s,
                mode="lines",
                name=f"Corr(Gold, {right_name})",
            )
        )
        cf.update_layout(
            height=320,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Rolling correlation", range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(cf, use_container_width=True)


def render_macro_tab(data: pd.DataFrame):
    st.subheader("CPI · US Dollar · Real Interest Rate vs Gold")

    # Load macro series
    cpi, dxy, real = load_macro_series()

    # Select which gold currency to use
    gold_currencies = [c for c in data.columns if c != "date"]
    macro_currency = st.selectbox("Gold currency", gold_currencies, index=0)

    # Build list of available macro series based on files actually present
    options = []
    if dxy is not None:
        options.append("USD Index")
    if cpi is not None:
        options.append("CPI")
    if real is not None:
        options.append("Real Rate")

    if not options:
        st.info("No macro comparison data available.")
        return

    compare_choice = st.selectbox(
        "Compare gold against",
        options,
        index=0,
    )

    gold_series = data[["date", macro_currency]]

    # RAW LEVEL CPI, DXY, REAL — no YoY, no fixed ranges
    if compare_choice == "USD Index":
        dxy_level = dxy.rename(columns={"value": "value"})
        _plot_pair(
            macro_currency=macro_currency,
            gold_series=gold_series,
            other=dxy_level,
            other_name="USD Index",
            y2_pct=False,
        )

    elif compare_choice == "CPI":
        cpi_level = cpi.rename(columns={"value": "value"})
        _plot_pair(
            macro_currency=macro_currency,
            gold_series=gold_series,
            other=cpi_level,
            other_name="CPI",
            y2_pct=False,
        )

    elif compare_choice == "Real Rate":
        real_level = real.rename(columns={"value": "value"})
        _plot_pair(
            macro_currency=macro_currency,
            gold_series=gold_series,
            other=real_level,
            other_name="Real Rate",
            y2_pct=False,
        )
