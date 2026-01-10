from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions import load_macro_series, join_on_date, detect_regimes_coarse


GOLD = "#DEB64B"
BG = "#0E1117"


def _to_month_end(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Resample to month-end with last available value."""
    x = df[["date", value_col]].dropna().sort_values("date").copy()
    x = x.set_index("date")
    m = x.resample("M").last().dropna()
    m = m.reset_index()
    return m


def _log_ret(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return np.log(s).diff()


def _pct_ret(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return s.pct_change()


def _ols(y: pd.Series, X: pd.DataFrame) -> Tuple[pd.Series, float]:
    """
    OLS with intercept using numpy lstsq.
    Returns: (coefficients including intercept as 'alpha', r2)
    """
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if df.shape[0] < max(30, X.shape[1] * 15):
        raise ValueError("Not enough overlapping data points for a stable regression.")

    yv = df["y"].to_numpy().reshape(-1, 1)
    Xv = df[X.columns].to_numpy()
    Xv = np.column_stack([np.ones(len(df)), Xv])  # intercept

    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    beta = beta.flatten()

    yhat = Xv @ beta
    ss_res = float(np.sum((df["y"].to_numpy() - yhat) ** 2))
    ss_tot = float(np.sum((df["y"].to_numpy() - df["y"].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    idx = ["alpha"] + list(X.columns)
    return pd.Series(beta, index=idx), r2


def _rolling_betas(y: pd.Series, X: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling OLS betas (with intercept) using a simple loop.
    Returns dataframe indexed by date with columns: alpha + X cols
    """
    df = pd.concat([y.rename("y"), X], axis=1).dropna().reset_index(drop=False)
    if "date" not in df.columns:
        # if the index was date, recover
        df = df.rename(columns={df.columns[0]: "date"})

    if df.shape[0] < window + 5:
        return pd.DataFrame(columns=["date", "alpha"] + list(X.columns))

    out_rows = []
    cols = list(X.columns)

    for i in range(window, len(df) + 1):
        w = df.iloc[i - window : i].copy()
        yv = w["y"].to_numpy().reshape(-1, 1)
        Xv = w[cols].to_numpy()
        Xv = np.column_stack([np.ones(len(w)), Xv])

        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        beta = beta.flatten()

        out_rows.append([w["date"].iloc[-1]] + beta.tolist())

    return pd.DataFrame(out_rows, columns=["date", "alpha"] + cols)


def render_driver_attribution_tab(data: pd.DataFrame):
    st.subheader("Driver Attribution")

    cpi, dxy, real = load_macro_series()

    if dxy is None and cpi is None and real is None:
        st.info("No macro series found (CPI/DXY/Real Rate). Add the CSVs in /data to enable attribution.")
        return

    gold_currencies = [c for c in data.columns if c != "date"]
    colA, colB, colC, colD = st.columns([1.2, 1.1, 1.1, 1.3])

    with colA:
        currency = st.selectbox("Gold currency", gold_currencies, index=0, key="attr_currency")
    with colB:
        freq = st.selectbox("Frequency", ["Daily", "Monthly"], index=1, key="attr_freq")
    with colC:
        ret_type = st.selectbox("Gold return", ["Log return", "Pct return"], index=0, key="attr_ret")
    with colD:
        roll_win = st.selectbox("Rolling beta window", [60, 126, 252], index=1, key="attr_roll")

    # Window controls
    st.markdown("#### Window")
    min_dt = data["date"].min().date()
    max_dt = data["date"].max().date()

    preset = st.radio(
    "Period",
    ["MAX", "YTD", "1Y", "5Y", "10Y", "Custom"],
    horizontal=True,
    index=3,
    key="attr_period_radio"
)
    today = max_dt

    if preset == "MAX":
        start_dt, end_dt = min_dt, max_dt
    elif preset == "YTD":
        start_dt, end_dt = date(today.year, 1, 1), today
    elif preset == "1Y":
        start_dt, end_dt = today - timedelta(days=365), today
    elif preset == "5Y":
        start_dt, end_dt = today - timedelta(days=365 * 5), today
    elif preset == "10Y":
        start_dt, end_dt = today - timedelta(days=365 * 10), today
    else:
        start_dt, end_dt = today - timedelta(days=365 * 5), today

    if preset == "Custom":
        start_dt, end_dt = st.date_input("Custom date range", value=(start_dt, end_dt), min_value=min_dt, max_value=max_dt)
        if isinstance(start_dt, (list, tuple)):
            start_dt, end_dt = start_dt

    # Build gold series
    gold = data[["date", currency]].rename(columns={currency: "gold"}).dropna().sort_values("date")

    # Macro series: standardize column names to avoid conflicts
    series = []
    if dxy is not None:
        series.append(dxy.rename(columns={"value": "dxy"}))
    if real is not None:
        series.append(real.rename(columns={"value": "real"}))
    if cpi is not None:
        series.append(cpi.rename(columns={"value": "cpi"}))

    # Align by date
    merged = join_on_date(gold, *series).sort_values("date").reset_index(drop=True)

    # Restrict window
    mask = (merged["date"].dt.date >= start_dt) & (merged["date"].dt.date <= end_dt)
    mwin = merged.loc[mask].copy().reset_index(drop=True)

    if mwin.empty:
        st.warning("No overlapping data in this window.")
        return

    # Frequency
    if freq == "Monthly":
        # month-end everything (gold + macro)
        m = mwin.copy()
        # separate to reuse helper
        gold_m = _to_month_end(m[["date", "gold"]].rename(columns={"gold": "value"}), "value").rename(columns={"value": "gold"})
        parts = [gold_m]

        if "dxy" in m.columns:
            parts.append(_to_month_end(m[["date", "dxy"]].rename(columns={"dxy": "value"}), "value").rename(columns={"value": "dxy"}))
        if "real" in m.columns:
            parts.append(_to_month_end(m[["date", "real"]].rename(columns={"real": "value"}), "value").rename(columns={"value": "real"}))
        if "cpi" in m.columns:
            parts.append(_to_month_end(m[["date", "cpi"]].rename(columns={"cpi": "value"}), "value").rename(columns={"value": "cpi"}))

        mwin = join_on_date(*parts).sort_values("date").reset_index(drop=True)

    # Returns/changes (features)
    if ret_type == "Log return":
        y = _log_ret(mwin["gold"]).rename("GoldRet")
    else:
        y = _pct_ret(mwin["gold"]).rename("GoldRet")

    X = pd.DataFrame(index=mwin.index)
    X["date"] = mwin["date"]

    if "dxy" in mwin.columns:
        # DXY return (log)
        X["DXYRet"] = _log_ret(mwin["dxy"])
    if "cpi" in mwin.columns:
        # CPI: percent change (monthly if resampled; daily CPI usually sparse anyway)
        X["CPIRet"] = _pct_ret(mwin["cpi"])
    if "real" in mwin.columns:
        # Real rate: change in level (basis points-ish if series is in %)
        X["RealChg"] = mwin["real"].astype(float).diff()

    # Drop date into separate df for regression
    Xr = X.drop(columns=["date"])

    # If not enough factors, stop
    if Xr.shape[1] == 0:
        st.info("No usable macro factors found in the selected window.")
        return

    # Fit OLS + rolling betas
    try:
        betas, r2 = _ols(y, Xr)
    except Exception as e:
        st.error(f"Regression failed: {e}")
        st.stop()

    # Contributions over the full window (factor move * beta)
    df_reg = pd.concat([mwin[["date"]], y, Xr], axis=1).dropna().reset_index(drop=True)
    if df_reg.empty:
        st.warning("Not enough overlapping return data after transformations.")
        return

    # For contributions, use cumulative factor move over window
    contrib = {}
    for col in Xr.columns:
        factor_move = float(df_reg[col].sum())  # sum of (log) returns or diffs over window
        contrib[col] = float(betas[col] * factor_move)

    explained = float(sum(contrib.values()))
    actual = float(df_reg["GoldRet"].sum())
    residual = actual - (betas["alpha"] * len(df_reg) + explained)

    # --- Display: Sensitivity table ---
    st.markdown("### Sensitivities (OLS betas)")
    pretty = betas.copy()

    # Rename to readable
    rename = {
        "alpha": "Alpha (intercept)",
        "DXYRet": "DXY (return)",
        "CPIRet": "CPI (change)",
        "RealChg": "Real rate (level change)",
    }
    pretty.index = [rename.get(i, i) for i in pretty.index]

    c1, c2, c3 = st.columns([1.6, 1.0, 1.2])
    with c1:
        st.dataframe(pretty.to_frame("Beta"), use_container_width=True)
    with c2:
        st.metric("R²", f"{r2:.3f}")
    with c3:
        st.caption(
            "Interpretation: sign shows direction; magnitude shows sensitivity. "
            "DXY and CPI use returns; Real uses level change."
        )

    # --- Contribution bars ---
    st.markdown("### Driver contribution over selected window")
    labels = []
    values = []
    for k, v in contrib.items():
        labels.append(rename.get(k, k))
        values.append(v)

    labels += ["Residual"]
    values += [residual]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=labels, y=values, name="Contribution"))

    fig_bar.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=GOLD),
        xaxis=dict(
            title="Driver",
            title_font=dict(color=GOLD),
            tickfont=dict(color=GOLD),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Contribution to Gold return (same units as GoldRet sum)",
            title_font=dict(color=GOLD),
            tickfont=dict(color=GOLD),
            gridcolor="rgba(255,255,255,0.08)",
        ),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        "Contributions are approximate: beta × (factor move over window). "
        "Residual captures what the factors didn’t explain."
    )

