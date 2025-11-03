# streamlit_app.py
# ------------------------------------------------------------
# Gold Dashboard — Tab 1 (Gold Prices)  [No sidebar, local CSV, currency picker]
# ------------------------------------------------------------

import io
import math
from datetime import date, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# -------------------------
# Page / layout
# -------------------------
st.set_page_config(page_title="Gold Dashboard", layout="centered")
st.title("🏅 Gold Dashboard")

# Local CSV path (change if you use a different filename)
DATA_FILE = Path("./gold_prices.csv")

# -------------------------
# Helpers
# -------------------------
def normalize_multi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a variety of CSV layouts into:
      - 'date' column (datetime)
      - 1+ numeric columns (treated as currencies/price series)
    Keeps ALL numeric columns as candidates for currency selection.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Find date-like column
    date_col = None
    for c in df.columns:
        if c.lower() in {"date", "time", "period"} or "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        # as a fallback, if first column looks like dates when parsed
        try:
            probe = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            if probe.notna().sum() > 0:
                date_col = df.columns[0]
        except Exception:
            pass
    if date_col is None:
        raise ValueError("No date-like column found.")

    # Parse date and clean
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Keep numeric columns as currency options (exclude the date col)
    value_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("No numeric columns found for prices/currencies.")

    out = df[[date_col] + value_cols].dropna().drop_duplicates(subset=[date_col]).reset_index(drop=True)
    out = out.rename(columns={date_col: "date"})
    return out

def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    s = df["price"].astype(float)
    rolling_peak = s.cummax()
    drawdown = s / rolling_peak - 1.0
    out = df.copy()
    out["rolling_peak"] = rolling_peak
    out["drawdown"] = drawdown
    return out


def window_stats(df: pd.DataFrame):
    """
    Returns:
      mdd_frac (negative, e.g. -0.32),
      peak_date, trough_date,
      peak_price, trough_price,
      mdd_abs (positive number = peak_price - trough_price)
    """
    if df.empty:
        return float("nan"), None, None, None, None, None

    s = df["price"].astype(float)
    roll = s.cummax()
    dd = s / roll - 1.0

    mdd = float(dd.min())
    trough_idx = int(dd.idxmin()) if not dd.isna().all() else None
    peak_idx = None
    if trough_idx is not None:
        roll_eq = roll.loc[:trough_idx]
        peak_val = roll_eq.max()
        peak_idx = int(roll_eq[roll_eq == peak_val].index[0])

    peak_date = df.loc[peak_idx, "date"] if peak_idx is not None else None
    trough_date = df.loc[trough_idx, "date"] if trough_idx is not None else None
    peak_price = float(df.loc[peak_idx, "price"]) if peak_idx is not None else None
    trough_price = float(df.loc[trough_idx, "price"]) if trough_idx is not None else None
    mdd_abs = (peak_price - trough_price) if (peak_price is not None and trough_price is not None) else None

    return mdd, peak_date, trough_date, peak_price, trough_price, mdd_abs


def detect_regimes(df: pd.DataFrame, bear_threshold: float = -0.20, correction_threshold: float = -0.10) -> pd.DataFrame:
    """
    Label rows as Bull / Correction / Bear by drawdown from most recent peak.
      drawdown ≤ bear_threshold       -> Bear
      correction_threshold < drawdown -> Bull
      else                            -> Correction
    Returns (labeled_df, segments_table).
    """
    x = compute_drawdown(df)
    dd = x["drawdown"].values
    labels = np.where(dd <= bear_threshold, "Bear",
              np.where(dd <= correction_threshold, "Correction", "Bull"))
    x["regime"] = labels

    # Compact segments where label is constant
    segs = []
    start_idx = 0
    for i in range(1, len(x)):
        if x.loc[i, "regime"] != x.loc[i-1, "regime"]:
            segs.append((start_idx, i-1))
            start_idx = i
    if len(x) > 0:
        segs.append((start_idx, len(x)-1))

    records = []
    for a, b in segs:
        start, end = x.loc[a, "date"], x.loc[b, "date"]
        p0, p1 = x.loc[a, "price"], x.loc[b, "price"]
        ret = (p1 / p0 - 1.0) if p0 and not math.isclose(p0, 0.0) else np.nan
        depth = x.loc[a:b, "drawdown"].min()
        records.append({
            "Start": start.date(),
            "End": end.date(),
            "Days": (end - start).days + 1,
            "Regime": x.loc[a, "regime"],
            "Return": ret,
            "Max Drawdown in Segment": depth,
        })

    seg_df = pd.DataFrame.from_records(records)
    return x, seg_df

def human_pct(x: float) -> str:
    return ("{:+.2f}%".format(100 * x)) if pd.notna(x) else "—"

# -------------------------
# Load local CSV
# -------------------------
if not DATA_FILE.exists():
    st.error(f"Local CSV not found: `{DATA_FILE.name}`. Put it next to this file or update DATA_FILE.")
    st.stop()

try:
    raw_df = pd.read_csv(DATA_FILE)
    data = normalize_multi(raw_df)  # -> date + many numeric columns (currencies)
except Exception as e:
    st.error(f"Failed to read/normalize `{DATA_FILE.name}`: {e}")
    st.stop()

min_dt = data["date"].min().date()
max_dt = data["date"].max().date()

# -------------------------
# Controls row (main page)
# -------------------------
st.subheader("Price Explorer")

# Currency options = all numeric columns except 'date'
currency_cols = [c for c in data.columns if c != "date" and pd.api.types.is_numeric_dtype(data[c])]
if not currency_cols:
    st.error("No numeric currency columns found in the CSV.")
    st.stop()

c1, c2, c3 = st.columns([1.3, 1.3, 2.4])
with c1:
    currency = st.selectbox("Currency column", currency_cols, index=0)
with c2:
    # Regime thresholds (moved from sidebar to main page)
    correction_thr = st.number_input("Correction threshold (≤)", value=-0.10, step=0.01, format="%.2f")
    bear_thr = st.number_input("Bear threshold (≤)", value=-0.20, step=0.01, format="%.2f")
with c3:
    # Period presets
    preset = st.radio("Period", ["MAX", "YTD", "1Y", "5Y", "10Y", "Custom"], horizontal=True, index=3)

# Determine window
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
    # sensible default for Custom
    start_dt, end_dt = data["date"].quantile(0.6).date(), today

if preset == "Custom":
    start_dt, end_dt = st.date_input("Custom date range", value=(start_dt, end_dt),
                                     min_value=min_dt, max_value=max_dt)
    if isinstance(start_dt, (list, tuple)):
        start_dt, end_dt = start_dt

# Build a working window with the selected currency
work = data[["date", currency]].rename(columns={currency: "price"})
mask = (work["date"].dt.date >= start_dt) & (work["date"].dt.date <= end_dt)
win = work.loc[mask].reset_index(drop=True)

if win.empty:
    st.warning("No data in the selected range. Adjust your dates.")
    st.stop()

# Drawdown + regimes for the window
dd_df = compute_drawdown(win)
regimes_df, seg_df = detect_regimes(win, bear_threshold=bear_thr, correction_threshold=correction_thr)

# ---------------- Chart -----------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=win["date"], y=win["price"], mode="lines",
    name=f"Price ({currency})",
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
))
# Shade Bear regimes
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
)
st.plotly_chart(fig, use_container_width=True)

# ------------- Metrics row --------------
mdd, peak_dt, trough_dt, peak_px, trough_px, mdd_abs = window_stats(win)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Max drawdown (%)", human_pct(mdd))
col2.metric("Max drawdown (abs)", f"{mdd_abs:,.2f} {currency}" if mdd_abs is not None else "—")
col3.metric("Peak before MDD", peak_dt.strftime("%Y-%m-%d") if peak_dt else "—")
col4.metric("Trough of MDD", trough_dt.strftime("%Y-%m-%d") if trough_dt else "—")
if peak_dt and trough_dt:
    col5.metric("Peak→Trough days", (trough_dt.date() - peak_dt.date()).days)
else:
    col5.metric("Peak→Trough days", "—")


st.divider()

# ------------- Price inspector -----------
st.markdown("### 🔎 Point-in-time price")
inspect_dt = st.date_input("Pick a date to inspect", value=end_dt, min_value=start_dt, max_value=end_dt)
sidx = win["date"].searchsorted(pd.to_datetime(inspect_dt))
candidates = []
if 0 <= sidx < len(win):
    candidates.append((abs((win.loc[sidx, "date"].date() - inspect_dt).days), sidx))
if sidx - 1 >= 0:
    candidates.append((abs((win.loc[sidx-1, "date"].date() - inspect_dt).days), sidx-1))
if candidates:
    _, best_idx = sorted(candidates)[0]
    row = win.loc[best_idx]
    st.info(f"**{row['date'].date()}** · {row['price']:.2f} ({currency})")

# ------------- Regime table --------------
st.markdown("### 📈 Regimes in window")
if not seg_df.empty:
    pretty = seg_df.copy()
    pretty["Return"] = pretty["Return"].map(human_pct)
    pretty["Max Drawdown in Segment"] = pretty["Max Drawdown in Segment"].map(human_pct)
    st.dataframe(pretty, use_container_width=True, hide_index=True)
else:
    st.write("No regimes identified in this window.")

st.caption("Shaded red bands indicate **Bear** periods using the thresholds above.")
