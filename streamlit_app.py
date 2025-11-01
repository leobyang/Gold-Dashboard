# streamlit_app.py
# ------------------------------------------------------------
# Gold Dashboard — Tab 1 (Gold Prices)
# ------------------------------------------------------------
# Features
# - Data source: paste Goldhub CSV URL or upload CSV
# - Period presets + custom date range picker
# - Interactive chart (Plotly) with hover tooltips and range slider
# - Point-in-time price lookup (date inspector)
# - Max drawdown for selected window (level + dates)
# - Bull/Bear/Correction regime detection with adjustable thresholds
# - Summary table of regimes inside the selected window
# ------------------------------------------------------------

import io
import math
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------------
# Page / layout
# -------------------------
st.set_page_config(
    page_title="Gold Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏅 Gold Dashboard")
st.caption("Tab 1 · Gold prices, drawdowns, and bull/bear regimes (1969–present)")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    return df

@st.cache_data(show_spinner=True)
def _fetch_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def normalize_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce a variety of Goldhub CSV layouts into [date, price] cols.
    Expected outputs:
      - date: datetime64[ns]
      - price: float (USD/oz or whatever the CSV provides)
    We try common column names from Goldhub and generic CSVs.
    """
    # Trim whitespace in headers
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Candidate date columns
    date_cols = [
        c for c in df.columns
        if str(c).lower() in {"date", "time", "period"} or "date" in str(c).lower()
    ]
    price_cols = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ["usd", "price", "value", "close", "gold"])
        and str(c).lower() not in {"date", "time", "period"}
    ]

    # Fallback: if two columns only, assume first is date, second is price
    if not date_cols and not price_cols and df.shape[1] >= 2:
        date_cols = [df.columns[0]]
        price_cols = [df.columns[1]]

    # Pick first candidate
    date_col = date_cols[0]
    # Prefer a USD-like column if available
    if price_cols:
        # heuristic: prefer columns with 'usd'
        price_cols_sorted = sorted(price_cols, key=lambda c: ("usd" not in str(c).lower(), str(c)))
        price_col = price_cols_sorted[0]
    else:
        # if no explicit price col, try the 2nd column
        price_col = df.columns[1]

    out = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "price"}).copy()

    # Parse date
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).sort_values("date")

    # Price to numeric
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"])

    # Drop duplicates and ensure daily (no need to resample by default)
    out = out.drop_duplicates(subset=["date"]).reset_index(drop=True)

    return out


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling peak, drawdown series, and peak/trough markers."""
    s = df["price"].astype(float)
    rolling_peak = s.cummax()
    drawdown = s / rolling_peak - 1.0
    out = df.copy()
    out["rolling_peak"] = rolling_peak
    out["drawdown"] = drawdown
    return out


def window_stats(df: pd.DataFrame) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return max drawdown (as -x.xx), peak date, trough date for the window."""
    if df.empty:
        return float("nan"), None, None
    s = df["price"].astype(float)
    roll = s.cummax()
    dd = s / roll - 1.0
    mdd = dd.min()
    trough_idx = dd.idxmin() if not dd.isna().all() else None
    peak_idx = None
    if trough_idx is not None:
        # peak is the last time the rolling max was set before trough
        roll_eq = roll.loc[:trough_idx]
        peak_val = roll_eq.max()
        # get first occurrence of that peak
        peak_idx = roll_eq[roll_eq == peak_val].index[0]
    peak_date = df.loc[peak_idx, "date"] if peak_idx is not None else None
    trough_date = df.loc[trough_idx, "date"] if trough_idx is not None else None
    return float(mdd), peak_date, trough_date


def detect_regimes(df: pd.DataFrame, bear_threshold: float = -0.20, correction_threshold: float = -0.10) -> pd.DataFrame:
    """Label each row as Bull / Correction / Bear based on drawdown from the most recent peak.
    - drawdown ≤ -20% → Bear
    - -20% < drawdown ≤ -10% → Correction
    - drawdown > -10% → Bull
    Returns df with a 'regime' column and compacted segments table.
    """
    x = compute_drawdown(df)
    dd = x["drawdown"].values
    labels = np.where(dd <= bear_threshold, "Bear",
              np.where(dd <= correction_threshold, "Correction", "Bull"))
    x["regime"] = labels

    # Build segments (start when label changes)
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
# Sidebar — Data source
# -------------------------
with st.sidebar:
    st.header("Data source")
    st.markdown(
        "Paste a **Goldhub CSV URL** (e.g., from the *Download data* button on their page),\n"
        "or upload your own CSV with columns like `Date, USD (AM)` or `date, price`."
    )
    default_url = ""
    csv_url = st.text_input("Gold price CSV URL (optional)", value=default_url, placeholder="https://.../gold-prices.csv")
    uploaded = st.file_uploader("...or upload a CSV", type=["csv"])\

    st.divider()
    st.subheader("Regime thresholds")
    col_a, col_b = st.columns(2)
    with col_a:
        correction_thr = st.number_input("Correction ≤", value=-0.10, step=0.01, format="%.2f")
    with col_b:
        bear_thr = st.number_input("Bear ≤", value=-0.20, step=0.01, format="%.2f")

    st.caption("Thresholds are applied to drawdown from the most recent peak in the selected window.")

# -------------------------
# Load data
# -------------------------
raw_df: Optional[pd.DataFrame] = None
err = None
if uploaded is not None:
    try:
        raw_df = _read_csv_from_bytes(uploaded.getvalue())
    except Exception as e:
        err = f"Failed to read uploaded CSV: {e}"
elif csv_url:
    try:
        raw_df = _fetch_csv(csv_url)
    except Exception as e:
        err = f"Failed to fetch CSV: {e}"

if err:
    st.error(err)

if raw_df is None:
    st.info("Provide a CSV URL from Goldhub or upload a CSV to begin.")
    st.stop()

# Normalize
try:
    gold = normalize_gold_df(raw_df)
except Exception as e:
    st.error(f"Could not normalize the CSV (expecting date/price columns). Error: {e}")
    st.stop()

if gold.empty:
    st.warning("No rows after parsing the CSV. Check the column names and data.")
    st.stop()

min_dt = gold["date"].min().date()
max_dt = gold["date"].max().date()

# -------------------------
# Tabs
# -------------------------
tab1, = st.tabs(["Gold Prices"])

with tab1:
    st.subheader("Price Explorer")

    # Presets
    preset = st.segmented_control(
        "Period",
        options=["MAX", "YTD", "1Y", "5Y", "10Y", "Custom"],
        default="5Y",
    )

    # Default window
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
        start_dt, end_dt = gold["date"].quantile(0.6).date(), today

    if preset == "Custom":
        start_dt, end_dt = st.date_input(
            "Custom date range",
            value=(start_dt, end_dt),
            min_value=min_dt,
            max_value=max_dt,
        )
        if isinstance(start_dt, (list, tuple)):
            start_dt, end_dt = start_dt

    mask = (gold["date"].dt.date >= start_dt) & (gold["date"].dt.date <= end_dt)
    win = gold.loc[mask].reset_index(drop=True)

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
        name="Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))

    # Shade Bear regimes
    # Build segments inside window where regime == 'Bear'
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
        yaxis_title="Price",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------- Metrics row --------------
    mdd, peak_dt, trough_dt = window_stats(win)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max drawdown", human_pct(mdd))
    col2.metric("Peak before MDD", peak_dt.strftime("%Y-%m-%d") if peak_dt is not None else "—")
    col3.metric("Trough of MDD", trough_dt.strftime("%Y-%m-%d") if trough_dt is not None else "—")
    if peak_dt is not None and trough_dt is not None:
        col4.metric("Peak→Trough days", (trough_dt.date() - peak_dt.date()).days)
    else:
        col4.metric("Peak→Trough days", "—")

    st.divider()

    # ------------- Price inspector -----------
    st.markdown("### 🔎 Point‑in‑time price")
    inspect_dt = st.date_input("Pick a date to inspect", value=end_dt, min_value=start_dt, max_value=end_dt)

    # nearest date (forward/backward)
    sidx = win["date"].searchsorted(pd.to_datetime(inspect_dt))
    candidates = []
    if 0 <= sidx < len(win):
        candidates.append((abs((win.loc[sidx, "date"].date() - inspect_dt).days), sidx))
    if sidx - 1 >= 0:
        candidates.append((abs((win.loc[sidx-1, "date"].date() - inspect_dt).days), sidx-1))

    if candidates:
        _, best_idx = sorted(candidates)[0]
        row = win.loc[best_idx]
        st.info(f"**{row['date'].date()}** · ${row['price']:.2f}")

    # ------------- Regime table --------------
    st.markdown("### 📈 Regimes in window")
    if not seg_df.empty:
        pretty = seg_df.copy()
        pretty["Return"] = pretty["Return"].map(human_pct)
        pretty["Max Drawdown in Segment"] = pretty["Max Drawdown in Segment"].map(human_pct)
        st.dataframe(pretty, use_container_width=True, hide_index=True)
    else:
        st.write("No regimes identified in this window.")

    st.caption(
        "Shaded red bands indicate **Bear** periods using your thresholds. "
        "You can tune these in the left sidebar."
    )

# -------------------------
# Notes
# -------------------------
# - To get the Goldhub CSV, visit gold.org → Goldhub → Data → Gold prices, click **Download data**.
# - If their CSV schema differs, the normalizer tries to detect `date` and a price-like column automatically.
# - You can swap Plotly for Altair if preferred. Plotly is used here for the range slider and rich hover.
