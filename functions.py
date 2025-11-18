from pathlib import Path
from datetime import date, timedelta
from typing import Optional, Tuple, List
import os
import math

import numpy as np
import pandas as pd
import streamlit as st

DATA_FILE = Path("./data/gold_prices.csv")
CPI_FILE  = Path("./data/cpi.csv")
DXY_FILE  = Path("./data/dxy.csv")
REAL_FILE = Path("./data/interest.csv")


def _find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"date", "time", "period"} or "date" in lc:
            return c
    probe = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    if probe.notna().sum() > 0:
        return df.columns[0]
    raise ValueError("No date-like column found.")


def normalize_multi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def _find_date_col_(_df: pd.DataFrame) -> str:
        for c in _df.columns:
            lc = str(c).lower()
            if lc in {"date", "time", "period"} or "date" in lc:
                return c
        probe = pd.to_datetime(_df.iloc[:, 0], errors="coerce")
        if probe.notna().sum() > 0:
            return _df.columns[0]
        raise ValueError("No date-like column found.")

    date_col = _find_date_col_(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    value_cols = [c for c in df.columns if c != date_col]
    for c in value_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        )

    out = df[[date_col] + value_cols].drop_duplicates(subset=[date_col]).reset_index(drop=True)
    out = out.rename(columns={date_col: "date"})
    return out


def normalize_single(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _find_date_col(df)
    cand = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not cand:
        raise ValueError("No numeric 'value' column found in single-series CSV.")
    val_col = cand[0]
    out = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)
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


def detect_regimes_coarse(df: pd.DataFrame, bear_threshold: float = -0.20, min_days: int = 60):
    x = compute_drawdown(df)
    dd = x["drawdown"].values
    labels = np.where(dd <= bear_threshold, "Bear", "Bull")
    x["regime"] = labels

    segs = []
    start_idx = 0
    for i in range(1, len(x)):
        if x.loc[i, "regime"] != x.loc[i-1, "regime"]:
            segs.append((start_idx, i-1))
            start_idx = i
    if len(x) > 0:
        segs.append((start_idx, len(x)-1))

    merged = []
    for a, b in segs:
        merged.append([a, b])

    i = 0
    while i < len(merged):
        a, b = merged[i]
        days = (x.loc[b, "date"].date() - x.loc[a, "date"].date()).days + 1
        if days < min_days:
            if i == 0 and len(merged) > 1:
                merged[i+1][0] = a
                merged.pop(i)
                continue
            elif i > 0:
                merged[i-1][1] = b
                merged.pop(i)
                i -= 1
                continue
        i += 1

    records = []
    reg_series = np.array([""] * len(x), dtype=object)
    for a, b in merged:
        reg = x.loc[a, "regime"]
        reg_series[a:b+1] = reg
        start, end = x.loc[a, "date"], x.loc[b, "date"]
        p0, p1 = x.loc[a, "price"], x.loc[b, "price"]
        ret = (p1 / p0 - 1.0) if p0 and not math.isclose(p0, 0.0) else np.nan
        depth = x.loc[a:b, "drawdown"].min()
        records.append({
            "Start": start.date(),
            "End": end.date(),
            "Days": (end - start).days + 1,
            "Regime": reg,
            "Return": ret,
            "Max Drawdown in Segment": depth,
        })
    x["regime"] = reg_series
    seg_df = pd.DataFrame.from_records(records)
    return x, seg_df


def human_pct(x: float) -> str:
    return ("{:+.2f}%".format(100 * x)) if pd.notna(x) else "—"


def yoy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["yoy"] = out["value"].pct_change(12)
    return out


def join_on_date(*dfs: pd.DataFrame) -> pd.DataFrame:
    out = None
    for d in dfs:
        out = d if out is None else out.merge(d, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def file_sig(p: Path) -> str:
    stt = os.stat(p)
    return f"{stt.st_mtime_ns}-{stt.st_size}"


@st.cache_data(show_spinner=True)
def load_csv_cached(path: Path, sig: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=True)
def read_and_normalize_single(path: Path, sig: str) -> Optional[pd.DataFrame]:
    df = pd.read_csv(path)
    return normalize_single(df)


def safe_read_single(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        st.warning(f"Missing `{label}` file: {path.name}")
        return None
    try:
        return read_and_normalize_single(path, file_sig(path))
    except Exception as e:
        st.error(f"Failed to read `{path.name}`: {e}")
        return None


def load_main_gold() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(f"Local CSV not found: `{DATA_FILE.name}`. Put it next to this file or update DATA_FILE.")
        st.stop()
    try:
        raw_df = load_csv_cached(DATA_FILE, file_sig(DATA_FILE))
        data = normalize_multi(raw_df)
    except Exception as e:
        st.error(f"Failed to read/normalize `{DATA_FILE.name}`: {e}")
        st.stop()
    return data


def load_macro_series():
    cpi  = safe_read_single(CPI_FILE,  "CPI")
    dxy  = safe_read_single(DXY_FILE,  "Dollar Index")
    real = safe_read_single(REAL_FILE, "Real Interest Rate")
    return cpi, dxy, real
